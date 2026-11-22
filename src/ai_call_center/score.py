"""`uv run score-harness`: score the prompts and the word lists against a made-up test set.

A change to a prompt or a word list is good only if a fixed measure says so. This command is that
measure. Run it before the change and after it, and compare the two results.

The test set is the folder `evals/`. Each line in it is made up; no caller's words are in it.

- `evals/live_call.json`: caller lines, each with the answer to "does this caller want a live call?".
  The word list of the transfer check must give that answer. This part makes no model request.
- `evals/calls.json`: made-up calls. Each one has the caller's lines and what the call must do. The
  agents answer the lines in the text mode, then the grader scores the call against its rules. This
  part makes model requests that cost money, so it runs only with `--calls`.

In a made-up call the caller's lines are fixed: the caller does not react to the agent. No email is
sent, no phone rings, and nothing is stored: the lead store is in memory. Jev is off, so the word
list alone decides a transfer. The agents' answers change from run to run, so one check of a made-up
call can pass in one run and fail in the next. The word list part gives the same result each time.

Add a new sample or a new call at the end of its file. Do not change or remove an old one: the result
of each check is compared by its name.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_call_center.agents.specs import DEFAULT_LEGS, SPECIALISTS, Welcome, companies_phrase, greeting_tip, greeting_words, load_profile, web_address
from ai_call_center.analytics import MESSAGE, NO_OUTCOME, OUTCOMES, TRANSFER, harness_version
from ai_call_center.callers import transcript_lines
from ai_call_center.config import Settings, get_settings
from ai_call_center.decision_grader import default_decision_grader
from ai_call_center.grading import FAIL, NA, RULES, Score, grade_call
from ai_call_center.graph.build import build_graph, default_model, default_specs
from ai_call_center.graph.state import CallContext
from ai_call_center.graph.tools import asked_for_the_owner, message_saved
from ai_call_center.leads import LeadStore
from ai_call_center.notify import RecordingNotifier
from ai_call_center.telephony import RecordingTelephony
from ai_call_center.writing import default_library

TEST_SET = Path("evals")
CALLER_NUMBER = "+15555550123"  # made up: 555 numbers are not in use
OWNER_PHONE = "+15555550199"  # made up; a transfer in a made-up call is recorded, never dialed
CALL_CONNECTED = "[The call connects. Greet the caller.]"  # the same first line as chat.py
EXPECT_KEYS = {"visits", "outcome", "forgotten"}


@dataclass(frozen=True)
class LiveCallSample:
    """The end of a call, and if the caller wants a live call. A line with no "agent:" at its start is the caller's."""

    lines: tuple[str, ...]
    wants_live_call: bool


@dataclass(frozen=True)
class MadeUpCall:
    id: str
    about: str
    caller: tuple[str, ...]  # what the caller says, one line for each turn
    visits: tuple[str, ...] = ()  # agents that must take part in the call
    outcome: str = ""  # "message", "transfer" or "none"; empty = not checked
    forgotten: bool | None = None  # True = the caller must be forgotten; None = not checked


@dataclass
class CallRun:
    """What one made-up call did."""

    lines: list[str] = field(default_factory=list)  # "caller: ..." and "agent: ..." lines
    visited: list[str] = field(default_factory=list)
    outcome: str = NO_OUTCOME
    forgotten: bool = False
    error: str = ""  # not empty when the call stopped on an error


def _read(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must hold a list")
    return data


def load_live_call(directory: Path = TEST_SET) -> list[LiveCallSample]:
    samples = []
    for number, item in enumerate(_read(directory / "live_call.json"), 1):
        lines = tuple(str(line) for line in item.get("lines", []))
        if not any(not line.startswith("agent:") for line in lines):
            raise ValueError(f"live_call.json, sample {number}: there is no caller line")
        if not isinstance(item.get("wants_live_call"), bool):
            raise ValueError(f"live_call.json, sample {number}: wants_live_call must be true or false")
        samples.append(LiveCallSample(lines=lines, wants_live_call=item["wants_live_call"]))
    return samples


def load_calls(directory: Path = TEST_SET) -> list[MadeUpCall]:
    calls: list[MadeUpCall] = []
    for item in _read(directory / "calls.json"):
        name = str(item.get("id", "")).strip()
        expect = item.get("expect", {})
        if not name or "/" in name or any(call.id == name for call in calls):
            raise ValueError(f"calls.json: the id {name!r} is empty, holds a '/', or is used two times")
        if not item.get("caller"):
            raise ValueError(f"calls.json, {name}: the caller says nothing")
        if set(expect) - EXPECT_KEYS:
            raise ValueError(f"calls.json, {name}: unknown key in expect: {sorted(set(expect) - EXPECT_KEYS)}")
        if set(expect.get("visits", [])) - set(SPECIALISTS):
            raise ValueError(f"calls.json, {name}: visits must hold names from {SPECIALISTS}")
        if expect.get("outcome", NO_OUTCOME) not in OUTCOMES:
            raise ValueError(f"calls.json, {name}: outcome must be one of {OUTCOMES}")
        calls.append(
            MadeUpCall(
                id=name,
                about=str(item.get("about", "")),
                caller=tuple(str(line) for line in item["caller"]),
                visits=tuple(expect.get("visits", [])),
                outcome=expect.get("outcome", ""),
                forgotten=expect.get("forgotten"),
            )
        )
    return calls


def sample_lines(lines: tuple[str, ...] | list[str], owner_name: str) -> list[str]:
    """A real call starts with the agent's greeting, which names the owner. Without it, "him" has no meaning."""
    greeting = f"agent: This is {owner_name}'s AI business card. How can I help?"
    return [greeting] + [line if line.startswith("agent:") else f"caller: {line}" for line in lines]


def word_list_checks(samples: list[LiveCallSample], owner_name: str) -> dict[str, bool]:
    """One check for each sample: the word list of the transfer check gives the expected answer."""
    return {f"live_call/{number:02d}": asked_for_the_owner(sample_lines(sample.lines, owner_name)) == sample.wants_live_call for number, sample in enumerate(samples, 1)}


async def run_call(call: MadeUpCall, graph: Any, settings: Settings) -> CallRun:
    """Say the caller's lines to the agents, one for each turn, in the text mode. The call stops when
    the lines end, when an agent ends the call, or when an agent transfers it."""
    store = LeadStore(":memory:")
    telephony = RecordingTelephony()
    call_id = f"score-{call.id}-{uuid4().hex[:6]}"
    store.start_call(call_id, CALLER_NUMBER)
    owner = settings.owner_name
    legs = dict(DEFAULT_LEGS)
    writing = bool(default_library().posts)
    ctx = CallContext(
        call_control_id=call_id,
        caller_number=CALLER_NUMBER,
        owner_phone=OWNER_PHONE,
        leads=store,
        telephony=telephony,
        notifier=RecordingNotifier(),
        legs=legs,
        mode="text",
        # One tip for every run: a random tip would change the agents' first answer between two runs.
        welcome=Welcome(owner=owner, companies=companies_phrase(legs), tip=greeting_tip(owner, writing=writing, rng=random.Random(0)), rag=writing),
        owner_name=owner,
        website_spoken=site.spoken if (site := web_address(settings.website_url, settings.brand_name)) else "",
    )
    config = {"configurable": {"thread_id": call_id}}
    hello = greeting_words(owner, settings.brand_name)
    turns = [f'{CALL_CONNECTED[:-1]} Say exactly these words: "{hello}"]', *call.caller]
    result: dict[str, Any] = {}
    error = ""
    try:
        for text in turns:
            result = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config, context=ctx)
            if result.get("transfer_requested") or result.get("call_ended"):
                break
    except Exception as exc:  # noqa: BLE001 - one failed call must not stop the score of the others
        error = f"{type(exc).__name__}: {exc}"
    try:
        return CallRun(
            lines=transcript_lines(result.get("messages", [])),
            visited=["classifier", *result.get("handoffs", [])],
            outcome=TRANSFER if telephony.transfers else MESSAGE if message_saved(ctx) else NO_OUTCOME,
            forgotten=ctx.forgotten,
            error=error,
        )
    finally:
        store.close()


def call_checks(call: MadeUpCall, run: CallRun, scores: list[Score], *, graded: bool) -> dict[str, bool]:
    """The checks of one made-up call: it ran, it did what the test set expects, and each rule of the grader.

    A rule that does not apply to the call gives no check. With `graded`, a model rule with no verdict
    is a failed check: a grader that gave no answer must not make the score better."""
    base = f"call/{call.id}"
    checks = {f"{base}/ran": not run.error}
    for agent in call.visits:
        checks[f"{base}/visits/{agent}"] = agent in run.visited
    if call.outcome:
        checks[f"{base}/outcome"] = run.outcome == call.outcome
    if call.forgotten is not None:
        checks[f"{base}/forgotten"] = run.forgotten == call.forgotten
    got = {score.rule: score.verdict for score in scores}
    for rule in RULES:
        verdict = got.get(rule.key)
        if verdict is None:
            if graded and rule.check and not rule.hears:  # the text mode has no audio, so an audio rule has no verdict
                checks[f"{base}/rule/{rule.key}"] = False
        elif verdict != NA:
            checks[f"{base}/rule/{rule.key}"] = verdict != FAIL
    return checks


async def score_calls(calls: list[MadeUpCall], settings: Settings, *, model: BaseChatModel, grader: BaseChatModel | None, decider: Any = None, show: bool = False) -> dict[str, bool]:
    """Run each made-up call and grade it. These are model requests that cost money."""
    graph = build_graph(model, default_specs(), checkpointer=InMemorySaver())
    profile = load_profile(settings.profile_path).full
    checks: dict[str, bool] = {}
    for call in calls:
        run = await run_call(call, graph, settings)
        scores = await grade_call(run.lines, visited=run.visited, profile=profile, owner=settings.owner_name, model=grader, decider=decider)
        checks.update(call_checks(call, run, scores, graded=grader is not None or decider is not None))
        if show:
            print(f"\n--- {call.id}: {call.about}")
            print("\n".join(run.lines))
            print(f"[agents: {', '.join(run.visited)}; outcome: {run.outcome}; forgotten: {run.forgotten}{'; error: ' + run.error if run.error else ''}]")
            for score in scores:
                if score.verdict == FAIL:
                    print(f"[failed {score.rule}: {score.note}]")
    return checks


def result_of(checks: dict[str, bool], *, version: str, calls: bool) -> dict[str, Any]:
    return {"version": version, "calls": calls, "passed": sum(checks.values()), "total": len(checks), "checks": dict(sorted(checks.items()))}


def compare(old: dict[str, Any], new: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(the checks that passed in `old` and fail in `new`, the checks that failed in `old` and pass in `new`).
    A check that is in only one of the two results is not compared."""
    before, after = old.get("checks", {}), new.get("checks", {})
    both = sorted(set(before) & set(after))
    return [name for name in both if before[name] and not after[name]], [name for name in both if not before[name] and after[name]]


def describe(result: dict[str, Any]) -> str:
    failed = [name for name, ok in result["checks"].items() if not ok]
    text = f"{result['passed']} of {result['total']} checks passed (version {result['version']})"
    return text + ("\nfailed:\n" + "\n".join(f"  {name}" for name in failed) if failed else "")


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    directory = Path(args.set)
    checks = word_list_checks(load_live_call(directory), settings.owner_name)
    if args.calls:
        calls = [call for call in load_calls(directory) if not args.only or call.id in args.only]
        calls = calls[: args.limit] if args.limit else calls
        grader = init_chat_model(settings.grader_model) if settings.grader_model else None
        # A made-up call has no audio, so the grader model is first here: on a transcript it found 3 of 3 planted
        # failures, and the decision grader 2 of 3 (`uv run check-grader`, 2026-11-21). With no grader model, the decision grader reads.
        decider = default_decision_grader(settings) if grader is None else None
        turns = sum(len(call.caller) + 1 for call in calls)
        print(f"{len(calls)} made-up call(s): {turns} turns for {settings.llm_model}, and {len(calls) if grader or decider else 0} request(s) for the grader. These cost money.", file=sys.stderr)
        if grader is None and decider is None:
            print("GRADER_MODEL is empty and the decision grader is off, so only the code rules score the calls.", file=sys.stderr)
        checks.update(await score_calls(calls, settings, model=default_model(), grader=grader, decider=decider, show=args.show))
    result = result_of(checks, version=harness_version(settings.harness_version), calls=args.calls)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2) if args.json else describe(result))
    if not args.compare:
        return 0
    old = json.loads(Path(args.compare).read_text(encoding="utf-8"))
    worse, better = compare(old, result)
    print(f"\ncompared with version {old.get('version', '?')}: {len(better)} check(s) now pass, {len(worse)} check(s) now fail", file=sys.stderr)
    for name in better:
        print(f"  now passes: {name}", file=sys.stderr)
    for name in worse:
        print(f"  now fails:  {name}", file=sys.stderr)
    return 1 if worse else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="score-harness", description="Score the prompts and the word lists against the made-up test set in evals/.")
    parser.add_argument("--calls", action="store_true", help="also run the made-up calls; these are model requests that cost money")
    parser.add_argument("--only", action="append", default=[], metavar="ID", help="with --calls: run only the call with this id (can be given more than one time)")
    parser.add_argument("--limit", type=int, default=0, metavar="N", help="with --calls: run only the first N calls")
    parser.add_argument("--show", action="store_true", help="with --calls: print each made-up call and its failed rules")
    parser.add_argument("--set", default=str(TEST_SET), metavar="DIR", help="the folder of the test set (default: evals)")
    parser.add_argument("--out", metavar="FILE", help="write the result to this JSON file")
    parser.add_argument("--compare", metavar="FILE", help="compare with an earlier result; exit 1 when a check that passed there fails now")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    os.environ["LANGSMITH_TRACING"] = "false"  # made-up calls stay out of the traces of real calls
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
