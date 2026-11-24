"""`uv run check-grader`: send one made-up call to the graders, to check the keys, the questions, and the verdicts.

The call below has three failures on purpose, and one rule that must pass. The run prints each rule
with the verdict of each grader and the verdict it must get. Run it after you change a rule,
GRADER_MODEL, or GRADER_DECISION_MODEL.

- The decision grader gets one request. The made-up call has no audio, so the model reads the transcript.
- The grader model gets one request.
- `--audio FILE` sends a WAV file with the transcript to the decision grader, so it hears the call.
  The sample transcript does not fit your file: read only the audio rules and the readings of that run.
- `--only decision` or `--only model` asks one grader.
- `--set` sends each made-up call of `evals/grader_calls.json`, and counts the failures that a grader found
  and its false alarms. Run it before you trust a change to a question, a rule, or the limit GRADER_FAIL_AT.
- `--min-found N` and `--max-false-alarms N` give `--set` a floor. The exit code is then 1 only below the floor.
  Without them, each miss and each false alarm gives 1. The weekly check (grader-check.yml) uses the floor.

The transcript is made up; no caller's words are sent. A file that you name with `--audio` is sent to Google.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from langchain.chat_models import init_chat_model

from ai_call_center.agents.specs import load_profile
from ai_call_center.config import get_settings
from ai_call_center.decision_grader import default_decision_grader, fail_probability, feedback_chances
from ai_call_center.grading import FAIL, NA, PASS, READINGS, RULES, Score, grade, model_rules
from ai_call_center.voice.call_audio import read_wav

VISITED = ["classifier", "profile", "contact"]
GRADER_CALLS = Path("evals") / "grader_calls.json"  # made-up calls with known verdicts, for `--set`

# {owner} is the owner's name. The medal is made up, so no profile holds it.
SAMPLE = """agent: This is {owner}'s AI business card. I remember callers, so first: who am I speaking with?
caller: This is Dana.
agent: Thanks, Dana. Do you have a project or a role in mind, or would you like to look around first?
caller: I run a dental clinic and I need a booking platform. Has he built something like that?
agent: Let me bring in the right agent for that.
agent: Hi, I'm the profile agent, running on Google Gemini. He has built platforms like that. He also won an Olympic medal in rowing in 2012.
caller: Okay. What does he charge?
agent: I do not know his rates. The contact agent can pass that question on to him.
caller: Sure, do that.
agent: Let me bring in the right agent for that.
agent: Hi, I'm the contact agent, running on OpenAI. Who should I say is calling?
caller: Dana, like I said.
agent: Thanks. And what is the project about?
caller: The booking platform for my clinic. I told the other agent.
agent: So you want {owner} to call you back about a booking platform for your dental clinic, and about his rates. Is that right?
caller: Yes."""

# What each rule must get. The code rules are here too: they need no model, and a change to them shows.
# An audio rule has no expected verdict: the made-up call has no audio.
EXPECTED = {
    "tool_name_not_spoken": PASS,
    "handoff_intro": PASS,
    "summary_first_try": PASS,
    "name_asked_once": FAIL,  # the contact agent asked for the name again
    "nothing_asked_again": FAIL,  # the contact agent asked what the project is
    "question_answered": PASS,  # "I do not know his rates" is an honest answer
    "caller_memory_right": NA,  # no agent speaks about an earlier call
    "facts_from_profile": FAIL,  # the medal
}
# For each planted failure: the agent at fault and the cause code that a grader must give.
EXPECTED_FAULT = {
    "name_asked_once": ("contact", "asked_again_after_handoff"),
    "nothing_asked_again": ("contact", "asked_again_after_handoff"),
    "facts_from_profile": ("profile", "invented_fact"),
}


def _row(rule_key: str, score: Score | None, extra: str = "") -> tuple[str, bool, bool]:
    """One printed line for one rule, and: is the verdict missing, is it wrong."""
    expected = EXPECTED.get(rule_key, "")
    if score is None:
        return f"  {rule_key:<22}{'-':<6}{expected:<9}{'NONE' if expected else '':<6}{extra}", bool(expected), False
    wrong = bool(expected) and score.verdict != expected
    fault = ""
    if score.verdict == FAIL:
        fault = f"{score.agent or '?'} / {score.cause}"
        if rule_key in EXPECTED_FAULT and (score.agent, score.cause) != EXPECTED_FAULT[rule_key]:
            fault += f"  (expected {' / '.join(EXPECTED_FAULT[rule_key])})"
    mark = "" if not expected else "WRONG" if wrong else "ok"
    return f"  {rule_key:<22}{score.verdict:<6}{expected:<9}{mark:<6}{extra}{fault}", False, wrong


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    owner = settings.owner_name
    lines = SAMPLE.format(owner=owner).split("\n")
    profile = load_profile(settings.profile_path).full
    audio = read_wav(Path(args.audio).read_bytes()) if args.audio else None
    decider = default_decision_grader(settings) if args.only != "model" else None
    model = init_chat_model(settings.grader_model) if settings.grader_model and args.only != "decision" and not args.audio else None
    if decider is None and args.only != "model":
        print("The decision grader is off: GOOGLE_API_KEY or GRADER_DECISION_MODEL is empty in .env.\n")
    if model is None and args.only != "decision" and not args.audio:
        print("GRADER_MODEL is empty in .env, so no text model reads a call.\n")
    bad = 0
    if decider is not None:
        graded = await grade(lines, visited=VISITED, summary_tries=1, profile=profile, owner=owner, decider=decider, audio=audio)
        got = {score.rule: score for score in graded.scores}
        print(f"The decision grader: {graded.by or 'no answer (see the log line above)'}")
        print(f"  {'rule':<22}{'got':<6}{'expected':<9}{'':<6}{'probability':<13}agent / cause")
        for rule in RULES:
            if rule.hears and audio is None:
                continue
            top = fail_probability(rule, decider.last_answers, decider.last_lines) if rule.check else None  # the probability of the most probable cause
            line, missing, wrong = _row(rule.key, got.get(rule.key), f"{f'{top[1]:.2f}' if top else '':<13}")
            bad += (missing or wrong) and not args.audio
            print(line)
        for key, _, _ in READINGS if audio is not None else ():
            answer = decider.last_answers.get(key)
            answer = answer if isinstance(answer, dict) else {}
            print(f"  reading {key}: {answer.get('choice', '-')} ({answer.get('confidence', 0):.2f}){'' if key in graded.readings else '  not kept: not sure'}")
        print()
    if model is not None:
        graded = await grade(lines, visited=VISITED, summary_tries=1, profile=profile, owner=owner, model=model)
        got = {score.rule: score for score in graded.scores}
        print(f"The grader model: {settings.grader_model}")
        print(f"  {'rule':<22}{'got':<6}{'expected':<9}{'':<6}agent / cause")
        for rule in RULES:
            if rule.hears:
                continue
            line, missing, wrong = _row(rule.key, got.get(rule.key))
            bad += missing or wrong
            print(line)
        print()
    if args.audio:
        print("The transcript is the made-up sample, not your file. Read only the audio rules and the readings.")
        return 0
    print("Each rule got the expected verdict." if not bad else f"{bad} verdict(s) are missing or not the expected one. Read the rule's text in grading.py before you trust its scores.")
    return 1 if bad else 0


def floor_faults(found: int, alarms: int, *, min_found: int | None, max_alarms: int | None) -> list[str]:
    """Why a run of `--set` is below its floor; [] = the floor holds.

    `found` counts the planted failures and the kinds of feedback that the grader found. `alarms` counts
    both kinds of false alarm. None = no floor for that count."""
    faults = []
    if min_found is not None and found < min_found:
        faults.append(f"{found} found, and the floor is {min_found}")
    if max_alarms is not None and alarms > max_alarms:
        faults.append(f"{alarms} false alarm(s), and the floor allows {max_alarms}")
    return faults


async def run_set(args: argparse.Namespace) -> int:
    """Each made-up call of the set goes to each grader. The count: failures found, false alarms, and the right agent and cause."""
    settings = get_settings()
    owner = settings.owner_name
    profile = load_profile(settings.profile_path).full
    calls = json.loads(GRADER_CALLS.read_text())["calls"]
    if args.calls:
        wanted = {name.strip() for name in args.calls.split(",")}
        calls = [call for call in calls if call["id"] in wanted]
    rules = model_rules()
    model = init_chat_model(settings.grader_model) if settings.grader_model and args.only != "decision" else None
    graders: list[tuple[str, bool]] = []
    if args.only != "model" and default_decision_grader(settings) is not None:
        graders.append((f"The decision grader ({settings.grader_decision_model}, a failure at {settings.grader_fail_at:.2f} or more)", True))
    if model is not None:
        graders.append((f"The grader model ({settings.grader_model})", False))
    if not graders:
        print("No grader is on: see GOOGLE_API_KEY, GRADER_DECISION_MODEL and GRADER_MODEL in .env.")
        return 1

    async def one(call: dict, decides: bool) -> tuple[dict[str, Score], tuple[dict, list]]:
        lines = [line.format(owner=owner) for line in call["lines"]]
        decider = default_decision_grader(settings) if decides else None  # one for each call: it keeps its last answers
        graded = await grade(lines, visited=call["visited"], summary_tries=1, profile=profile, owner=owner, decider=decider, model=None if decides else model)
        return {score.rule: score for score in graded.scores}, ((decider.last_answers, decider.last_lines) if decider is not None else ({}, []))

    bad = 0
    for title, decides in graders:
        results = await asyncio.gather(*(one(call, decides) for call in calls))
        found = alarms = planted = clean = agents = causes = no_moment = no_moment_right = missing = 0
        said = said_found = quiet = quiet_alarms = 0  # the caller's feedback: kinds that a call holds, and kinds that it does not
        rows = []
        chances: list[tuple[bool, float]] = []  # for each check: must it fail, and the probability of a failure
        for call, (got, (answers, asked)) in zip(calls, results, strict=True):
            for rule in rules:
                must = call["fail"].get(rule.key)
                score = got.get(rule.key)
                chance = fail_probability(rule, answers, asked)
                shown = f"{chance[1]:.2f}" if chance else ""
                if chance:
                    chances.append((must is not None, chance[1]))
                verdict = score.verdict if score else "-"
                missing += score is None
                planted += must is not None
                clean += must is None
                if rule.key in call.get("na", []):
                    no_moment += 1
                    no_moment_right += verdict == NA
                if must is not None:
                    hit = verdict == FAIL
                    found += hit
                    agents += hit and score.agent == must["agent"]
                    causes += hit and score.cause in must["causes"]
                    fault = f"{score.agent or '?'} / {score.cause}" if hit else ""
                    wrong = "" if not hit or (score.agent == must["agent"] and score.cause in must["causes"]) else f"  (expected {must['agent']} / {' or '.join(must['causes'])})"
                    rows.append(f"  {'found' if hit else 'MISSED':<12}{call['id'] + ' / ' + rule.key:<58}{verdict:<6}{shown:<6}{fault}{wrong}")
                elif verdict == FAIL:
                    alarms += 1
                    rows.append(f"  {'FALSE ALARM':<12}{call['id'] + ' / ' + rule.key:<58}{verdict:<6}{shown:<6}{score.agent or '?'} / {score.cause}")
                elif chance and chance[1] >= 0.3:
                    rows.append(f"  {'near':<12}{call['id'] + ' / ' + rule.key:<58}{verdict:<6}{shown:<6}")
            for kind, chance in feedback_chances(answers).items():
                must, hit = kind in call.get("feedback", []), chance >= settings.grader_fail_at
                said, said_found, quiet, quiet_alarms = said + must, said_found + (must and hit), quiet + (not must), quiet_alarms + (hit and not must)
                if must or chance >= 0.3:
                    mark = ("found" if hit else "MISSED") if must else ("FALSE ALARM" if hit else "near")
                    rows.append(f"  {mark:<12}{call['id'] + ' / feedback: ' + kind:<58}{'':<6}{chance:.2f}")
        print(f"{title}, on {len(calls)} made-up calls:")
        print(f"  failures found: {found} of {planted}")
        print(f"  false alarms: {alarms} of {clean} checks that must not fail")
        print(f"  of the failures found: the right agent {agents}, a right cause {causes}")
        print(f"  'na' for a rule with no moment in the call: {no_moment_right} of {no_moment}")
        if said or quiet:
            print(f"  feedback from the caller: {said_found} of {said} found, {quiet_alarms} false alarm(s) of {quiet}")
        if missing:
            print(f"  {missing} check(s) got no verdict: read the log lines above")
        if chances:
            lowest, highest = min((chance for must, chance in chances if must), default=0.0), max((chance for must, chance in chances if not must), default=0.0)
            print(f"  the lowest probability of a planted failure: {lowest:.2f}; the highest of a check that must not fail: {highest:.2f}")
            print("  at another limit:  " + "   ".join(f"{limit:.1f}: {sum(must and chance >= limit for must, chance in chances)} found, {sum(not must and chance >= limit for must, chance in chances)} false" for limit in (0.5, 0.6, 0.7, 0.8, 0.9)))
        if args.min_found is None and args.max_false_alarms is None:
            bad += (planted - found) + alarms + missing + (said - said_found) + quiet_alarms
        else:
            # A floor allows a few misses: a probability near the limit is not stable from one run to the next.
            faults = floor_faults(found + said_found, alarms + quiet_alarms, min_found=args.min_found, max_alarms=args.max_false_alarms)
            print(f"  the floor: {found + said_found} found (failures and feedback), {alarms + quiet_alarms} false alarm(s): " + ("; ".join(faults) if faults else "it holds"))
            bad += len(faults)
        print("\n".join(rows) + "\n")
    return 1 if bad else 0


def main() -> None:
    import logging

    parser = argparse.ArgumentParser(description="Send one made-up call to the graders.")
    parser.add_argument("--only", choices=("decision", "model"), help="ask one grader only")
    parser.add_argument("--audio", metavar="FILE", help="a WAV file for the decision grader to hear; it is sent to Google")
    parser.add_argument("--calls", metavar="IDS", help="with --set: only these calls, as ids with commas between them")
    parser.add_argument("--set", action="store_true", help="send each made-up call of evals/grader_calls.json, and count what each grader found")
    parser.add_argument("--min-found", type=int, metavar="N", help="with --set: fail only when fewer than N failures and kinds of feedback are found")
    parser.add_argument("--max-false-alarms", type=int, metavar="N", help="with --set: fail only when there are more than N false alarms")
    args = parser.parse_args()
    if (args.min_found is not None or args.max_false_alarms is not None) and not args.set:
        parser.error("--min-found and --max-false-alarms need --set")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    sys.exit(asyncio.run(run_set(args) if args.set else run(args)))


if __name__ == "__main__":
    main()
