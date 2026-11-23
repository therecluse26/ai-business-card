"""Terminal simulator: you type as the caller; the graph answers as the active agent.

Run: uv run chat
     uv run chat --crm    the end of the call also writes to the owner's CRM, as a phone call does (crm.py)
"""

from __future__ import annotations

import asyncio
import select
import sys
from datetime import UTC, datetime
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_call_center import tracing
from ai_call_center.agents.specs import (
    DEFAULT_LEGS,
    Welcome,
    companies_phrase,
    greeting_tip,
    greeting_words,
    load_profile,
    web_address,
)
from ai_call_center.analytics import MESSAGE, NO_OUTCOME, TEXT, TRANSFER, CallAnalytics, call_day, forget_call, harness_version, record_call
from ai_call_center.callers import summarize_call, transcript_lines
from ai_call_center.config import get_settings
from ai_call_center.crm import CrmStore, default_crm, sync_call
from ai_call_center.decision_grader import default_decision_grader
from ai_call_center.decisions import default_decider
from ai_call_center.grading import describe_scores
from ai_call_center.graph.build import build_graph, default_model, default_specs
from ai_call_center.graph.state import CallContext
from ai_call_center.graph.tools import (
    message_saved,
    note_refused_transfer_at_call_end,
    send_warm_lead_at_call_end,
)
from ai_call_center.guardrails import TransferHours
from ai_call_center.leads import LeadStore
from ai_call_center.notify import RecordingNotifier, notifier_from_settings
from ai_call_center.telephony import RecordingTelephony
from ai_call_center.writing import default_library

CALL_CONNECTED = "[The call connects. Greet the caller.]"

try:  # line editing, history, and clean handling of bracketed paste in the terminal
    import readline  # noqa: F401
except ImportError:  # pragma: no cover - platforms without readline
    pass

_PASTE_MARKERS = ("\x1b[200~", "\x1b[201~")


def _strip_paste_markers(text: str) -> str:
    for marker in _PASTE_MARKERS:
        text = text.replace(marker, "")
    return text


def read_caller_input(prompt: str) -> str:
    """Read one caller message. A pasted block of several lines becomes one message."""
    lines = [_strip_paste_markers(input(prompt))]
    # Lines that are already waiting (a multi-line paste) belong to the same message.
    while select.select([sys.stdin], [], [], 0.05)[0]:
        more = sys.stdin.readline()
        if not more:
            break
        lines.append(_strip_paste_markers(more.rstrip("\n")))
    return "\n".join(line for line in lines if line.strip()).strip()


def print_new_messages(messages: list, start: int) -> None:
    for msg in messages[start:]:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                print(f"  [tool] {call['name']}({args})")
            text = msg.text if isinstance(msg.text, str) else msg.text()
            if text.strip():
                print(f"agent> {text.strip()}")
        elif isinstance(msg, ToolMessage):
            print(f"  [result] {msg.content}")


async def run(caller_number: str = "+10000000000", *, crm: bool = False) -> None:
    settings = get_settings()
    tracing.start(settings, voice=False)
    specs = default_specs()
    graph = build_graph(default_model(), specs, checkpointer=InMemorySaver())
    telephony = RecordingTelephony()
    notifier = notifier_from_settings(settings)
    call_id = f"chat-{uuid4().hex[:8]}"
    store = LeadStore(settings.leads_db)
    # `uv run chat --from +15551234567` pretends to call from that number, to test caller memory.
    store.start_call(call_id, caller_number)
    ctx = CallContext(
        call_control_id=call_id,
        caller_number=caller_number,
        owner_phone=settings.owner_phone,
        leads=store,
        telephony=telephony,
        notifier=notifier,
        legs=dict(DEFAULT_LEGS),
        mode="text",
        welcome=Welcome(
            owner=settings.owner_name,
            companies=companies_phrase(dict(DEFAULT_LEGS)),
            tip=greeting_tip(settings.owner_name, writing=bool(default_library().posts)),
            rag=bool(default_library().posts),
        ),
        owner_name=settings.owner_name,
        decider=default_decider(settings),
        crm=default_crm(settings) if crm else None,  # `--crm`: a text call is a test, so it writes to the owner's CRM only on request
        earlier_calls=store.call_count_for(caller_number, exclude=call_id),
        transfer_hours=TransferHours.parse(settings.transfer_hours, settings.transfer_days),
        website_spoken=site.spoken if (site := web_address(settings.website_url, settings.brand_name)) else "",
    )
    config = {"configurable": {"thread_id": ctx.call_control_id}, **tracing.run_config(ctx.call_control_id, "chat-turn")}
    print(f"Voice business card, text mode. Model: {settings.llm_model}. Type /quit to hang up.\n")

    seen = 0
    result_messages: list = []
    handoffs: list[str] = []
    hello = greeting_words(settings.owner_name, settings.brand_name)
    text = f'{CALL_CONNECTED[:-1]} Say exactly these words: "{hello}"]'
    active = "classifier"
    while True:
        try:
            result = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config, context=ctx)
        except Exception as exc:  # noqa: BLE001 - keep the session alive; show the cause
            print(f"\n[error] {type(exc).__name__}: {exc}")
            print("[the call stays open; try again or type /quit]")
            text = ""
            try:
                text = read_caller_input(f"\ncaller ({active})> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text in {"/quit", "/q", "/exit"}:
                break
            text = text or "[silence]"
            continue
        print_new_messages(result["messages"], seen)
        result_messages = result["messages"]
        handoffs = result.get("handoffs", [])
        seen = len(result["messages"])
        active = result.get("active_agent", "classifier")
        if result.get("transfer_requested"):
            print(f"\n[call transferred to {telephony.transfers[-1].to}; hanging up]")
            break
        if result.get("call_ended"):
            print("\n[the agent ended the call]")
            break
        try:
            text = read_caller_input(f"\ncaller ({active})> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text in {"/quit", "/q", "/exit"}:
            break
        if not text:
            text = "[silence]"
    visited = ["classifier", *handoffs]
    transferred = bool(telephony.transfers)
    if ctx.forgotten:
        await note_refused_transfer_at_call_end(ctx, lead_saved=message_saved(ctx), visited=visited)
        store.finish_call(call_id, agents=visited, summary="")
        forget_call(store.connection, call_id)
        print("\n[the caller asked to be forgotten: no summary saved]")
    elif result_messages:
        summary = await summarize_call(result_messages, init_chat_model(settings.summary_model), config=tracing.run_config(call_id, "call-summary"))
        noted = await note_refused_transfer_at_call_end(ctx, lead_saved=message_saved(ctx), visited=visited, summary=summary)
        store.finish_call(call_id, agents=visited, summary=summary.summary, caller_name=ctx.caller_name or summary.caller_name, topic=summary.topic)
        if summary.summary:
            print(f"\n[call summary saved for {caller_number}: {summary.summary} (topic: {summary.topic or '-'}, interest: {summary.interest})]")
        if await send_warm_lead_at_call_end(ctx, summary, lead_saved=bool(noted) or message_saved(ctx), visited=visited):
            print("[warm lead email sent: the caller talked about a project or a role and left no message]")
        if ctx.crm is not None:
            await sync_call(ctx, summary, visited=visited, transferred=transferred)
            row = CrmStore(store.connection).get(call_id)
            print(f"[CRM: {'this call left no entry' if row is None else 'entry ' + row.state + (', person ' + row.person_id if row.person_id else '')}]")
        # The grader scores the call, as it does after a phone call. A call from this simulator is marked as a test.
        try:
            record = await record_call(
                store.connection,
                CallAnalytics(
                    call_id=call_id,
                    day=call_day(ctx.started_at),
                    channel=TEXT,
                    version=harness_version(settings.harness_version),
                    test=True,
                    duration_secs=int((datetime.now(UTC) - ctx.started_at).total_seconds()),
                    agents=visited,
                    outcome=TRANSFER if transferred else MESSAGE if message_saved(ctx) else NO_OUTCOME,
                ),
                transcript_lines(result_messages),
                summary_tries=summary.tries,
                profile=load_profile(settings.profile_path).full,
                owner=settings.owner_name,
                grader=init_chat_model(settings.grader_model) if settings.grader_model else None,
                config=tracing.run_config(call_id, "call-grade"),
                decider=None if settings.grader_model else default_decision_grader(settings),  # the text mode has no audio: the grader model is first (see score.py)
            )
            print(f"[grade: {describe_scores(record.scores)}]")
        except Exception as exc:  # noqa: BLE001 - the grade must not hide the end of the session
            print(f"[the call was not graded: {type(exc).__name__}: {exc}]")
        if ctx.feedback_filed:
            # Only a phone call turns a filed report into a GitHub issue (voice/call_session.py).
            print(f"[{len(ctx.feedback_filed)} report(s) filed by the feedback agent; a text call opens no GitHub issue]")
    leads = ctx.leads.all()
    print(f"\n[hung up] active agent: {active}; leads saved this process: {len(leads)}")
    if isinstance(notifier, RecordingNotifier) and notifier.sent:
        print(f"[{len(notifier.sent)} lead email(s) recorded, not sent: set OWNER_EMAIL and SMTP_HOST to send]")


def main() -> None:
    try:
        args = sys.argv[1:]
        number = args[args.index("--from") + 1] if "--from" in args and len(args) > args.index("--from") + 1 else "+10000000000"
        asyncio.run(run(number, crm="--crm" in args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
