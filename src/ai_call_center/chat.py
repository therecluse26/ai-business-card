"""Terminal simulator: you type as the caller; the graph answers as the active agent.

Run: uv run chat
"""

from __future__ import annotations

import asyncio
import select
import sys
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_call_center.agents.specs import DEFAULT_LEGS
from ai_call_center.config import get_settings
from ai_call_center.graph.build import build_graph, default_model, default_specs
from ai_call_center.graph.state import CallContext
from ai_call_center.leads import LeadStore
from ai_call_center.notify import RecordingNotifier, notifier_from_settings
from ai_call_center.telephony import RecordingTelephony

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


async def run() -> None:
    settings = get_settings()
    specs = default_specs()
    graph = build_graph(default_model(), specs, checkpointer=InMemorySaver())
    telephony = RecordingTelephony()
    notifier = notifier_from_settings(settings)
    ctx = CallContext(
        call_control_id=f"chat-{uuid4().hex[:8]}",
        caller_number="+10000000000",
        owner_phone=settings.owner_phone,
        leads=LeadStore(settings.leads_db),
        telephony=telephony,
        notifier=notifier,
        legs=dict(DEFAULT_LEGS),
        mode="text",
    )
    config = {"configurable": {"thread_id": ctx.call_control_id}}
    print(f"Voice business card, text mode. Model: {settings.llm_model}. Type /quit to hang up.\n")

    seen = 0
    text = CALL_CONNECTED
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
        seen = len(result["messages"])
        active = result.get("active_agent", "classifier")
        if result.get("transfer_requested"):
            print(f"\n[call transferred to {telephony.transfers[-1].to}; hanging up]")
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
    leads = ctx.leads.all()
    print(f"\n[hung up] active agent: {active}; leads saved this process: {len(leads)}")
    if isinstance(notifier, RecordingNotifier) and notifier.sent:
        print(f"[{len(notifier.sent)} lead email(s) recorded, not sent: set OWNER_EMAIL and SMTP_HOST to send]")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
