"""Returning-caller memory: what the agents may know about a phone number before the greeting,
and the end-of-call summary that feeds the next call.

A phone number can be shared, so a number is not a person. Before the caller says who they are,
the greeting agent knows one thing: the first name the earlier caller gave. It asks "Am I speaking
with <name>?" and nothing else. Only after a yes does `confirm_returning_caller` release the
history: company, saved interest, and a one- or two-sentence summary of each recent call.
What never reaches a prompt or a tool result: the email address and the phone number.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ai_call_center.leads import CallRecord, LeadStore, NumberInfo

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
UNKNOWN_NUMBERS = {"", "unknown", "anonymous", "restricted", "+10000000000"}


@dataclass(frozen=True)
class CallerMemory:
    name: str  # full name as given, or ""
    company: str
    need: str  # from the saved lead, or ""
    previous_calls: int
    recent: list[CallRecord] = field(default_factory=list)
    contact_on_file: bool = False  # a saved lead holds an email address for this number

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name.strip() else ""


def recall(store: LeadStore, caller_number: str, *, exclude_call: str = "") -> CallerMemory | None:
    """What the store knows about this number. None for a first-time or hidden number."""
    if caller_number in UNKNOWN_NUMBERS:
        return None
    lead = store.latest_lead_for(caller_number)
    recent = store.calls_for(caller_number, exclude=exclude_call)
    count = store.call_count_for(caller_number, exclude=exclude_call)
    if lead is None and not recent:
        return None
    name = lead.name if lead else next((c.caller_name for c in recent if c.caller_name), "")
    return CallerMemory(
        name=name,
        company=lead.company if lead and lead.company.lower() != "none" else "",
        need=lead.need if lead else "",
        previous_calls=count,
        recent=recent,
        contact_on_file=store.latest_lead_for(caller_number, with_email=True) is not None,
    )


def _when(iso_utc: str) -> str:
    try:
        when = datetime.fromisoformat(iso_utc).astimezone(EASTERN)
    except ValueError:
        return iso_utc
    return when.strftime("%b %-d at %-I:%M %p Eastern")


def greeting_note(memory: CallerMemory) -> str:
    """For the greeting agent only, before the caller has said who they are.

    Holds the first name and nothing else about earlier calls. Empty when no name is known:
    with no name there is nothing to confirm, so the caller is greeted as a new caller.
    """
    name = memory.first_name
    if not name:
        return ""
    return "\n".join(
        [
            "Returning caller note:",
            f"- This phone number has called before. The earlier caller gave the first name {name}.",
            "- A phone number can be shared. You do not know yet who is on the line, and you know nothing else about earlier calls yet.",
            f'- This replaces the normal greeting. Say you are the AI assistant on this line. Then ask only: "Am I speaking with {name}?" Stop and wait for the answer.',
            '- Until the caller says yes, say nothing about an earlier call: no topic, no company, no date, and no "welcome back".',
            "- If the caller says yes, call confirm_returning_caller at once, without saying anything first. Its result gives you the notes from their earlier calls.",
            '- If the caller says no, or the answer is not clear, say "No problem" and ask what brings them to call. '
            "Never call confirm_returning_caller then. Treat them as a new caller for the whole call.",
            "- If the caller asks who called before, or what an earlier call was about, say you cannot share anything about other calls.",
        ]
    )


def history_text(memory: CallerMemory) -> str:
    """The notes from earlier calls. Released by `confirm_returning_caller`, after the caller says who they are."""
    lines = [f"The caller confirmed they are {memory.first_name or 'the earlier caller'}. Notes from their earlier calls:"]
    if memory.name:
        who = memory.name + (f" from {memory.company}" if memory.company else "")
        lines.append(f"- Known as {who}.")
    if memory.need:
        lines.append(f"- Their saved interest: {memory.need}")
    for call in memory.recent:
        if call.summary:
            lines.append(f"- Call on {_when(call.started_at)}: {call.summary}")
    if memory.previous_calls > len(memory.recent):
        lines.append(f"- {memory.previous_calls} earlier calls in total.")
    if memory.contact_on_file:
        lines.append('- Their contact details are on file. To reuse them, call save_lead with email "on file". Ask only for what is new.')
    lines.append(
        "Now welcome them back in one sentence. Name the topic of their last call in a few words, and ask if they are calling about that or about something new. "
        "Do not read these notes out. Never say an email address or a phone number aloud. "
        "You are still the same agent, and this was not a handoff: do not introduce yourself, and do not say that you looked anything up."
    )
    return "\n".join(lines)


def lookup_text(info: NumberInfo) -> str:
    """The number-lookup block for the prompt. Empty when the lookup found nothing."""
    if info.empty:
        return ""
    lines = ["Public carrier data for the caller's phone number (from a number lookup; it can be wrong):"]
    if info.caller_name:
        lines.append(f"- Registered name: {info.caller_name.title()}")
    line = " ".join(part for part in (info.line_type, f"on {info.carrier}" if info.carrier else "") if part)
    if line:
        lines.append(f"- Line: {line}")
    area = ", ".join(part for part in (info.city.title(), info.state) if part)
    if area:
        lines.append(f"- Area: {area}")
    lines.append(
        "How to use it: do not open the call with it. If the caller asks what you know about them, or asks what this system can do, "
        "you may say the area and the line type as a live demonstration. "
        "When you need the caller's name, you may offer the registered name as a guess and ask if it is right. Accept a no without comment. "
        "The registered name is the account holder; never treat it as confirmed."
    )
    return "\n".join(lines)


def transcript_lines(messages: list[Any]) -> list[str]:
    """Plain 'caller:' / 'agent:' lines from Pipecat or LangChain context messages. Tool traffic is left out."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role, content = msg.get("role"), msg.get("content")
            if msg.get("tool_calls"):
                continue
        else:  # a LangChain message
            role = getattr(msg, "type", None)
            role = {"human": "user", "ai": "assistant"}.get(role, role)
            content = getattr(msg, "content", None)
            if getattr(msg, "tool_calls", None):
                continue
        if role not in {"user", "assistant"}:
            continue
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"})
        text = " ".join(str(content or "").split())
        if text and not text.startswith("["):
            lines.append(f"{'caller' if role == 'user' else 'agent'}: {text}")
    return lines


SUMMARY_PROMPT = """Below is the transcript of a phone call to an AI assistant line.
Write a JSON object with two keys:
- "summary": at most 40 words, in one or two plain sentences: what the caller wanted, and how the call ended. Write it for the assistant that will take this caller's next call. Do not include email addresses or phone numbers.
- "caller_name": the caller's name if they gave one, else "".
Return only the JSON object.

Transcript:
{transcript}"""


async def summarize_call(messages: list[Any], model: BaseChatModel) -> tuple[str, str]:
    """(summary, caller_name) for the call log. Empty strings when there is nothing to summarise or the model fails."""
    lines = transcript_lines(messages)
    if not any(line.startswith("caller:") for line in lines):
        return "", ""
    try:
        reply = await model.ainvoke([HumanMessage(SUMMARY_PROMPT.format(transcript="\n".join(lines)))])
        text = reply.text if isinstance(reply.text, str) else reply.text()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        return str(data.get("summary", "")).strip(), str(data.get("caller_name", "")).strip()
    except Exception:
        log.exception("call summary failed")
        return "", ""
