"""Returning-caller memory: what the agents may know about an earlier caller, and the end-of-call
summary that feeds the next call.

A phone number can be shared, so a number is not a person. Every caller hears the same greeting,
which asks who they are. The name they give goes to `identify_caller`; code (`names.same_person`)
compares it with the names on record for the number. A match makes the caller "recognised": the
agent says "Welcome back", and gets no notes. A phone number can be faked and a name is no secret,
so the notes need a passphrase that the caller set on an earlier call (`passphrase.py`). Only after
that check does the agent get that person's notes: company, saved interest, and a short summary and
topic of each recent call that was made with that passphrase.
Calls and leads that another name left on the same number never reach the agent.
What never reaches a prompt or a tool result: the email address, the phone number, and the
name on the phone account from a number lookup.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ai_call_center.leads import CallRecord, Lead, LeadStore, NumberInfo
from ai_call_center.names import same_person

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
UNKNOWN_NUMBERS = {"", "unknown", "anonymous", "restricted", "+10000000000"}


@dataclass(frozen=True)
class CallerMemory:
    """One person's records on one number."""

    name: str  # the name as they gave it before
    company: str
    need: str  # from the saved lead, or ""
    previous_calls: int
    recent: list[CallRecord] = field(default_factory=list)
    contact_on_file: bool = False  # a saved lead holds an email address for this person
    topic: str = ""  # the topic of their last call that has one, for "Last time you asked about ..."
    leads: list[Lead] = field(default_factory=list)  # all their leads, for forget_caller
    calls: list[CallRecord] = field(default_factory=list)  # all their calls, for forget_caller

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name.strip() else ""


def recall(store: LeadStore, caller_number: str, name: str, *, exclude_call: str = "", only_calls: frozenset[str] | None = None) -> CallerMemory | None:
    """The records of the person called `name` on this number. None when there is no match.

    A record matches when `same_person` says its name is this name. A call with no name in the log
    still belongs to them when their lead came from that call. With `only_calls`, only the records
    of those calls count: the ones that the caller's passphrase opens.
    """
    if caller_number in UNKNOWN_NUMBERS or not name.strip():
        return None
    # A lead saved on the call in progress is not history yet.
    leads = [lead for lead in store.leads_for(caller_number, limit=HISTORY_ROWS) if lead.call_control_id != exclude_call]
    calls = store.calls_for(caller_number, exclude=exclude_call, limit=HISTORY_ROWS)
    if only_calls is not None:
        leads = [lead for lead in leads if lead.call_control_id in only_calls]
        calls = [call for call in calls if call.call_control_id in only_calls]
    own_leads = [lead for lead in leads if same_person(name, lead.name)]
    lead_calls = {lead.call_control_id for lead in own_leads}
    own_calls = [call for call in calls if same_person(name, call.caller_name) or (not call.caller_name.strip() and call.call_control_id in lead_calls)]
    if not own_leads and not own_calls:
        return None
    lead = own_leads[0] if own_leads else None
    named = [call.caller_name for call in own_calls if call.caller_name.strip()]
    return CallerMemory(
        # A lead holds the name as the caller confirmed it; the call log holds what was heard.
        name=lead.name if lead else (named[0] if named else name.strip()),
        company=lead.company if lead and lead.company.lower() != "none" else "",
        need=lead.need if lead else "",
        previous_calls=len(own_calls),
        recent=own_calls[:RECENT_CALLS],
        contact_on_file=any(lead.email for lead in own_leads),
        topic=next((call.topic for call in own_calls if call.topic.strip()), ""),
        leads=own_leads,
        calls=own_calls,
    )


RECENT_CALLS = 3  # call summaries that the history holds
HISTORY_ROWS = 30  # calls read per number, to find one person's calls among them


def _moment(pair: tuple[str, str]) -> datetime:
    try:
        when = datetime.fromisoformat(pair[0])
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def _when(iso_utc: str) -> str:
    try:
        when = datetime.fromisoformat(iso_utc).astimezone(EASTERN)
    except ValueError:
        return iso_utc
    return when.strftime("%b %-d at %-I:%M %p Eastern")


def history_text(memory: CallerMemory) -> str:
    """The notes from earlier calls, for the agents only. `verify_passphrase` gives them after the passphrase check."""
    lines = [f"Notes from {memory.first_name or 'this caller'}'s earlier calls, for you and the other agents only:"]
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
    lines.append("Do not read these notes out. Never say an email address or a phone number aloud. Do not say that you looked anything up.")
    return "\n".join(lines)


def lookup_text(info: NumberInfo) -> str:
    """The number-lookup block for the prompt. Empty when the lookup found nothing."""
    if info.empty:
        return ""
    # The name on the phone account stays out: it can be another person, and saying it would leak it.
    if not (info.line_type or info.carrier or info.city or info.state):
        return ""
    lines = ["Public carrier data for the caller's phone number (from a number lookup; it can be wrong):"]
    line = " ".join(part for part in (info.line_type, f"on {info.carrier}" if info.carrier else "") if part)
    if line:
        lines.append(f"- Line: {line}")
    area = ", ".join(part for part in (info.city.title(), info.state) if part)
    if area:
        lines.append(f"- Area: {area}")
    lines.append(
        "How to use it: do not open the call with it. If the caller asks what you know about them, or asks what this system can do, "
        "you may say the area and the line type as a live demonstration."
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
Write a JSON object with these keys:
- "summary": at most 40 words in total, in one or two short sentences: what the caller wanted, and how the call ended. Leave the specifics to "owner_note". Write it for the assistant that will take this caller's next call. Do not include email addresses or phone numbers.
- "caller_name": the caller's name if they gave one, else "".
- "topic": at most 8 words that finish the sentence "Last time you asked about ...", for example "a new booking platform". No names, no companies, no numbers, no email addresses. "" if the caller asked about nothing.
- "interest": "project" if the caller talked about work they want done (a product, a system, a project, help for their business); "role" if they talked about a job or a role to fill; else "none".
- "owner_note": a note for the owner of the line, at most 120 words, in plain sentences. Give every specific the caller said about the work: who they are, their company and what it does, the project or the role, what they want from the owner, when it starts, how long it runs, the rate or budget, and their questions. Only what the caller said; never guess. No email addresses or phone numbers. "" if the caller said nothing about work.
- "timeline": when the caller wants the work to start, in their words, for example "in two weeks". "" if they did not say.
- "size": how big or how long the work is, in their words, for example "ongoing, no end date". "" if they did not say.
Return only the JSON object.

Transcript:
{transcript}"""

SUMMARY_RETRY = """Your answer broke these rules:
{faults}
Write the JSON object again, with every rule kept. Return only the JSON object."""

INTERESTS = ("project", "role", "none")
SUMMARY_WORDS = 40  # the next call's agents read the summary, so its size is fixed
SUMMARY_TRIES = 2  # the first answer, and one more when the check finds a fault
CHECKED_FIELDS = ("summary", "topic", "owner_note", "timeline", "size")  # the texts that are stored, spoken, or emailed
REMOVED = "[removed]"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
PHONE_DIGITS = 9  # a shorter run of digits is a date, a price, or a count


@dataclass(frozen=True)
class CallSummary:
    summary: str = ""
    caller_name: str = ""
    topic: str = ""
    interest: str = "none"  # one of INTERESTS
    owner_note: str = ""  # every specific about the work, for the owner's email; never spoken
    timeline: str = ""
    size: str = ""
    tries: int = 0  # the answers the summary model gave: 0 = no summary was asked for, 2 = the check sent the first one back
    first_faults: tuple[str, ...] = ()  # what the check found in the first answer; fixed texts of `summary_faults`, for the fault report


def _short(text: str, words: int) -> str:
    return " ".join(str(text or "").split()[:words]).strip(" .")


def contact_details(text: str) -> list[str]:
    """The email addresses and phone numbers that are written in a text."""
    return [*_EMAIL.findall(text), *(run for run in _PHONE.findall(text) if sum(ch.isdigit() for ch in run) >= PHONE_DIGITS)]


def without_contact_details(text: str) -> str:
    for item in contact_details(text):
        text = text.replace(item, REMOVED)
    return " ".join(text.split())


def _fit(text: str, words: int) -> str:
    """The text in at most `words` words: whole sentences while they fit, else the first words."""
    if len(text.split()) <= words:
        return text
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.?!])\s+", text):
        if len(" ".join([*kept, sentence]).split()) > words:
            break
        kept.append(sentence)
    return " ".join(kept) if kept else " ".join(text.split()[:words])


def _parse_summary(text: str) -> CallSummary:
    """The model's answer as a summary. Empty when the answer holds no JSON object."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    interest = str(data.get("interest", "none")).strip().lower()
    return CallSummary(
        summary=str(data.get("summary", "")).strip(),
        caller_name=str(data.get("caller_name", "")).strip(),
        # The topic is spoken to the caller next time. Code keeps it short, whatever the model wrote.
        topic=_short(data.get("topic", ""), 8),
        interest=interest if interest in INTERESTS else "none",
        owner_note=" ".join(str(data.get("owner_note", "")).split()),
        timeline=_short(data.get("timeline", ""), 12),
        size=_short(data.get("size", ""), 12),
    )


def summary_faults(summary: CallSummary) -> list[str]:
    """The rules of the summary prompt that this summary breaks. Code checks them; the prompt alone is a request, not a guard."""
    faults: list[str] = []
    if not summary.summary:
        faults.append('- The "summary" is empty, or the answer was not a JSON object.')
    if len(summary.summary.split()) > SUMMARY_WORDS:
        faults.append(f'- The "summary" has more than {SUMMARY_WORDS} words.')
    for key in CHECKED_FIELDS:
        if contact_details(getattr(summary, key)):
            faults.append(f'- The "{key}" holds an email address or a phone number.')
    if any(ch.isdigit() for ch in summary.topic):
        faults.append('- The "topic" holds a number.')
    return faults


def _repaired(summary: CallSummary) -> CallSummary:
    """The summary with each remaining fault removed by code. After this, `summary_faults` finds only an empty summary."""
    clean = {key: without_contact_details(getattr(summary, key)) for key in CHECKED_FIELDS}
    clean["summary"] = _fit(clean["summary"], SUMMARY_WORDS)
    if REMOVED in clean["topic"] or any(ch.isdigit() for ch in clean["topic"]):
        clean["topic"] = ""  # a caller hears the topic on the next call; a cut topic is worse than none
    return replace(summary, **clean)


async def summarize_call(messages: list[Any], model: BaseChatModel, config: dict[str, Any] | None = None) -> CallSummary:
    """The call log entry. Empty when there is nothing to summarise or the model fails.

    A check in code reads each answer (`summary_faults`). An answer with a fault goes back to the model
    one time, with the faults named. What is still wrong after that, code removes: the summary is
    stored, read by the next call's agents, and emailed, so a fault must not pass.

    `config` is the LangChain run config, for example the call's LangSmith thread (`tracing.run_config`)."""
    lines = transcript_lines(messages)
    if not any(line.startswith("caller:") for line in lines):
        return CallSummary()
    asked: list[Any] = [HumanMessage(SUMMARY_PROMPT.format(transcript="\n".join(lines)))]
    best, best_faults, tries = CallSummary(), [""], 0
    first_faults: tuple[str, ...] = ()
    try:
        while tries < SUMMARY_TRIES:
            reply = await model.ainvoke(asked, config=config)
            tries += 1
            text = reply.text if isinstance(reply.text, str) else reply.text()
            summary = _parse_summary(text)
            faults = summary_faults(summary)
            if tries == 1:
                first_faults = tuple(faults)
            if tries == 1 or len(faults) < len(best_faults):
                best, best_faults = summary, faults
            if not faults:
                break
            log.warning("call summary, try %d of %d, broke %d rule(s): %s", tries, SUMMARY_TRIES, len(faults), " ".join(faults))
            asked = [*asked, AIMessage(text), HumanMessage(SUMMARY_RETRY.format(faults="\n".join(faults)))]
    except Exception:
        # The answer of an earlier try stands. With no answer at all, the summary is empty.
        log.exception("call summary failed")
    return replace(_repaired(best), tries=tries, first_faults=first_faults)
