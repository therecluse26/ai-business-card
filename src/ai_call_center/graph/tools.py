"""Tools with side effects. LangGraph executes them; the voice model only requests them."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ai_call_center.agents.specs import (
    LEGS,
    PASSPHRASE_AGAIN,
    PASSPHRASE_OFFER,
    PASSPHRASE_SET,
    PASSPHRASE_SKIPPED,
    PASSPHRASE_START,
    PASSPHRASE_WRONG,
    RETURNING_QUESTION,
    WELCOME_QUESTION,
    feedback_filed,
    passphrase_question,
    passphrase_read_back,
    passphrase_right,
)
from ai_call_center.analytics import AnalyticsStore
from ai_call_center.caller_voice import voice_sentence, voice_summary
from ai_call_center.crm import CrmStore, request_status
from ai_call_center.decisions import Decision
from ai_call_center.feedback import FAULT, IDEA, MAX_ITEMS
from ai_call_center.graph.state import CallContext, CallState
from ai_call_center.callers import UNKNOWN_NUMBERS, CallSummary, history_text, recall, transcript_lines
from ai_call_center.leads import Lead, now_iso
from ai_call_center.names import same_person
from ai_call_center.notify import CallDetails
from ai_call_center.passphrase import MIN_LETTERS, TRIES, PhraseStore, plain, sound_key, usable
from ai_call_center.writing import default_library, search_result

log = logging.getLogger(__name__)


ON_FILE = {"on file", "onfile", "on-file"}


SETTLED = "Who the caller is is now settled for the whole call. No agent asks for their name again, or asks them to confirm it."


def _say(words: str) -> str:
    return f'Say exactly these words to the caller, and nothing else: "{words}"'


# A recognised caller with no passphrase check. The agent has no notes, and says why when the caller asks.
NO_NOTES = (
    "You have no notes of this caller's earlier calls, so say nothing about what an earlier call was about. "
    "If the caller asks about an earlier call, say in one sentence: a phone number can be faked, so this line uses the notes of an earlier call "
    "only after a passphrase check, and they can set a passphrase after they leave a message."
)
PHRASE_ASKED = (
    "The caller's reply is the passphrase, even a full sentence: call verify_passphrase with all of it, word for word, in silence. "
    "Only if they say they do not know it, do not remember it, or will not say it, call verify_passphrase with an empty passphrase. Say nothing about an earlier call before its result."
)


def _start_asked(ask: str) -> str:
    return f'If the caller says a word or phrase, call set_passphrase with it, in silence. If they skip it or ask for something, answer them, or ask: "{ask}"\n'


PHRASE_CLOSED = "You have no notes of the caller's earlier calls. Do not ask for the passphrase again on this call, and say nothing about what an earlier call was about."


def identify(ctx: CallContext, name: str) -> str:
    """The caller said who they are. Code compares the name with the names on record for this number.
    A match makes the caller "recognised": a welcome back, and no notes. The notes need the passphrase
    check (`verify`). Shared by text and voice modes."""
    name = " ".join(name.split())
    if ctx.identified:
        who = f" as {ctx.caller_name}" if ctx.caller_name else ""
        return f"The caller is already identified{who}. Do not call identify_caller again, and do not ask for their name. Answer what they last said."
    ctx.identified = True
    welcome = ctx.welcome
    if not name:
        words = welcome.new_caller() if welcome else ""
        follow = (
            "No name was given. Do not ask for it again. If the caller asked for something, answer it or route them, and do not say the words below. "
            "If they asked for nothing, "
        )
        return follow + (_say(words) if words else "ask what they would like to do.")
    memory = recall(ctx.leads, ctx.caller_number, name, exclude_call=ctx.call_control_id)
    # A caller with no passphrase to check is asked to set one at the start, so the next call can verify them.
    offer = ctx.phrase_offer and ctx.caller_number not in UNKNOWN_NUMBERS and not ctx.forgotten
    if memory is None:
        ctx.caller_name = name
        if welcome:
            say = _say(welcome.new_caller(name, PASSPHRASE_START if offer else WELCOME_QUESTION))
        else:
            say = f"Greet {name} by name. {_say(PASSPHRASE_START)}" if offer else f"Greet {name} by name, and ask what they would like to do."
        return f"{say}\n{_start_asked(WELCOME_QUESTION) if offer else ''}{SETTLED}"
    ctx.identity_confirmed = True
    ctx.returning_name = memory.name
    # The agents get the name that the caller said on this call. The name on record is a detail of an earlier
    # caller: two people on one phone can have the same first name, and a number can be faked. `verify` gives it.
    ctx.caller_name = name
    on_file = PhraseStore(ctx.leads.connection).find(ctx.caller_number, memory.name)
    if on_file is not None and not on_file.locked:
        return f"{_say(passphrase_question(name))}\n{PHRASE_ASKED}\n{SETTLED}"
    ctx.phrase_closed = True
    if welcome:
        say = _say(welcome.returning_caller(name, PASSPHRASE_START if offer else RETURNING_QUESTION))
    else:
        say = f"Welcome {name} back. {_say(PASSPHRASE_START)}" if offer else f"Welcome {name} back, and ask what they would like to do."
    return f"{say}\n{_start_asked(RETURNING_QUESTION) if offer else ''}{NO_NOTES}\n{SETTLED}"


def verify(ctx: CallContext, phrase: str) -> str:
    """The caller said the passphrase, or said that they have none to give. Code compares it; only after a
    match does the agent get that person's notes. Shared by text and voice modes."""
    phrase = " ".join(phrase.split())
    if ctx.identity_verified:
        return "The caller is verified already. Do not ask for the passphrase again. Answer what they last said."
    store = PhraseStore(ctx.leads.connection)
    name = ctx.returning_name
    known = ctx.identity_confirmed and bool(name) and ctx.caller_number not in UNKNOWN_NUMBERS and not ctx.forgotten
    on_file = store.find(ctx.caller_number, name) if known else None
    if on_file is None or on_file.locked or ctx.phrase_closed:
        return "There is no passphrase to check on this call. Do not ask for one. Answer what the caller last said."
    if not plain(phrase):
        ctx.phrase_closed = True
        return f"{_say(PASSPHRASE_SKIPPED)}\n{PHRASE_CLOSED}"
    ctx.spoken_phrases.append(phrase)
    ctx.phrase_tries += 1
    right = store.check(ctx.caller_number, name, phrase)
    if right is None:
        if ctx.phrase_tries < TRIES:
            return f"{_say(PASSPHRASE_AGAIN)}\n{PHRASE_ASKED}"
        ctx.phrase_closed = True
        return f"{_say(PASSPHRASE_WRONG)}\n{PHRASE_CLOSED}"
    ctx.identity_verified, ctx.open_calls = True, sorted(right.calls - {ctx.call_control_id})
    store.add_call(right.row_id, ctx.call_control_id)  # this call was made with the passphrase, so the passphrase opens it next time
    memory = recall(ctx.leads, ctx.caller_number, name, exclude_call=ctx.call_control_id, only_calls=right.calls)
    ctx.caller_history = history_text(memory) if memory else ""
    if memory:
        ctx.caller_name = memory.name  # the spelling from the calls that their passphrase opens, which they confirmed then
    return f"{_say(passphrase_right(memory.topic if memory else ''))}\n\n{ctx.caller_history}".rstrip()


def set_phrase(ctx: CallContext, phrase: str) -> str:
    """The caller gave a word or phrase to be known by on the next call. The first call says it back; the
    second call, with the same passphrase, keeps it. Shared by text and voice modes."""
    phrase = " ".join(phrase.split())
    name = ctx.caller_name
    if ctx.forgotten:
        return "Not kept: the caller asked to be forgotten, so this line keeps nothing about them. Say that in one sentence."
    if ctx.caller_number in UNKNOWN_NUMBERS:
        return "Not kept: the caller's number is hidden, so this line cannot know them on the next call. Say that in one sentence."
    if not name:
        return "Not kept: this line needs the caller's name first. Ask for it, then call set_passphrase again."
    if not usable(phrase, name=name):
        ctx.held_phrase = ""
        return f"Not kept: a passphrase needs {MIN_LETTERS} letters or more, and it cannot be the caller's name. Ask for another one."
    ctx.spoken_phrases.append(phrase)
    key = sound_key(phrase)
    if key != ctx.held_phrase:
        ctx.held_phrase = key
        return (
            f"{_say(passphrase_read_back(phrase))} After a yes, call set_passphrase again with the same passphrase, in silence. "
            "If the caller corrects it, call set_passphrase with the corrected one."
        )
    store = PhraseStore(ctx.leads.connection)
    # A new passphrase opens this call. A verified caller who changes theirs keeps what the old one opened.
    store.set(ctx.caller_number, name, phrase, calls=[ctx.call_control_id, *(ctx.open_calls if ctx.identity_verified else [])])
    ctx.held_phrase, ctx.phrase_set = "", True
    log.info("call %s: the caller set a passphrase", ctx.call_control_id)
    then = "Then ask if there is anything else." if message_saved(ctx) else "Then ask what you can do for them."
    return f"{_say(PASSPHRASE_SET)} {then} Never say the passphrase again."


def forget(ctx: CallContext) -> tuple[str, list[str]]:
    """The caller asked to be forgotten. Their leads are deleted and their call rows are blanked; this call is
    not summarised or saved. Returns the text for the model and the ids of the calls that were forgotten."""
    name = ctx.returning_name or ctx.caller_name
    memory = recall(ctx.leads, ctx.caller_number, name, exclude_call=ctx.call_control_id) if name else None
    leads = list(memory.leads) if memory else []
    if (current := ctx.leads.lead_for_call(ctx.call_control_id)) is not None:
        leads.append(current[1])
    call_ids = [call.call_control_id for call in memory.calls] if memory else []
    if ctx.caller_number not in UNKNOWN_NUMBERS:
        ctx.leads.forget(ctx.caller_number, call_ids=call_ids, leads=leads)
        if name:
            PhraseStore(ctx.leads.connection).forget(ctx.caller_number, name)
        # The grader's notes can say what happened on those calls. The call in progress gets no record at all.
        AnalyticsStore(ctx.leads.connection).drop_notes(call_ids)
    # The links to the caller's CRM records go, and so does an entry that still waits. What the CRM has stays with the owner, as the email does.
    CrmStore(ctx.leads.connection).forget([*call_ids, ctx.call_control_id])
    ctx.forgotten = True
    ctx.caller_history, ctx.identity_confirmed, ctx.returning_name = "", False, ""
    ctx.identity_verified, ctx.open_calls, ctx.phrase_closed, ctx.held_phrase, ctx.phrase_set = False, [], True, "", False
    ctx.held_message, ctx.refused_transfer = None, None
    ctx.feedback_filed.clear()  # a forgotten call is not graded, so no report of it reaches GitHub
    ctx.voice_readings.clear()  # and the voice reader hears no more of this call
    log.info("call %s: the caller asked to be forgotten (%d earlier calls, %d messages)", ctx.call_control_id, len(call_ids), len(leads))
    owner = ctx.owner_name.split()[0] if ctx.owner_name.strip() else "the owner"
    words = f"Done. This line won't remember you. Any message already sent to {owner} stays with {owner}."
    return f"{_say(words)} Then ask if there is anything else you can help with.", [*call_ids, ctx.call_control_id]


def message_saved(ctx: CallContext) -> bool:
    """True when the caller's own message is saved on this call. A note made only because they asked
    for the owner does not count: it says who called, not what they want."""
    return ctx.leads.lead_for_call(ctx.call_control_id) is not None and not ctx.transfer_note_id


async def send_warm_lead_at_call_end(ctx: CallContext, summary: CallSummary, *, lead_saved: bool, visited: list[str] | None = None) -> bool:
    """At the end of a call: a caller who talked about a project or a role but left no message still
    reaches the owner, by email. `lead_saved`: the owner has the caller's message already. True when an email went out.

    On 2026-11-17 a caller gave the consultant every detail of a role, said yes to the read-back, and
    the voice model never called save_lead. The owner got only "wants to connect now". The summary
    model reads the whole call, so this email has the details whatever the agents did."""
    if ctx.forgotten or lead_saved or summary.interest not in ("project", "role") or ctx.caller_number in UNKNOWN_NUMBERS:
        return False
    name = ctx.caller_name or summary.caller_name
    prior = ctx.leads.lead_for_call(ctx.call_control_id)
    lead = Lead(
        name=name or "A caller who gave no name",
        email=prior[1].email if prior else "",
        company=prior[1].company if prior else "",
        need=summary.owner_note or summary.summary or "The call summary is empty.",
        caller_number=ctx.caller_number,
        call_control_id=ctx.call_control_id,
        created_at=now_iso(),
        timeline=summary.timeline,
        size=summary.size,
    )
    await ctx.notifier.send_lead(lead, row_id=0, details=call_details(ctx, kind=f"warm-{summary.interest}", notes="", visited=visited or []))
    log.info("call %s: warm lead email sent (%s)", ctx.call_control_id, summary.interest)
    return True


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.casefold()) if len(w) > 3}


def says_the_same(notes: str, message: str) -> bool:
    """True when the notes only repeat the message: half of their words or more are in it."""
    words = _words(notes)
    return bool(words) and len(words & _words(message)) / len(words) >= 0.5


def call_details(ctx: CallContext, *, kind: str, notes: str, visited: list[str], message: str = "") -> CallDetails:
    """Facts about this call for the lead email. None of it goes back to a model."""
    info = ctx.number_info
    return CallDetails(
        kind=kind,
        # A voice model often fills the notes with the message again. The email says it once.
        notes="" if says_the_same(notes, message) else notes.strip(),
        agents=tuple(visited),
        seconds=int((datetime.now(UTC) - ctx.started_at).total_seconds()),
        earlier_calls=ctx.earlier_calls,
        confirmed_returning=ctx.identity_confirmed,
        verified=ctx.identity_verified,
        line=" ".join(part for part in (info.line_type, f"on {info.carrier}" if info.carrier else "") if part) if info else "",
        area=", ".join(part for part in (info.city.title(), info.state) if part) if info else "",
        registered_name=info.caller_name if info else "",
        voice=voice_summary(ctx.voice_readings),
    )


async def save_lead_record(
    ctx: CallContext,
    *,
    name: str,
    need: str,
    email: str = "",
    company: str = "",
    callback_number: str = "",
    call_notes: str = "",
    timeline: str = "",
    size: str = "",
    visited: list[str] | None = None,
    kind: str = "message",
) -> tuple[str, dict[str, str]]:
    """Store the lead and email it. Returns the text for the model and the lead as saved.

    Only a name and a need are required: the owner can call back on the number the call came from.
    A caller who said their passphrase can reuse the email address on file; the address goes
    from the store to the new row and never enters a prompt or a tool result. Only details that
    the same person left are reused, and only from the calls that the passphrase opens.

    `call_notes`, `visited`, and `kind` go to the owner's email only. They are not stored.

    Shared by the LangGraph tool (text mode) and the voice tool handler (phone mode).
    """
    name, need, company, callback_number = name.strip(), need.strip(), company.strip(), callback_number.strip()
    email = email.strip().lower()
    wants_file = email in ON_FILE
    if wants_file:
        email = ""
    confirmed = ctx.returning_name
    if ctx.identity_verified and confirmed and ctx.caller_number not in UNKNOWN_NUMBERS and (not name or same_person(name, confirmed)):
        own = [lead for lead in ctx.leads.leads_for(ctx.caller_number, limit=30) if lead.call_control_id in ctx.open_calls and same_person(confirmed, lead.name)]
        prior = next((lead for lead in own if lead.email), own[0] if own else None)
        if prior is not None:
            email = email or prior.email
            company = company or prior.company
            name = name or prior.name
    if not name or not need:
        return "Not saved: the caller's name and their message are both needed. Ask for what is missing, then call save_lead again.", {}
    hidden = ctx.caller_number in UNKNOWN_NUMBERS
    if hidden and not email and not callback_number:
        return (
            "Not saved yet: the caller's number is hidden, so the owner has no way to reach them. "
            "Ask for an email address or a callback number, then call save_lead again."
        ), {}
    # One record per call. A second save with nothing new (for example after a transfer that
    # already saved the caller) would send the owner the same lead twice.
    if (earlier := ctx.leads.lead_for_call(ctx.call_control_id)) is not None:
        row_id, saved = earlier
        same_caller = saved.name.casefold() == name.casefold()
        nothing_new = not callback_number and email in ("", saved.email) and company.casefold() in ("", saved.company.casefold())
        # A record made before a transfer says only that the caller asked for the owner. A message
        # that the caller leaves after it is new. On 2026-11-16 such a message was dropped here.
        new_message = kind == "message" and row_id == ctx.transfer_note_id and not says_the_same(need, saved.need)
        if same_caller and nothing_new and not new_message:
            return (
                f"Already saved on this call as record {row_id}. Do not save it again. Tell the caller the owner has their details and will follow up.",
                {"id": str(row_id), "name": saved.name, "email": saved.email, "company": saved.company, "need": saved.need},
            )
    if callback_number:
        need = f"{need} (Callback number given on the call: {callback_number})"
    lead = Lead(
        name=name,
        email=email,
        company=company,
        need=need,
        caller_number=ctx.caller_number,
        call_control_id=ctx.call_control_id,
        created_at=now_iso(),
        timeline=" ".join(timeline.split()),
        size=" ".join(size.split()),
    )
    row_id = ctx.leads.save(lead)
    ctx.caller_name = lead.name
    ctx.transfer_note_id = row_id if kind == "transfer" else 0
    ctx.refused_transfer = None  # this record tells the owner who called
    ctx.held_message = None
    reach = "by email" if email else ("on the callback number they gave" if callback_number else "on the number they are calling from")
    if wants_file and not email:
        reach += "; no email address was on file"
    # The result text never contains the owner's address, so the agent cannot repeat it.
    try:
        await ctx.notifier.send_lead(lead, row_id=row_id, details=call_details(ctx, kind=kind, notes=call_notes, visited=visited or [], message=need))
        if not getattr(ctx.notifier, "delivers", True):
            # Email is not set up on this server. The agent must not claim an email was sent.
            result = f"Lead saved as record {row_id}. The owner will see it. Say the owner has their details; do not say an email was sent."
        else:
            result = f"Lead saved as record {row_id} and emailed to the owner."
        result += f" The owner can reach the caller {reach}."
        # The next step comes from here, not from the model's memory of its instructions.
        # A caller who left a message did not ask for a live call, so no transfer follows from a save.
        owner = ctx.owner_name.split()[0] if ctx.owner_name.strip() else "the owner"
        result += f" Next, in this same turn: tell the caller, in one sentence, that {owner} has their message and will follow up. "
        offer = kind == "message" and ctx.phrase_offer and not (ctx.phrase_offered or ctx.phrase_set or ctx.identity_verified or ctx.forgotten or hidden)
        if offer:
            # The caller has left something to remember. Without a passphrase, no agent gets it on the next call.
            ctx.phrase_offered = True
            result += (
                f'Then make this offer, in these words: "{PASSPHRASE_OFFER}" '
                "If the caller says a word or phrase, call set_passphrase with it, in silence. If they say no, ask if there is anything else. "
            )
        else:
            result += "Then ask if there is anything else. "
        result += "If they say no or say goodbye, call end_call. Do not call request_owner_transfer, and do not offer a transfer: the caller chose to leave a message."
    except Exception:
        log.exception("Lead %s saved, but the owner email failed", row_id)
        result = (
            f"Lead saved as record {row_id}. The email to the owner failed, but the details are stored. "
            "Do not mention the failure. Tell the caller the owner will see their details."
        )
    return result, {"id": str(row_id), "name": lead.name, "email": lead.email, "company": lead.company, "need": lead.need}


def describe_call(ctx: CallContext, *, visited: list[str], active_agent: str, lead_saved: bool, transfer_requested: bool, figures: str = "") -> str:
    """Live facts about the call, as one spoken paragraph. Shared by text and voice modes.
    `figures` are the measured speed, tokens, and cost of a phone call (`voice/stats.py`)."""
    seconds = int((datetime.now(UTC) - ctx.started_at).total_seconds())
    parts = []
    for agent in visited:
        leg = ctx.legs.get(agent)
        parts.append(f"{agent} on {LEGS.get(leg, leg)}" if leg else agent)
    mode = "a text conversation for testing" if ctx.mode == "text" else "a live phone call"
    return (
        f"This is {mode}. It has run for {seconds} seconds. "
        f"Agents so far, in order: {'; '.join(parts)}. "
        f"Current agent: {active_agent}. "
        f"Lead saved: {'yes' if lead_saved else 'no'}. "
        f"Transfer requested: {'yes' if transfer_requested else 'no'}."
        + (f" {figures}" if figures else "")
        + "".join(
            f" A decision model, Jev from TypeSafe AI, read the caller's request for a live call: {round(d.probability * 100)} percent yes, in {d.millis} milliseconds."
            for d in ctx.decisions[-1:]
        )
        + (f" {heard}" if (heard := voice_sentence(ctx.voice_readings)) else "")
    )


@tool
async def save_lead(
    name: str,
    need: str,
    runtime: ToolRuntime[CallContext, CallState],
    email: str = "",
    company: str = "",
    callback_number: str = "",
    call_notes: str = "",
    timeline: str = "",
    size: str = "",
) -> Command:
    """Send the caller's message to the owner: it is stored, and emailed to the owner's inbox at once. Use this when the caller wants to leave a message, send or email the owner something, or get a follow-up. Call it once, after the caller confirms your one-sentence read-back.

    Args:
        name: The caller's name.
        need: The caller's message for the owner, with every detail the caller gave, such as the project, the role, the timing, and their questions. Use only what the caller said: add nothing, guess nothing, and do not list what they left out. One sentence is right when the caller said little.
        email: The caller's email address, only if they gave one. "on file" reuses the address from an earlier call. Leave empty otherwise.
        company: The caller's company, only if they gave one. Leave empty otherwise.
        callback_number: A phone number the caller asked to be called on, only if it differs from the number they are calling from.
        call_notes: One sentence for the owner on what the caller asked other agents earlier on this call, for example "Asked the tour agent how the system works and what a call costs." Never repeat the message here. Leave empty if the caller only left the message.
        timeline: When the caller wants the work to start, in their words, only if they said it on this call. Leave empty otherwise.
        size: How big the work is, in the caller's words, for example "a few weeks" or "a long engagement", only if they said it on this call. Leave empty otherwise.
    """
    state = runtime.state
    result, lead = await save_lead_record(
        runtime.context,
        name=name,
        need=need,
        email=email,
        company=company,
        callback_number=callback_number,
        call_notes=call_notes,
        timeline=timeline,
        size=size,
        visited=["classifier", *state.get("handoffs", [])],
    )
    update: dict = {"messages": [ToolMessage(content=result, tool_call_id=runtime.tool_call_id)]}
    if lead:
        update["lead"] = lead
    return Command(update=update)


@tool
def identify_caller(name: str, runtime: ToolRuntime[CallContext, CallState]) -> str:
    """Tell the system who the caller is, right after they say their name. Code checks the name against earlier callers on this number. The result gives the exact words to say next.

    Args:
        name: The name the caller just gave, as you heard it. Empty when they gave no name.
    """
    return identify(runtime.context, name)


@tool
def verify_passphrase(passphrase: str, runtime: ToolRuntime[CallContext, CallState]) -> str:
    """Check the passphrase of a returning caller. Call it only after the identify_caller result told you to ask for the passphrase. Code compares it. The result gives the exact words to say next.

    Args:
        passphrase: The word or phrase the caller said, as you heard it. Empty when they do not know it or will not say it.
    """
    return verify(runtime.context, passphrase)


@tool
def set_passphrase(passphrase: str, runtime: ToolRuntime[CallContext, CallState]) -> str:
    """Keep a passphrase for the caller, so that this line can use what they said today on their next call. Call it only after the caller said a word or phrase for it. The result gives the exact words to say next.

    Args:
        passphrase: The word or phrase the caller chose, as you heard it.
    """
    return set_phrase(runtime.context, passphrase)


@tool
def forget_caller(runtime: ToolRuntime[CallContext, CallState]) -> str:
    """Delete what this line knows about the caller, and do not save this call. Call it only after the caller asked to be forgotten and said yes to your one question. The result gives the exact words to say."""
    return forget(runtime.context)[0]


def file_report(ctx: CallContext, kind: str, summary: str) -> str:
    """The feedback agent read a report back, and the caller said yes. It is kept in the call context only:
    no request leaves the server during the call. After hangup it makes `feedback.report_feedback` run.
    Shared by text and voice modes."""
    kind, summary = kind.strip().lower(), " ".join(summary.split())
    if ctx.forgotten:
        return "Not filed: the caller asked to be forgotten, so this line keeps nothing from this call. Say that in one sentence, and ask if there is anything else."
    if kind not in (FAULT, IDEA):
        return f'Not filed: kind must be "{FAULT}" or "{IDEA}". Call file_feedback again, in silence, with one of them.'
    if not summary:
        return "Not filed: the summary is empty. Call file_feedback again, in silence, with the one sentence that the caller said yes to."
    if (kind, summary) in ctx.feedback_filed:
        return "Filed already on this call. Do not file it again. Ask if there is anything else."
    if len(ctx.feedback_filed) >= MAX_ITEMS:
        return f"Not filed: one call can file {MAX_ITEMS} reports at most, and this call has them. Say that in one sentence, and ask if there is anything else."
    ctx.feedback_filed.append((kind, summary))
    log.info("call %s: the feedback agent filed a report (%s)", ctx.call_control_id, kind)
    return f"{_say(feedback_filed(ctx.owner_name))} Then ask if there is anything else. Never promise a fix or a date."


@tool
def file_feedback(kind: str, summary: str, runtime: ToolRuntime[CallContext, CallState]) -> str:
    """File the caller's report about this phone line: a fault, or an idea for a feature. Call it once for each report, in silence, after the caller said yes to your one-sentence read-back. The result gives the exact words to say next.

    Args:
        kind: "fault" when something on the line went wrong, "idea" when the caller asks for a feature.
        summary: Your one-sentence report, as the caller confirmed it.
    """
    return file_report(runtime.context, kind, summary)


async def text_portfolio_link_to_caller(ctx: CallContext, *, owner_name: str) -> str:
    """Text the fixed portfolio link to the number the caller is calling from. Returns the text for the model.

    The body is fixed and the recipient is always the calling number, so a model cannot be talked
    into texting other content or other people. One text per call.
    """
    unavailable = "Texting is not available on this call. Do not promise a text. Offer to take their details instead."
    if ctx.texter is None or not ctx.portfolio_url or ctx.caller_number in UNKNOWN_NUMBERS:
        return unavailable
    if ctx.texts_sent:
        return "The link was already texted on this call. Tell the caller to check their messages."
    if ctx.caller_line_type and "mobile" not in ctx.caller_line_type.lower() and "voip" not in ctx.caller_line_type.lower():
        return "The caller's number looks like a landline, which cannot receive texts. Say so, and offer to take an email address instead."
    try:
        await ctx.texter.send_text(ctx.caller_number, f"{owner_name}'s AI line here. The link you asked for: {ctx.portfolio_url}")
    except Exception:
        log.exception("text to the caller failed")
        return "The text could not be sent. Apologise in a few words and offer to take their details instead."
    ctx.texts_sent += 1
    return "Text sent to the number the caller is calling from. Tell them it is on its way."


async def text_owner_before_transfer(ctx: CallContext, reason: str) -> None:
    """Best effort: tell the owner who is about to be connected. Never blocks or fails the transfer."""
    if ctx.texter is None:
        return
    who = ctx.caller_name or "A caller"
    try:
        await ctx.texter.send_text(ctx.owner_phone, f"AI line: connecting {who} to you now. Reason: {' '.join(reason.split())[:200]}")
    except Exception:
        log.exception("pre-transfer text to the owner failed")


async def note_caller_before_transfer(
    ctx: CallContext,
    *,
    lead_saved: bool,
    caller_name: str,
    company: str,
    reason: str,
    visited: list[str] | None = None,
    timeline: str = "",
    size: str = "",
) -> dict[str, str]:
    """Before the owner's phone rings, make sure the owner has a record of who called, so a call
    that is not answered still leaves a lead. Returns the lead saved now, or {} when one exists
    already or no name is known. Call it only when the transfer goes ahead: see `remember_refused_transfer`."""
    ctx.refused_transfer = None
    name = caller_name.strip() or ctx.caller_name
    if lead_saved or not name:
        return {}
    _result, lead = await save_lead_record(
        ctx, name=name, need=reason.strip() or "Asked to speak to the owner.", company=company, visited=visited, kind="transfer", timeline=timeline, size=size
    )
    return lead


def remember_refused_transfer(ctx: CallContext, *, caller_name: str, company: str, reason: str) -> None:
    """A refused transfer sends no email at once. The agent offers to take a message next, and that
    message is the owner's email. An email sent here went out before the caller said any message."""
    ctx.refused_transfer = {"name": caller_name.strip(), "company": company.strip(), "reason": reason.strip()}
    ctx.caller_name = ctx.caller_name or caller_name.strip()  # the next agent on this call does not ask again


async def note_refused_transfer_at_call_end(
    ctx: CallContext, *, lead_saved: bool, visited: list[str] | None = None, summary: CallSummary | None = None
) -> dict[str, str]:
    """At the end of a call: a caller who asked for the owner, was refused, and left no message
    still leaves a record. With a call summary, the record holds what the caller said about the work,
    not only the agent's one-line reason. Returns the lead saved now, or {}."""
    note, ctx.refused_transfer = ctx.refused_transfer, None
    if note is None:
        return {}
    reason = summary.owner_note if summary and summary.owner_note else note["reason"]
    return await note_caller_before_transfer(
        ctx,
        lead_saved=lead_saved,
        caller_name=note["name"],
        company=note["company"],
        reason=reason,
        visited=visited,
        timeline=summary.timeline if summary else "",
        size=summary.size if summary else "",
    )


# The fallback for the transfer check, when Jev is off or gave no decision (see `transfer_question`).
# Words a caller uses to ask for a live call. "Leave a message", "email him", and "have him call
# me" hold none of them. "Can you call Brad?" asks the agent to ring someone now; "can you call me
# back" and "can you call my office" do not.
LIVE_REQUEST = re.compile(
    r"\b(speak|spoke|talk|transfer|connect|patch)\w*|\bput me through\b|\bon the (phone|line)\b|\b(real|live) person\b|\bhuman\b"
    r"|\b(can|could|will|would) you (\w+ )?(call|ring|phone|dial) (?!me\b|us\b|my\b|back\b)\w+",
    flags=re.IGNORECASE,
)
CALLER_TURNS_CHECKED = 4


def asked_for_the_owner(lines: list[str]) -> bool:
    """True when one of the caller's last turns asks, in the caller's own words, for a live call."""
    said = [line for line in lines if line.startswith("caller:")][-CALLER_TURNS_CHECKED:]
    return any(LIVE_REQUEST.search(line.removeprefix("caller:")) for line in said)


# Jev gives the probability that the caller asked for a live call.
# A transfer rings the owner's phone, so a yes needs more than the 0.7 that TypeSafe gives as a start.
# Between the two limits nobody knows: the agent asks the caller.
LIVE_CALL_YES = 0.8
LIVE_CALL_NO = 0.3

NOT_ASKED = (
    "No transfer was made, and nothing was saved. The caller did not ask, in their own words, to speak to the owner now. "
    "A caller who wants to leave a message, send an email, or get a call back wants a message: take it and call save_lead. "
    "If you are not sure which one they want, ask one question: do they want to leave a message, or be connected to the owner's phone now? Use the owner's name. "
    "Call request_owner_transfer again only if they choose the live connection."
)
ASK_WHICH = (
    "No transfer was made, and nothing was saved. It is not clear if the caller wants a live call or a message. "
    "Ask one question now: do they want to leave a message, or be connected to the owner's phone now? Use the owner's name. "
    "Call request_owner_transfer again only if they choose the live connection. If they choose a message, take it and call save_lead."
)


async def read_live_call(ctx: CallContext, lines: list[str]) -> Decision | None:
    """Jev's reading of the caller's request. None when Jev is off or gave no decision in time.
    A caller who asked to be forgotten is not sent to Jev."""
    if ctx.decider is None or ctx.forgotten or not any(line.startswith("caller:") for line in lines):
        return None
    decision = await ctx.decider.asked_for_live_call(lines, owner_name=ctx.owner_name)
    if decision is not None:
        ctx.decisions.append(decision)
    return decision


async def transfer_question(ctx: CallContext, lines: list[str]) -> str | None:
    """None when the transfer can go ahead. Otherwise the text for the model: no transfer yet, ask first.

    A transfer rings the owner's phone, so the voice model's reading of the caller is not enough.
    The caller's own words must ask for a live call. Jev reads them for their meaning (`decisions.py`):
    a high probability lets the transfer go ahead, a middle one makes the agent ask one question,
    and a low one means no transfer. One word can mean opposite things ("can I talk to him?", "I can't
    talk right now"), so the word list decides only when Jev is off or gave no decision in time.
    The caller's answer to the agent's question decides, so the check passes on the next request.
    """
    if ctx.transfer_question_asked:
        return None
    listed = asked_for_the_owner(lines)
    decision = await read_live_call(ctx, lines)
    if decision is None:  # Jev is off, too slow, or failed: the word list is the fallback
        ctx.transfer_question_asked = not listed
        return None if listed else NOT_ASKED
    if decision.probability >= LIVE_CALL_YES:
        outcome, question = "the transfer goes ahead", None
    elif decision.probability > LIVE_CALL_NO:
        outcome, question = "the agent asks the caller", ASK_WHICH
    else:
        outcome, question = "no transfer", NOT_ASKED
    log.info(
        "call %s: Jev read the transfer request: %.2f in %d ms (%s; word list: %s): %s",
        ctx.call_control_id, decision.probability, decision.millis, decision.model, "match" if listed else "miss", outcome,
    )  # fmt: skip
    ctx.transfer_question_asked = question is not None
    return question


# A short yes to the agent's "is that right?". A long turn holds new details, so it is not a yes.
CONFIRMS = re.compile(
    r"\b(yes|yeah|yep|yup|yea|correct|right|exactly|perfect|sure|okay|ok|sounds good|go ahead|please do|you got it|that's it|that is it|uh huh)\b",
    flags=re.IGNORECASE,
)
REJECTS = re.compile(r"\b(no|not|nope|wrong|wait|actually|instead|change)\b", flags=re.IGNORECASE)
CONFIRM_MAX_WORDS = 10
# The agent's question before that yes. A yes to "what's your name?" confirms nothing.
ASKS_IF_RIGHT = re.compile(
    r"\b(right|correct|accurate|confirm|sounds? (good|okay|ok)|looks? good|should i (save|send|pass))\b[^.?!]*\?", flags=re.IGNORECASE
)


def confirmed_the_message(lines: list[str]) -> bool:
    """True when the agent asked if the message is right, and the caller's words after it are a short yes."""
    rest = list(lines)
    after: list[str] = []
    while rest and rest[-1].startswith("caller:"):
        after.insert(0, rest.pop().removeprefix("caller:"))
    asked: list[str] = []
    while rest and rest[-1].startswith("agent:"):
        asked.insert(0, rest.pop().removeprefix("agent:"))
    said = " ".join(after)
    yes = bool(said) and len(said.split()) <= CONFIRM_MAX_WORDS and bool(CONFIRMS.search(said)) and not REJECTS.search(said)
    return yes and bool(ASKS_IF_RIGHT.search(" ".join(asked)))


SAVE_NOW = (
    "The caller said yes to the message, but nothing is saved and nothing was sent. "
    "Call save_lead now, with the details you read back. Say no words before it."
)


def missed_save(ctx: CallContext, lines: list[str]) -> int:
    """The agent asked "Is that right?", the caller said yes, and the agent answered without saving.
    Returns the number of lines up to the caller's yes (a key, so the agent is told once per yes), or 0.

    On 2026-11-17 a voice agent heard "Yeah." and said "Perfect, thanks. I'll take that as confirming",
    with no save_lead call. The caller's message never reached the owner.
    """
    if message_saved(ctx):
        return 0
    rest = list(lines)
    while rest and rest[-1].startswith("agent:"):
        rest.pop()
    if len(rest) == len(lines):  # the agent has not answered the yes yet
        return 0
    return len(rest) if confirmed_the_message(rest) else 0


def message_confirm_question(ctx: CallContext, lines: list[str]) -> str | None:
    """None when the message can be saved. Otherwise the text for the model: nothing saved yet, say it back first.

    A save emails the owner at once, so the model's memory of its instructions is not enough. On
    2026-11-16 an agent saved a message that the caller had not heard back. The check holds a save
    back once only: the caller's answer decides, so the next save_lead call passes.
    """
    if ctx.message_confirm_asked or confirmed_the_message(lines):
        return None
    ctx.message_confirm_asked = True
    return (
        "Not saved yet, and nothing was sent: the caller has not confirmed the message. Do not say that anything is saved. "
        "Say the message back in one sentence, and say how the owner will reach them. Then ask: \"Is that right?\" "
        "Do not ask for their name again. After they say yes, call save_lead again with the same details. If they change something, use the new details."
    )


NOT_CONFIRMED_NOTE = "The call ended before the caller confirmed this message."


async def save_held_message_at_call_end(ctx: CallContext, *, visited: list[str] | None = None) -> dict[str, str]:
    """At the end of a call: a message that waited for the caller's yes is saved as it is. An
    unconfirmed message is better for the owner than a lost one. Returns the lead saved now, or {}."""
    held, ctx.held_message = ctx.held_message, None
    # forget_caller clears a held message, so one that is here now came after it: the caller wants it kept.
    if not held:
        return {}
    notes = " ".join(part for part in (held.get("call_notes", "").strip(), NOT_CONFIRMED_NOTE) if part)
    _result, lead = await save_lead_record(ctx, **{**held, "call_notes": notes}, visited=visited)
    return lead


def transfer_closed_message(ctx: CallContext, now: datetime | None = None, caller_name: str = "") -> str | None:
    """None when a live transfer is allowed now. Otherwise the text for the model: no transfer, take a message."""
    name = caller_name.strip() or ctx.caller_name
    # The agent has the name already. Without this line it asked "what's your name?" again for the message.
    have_name = f" You have the caller's name ({name}). Do not ask for it again: ask only for the message." if name else ""
    if ctx.transfer_unavailable:
        return (
            f"No transfer was made. A live transfer is not available right now ({ctx.transfer_unavailable}). "
            "Do not give the caller the technical reason. Say the owner cannot take the call right now, and that the owner will follow up." + have_name
        )
    hours = ctx.transfer_hours
    if hours is None or hours.is_open(now):
        return None
    return (
        f"No transfer was made. The owner takes live transfers {hours.describe()}, and it is outside those hours now. "
        "Tell the caller that in one sentence. Then offer to take a message, so the owner can call them back." + have_name
    )


# What a caller says when the call is over for them. Checked on the caller's own words, because a
# hang-up in the middle of a conversation is worse than a call that stays open.
GOODBYE = re.compile(
    r"\b(bye|goodbye|good night|nothing else|take care|see you|all set|hang up|gotta go)\b"
    r"|\b(that's|that is|that'll be|that will be) (all|it|everything)\b"
    r"|\bno,? (thanks|thank you)\b"
    r"|\bhave a (good|great|nice)\b"
    r"|\btalk (to you )?(later|soon)\b"
    r"|\b(i'm|i am|we're|we are) (done|good|fine|finished)\b"
    r"|\b(have|got|need) to go\b"
    r"|\bend (the|this) call\b",
    flags=re.IGNORECASE,
)
# The answer to "anything else?". A plain "No, I don't." missed here, and the agent asked again.
ONLY_NO = re.compile(r"^\W*(no|nope|nah|not really|nothing)(\W+(i don't|i do not|not really|nothing|there isn't|there's not))?\W*$", flags=re.IGNORECASE)
GOODBYE_TURNS_CHECKED = 2


def said_goodbye(lines: list[str]) -> bool:
    """True when one of the caller's last two turns says the call is over for them."""
    said = [line.removeprefix("caller:") for line in lines if line.startswith("caller:")][-GOODBYE_TURNS_CHECKED:]
    return any(GOODBYE.search(text) or ONLY_NO.match(text) for text in said)


CALL_NOT_ENDED = (
    "The call was not ended: the caller did not say goodbye, and did not say they are done. Do not say goodbye. "
    "Answer what they last said, or ask if there is anything else you can help with."
)
SAY_GOODBYE = (
    'Say one short goodbye line now, for example: "Thanks for calling. Goodbye." Say nothing after it. '
    "The call ends when you stop speaking."
)


def say_goodbye(website_spoken: str = "") -> str:
    """The end_call result. With a website, the goodbye names it, so every caller leaves with the address."""
    if not website_spoken:
        return SAY_GOODBYE
    return (
        f'Say one short goodbye line now that gives the website, for example: "Thanks for calling. There\'s more at {website_spoken}. Goodbye." '
        "Say nothing after it. The call ends when you stop speaking."
    )


@tool
async def end_call(runtime: ToolRuntime[CallContext, CallState]) -> Command:
    """Hang up the phone call. Call this when the caller says goodbye, or says they need nothing else. Say nothing before you call it: the result gives you the goodbye line to say. Never call it while the caller still has a question."""
    ctx = runtime.context
    if not said_goodbye(transcript_lines(runtime.state.get("messages", []))):
        return Command(update={"messages": [ToolMessage(content=CALL_NOT_ENDED, tool_call_id=runtime.tool_call_id)]})
    await ctx.telephony.hangup(ctx.call_control_id)
    return Command(update={"call_ended": True, "messages": [ToolMessage(content=say_goodbye(ctx.website_spoken), tool_call_id=runtime.tool_call_id)]})


@tool
async def text_portfolio_link(runtime: ToolRuntime[CallContext, CallState]) -> str:
    """Text the caller a link to the owner's work, to the number they are calling from. Call this only after the caller says yes to a text."""
    from ai_call_center.config import get_settings

    return await text_portfolio_link_to_caller(runtime.context, owner_name=get_settings().owner_name)


@tool
async def request_owner_transfer(reason: str, runtime: ToolRuntime[CallContext, CallState], caller_name: str = "", company: str = "") -> Command:
    """Ring the owner's phone and move this live call to it. Call this only when the caller asked, in their own words, to speak to, talk to, or be connected to the owner now. Never call it for a caller who wants to leave a message, send or email the owner something, or get a call back: that is save_lead. It also saves the caller's name and reason for the owner, so save_lead is not needed first. Say nothing before you call it: the result gives you the one line to say.

    Args:
        reason: One line on why the caller wants to speak to the owner.
        caller_name: The caller's name. Ask for it first if you do not have it.
        company: The caller's company, only if they gave one.
    """
    ctx = runtime.context
    update: dict = {}
    state = runtime.state
    if question := await transfer_question(ctx, transcript_lines(state.get("messages", []))):
        return Command(update={"messages": [ToolMessage(content=question, tool_call_id=runtime.tool_call_id)]})
    if closed := transfer_closed_message(ctx, caller_name=caller_name):
        remember_refused_transfer(ctx, caller_name=caller_name, company=company, reason=reason)
        return Command(update={"messages": [ToolMessage(content=closed, tool_call_id=runtime.tool_call_id)]})
    visited = ["classifier", *state.get("handoffs", [])]
    if lead := await note_caller_before_transfer(
        ctx, lead_saved=bool(state.get("lead")), caller_name=caller_name, company=company, reason=reason, visited=visited
    ):
        update["lead"] = lead
    await text_owner_before_transfer(ctx, reason)
    await ctx.telephony.transfer(ctx.call_control_id, to=ctx.owner_phone)
    return Command(
        update={
            **update,
            "transfer_requested": True,
            "messages": [
                ToolMessage(
                    content=f"Transfer to the owner requested ({reason.strip()}). Tell the caller you are connecting them now, in one sentence.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def get_call_stats(runtime: ToolRuntime[CallContext, CallState]) -> str:
    """Get live facts about this call: how long it has run, which agents have spoken, and on which voice provider each one ran.
    On a phone call it also gives the measured answer speed, the handoff time, the tokens used, and an estimate of the model cost."""
    state = runtime.state
    return describe_call(
        runtime.context,
        visited=["classifier", *state.get("handoffs", [])],
        active_agent=state.get("active_agent", "classifier"),
        lead_saved=bool(state.get("lead")),
        transfer_requested=bool(state.get("transfer_requested")),
    )


@tool
async def crm_lookup(runtime: ToolRuntime[CallContext, CallState]) -> str:
    """Look up where the caller's own earlier request stands in the owner's CRM, over MCP. Call it when a returning caller asks if the owner got their message, or where their project or role stands.
    It reads a record only after the passphrase check of this call. The result gives you what to say."""
    return await request_status(runtime.context)


@tool
def search_writing(query: str) -> str:
    """RAG search over the owner's published essays on software engineering, architecture, teams, hiring, and AI agents.
    Returns the few passages that best match the question, never whole essays.
    query: the caller's question, in plain words, with the topic in it."""
    return search_result(default_library(), query)
