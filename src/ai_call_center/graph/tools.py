"""Tools with side effects. LangGraph executes them; the voice model only requests them."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ai_call_center.agents.specs import LEGS
from ai_call_center.graph.state import CallContext, CallState
from ai_call_center.callers import UNKNOWN_NUMBERS
from ai_call_center.leads import Lead, now_iso

log = logging.getLogger(__name__)


ON_FILE = {"on file", "onfile", "on-file"}


def release_caller_history(ctx: CallContext) -> str:
    """The caller said they are the earlier caller: hand the notes to the agent. Shared by text and voice modes."""
    if not ctx.caller_history:
        return "No earlier calls are on record for this caller. Treat them as a new caller."
    ctx.identity_confirmed = True
    return ctx.caller_history


async def save_lead_record(
    ctx: CallContext, *, name: str, need: str, email: str = "", company: str = "", callback_number: str = ""
) -> tuple[str, dict[str, str]]:
    """Store the lead and email it. Returns the text for the model and the lead as saved.

    Only a name and a need are required: the owner can call back on the number the call came from.
    A caller who confirmed their identity can reuse the email address on file; the address goes
    from the store to the new row and never enters a prompt or a tool result.

    Shared by the LangGraph tool (text mode) and the voice tool handler (phone mode).
    """
    name, need, company, callback_number = name.strip(), need.strip(), company.strip(), callback_number.strip()
    email = email.strip().lower()
    wants_file = email in ON_FILE
    if wants_file:
        email = ""
    if ctx.identity_confirmed and ctx.caller_number not in UNKNOWN_NUMBERS:
        prior = ctx.leads.latest_lead_for(ctx.caller_number, with_email=True) or ctx.leads.latest_lead_for(ctx.caller_number)
        if prior is not None:
            email = email or prior.email
            company = company or prior.company
            name = name or prior.name
    if not name or not need:
        return "Not saved: a name and one line on what the caller wants are both needed. Ask for what is missing, then call save_lead again.", {}
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
        same_person = saved.name.casefold() == name.casefold()
        nothing_new = not callback_number and email in ("", saved.email) and company.casefold() in ("", saved.company.casefold())
        if same_person and nothing_new:
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
    )
    row_id = ctx.leads.save(lead)
    ctx.caller_name = lead.name
    reach = "by email" if email else ("on the callback number they gave" if callback_number else "on the number they are calling from")
    if wants_file and not email:
        reach += "; no email address was on file"
    # The result text never contains the owner's address, so the agent cannot repeat it.
    try:
        await ctx.notifier.send_lead(lead, row_id=row_id)
        if not getattr(ctx.notifier, "delivers", True):
            # Email is not set up on this server. The agent must not claim an email was sent.
            result = f"Lead saved as record {row_id}. The owner will see it. Say the owner has their details; do not say an email was sent."
        else:
            result = f"Lead saved as record {row_id} and emailed to the owner."
        result += f" The owner can reach the caller {reach}."
        # The next step comes from here, not from the model's memory of its instructions.
        if transfer_closed_message(ctx) is None:
            result += (
                " Next, in this same turn: tell them the owner has their details, and offer once to connect them to the owner now. "
                "If they already asked to speak to the owner, call request_owner_transfer now instead."
            )
        else:
            result += " A live transfer is not possible right now, so do not offer one. Tell the caller the owner has their details and will follow up."
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
    )


@tool
async def save_lead(
    name: str,
    need: str,
    runtime: ToolRuntime[CallContext, CallState],
    email: str = "",
    company: str = "",
    callback_number: str = "",
) -> Command:
    """Save the caller's details as a lead for the owner. Call this once, after the caller confirms your one-sentence read-back.

    Args:
        name: The caller's name.
        need: One or two sentences on what the caller wants.
        email: The caller's email address, only if they gave one. "on file" reuses the address from an earlier call. Leave empty otherwise.
        company: The caller's company, only if they gave one. Leave empty otherwise.
        callback_number: A phone number the caller asked to be called on, only if it differs from the number they are calling from.
    """
    result, lead = await save_lead_record(runtime.context, name=name, need=need, email=email, company=company, callback_number=callback_number)
    update: dict = {"messages": [ToolMessage(content=result, tool_call_id=runtime.tool_call_id)]}
    if lead:
        update["lead"] = lead
    return Command(update=update)


@tool
def confirm_returning_caller(runtime: ToolRuntime[CallContext, CallState]) -> str:
    """Get the notes from this caller's earlier calls. Call this only when a "Returning caller note" in your instructions names a first name, and the caller has just said yes, they are that person. Never call it after a no or an unclear answer. Say nothing before you call it; speak after you have the result."""
    return release_caller_history(runtime.context)


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


async def note_caller_before_transfer(ctx: CallContext, *, lead_saved: bool, caller_name: str, company: str, reason: str) -> dict[str, str]:
    """Before a live transfer, make sure the owner has a record of who called, so a call that is
    not answered still leaves a lead. Returns the lead saved now, or {} when one exists already
    or no name is known."""
    name = caller_name.strip() or ctx.caller_name
    if lead_saved or not name:
        return {}
    _result, lead = await save_lead_record(ctx, name=name, need=reason.strip() or "Asked to speak to the owner.", company=company)
    return lead


def transfer_closed_message(ctx: CallContext, now: datetime | None = None) -> str | None:
    """None when a live transfer is allowed now. Otherwise the text for the model: no transfer, take a message."""
    if ctx.transfer_unavailable:
        return (
            f"No transfer was made. A live transfer is not available right now ({ctx.transfer_unavailable}). "
            "Do not give the caller the technical reason. Say the owner cannot take the call right now, and that the owner will follow up."
        )
    hours = ctx.transfer_hours
    if hours is None or hours.is_open(now):
        return None
    return (
        f"No transfer was made. The owner takes live transfers {hours.describe()}, and it is outside those hours now. "
        "Tell the caller that in one sentence. Then offer to take their details and a message, so the owner can call them back."
    )


@tool
async def text_portfolio_link(runtime: ToolRuntime[CallContext, CallState]) -> str:
    """Text the caller a link to the owner's work, to the number they are calling from. Call this only after the caller says yes to a text."""
    from ai_call_center.config import get_settings

    return await text_portfolio_link_to_caller(runtime.context, owner_name=get_settings().owner_name)


@tool
async def request_owner_transfer(reason: str, runtime: ToolRuntime[CallContext, CallState], caller_name: str = "", company: str = "") -> Command:
    """Transfer this live call to the owner's phone. Call this when the caller wants to speak to the owner. It also saves the caller's name and reason for the owner, so save_lead is not needed first. Say nothing before you call it: the result gives you the one line to say.

    Args:
        reason: One line on why the caller wants to speak to the owner.
        caller_name: The caller's name. Ask for it first if you do not have it.
        company: The caller's company, only if they gave one.
    """
    ctx = runtime.context
    update: dict = {}
    if lead := await note_caller_before_transfer(ctx, lead_saved=bool(runtime.state.get("lead")), caller_name=caller_name, company=company, reason=reason):
        update["lead"] = lead
    if closed := transfer_closed_message(ctx):
        return Command(update={**update, "messages": [ToolMessage(content=closed, tool_call_id=runtime.tool_call_id)]})
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
