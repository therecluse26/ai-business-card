"""Tools with side effects. LangGraph executes them; the voice model only requests them."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ai_call_center.agents.specs import LEGS
from ai_call_center.graph.state import CallContext, CallState
from ai_call_center.leads import Lead, now_iso

log = logging.getLogger(__name__)


async def save_lead_record(ctx: CallContext, *, name: str, email: str, company: str, need: str) -> tuple[str, dict[str, str]]:
    """Store the lead and email it. Returns the text for the model and the lead as saved.

    Shared by the LangGraph tool (text mode) and the voice tool handler (phone mode).
    """
    lead = Lead(
        name=name.strip(),
        email=email.strip().lower(),
        company=company.strip(),
        need=need.strip(),
        caller_number=ctx.caller_number,
        call_control_id=ctx.call_control_id,
        created_at=now_iso(),
    )
    row_id = ctx.leads.save(lead)
    # The result text never contains the owner's address, so the agent cannot repeat it.
    try:
        await ctx.notifier.send_lead(lead, row_id=row_id)
        result = f"Lead saved as record {row_id} and emailed to the owner."
    except Exception:
        log.exception("Lead %s saved, but the owner email failed", row_id)
        result = (
            f"Lead saved as record {row_id}. The email to the owner failed, but the details are stored. "
            "Do not mention the failure. Tell the caller the owner will see their details."
        )
    return result, {"id": str(row_id), "name": lead.name, "email": lead.email, "company": lead.company, "need": lead.need}


def describe_call(ctx: CallContext, *, visited: list[str], active_agent: str, lead_saved: bool, transfer_requested: bool) -> str:
    """Live facts about the call, as one spoken paragraph. Shared by text and voice modes."""
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
    )


@tool
async def save_lead(name: str, email: str, company: str, need: str, runtime: ToolRuntime[CallContext, CallState]) -> Command:
    """Save the caller's contact details as a lead and email them to the owner. Call this only after the caller confirms the read-back.

    Args:
        name: The caller's full name.
        email: The caller's email address, confirmed letter by letter.
        company: The caller's company or organisation. Use "none" if they have none.
        need: One or two sentences on what the caller is interested in or needs.
    """
    result, lead = await save_lead_record(runtime.context, name=name, email=email, company=company, need=need)
    return Command(update={"lead": lead, "messages": [ToolMessage(content=result, tool_call_id=runtime.tool_call_id)]})


@tool
async def request_owner_transfer(reason: str, runtime: ToolRuntime[CallContext, CallState]) -> Command:
    """Transfer this live call to the owner's phone. Call this only after the caller agrees to the transfer.

    Args:
        reason: One line on why the caller wants to speak to the owner.
    """
    ctx = runtime.context
    await ctx.telephony.transfer(ctx.call_control_id, to=ctx.owner_phone)
    return Command(
        update={
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
    """Get live facts about this call: how long it has run, which agents have spoken, and on which voice provider each one ran."""
    state = runtime.state
    return describe_call(
        runtime.context,
        visited=["classifier", *state.get("handoffs", [])],
        active_agent=state.get("active_agent", "classifier"),
        lead_saved=bool(state.get("lead")),
        transfer_requested=bool(state.get("transfer_requested")),
    )
