"""Call state (checkpointed) and call context (per run, not checkpointed)."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Literal, NotRequired

from langchain.agents import AgentState

from ai_call_center.agents.specs import Welcome
from ai_call_center.guardrails import TransferHours
from ai_call_center.leads import LeadStore, NumberInfo
from ai_call_center.notify import Notifier
from ai_call_center.telephony import Telephony, Texter


class CallState(AgentState):
    active_agent: NotRequired[str]
    handoffs: NotRequired[Annotated[list[str], operator.add]]
    lead: NotRequired[dict[str, str] | None]
    transfer_requested: NotRequired[bool]
    call_ended: NotRequired[bool]


@dataclass
class CallContext:
    call_control_id: str
    caller_number: str
    owner_phone: str
    leads: LeadStore
    telephony: Telephony
    notifier: Notifier
    legs: dict[str, str] = field(default_factory=dict)  # agent name -> leg id
    mode: Literal["text", "voice"] = "text"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    caller_history: str = ""  # notes from earlier calls; set by identify_caller after a name match
    identity_confirmed: bool = False  # the name the caller gave matched an earlier caller on this number
    returning_name: str = ""  # the matched earlier caller's name; only this person's details are reused
    identified: bool = False  # identify_caller has run on this call (with a name or without one)
    welcome: Welcome | None = None  # the words after the caller says who they are; None = no fixed words
    forgotten: bool = False  # the caller asked to be forgotten: nothing about this call is kept
    earlier_calls: int = 0  # finished calls from this number before this one, for the lead email
    number_info: NumberInfo | None = None  # the number lookup result, for the lead email
    # "caller: ..." and "agent: ..." lines of the call so far. The transfer check reads the caller's own words from it.
    transcript: Callable[[], list[str]] | None = None
    transfer_question_asked: bool = False  # the agent was told to ask "connect now, or take a message?"
    message_confirm_asked: bool = False  # the agent was told to say the message back before it is saved
    held_message: dict[str, str] | None = None  # a save_lead call that waits for the caller's yes; saved at the end of the call if no yes came
    caller_lookup: str = ""  # public carrier data for the number, for every agent's prompt
    caller_name: str = ""  # set when the caller confirms who they are or saves a lead; "" when unknown
    transfer_unavailable: str = ""  # why a live transfer cannot be made on this server; "" when it can
    transfer_note_id: int = 0  # the record that only says "asked to speak to the owner"; 0 when there is none
    refused_transfer: dict[str, str] | None = None  # who asked for a live call that was refused; emailed at the end of the call if no message came
    caller_line_type: str = ""  # from the number lookup: "mobile", "fixed line", ...
    texter: Texter | None = None  # None when texting is off
    portfolio_url: str = ""
    owner_name: str = ""  # for words that name the owner, for example after forget_caller
    website_spoken: str = ""  # the owner's website as a voice says it; the goodbye line names it
    texts_sent: int = 0
    transfer_hours: TransferHours | None = None  # None = a live transfer is allowed at any time
