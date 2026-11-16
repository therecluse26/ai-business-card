"""Call state (checkpointed) and call context (per run, not checkpointed)."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Literal, NotRequired

from langchain.agents import AgentState

from ai_call_center.guardrails import TransferHours
from ai_call_center.leads import LeadStore
from ai_call_center.notify import Notifier
from ai_call_center.telephony import Telephony, Texter


class CallState(AgentState):
    active_agent: NotRequired[str]
    handoffs: NotRequired[Annotated[list[str], operator.add]]
    lead: NotRequired[dict[str, str] | None]
    transfer_requested: NotRequired[bool]


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
    caller_greeting: str = ""  # for the greeting agent only: the first name to confirm, nothing else
    caller_history: str = ""  # notes from earlier calls; released by confirm_returning_caller after a yes
    identity_confirmed: bool = False  # the caller said they are the earlier caller
    caller_lookup: str = ""  # public carrier data for the number, for every agent's prompt
    caller_name: str = ""  # set when the caller confirms who they are or saves a lead; "" when unknown
    transfer_unavailable: str = ""  # why a live transfer cannot be made on this server; "" when it can
    caller_line_type: str = ""  # from the number lookup: "mobile", "fixed line", ...
    texter: Texter | None = None  # None when texting is off
    portfolio_url: str = ""
    texts_sent: int = 0
    transfer_hours: TransferHours | None = None  # None = a live transfer is allowed at any time
