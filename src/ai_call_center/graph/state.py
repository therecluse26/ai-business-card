"""Call state (checkpointed) and call context (per run, not checkpointed)."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Literal, NotRequired

from langchain.agents import AgentState

from ai_call_center.leads import LeadStore
from ai_call_center.notify import Notifier
from ai_call_center.telephony import Telephony


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
