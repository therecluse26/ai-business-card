"""Telephony seam. M1 ships a recording fake; M2 adds the Telnyx client behind the same protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Telephony(Protocol):
    async def transfer(self, call_control_id: str, *, to: str) -> None:
        """Transfer the live call to a phone number or SIP URI."""


@dataclass
class TransferRequest:
    call_control_id: str
    to: str


@dataclass
class RecordingTelephony:
    """Records transfer requests instead of making them. Used by tests and chat.py."""

    transfers: list[TransferRequest] = field(default_factory=list)

    async def transfer(self, call_control_id: str, *, to: str) -> None:
        self.transfers.append(TransferRequest(call_control_id=call_control_id, to=to))
