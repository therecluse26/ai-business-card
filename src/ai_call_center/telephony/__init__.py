"""Telephony seam. M1 ships a recording fake; M2 adds the Telnyx client behind the same protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Telephony(Protocol):
    async def transfer(self, call_control_id: str, *, to: str) -> None:
        """Transfer the live call to a phone number or SIP URI."""

    async def hangup(self, call_control_id: str) -> None:
        """End the live call."""


@runtime_checkable
class Texter(Protocol):
    async def send_text(self, to: str, body: str) -> None:
        """Send one SMS from the line's number. Raise on failure."""


@dataclass
class TransferRequest:
    call_control_id: str
    to: str


@dataclass
class RecordingTelephony:
    """Records transfer requests instead of making them. Used by tests and chat.py."""

    transfers: list[TransferRequest] = field(default_factory=list)
    texts: list[tuple[str, str]] = field(default_factory=list)  # (to, body)
    hangups: list[str] = field(default_factory=list)  # call ids

    async def transfer(self, call_control_id: str, *, to: str) -> None:
        self.transfers.append(TransferRequest(call_control_id=call_control_id, to=to))

    async def send_text(self, to: str, body: str) -> None:
        self.texts.append((to, body))

    async def hangup(self, call_control_id: str) -> None:
        self.hangups.append(call_control_id)
