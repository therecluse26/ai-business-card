"""Live transfer to the owner's phone, and the way back when the owner does not pick up.

Telnyx refuses a transfer while a media stream is on (error 90045), so the stream stops first.
Stopping it closes our media WebSocket, which ends the AI session. If the transfer then fails, or
the owner's phone rings out, the caller would sit in silence. So the manager starts the media
stream again on the caller's leg, and the contact agent comes back with the conversation so far.
Telnyx's built-in voice speaks only if that restart fails too.

How the manager learns the outcome: the transfer command tags the new leg (the owner's phone)
with `client_state`, and Telnyx sends that state back on every webhook for that leg.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

APOLOGY = "Sorry, I could not connect the call. The owner will follow up with you. Goodbye."


class TransferTelephony(Protocol):
    async def stop_streaming(self, call_control_id: str) -> None: ...
    async def start_streaming(self, call_control_id: str, *, stream_url: str) -> None: ...
    async def transfer(self, call_control_id: str, *, to: str, timeout_secs: int | None = None, state: dict | None = None) -> None: ...
    async def speak(self, call_control_id: str, text: str, *, then_hangup: bool = False) -> None: ...


@dataclass
class Snapshot:
    """What the AI session knew when it handed the call over. A resumed session starts from it."""

    messages: list[Any] = field(default_factory=list)  # the shared conversation context
    handoffs: list[str] = field(default_factory=list)
    lead_saved: bool = False
    identity_confirmed: bool = False
    caller_name: str = ""
    reason: str = ""  # set on resume: "no_answer" or "failed"


@dataclass
class _Pending:
    snapshot: Snapshot
    answered: bool = False


class TransferManager:
    """One per server. Runs outside the call's Pipecat pipeline, because the pipeline ends when the stream stops."""

    def __init__(self, telephony: TransferTelephony, *, stream_url: str, ring_secs: int = 25) -> None:
        self.telephony = telephony
        self.stream_url = stream_url
        self.ring_secs = ring_secs
        self.unavailable = ""  # why no transfer can be made on this server; "" when one can
        self._pending: dict[str, _Pending] = {}
        self._resumes: dict[str, Snapshot] = {}
        self._tasks: set[asyncio.Task] = set()

    def start(self, call_control_id: str, *, to: str, snapshot: Snapshot) -> asyncio.Task:
        """Stop the stream and transfer the call. Returns at once; the work runs in its own task."""
        self._pending[call_control_id] = _Pending(snapshot)
        task = asyncio.create_task(self._run(call_control_id, to))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(self, call_control_id: str, to: str) -> None:
        try:
            await self.telephony.stop_streaming(call_control_id)
        except Exception as exc:  # noqa: BLE001 - most likely the caller hung up during the goodbye line
            log.warning("call %s: could not stop the media stream before the transfer (%s); no transfer made", call_control_id, exc)
            self._pending.pop(call_control_id, None)
            return
        try:
            await self.telephony.transfer(call_control_id, to=to, timeout_secs=self.ring_secs, state={"transfer_of": call_control_id})
            log.info("call %s: transfer command accepted; the owner's phone rings for up to %d s", call_control_id, self.ring_secs)
        except Exception:
            log.exception("call %s: transfer failed after the stream stopped", call_control_id)
            await self._resume(call_control_id, "failed")

    async def on_event(self, event_type: str | None, call_control_id: str | None, state: dict) -> None:
        """Feed every Telnyx webhook here. `state` is the decoded `client_state` of the event."""
        caller_leg = state.get("transfer_of")
        if caller_leg:  # an event for the owner's leg
            pending = self._pending.get(caller_leg)
            if pending is None:
                return
            if event_type in {"call.answered", "call.bridged"}:
                pending.answered = True
            elif event_type == "call.hangup":
                if pending.answered:
                    self._pending.pop(caller_leg, None)  # a normal end: both sides talked
                else:
                    log.info("call %s: the owner did not pick up; bringing the agent back", caller_leg)
                    await self._resume(caller_leg, "no_answer")
        elif event_type == "call.hangup" and call_control_id:
            # The caller hung up. Nothing to resume.
            self._pending.pop(call_control_id, None)
            self._resumes.pop(call_control_id, None)

    async def _resume(self, call_control_id: str, reason: str) -> None:
        pending = self._pending.pop(call_control_id, None)
        if pending is None:
            return
        pending.snapshot.reason = reason
        self._resumes[call_control_id] = pending.snapshot
        try:
            await self.telephony.start_streaming(call_control_id, stream_url=self.stream_url)
        except Exception as exc:  # noqa: BLE001
            self._resumes.pop(call_control_id, None)
            log.warning("call %s: could not restart the media stream (%s); using the spoken apology", call_control_id, exc)
            try:
                await self.telephony.speak(call_control_id, APOLOGY, then_hangup=True)
            except Exception as exc2:  # noqa: BLE001 - the caller has most likely hung up
                log.warning("call %s: the spoken apology failed too (%s)", call_control_id, exc2)

    def take_resume(self, call_control_id: str) -> Snapshot | None:
        """The snapshot for a media stream that has just started again, or None for a new call."""
        return self._resumes.pop(call_control_id, None)
