"""Live transfer to the owner's phone, and the way back when the owner does not pick up.

Telnyx refuses a transfer while a media stream is on (error 90045), so the stream stops first.
Stopping it closes our media WebSocket, which ends the AI session. If the transfer then fails, or
the owner's phone rings out, the caller would sit in silence. So the manager starts the media
stream again on the caller's leg, and the contact agent comes back with the conversation so far.
Telnyx's built-in voice speaks only if that restart fails too.

The same restart is the way out when a voice provider stops answering: `restart` stops the
stream and starts it again, and the new AI session builds its agents on another provider. When
no provider is left, `last_resort` uses Telnyx's voice to say so, then transfers the call to the
owner or ends it.

How the manager learns the outcome: the transfer command tags the new leg (the owner's phone)
with `client_state`, and Telnyx sends that state back on every webhook for that leg.

Call screening (`screen`, the default on the server): the manager does not use Telnyx's
transfer. It calls the owner as a separate leg, and the caller hears a ring tone the whole
time, never the owner's phone. When that leg is answered, a Telnyx voice says who is calling
and asks for a key. On 1 the manager connects the two legs. On 2, on no key, or when the phone
rings out, it hangs up the owner's leg and the agent comes back. A voicemail cannot press 1, so
the caller never hears it. With `detect_voicemail`, Telnyx also reports a machine, and the agent
comes back a few seconds sooner.

Without screening, voicemail: the owner's voicemail answers like a person does, so "answered" does not mean the
owner is there. With `detect_voicemail`, Telnyx listens after the answer and reports "human" or
"machine". On "machine" the manager hangs up the owner's leg and the agent comes back, as for a
phone that rang out. The caller can hear the first seconds of the voicemail greeting. The same
option keeps the caller's leg alive when the owner's leg ends, so the manager hangs up the
caller's leg itself after a normal conversation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from ai_call_center.telephony.stream_tokens import StreamTokens

log = logging.getLogger(__name__)

APOLOGY = "Sorry, I could not connect the call. The owner will follow up with you. Goodbye."
KEYS = "Press 1 to accept. Press 2 to decline."
FALLBACK_ANNOUNCE = f"A caller is on your voice card line, and the AI agents have a technical problem. {KEYS}"
ANNOUNCE_TRIES = 2  # the question is asked twice: the first words can be lost when a mobile phone answers


def screening_prompt(caller_name: str, reason: str = "") -> str:
    """What the owner hears when they answer: who is calling, why, and the two keys."""
    name = " ".join(caller_name.split())[:60] or "a caller"
    why = " ".join(reason.split())[:200].rstrip(".")
    return f"Voice card call from {name}. {why + '. ' if why else ''}{KEYS}"


class TransferTelephony(Protocol):
    async def stop_streaming(self, call_control_id: str) -> None: ...
    async def start_streaming(self, call_control_id: str, *, stream_url: str) -> None: ...
    async def transfer(
        self, call_control_id: str, *, to: str, timeout_secs: int | None = None, state: dict | None = None, detect_voicemail: bool = False
    ) -> None: ...
    async def hangup(self, call_control_id: str) -> None: ...
    async def dial(self, *, to: str, timeout_secs: int, state: dict, detect_voicemail: bool = False) -> str: ...
    async def play_audio(self, call_control_id: str, *, url: str) -> None: ...
    async def stop_audio(self, call_control_id: str) -> None: ...
    async def ask_digit(self, call_control_id: str, text: str, *, digits: str, timeout_secs: int, tries: int, state: dict) -> None: ...
    async def bridge(self, call_control_id: str, other: str) -> None: ...
    async def speak(self, call_control_id: str, text: str, *, then_hangup: bool = False, state: dict | None = None) -> None: ...


@dataclass
class Snapshot:
    """What the AI session knew when it handed the call over. A resumed session starts from it."""

    messages: list[Any] = field(default_factory=list)  # the shared conversation context
    handoffs: list[str] = field(default_factory=list)
    lead_saved: bool = False
    transfer_note_id: int = 0  # see `CallContext.transfer_note_id`
    refused_transfer: dict[str, str] | None = None  # see `CallContext.refused_transfer`
    identity_confirmed: bool = False
    identity_verified: bool = False  # see `CallContext.identity_verified`
    open_calls: list[str] = field(default_factory=list)  # see `CallContext.open_calls`
    phrase_closed: bool = False  # see `CallContext.phrase_closed`
    phrase_offered: bool = False  # see `CallContext.phrase_offered`
    phrase_set: bool = False  # see `CallContext.phrase_set`
    spoken_phrases: list[str] = field(default_factory=list)  # see `CallContext.spoken_phrases`
    caller_name: str = ""
    identified: bool = False  # identify_caller has run; see `CallContext.identified`
    returning_name: str = ""  # see `CallContext.returning_name`
    forgotten: bool = False  # see `CallContext.forgotten`
    reason: str = ""  # set on resume: "no_answer" or "failed" (a transfer), or "provider_failed"
    entry_agent: str = "contact"  # the agent that speaks first when the session starts again
    failovers: int = 0  # how many times this call has moved to another voice provider
    stats: object | None = None  # the call's live figures (`voice/stats.py`), kept across the restart


@dataclass
class Fallback:
    """What Telnyx's built-in voice says when no AI agent can speak, and where the call goes then."""

    to: str = ""  # the owner's phone; "" = no transfer, say goodbye and hang up
    connecting: str = "Sorry, the AI agents on this line have a technical problem. I am connecting you to the owner now."
    goodbye: str = "Sorry, the AI agents on this line have a technical problem right now. Please call again in a few minutes. Goodbye."
    no_answer: str = "Sorry, the owner could not take the call. Please call again later. Goodbye."


@dataclass
class _Pending:
    snapshot: Snapshot
    answered: bool = False
    voicemail: bool = False  # a machine answered the owner's leg
    announce: str = ""  # what the owner hears before they accept, with call screening
    screened: bool = False  # this transfer uses call screening
    owner_leg: str = ""
    asked: bool = False  # the owner's leg was answered, and the question was asked
    declined: bool = False  # the owner pressed 2, or pressed nothing
    failed: bool = False  # the owner accepted, and the two legs could not be connected
    fallback: Fallback | None = None  # set for a transfer made without an AI session to come back to


class TransferManager:
    """One per server. Runs outside the call's Pipecat pipeline, because the pipeline ends when the stream stops."""

    def __init__(
        self,
        telephony: TransferTelephony,
        *,
        stream_url: str,
        ring_secs: int = 25,
        detect_voicemail: bool = False,
        screen: bool = False,
        hold_audio_url: str = "",
        accept_secs: int = 6,
        tokens: StreamTokens | None = None,
    ) -> None:
        self.telephony = telephony
        self.stream_url = stream_url
        self.tokens = tokens  # with tokens, each restarted stream gets a URL with its own one-time token
        self.ring_secs = ring_secs
        self.detect_voicemail = detect_voicemail
        self.screen = screen
        self.hold_audio_url = hold_audio_url
        self.accept_secs = accept_secs
        self._owner_legs: dict[str, str] = {}  # the owner's leg -> the caller's leg, while a transfer is pending
        self.unavailable = ""  # why no transfer can be made on this server; "" when one can
        self.checked: asyncio.Future | None = None  # the start-up check that can set `unavailable`
        self._pending: dict[str, _Pending] = {}
        self._resumes: dict[str, Snapshot] = {}
        self._after_speech: dict[str, Fallback] = {}  # calls where Telnyx's voice speaks before a transfer
        self.restart_attempts = 3
        self.restart_retry_secs = 0.4
        self._tasks: set[asyncio.Task] = set()

    def _stream_url_for(self, call_control_id: str) -> str:
        return self.tokens.url_for(call_control_id) if self.tokens is not None else self.stream_url

    async def settled(self, timeout: float = 3.0) -> None:
        """Wait until `unavailable` is final. The server checks the Telnyx application while it already takes
        calls, so the first call after a start can arrive before that check ends. After `timeout` the call
        goes on with the value so far."""
        if self.checked is None or self.checked.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(self.checked), timeout)
        except Exception:  # noqa: BLE001 - a late or failed check must not stop a call
            log.warning("the start-up check of the Telnyx application did not end in %.0f s; the call goes on", timeout)

    def _spawn(self, coro) -> asyncio.Task:
        # Its own task: the call's pipeline, and every task inside it, is cancelled when the stream stops.
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def restart(self, call_control_id: str, *, snapshot: Snapshot, fallback: Fallback) -> asyncio.Task:
        """Stop the media stream and start it again, so a new AI session takes the call from the
        snapshot. Used when a voice provider stops answering. Returns at once."""
        return self._spawn(self._restart(call_control_id, snapshot, fallback))

    async def _restart(self, call_control_id: str, snapshot: Snapshot, fallback: Fallback) -> None:
        try:
            await self.telephony.stop_streaming(call_control_id)
        except Exception as exc:  # noqa: BLE001 - the stream may be gone already; the start below decides
            log.warning("call %s: could not stop the media stream before the restart (%s)", call_control_id, exc)
        self._resumes[call_control_id] = snapshot
        # Telnyx can refuse a start while the old stream is still closing, so try a few times.
        for attempt in range(1, self.restart_attempts + 1):
            try:
                await self.telephony.start_streaming(call_control_id, stream_url=self._stream_url_for(call_control_id))
                log.info("call %s: media stream restarted (%s)", call_control_id, snapshot.reason)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("call %s: could not restart the media stream, attempt %d of %d (%s)", call_control_id, attempt, self.restart_attempts, exc)
                if attempt < self.restart_attempts:
                    await asyncio.sleep(self.restart_retry_secs)
        self._resumes.pop(call_control_id, None)
        await self._last_resort(call_control_id, fallback)

    def last_resort(self, call_control_id: str, *, fallback: Fallback) -> asyncio.Task:
        """No AI agent can speak. Telnyx's voice says so; then the call goes to the owner, or ends. Returns at once."""
        return self._spawn(self._last_resort(call_control_id, fallback))

    async def _last_resort(self, call_control_id: str, fallback: Fallback) -> None:
        # A transfer needs the stream to be stopped. Stopping a stream that is already stopped is harmless.
        try:
            await self.telephony.stop_streaming(call_control_id)
        except Exception as exc:  # noqa: BLE001
            log.info("call %s: no media stream to stop before the spoken fallback (%s)", call_control_id, exc)
        try:
            if fallback.to:
                # The transfer starts when Telnyx reports that the sentence has ended (`on_event`).
                self._after_speech[call_control_id] = fallback
                await self.telephony.speak(call_control_id, fallback.connecting, state={"then": "transfer"})
            else:
                await self.telephony.speak(call_control_id, fallback.goodbye, then_hangup=True)
            log.warning("call %s: spoken fallback (%s)", call_control_id, "then a transfer to the owner" if fallback.to else "then a hang-up")
        except Exception as exc:  # noqa: BLE001 - the caller has most likely hung up
            self._after_speech.pop(call_control_id, None)
            log.warning("call %s: the spoken fallback failed (%s)", call_control_id, exc)

    async def _transfer_after_speech(self, call_control_id: str, fallback: Fallback) -> None:
        self._pending[call_control_id] = _Pending(Snapshot(), fallback=fallback, announce=FALLBACK_ANNOUNCE)
        try:
            await self._ring(call_control_id, fallback.to)
            log.info("call %s: fallback transfer accepted; the owner's phone rings for up to %d s", call_control_id, self.ring_secs)
        except Exception:
            log.exception("call %s: the fallback transfer failed", call_control_id)
            await self._resume(call_control_id, "failed")

    def start(self, call_control_id: str, *, to: str, snapshot: Snapshot, announce: str = "") -> asyncio.Task:
        """Stop the stream and transfer the call. Returns at once; the work runs in its own task.
        `announce` is what the owner hears before they accept, with call screening."""
        self._pending[call_control_id] = _Pending(snapshot, announce=announce or screening_prompt(snapshot.caller_name))
        return self._spawn(self._run(call_control_id, to))

    async def _run(self, call_control_id: str, to: str) -> None:
        try:
            await self.telephony.stop_streaming(call_control_id)
        except Exception as exc:  # noqa: BLE001 - most likely the caller hung up during the goodbye line
            log.warning("call %s: could not stop the media stream before the transfer (%s); no transfer made", call_control_id, exc)
            self._pending.pop(call_control_id, None)
            return
        try:
            await self._ring(call_control_id, to)
            log.info("call %s: transfer command accepted; the owner's phone rings for up to %d s", call_control_id, self.ring_secs)
        except Exception:
            log.exception("call %s: transfer failed after the stream stopped", call_control_id)
            await self._resume(call_control_id, "failed")

    async def _transfer(self, call_control_id: str, to: str) -> None:
        extra = {"detect_voicemail": True} if self.detect_voicemail else {}
        await self.telephony.transfer(call_control_id, to=to, timeout_secs=self.ring_secs, state={"transfer_of": call_control_id}, **extra)

    async def _ring(self, caller_leg: str, to: str) -> None:
        """Ring the owner's phone: with call screening when it is on, else with Telnyx's transfer."""
        pending = self._pending.get(caller_leg)
        if not self.screen or pending is None:
            await self._transfer(caller_leg, to)
            return
        pending.screened = True
        if self.hold_audio_url:
            try:
                await self.telephony.play_audio(caller_leg, url=self.hold_audio_url)
            except Exception as exc:  # noqa: BLE001 - the caller waits in silence; the call to the owner still goes out
                log.warning("call %s: could not start the ring tone for the caller (%s)", caller_leg, exc)
        owner_leg = await self.telephony.dial(
            to=to, timeout_secs=self.ring_secs, state={"transfer_of": caller_leg}, detect_voicemail=self.detect_voicemail
        )
        if owner_leg:
            pending.owner_leg = owner_leg
            self._owner_legs[owner_leg] = caller_leg

    async def _quiet(self, what: str, call_control_id: str, command) -> None:
        """Run a command that may find the call gone already. Such a failure changes nothing."""
        try:
            await command
        except Exception as exc:  # noqa: BLE001
            log.info("call %s: %s did nothing (%s)", call_control_id, what, exc)

    async def _ask_owner(self, caller_leg: str, owner_leg: str, pending: _Pending) -> None:
        try:
            await self.telephony.ask_digit(
                owner_leg, pending.announce, digits="12", timeout_secs=self.accept_secs, tries=ANNOUNCE_TRIES, state={"transfer_of": caller_leg}
            )
            log.info("call %s: the owner's phone was answered; asking for a key", caller_leg)
        except Exception as exc:  # noqa: BLE001
            log.warning("call %s: could not ask the owner for a key (%s)", caller_leg, exc)
            pending.declined = True
            await self._quiet("the hang-up of the owner's leg", caller_leg, self.telephony.hangup(owner_leg))  # its hang-up event brings the agent back

    async def _owner_answered_question(self, caller_leg: str, owner_leg: str, pending: _Pending, digits: str) -> None:
        if pending.answered or pending.declined:
            return
        if digits == "1":
            await self._quiet("the stop of the ring tone", caller_leg, self.telephony.stop_audio(caller_leg))
            try:
                await self.telephony.bridge(caller_leg, owner_leg)
                pending.answered = True
                log.info("call %s: the owner accepted; the two calls are connected", caller_leg)
                return
            except Exception:
                log.exception("call %s: the owner accepted, and the two calls could not be connected", caller_leg)
                pending.failed = True
        else:
            log.info("call %s: the owner did not accept (%s)", caller_leg, "pressed 2" if digits == "2" else "no key; a voicemail gives none")
        pending.declined = True
        await self._quiet("the hang-up of the owner's leg", caller_leg, self.telephony.hangup(owner_leg))  # its hang-up event brings the agent back

    async def on_event(self, event_type: str | None, call_control_id: str | None, state: dict, result: str = "", digits: str = "") -> None:
        """Feed every Telnyx webhook here. `state` is the decoded `client_state` of the event.
        `result` is the payload's `result`: who answered, on `call.machine.detection.ended`.
        `digits` is the payload's `digits`: the key the owner pressed, on `call.gather.ended`."""
        # The owner's leg is known by its tag. Its call id is kept too, for an event that comes without the tag.
        caller_leg = state.get("transfer_of") or self._owner_legs.get(call_control_id or "")
        if caller_leg:  # an event for the owner's leg
            pending = self._pending.get(caller_leg)
            if pending is None:
                self._owner_legs.pop(call_control_id or "", None)
                return
            if call_control_id and call_control_id != caller_leg:
                self._owner_legs[call_control_id] = caller_leg
                pending.owner_leg = call_control_id
            owner_leg = call_control_id or pending.owner_leg
            if pending.screened:
                if event_type == "call.answered" and not pending.asked:
                    pending.asked = True  # "answered" can be a voicemail: only the key 1 connects the caller
                    await self._ask_owner(caller_leg, owner_leg, pending)
                elif event_type == "call.gather.ended":
                    await self._owner_answered_question(caller_leg, owner_leg, pending, digits)
                elif event_type == "call.machine.detection.ended":
                    log.info("call %s: the owner's leg was answered by: %s", caller_leg, result or "unknown")
                    if result == "machine" and not pending.answered and not pending.declined:
                        # No need to wait for the question to run out: a voicemail presses no key.
                        pending.voicemail = pending.declined = True
                        await self._quiet("the hang-up of the owner's voicemail", caller_leg, self.telephony.hangup(owner_leg))
            elif event_type in {"call.answered", "call.bridged"}:
                pending.answered = True
            elif event_type == "call.machine.detection.ended":
                log.info("call %s: the owner's leg was answered by: %s", caller_leg, result or "unknown")
                if result == "machine" and not pending.voicemail and call_control_id:
                    pending.voicemail = True
                    try:
                        await self.telephony.hangup(call_control_id)  # the hang-up event below brings the agent back
                    except Exception as exc:  # noqa: BLE001
                        log.warning("call %s: could not hang up the owner's voicemail (%s)", caller_leg, exc)
            if event_type == "call.hangup":
                self._owner_legs.pop(call_control_id or "", None)
                if pending.answered and not pending.voicemail:
                    self._pending.pop(caller_leg, None)  # a normal end: both sides talked
                    if self.detect_voicemail or pending.screened:
                        # The caller's leg can outlive the owner's leg. End it here.
                        await self._quiet("the hang-up after the owner left", caller_leg, self.telephony.hangup(caller_leg))
                else:
                    why = "the owner's voicemail answered" if pending.voicemail else ("the owner did not accept" if pending.declined else "the owner did not pick up")
                    log.info("call %s: %s; bringing the agent back", caller_leg, why)
                    await self._resume(caller_leg, "failed" if pending.failed else "no_answer")
        elif event_type == "call.speak.ended" and call_control_id in self._after_speech:
            await self._transfer_after_speech(call_control_id, self._after_speech.pop(call_control_id))
        elif event_type == "call.hangup" and call_control_id:
            # The caller hung up. Nothing to resume.
            gone = self._pending.pop(call_control_id, None)
            if gone is not None and gone.screened and gone.owner_leg:
                # With call screening the owner's leg is a call of its own: it would go on ringing.
                await self._quiet("the hang-up of the owner's leg", call_control_id, self.telephony.hangup(gone.owner_leg))
            for owner_leg in [leg for leg, caller in self._owner_legs.items() if caller == call_control_id]:
                self._owner_legs.pop(owner_leg, None)
            self._resumes.pop(call_control_id, None)
            self._after_speech.pop(call_control_id, None)

    async def _resume(self, call_control_id: str, reason: str) -> None:
        pending = self._pending.pop(call_control_id, None)
        if pending is None:
            return
        if pending.screened and self.hold_audio_url:
            await self._quiet("the stop of the ring tone", call_control_id, self.telephony.stop_audio(call_control_id))
        if pending.fallback is not None:
            # No AI session to come back to: Telnyx's voice ends the call.
            try:
                await self.telephony.speak(call_control_id, pending.fallback.no_answer, then_hangup=True)
            except Exception as exc:  # noqa: BLE001 - the caller has most likely hung up
                log.warning("call %s: the spoken goodbye failed (%s)", call_control_id, exc)
            return
        pending.snapshot.reason = reason
        self._resumes[call_control_id] = pending.snapshot
        try:
            await self.telephony.start_streaming(call_control_id, stream_url=self._stream_url_for(call_control_id))
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
