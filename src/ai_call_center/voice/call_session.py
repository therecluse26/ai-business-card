"""One live call: Telnyx WebSocket ⇄ Pipecat pipeline ⇄ voice legs.

The speech-to-speech model hears and speaks. It asks for tools; this module runs them.

Every agent has its own voice service, built at call start and connected before the greeting,
behind Pipecat's `LLMSwitcher`. A handoff switches the active service and asks the new one to
answer from the shared context, so the caller hears a new voice within about a second.
`save_lead` and `get_call_stats` call the same code as the LangGraph tools.
`request_owner_transfer` arms a gate that moves the call to the owner's phone once the agent
has finished its goodbye line.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import cached_property

from fastapi import WebSocket
from langchain.chat_models import init_chat_model
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    FunctionCallResultProperties,
    InputAudioRawFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    ManuallySwitchServiceFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from ai_call_center import tracing
from ai_call_center.agents.specs import (
    LEG_PLATFORMS,
    LEGS,
    VOICE_LEVELS_DBFS,
    AgentSpec,
    Welcome,
    companies_phrase,
    greeting_tip,
    greeting_words,
    load_profile,
    web_address,
)
from ai_call_center.analytics import MESSAGE, NO_OUTCOME, TRANSFER, VOICE, CallAnalytics, call_day, forget_call, harness_version, record_call
from ai_call_center.caller_voice import VoiceReading, keep_reading
from ai_call_center.callers import (
    UNKNOWN_NUMBERS,
    CallSummary,
    lookup_text,
    summarize_call,
    transcript_lines,
)
from ai_call_center.config import Settings
from ai_call_center.decision_grader import default_decision_grader, feedback_chances
from ai_call_center.crm import default_crm, request_status, sync_call
from ai_call_center.decisions import default_decider
from ai_call_center.graph.build import ENTRY_AGENT, default_specs
from ai_call_center.graph.handoffs import handoff_tool_name
from ai_call_center.graph.state import CallContext
from ai_call_center.grading import describe_scores, differences, grade_call, rule_of
from ai_call_center.fault_report import FaultReport, summary_report, write_report
from ai_call_center.feedback import report_feedback
from ai_call_center.fix_check import check_fixes
from ai_call_center.improve import FixRequest, request_fixes
from ai_call_center.publish import publish_report
from ai_call_center.graph.tools import (
    CALL_NOT_ENDED,
    SAVE_NOW,
    asked_for_the_owner,
    confirmed_the_message,
    describe_call,
    file_report,
    forget,
    identify,
    message_confirm_question,
    message_saved,
    missed_save,
    note_caller_before_transfer,
    note_refused_transfer_at_call_end,
    remember_refused_transfer,
    said_goodbye,
    save_held_message_at_call_end,
    save_lead_record,
    say_goodbye,
    send_warm_lead_at_call_end,
    set_phrase,
    text_owner_before_transfer,
    text_portfolio_link_to_caller,
    transfer_closed_message,
    verify,
    transfer_question,
)
from ai_call_center.guardrails import HIDDEN_NUMBER, TransferHours
from ai_call_center.leads import LeadStore
from ai_call_center.notify import notifier_from_settings
from ai_call_center.passphrase import masked, private_words
from ai_call_center.telephony.lookup import CallerLookup
from ai_call_center.telephony.telnyx_client import TelnyxTelephony
from ai_call_center.telephony.transfers import (
    Fallback,
    Snapshot,
    TransferManager,
    screening_prompt,
)
from ai_call_center.voice.call_audio import call_audio, recorder
from ai_call_center.voice.caller_audio import CallerAudioFile, caller_audio_path
from ai_call_center.voice.cutoff import CallerSoundWatch
from ai_call_center.voice.health import ProviderHealth
from ai_call_center.voice.legs import (
    adopt_context,
    build_leg,
    ensure_connected,
    first_answer_needs_run_frame,
    forget_note,
    ready_legs,
    replay_and_respond,
    reset_session,
    resolve_spec,
    self_heals,
    session_clock,
    tell_now,
)
from ai_call_center.voice.leveler import VoiceLeveler
from ai_call_center.voice.phone_messages import CALL_SO_FAR_MARK, CONNECT_MARK, GREETING_DONE, connect_message  # noqa: F401 - importable from here too
from ai_call_center.voice.respell import Respelling
from ai_call_center.voice.spoken_tool import SpokenToolWatch
from ai_call_center.voice.stats import CallStats
from ai_call_center.voice.tools import tools_for_agent, voice_tool_schemas
from ai_call_center.voice.voice_reader import CallerVoiceReader, FrustrationNote, default_voice_model
from ai_call_center.writing import default_library, search_result

log = logging.getLogger(__name__)

TELNYX_SAMPLE_RATE = 8000
TRANSFER_FALLBACK_SECS = 4.0  # transfer anyway if the agent never says its goodbye line
SWITCH_TIMEOUT_SECS = 2.0
CALLER_WORDS_SECS = 6.0  # how long words the caller just said count as "not in the transcript yet"
# How long a handoff waits for those words to reach the shared context. Measured 2026-11-16 over
# 10 handoffs from Nova: Pipecat committed the words within 0.1 s twice, after 1.0 s twice, and
# not within 1.5 s six times. After the wait, the handoff adds the words itself.
CALLER_WORDS_WAIT_SECS = 0.5
SPOKEN_TOOL_WAIT_SECS = 0.4  # after the agent said a tool's name: how long the server waits for the real tool call
SPOKE_WAIT_SECS = 0.5  # how long a handoff waits for a sign that the caller spoke to the agent, before it is refused
OPENING_QUIET_SECS = 3.0  # the longest time the caller's audio is replaced by silence while the greeting has not started
TRANSCRIPT_WAIT_SECS = 0.8  # how long a transfer request waits for the transcript of the caller's last turn
ANSWER_DEADLINE_SECS = 8.0  # an agent asked to speak (greeting, handoff) that stays silent this long has failed
RECOVERIES_PER_SERVICE = 2
RELAPSE_SECS = 10.0  # an error this soon after a recovery means the recovery did not help
MAX_FAILOVERS = 2  # moves to another voice provider in one call; after that, the spoken fallback
# An error with one of these marks will not go away with a reconnect: a bad key, or an account out of credit.
PERMANENT_ERROR_MARKS = ("invalid_api_key", "incorrect api key", "insufficient_quota", "billing", "api key not valid", "permission_denied")
PROVIDER_FAILED = "provider_failed"  # the active agent went silent; the caller heard a pause
HANDOFF_REROUTED = "handoff_rerouted"  # a handoff's target was not answering; the caller heard no failure
FAILOVER_REASONS = {PROVIDER_FAILED, HANDOFF_REROUTED}


def voice_instructions(spec: AgentSpec, leg: str, *notes: str) -> str:
    """The agent's instructions for this call, followed by the notes that apply to it (empty notes are dropped)."""
    return "\n\n".join([f"{spec.instructions}\n\nIn this call you run on {LEGS.get(leg, leg)}.", *(n for n in notes if n)])


def known_name_note(caller_name: str) -> str:
    """Tells an agent that takes the call who the caller is. On 2026-11-16 the contact agent asked
    "who should I say is calling?" and "what's your name?" after the caller had confirmed their name."""
    name = caller_name.strip()
    if not name:
        return ""
    return (
        f" The caller's name is {name}. They gave or confirmed it earlier on this call. "
        "Never ask for their name, never ask who is calling, and never ask them to confirm it."
    )


MISSED_SAVE_WAIT_SECS = 2.5  # after the agent stops speaking, how long a save_lead call may take to arrive
CALL_SO_FAR_LINES = 16  # how many of the last turns the next agent reads again in the handoff message
CALL_SO_FAR_LINE_CHARS = 320


def call_so_far(lines: list[str] | None) -> str:
    """The last turns of the call, word for word, for the agent that takes over.

    The agent has the whole call in its context already. An OpenAI Realtime session gets it as one
    JSON text that Pipecat calls "a previously saved conversation". On 2026-11-17 the contact agent
    got a caller's project that way, asked what the project was, and then said "I don't have
    access to any of that information". So the handoff message, which the agent acts on, says it again.
    """
    recent = [line if len(line) <= CALL_SO_FAR_LINE_CHARS else line[:CALL_SO_FAR_LINE_CHARS].rstrip() + "..." for line in (lines or [])[-CALL_SO_FAR_LINES:]]
    if not recent:
        return ""
    return (
        f"\n\n{CALL_SO_FAR_MARK}, newest last. The caller said this on this call, to you or to another agent:\n"
        + "\n".join(recent)
        + "\n\nYou know all of this. Use it in your answer. Never ask the caller to say it again, and never say that you cannot see the earlier conversation."
    )


def handoff_instruction(target: str, platform: str, caller_name: str = "", lines: list[str] | None = None) -> str:
    """What the new agent reads last when it takes over the call.

    Nova and Gemini read this as a caller's turn, so it says that it is not one. Without that,
    Nova took "the call is now with the consultant agent" as a request: it said "Let me bring in
    the right agent for that" again and called a handoff tool. Measured 2026-11-16.
    """
    return (
        "This is a message from the phone system, not from the caller. "
        f"The handoff is complete: you are the {target} agent, with a new voice, and the call is yours. "
        "Do not say that you will bring in an agent, and do not call a transfer_to tool in this answer. Speak to the caller now. "
        f'Your first sentence must be exactly: "Hi, I\'m the {target} agent, running on {platform}." '
        "Say it one time only, before anything else, also before you call a tool. "
        "Then answer what the caller last asked. Do not greet again." + known_name_note(caller_name) + call_so_far(lines)
    )


def handoff_refused(agent: str, platform: str) -> str:
    """What an agent reads when it tries to hand the call on before the caller has said anything to it."""
    return (
        f"No handoff was made. You are the {agent} agent. You took over this call a moment ago, and the caller has not spoken since. "
        "The caller was sent to you on purpose. Do not mention this, and do not announce another agent. "
        f'If you have not said it yet, say: "Hi, I\'m the {agent} agent, running on {platform}." '
        "Then answer the caller's last words yourself. You may hand off after the caller speaks again."
    )


def retire_connect_message(messages: list) -> bool:
    """After the greeting, the order to say it must not reach the next agent. True when changed.

    A handoff replays the shared context into a new session. In a short call the order "Say
    exactly these words" stood next to the handoff's own "Your first sentence must be exactly",
    and the next agent said its intro line two times (3 of 6 simulated calls, 2026-11-17).
    """
    first = messages[0] if messages else None
    if isinstance(first, dict) and first.get("role") == "developer" and str(first.get("content", "")).startswith(CONNECT_MARK):
        first["content"] = GREETING_DONE
        return True
    return False


def failover_instruction(agent: str, platform: str) -> str:
    """What an agent reads last when it takes the call back on another provider, after its first provider went silent."""
    return (
        f"The line went silent for a few seconds because of a technical problem. The call is still with the {agent} agent. "
        f"You are the {agent} agent, now with a new voice. Speak now. "
        f'Your first sentence must be exactly: "Sorry about that pause. I\'m the {agent} agent, now running on {platform}." '
        "Then, if the caller's last question has no answer yet, answer it. If it was answered, ask in one short sentence what else they want to know. Do not greet again."
    )


TRANSFER_DID_NOT_CONNECT = "The transfer was attempted. It did not connect."


def after_transfer_attempt(messages: list) -> list:
    """The conversation as the contact agent reads it when the call comes back to it.

    The transfer tool's result told the agent to say a goodbye line. That instruction is now
    wrong, so it is replaced with what happened.
    """
    transfer_calls = {
        call.get("id")
        for m in messages
        if isinstance(m, dict)
        for call in (m.get("tool_calls") or [])
        if (call.get("function") or {}).get("name") == "request_owner_transfer"
    }
    return [
        {**m, "content": json.dumps(TRANSFER_DID_NOT_CONNECT)} if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id") in transfer_calls else m
        for m in messages
    ]


def resume_message(reason: str, owner: str, lead_saved: bool) -> str:
    """What the contact agent reads when the call comes back to it after a transfer that did not connect."""
    what = f"{owner} did not pick up" if reason == "no_answer" else f"the call to {owner} could not be connected"
    if lead_saved:
        follow = f'Then say to the caller: "Your details are saved, and {owner} will call you back." Then ask if there is anything else you can help with.'
    else:
        follow = f"Then offer to take their name and a message for {owner}."
    return (
        f"The live transfer did not go through: {what}. The caller is back on the line with you, the contact agent. Speak to the caller now. "
        f"Say in one sentence that {what}. {follow} "
        "Do not introduce yourself again. Do not offer another transfer, and do not call save_lead for details that are already saved. "
        "Try the transfer again only if the caller asks for it."
    )


@dataclass
class VoiceCallState:
    """What LangGraph's CallState holds in text mode, kept here for the voice session."""

    active_agent: str = ENTRY_AGENT
    handoffs: list[str] = field(default_factory=list)
    lead_saved: bool = False
    transfer_requested: bool = False
    end_reason: str = ""  # why the session ended: "caller", "agent", "transfer", "limit", "failover"; "" while it runs
    ending: bool = False  # end_call was accepted; the hang-up waits for the goodbye line
    transfer_announce: str = ""  # what the owner hears before they accept the call: who is calling, and why
    took_over_at: float = 0.0  # when the active agent got the call by a handoff (monotonic); 0.0 for the first agent


class AgentStoppedWatch(FrameProcessor):
    """Calls `on_stopped` each time the agent stops speaking. Sits after the transport output."""

    def __init__(self, on_stopped) -> None:
        super().__init__()
        self._on_stopped = on_stopped

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStoppedSpeakingFrame):
            await self._on_stopped()
        await self.push_frame(frame, direction)


class TransferGate(FrameProcessor):
    """Sits after the transport output. When armed, it transfers the call as soon as the bot
    stops speaking, so the caller hears the whole goodbye line first. `arm(action)` runs another
    action at that moment instead: the hang-up of `end_call`."""

    def __init__(self, on_transfer) -> None:
        super().__init__()
        self._on_transfer = on_transfer
        self._armed = False
        self._fired = False
        self._speaking = False
        self.last_started = 0.0  # when the bot last began to speak (time.monotonic)

    @property
    def speaking(self) -> bool:
        return self._speaking

    def arm(self, action=None) -> None:
        if self._armed:
            return
        self._armed = True
        if action is not None:
            self._on_transfer = action
        self.create_task(self._fire_after(TRANSFER_FALLBACK_SECS))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._speaking = True
            self.last_started = time.monotonic()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking = False
            if self._armed:
                await self._fire()
        await self.push_frame(frame, direction)

    async def _fire_after(self, secs: float) -> None:
        await asyncio.sleep(secs)
        if self._speaking:  # the goodbye line is still playing; BotStoppedSpeakingFrame fires it
            return
        await self._fire()

    async def _fire(self) -> None:
        if self._fired:
            return
        self._fired = True
        await self._on_transfer()


def _letters(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


class OpeningQuiet(FrameProcessor):
    """Replaces the caller's audio with silence until the greeting starts to play.

    A phone line can click when the media stream opens. A voice provider takes the click as a
    caller's turn and answers it, so the greeting plays twice. Measured 2026-11-16: a noise of
    170 ms, 60 ms after the stream opened, and a second greeting right after the first.
    Before the greeting starts, the caller has nothing to answer. After it starts, the caller's
    audio passes, so the caller can interrupt. `limit_secs` opens the audio if no greeting comes.
    """

    def __init__(self, limit_secs: float = OPENING_QUIET_SECS) -> None:
        super().__init__()
        self._limit = limit_secs
        self._first_audio_at: float | None = None
        self.open = False

    def screen(self, frame: Frame) -> None:
        """Open on the first sound from the agent. Until then, turn the caller's audio into silence."""
        if self.open:
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            self.open = True
            log.info("caller audio opened: the greeting started")
        elif isinstance(frame, InputAudioRawFrame):
            now = time.monotonic()
            self._first_audio_at = self._first_audio_at or now
            if now - self._first_audio_at >= self._limit:
                self.open = True
                log.warning("caller audio opened after %.1f s with no greeting", self._limit)
            else:
                frame.audio = bytes(len(frame.audio))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.screen(frame)
        await self.push_frame(frame, direction)


class CallerWords(FrameProcessor):
    """Keeps the words the caller said in the last seconds, as the voice service reports them.

    The shared context gets a caller's turn only when Pipecat's aggregator commits it. That can be
    a second after the service heard the words, and after a tool call that the words caused. A
    check on "the caller's own words", or a handoff, must not miss those words.
    """

    def __init__(self) -> None:
        super().__init__()
        self._heard: list[tuple[float, str]] = []
        self.last_spoke_at = 0.0  # when a service last reported the caller's words or the start of a caller's turn (monotonic)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.UPSTREAM and frame.text.strip():
            now = time.monotonic()
            self._heard = [(at, text) for at, text in self._heard if now - at <= CALLER_WORDS_SECS] + [(now, frame.text)]
            self.last_spoke_at = now
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.last_spoke_at = time.monotonic()
        await self.push_frame(frame, direction)

    def spoke_since(self, when: float) -> bool:
        """True when the caller said something after `when` (monotonic)."""
        return self.last_spoke_at > when

    def missing_from(self, lines: list[str]) -> str:
        """The caller's newest words, when the transcript `lines` do not hold them yet. Else ""."""
        now = time.monotonic()
        recent = " ".join(" ".join(text.split()) for at, text in self._heard if now - at <= CALLER_WORDS_SECS)
        if not _letters(recent):
            return ""
        last = next((line for line in reversed(lines) if line.startswith("caller: ")), "")
        return "" if _letters(recent) in _letters(last) else recent

    def with_newest(self, lines: list[str]) -> list[str]:
        """The transcript, plus the caller's newest words when it does not hold them yet."""
        missing = self.missing_from(lines)
        return [*lines, f"caller: {missing}"] if missing else lines


class ServiceWatchdog(FrameProcessor):
    """Sits before the switcher and watches the error frames the voice services push upstream.

    Pipecat ends a realtime service's receive loop on any error event from OpenAI, even one
    that only rejects a single client event. Without this, the active session goes silent for
    the rest of the call. The watchdog reconnects the session; if it is the active one, it
    replays the context and asks for a response. A Gemini service reconnects by itself, so the
    watchdog leaves it alone.

    A service is dead when Pipecat has given up on it (`is_usable` is False), when a recovery
    fails, when it errors again right after a recovery, or after `RECOVERIES_PER_SERVICE`
    recoveries. The watchdog then calls `on_dead(leg, reason)`, and the session moves the call
    to another provider.
    """

    def __init__(self, legs: dict[str, VoiceLeg], switcher: LLMSwitcher, context: LLMContext, *, on_dead=None) -> None:
        super().__init__()
        self._legs = legs
        self._switcher = switcher
        self._context = context
        self._on_dead = on_dead
        self._recoveries: dict[str, int] = {}
        self._recovered_at: dict[str, float] = {}
        self._recovering: set[str] = set()
        self._dead: set[str] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame) and direction == FrameDirection.UPSTREAM:
            self._check(frame)
        await self.push_frame(frame, direction)

    def _check(self, frame: ErrorFrame) -> None:
        # The switcher reports the active service's last error as its own, so look at every
        # service, not only at the one named in the frame.
        for vl in self._legs.values():
            if not vl.service.is_usable:
                self._declare_dead(vl, f"the service gave up: {frame.error}")
        vl = next((vl for vl in self._legs.values() if vl.service is frame.processor), None)
        if vl is None or frame.fatal or vl.agent in self._dead or vl.agent in self._recovering or self_heals(vl.service):
            return
        if any(mark in str(frame.error).lower() for mark in PERMANENT_ERROR_MARKS):
            self._declare_dead(vl, f"the provider refused the account: {frame.error}")
            return
        relapse = time.monotonic() - self._recovered_at.get(vl.agent, float("-inf")) < RELAPSE_SECS
        if relapse or self._recoveries.get(vl.agent, 0) >= RECOVERIES_PER_SERVICE:
            self._declare_dead(vl, f"{'the error came back after a recovery' if relapse else 'too many recoveries'}: {frame.error}")
            return
        self._recovering.add(vl.agent)
        self.create_task(self._recover(vl, frame.error))

    def _declare_dead(self, vl: VoiceLeg, reason: str) -> None:
        if vl.agent in self._dead:
            return
        self._dead.add(vl.agent)
        log.error("%s: voice service on %s is dead (%s)", vl.agent, vl.leg, reason[:200])
        if self._on_dead is not None:
            self._on_dead(vl, reason)

    async def _recover(self, vl: VoiceLeg, error: str) -> None:
        self._recoveries[vl.agent] = self._recoveries.get(vl.agent, 0) + 1
        t0 = time.monotonic()
        active = self._switcher.strategy.active_service is vl.service
        log.warning("%s: session error (%s); reconnecting%s", vl.agent, error[:200], " and replaying" if active else "")
        try:
            async with vl.lock:
                await reset_session(vl.service)
                if active:
                    await replay_and_respond(vl.service, self._context)
            log.info("%s: recovered in %.0f ms", vl.agent, (time.monotonic() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            log.exception("%s: recovery failed", vl.agent)
            self._declare_dead(vl, f"recovery failed: {exc}")
        finally:
            self._recovered_at[vl.agent] = time.monotonic()
            self._recovering.discard(vl.agent)


class NoToolCall:
    """Stands in for a tool call when the server makes a handoff that the agent only said aloud."""

    async def result_callback(self, result, *, properties=None) -> None:
        log.info("handoff by the server: %s", result)
        if properties is not None and properties.on_context_updated is not None:
            await properties.on_context_updated()


@dataclass
class VoiceLeg:
    """One agent's voice service and what the session knows about it."""

    agent: str
    leg: str
    service: LLMService
    used: bool = False  # it has answered in this call; its server-side session holds history
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class CallSession:
    websocket: WebSocket
    call_data: dict
    settings: Settings
    telephony: TelnyxTelephony
    specs: dict[str, AgentSpec] = field(default_factory=default_specs)
    lookups: CallerLookup | None = None
    transfers: TransferManager | None = None
    resume: Snapshot | None = None  # set when the stream restarts: after a transfer that did not connect, or to change provider
    health: ProviderHealth | None = None  # which voice providers are answering; shared by all calls

    @cached_property
    def respell(self) -> Respelling:
        """The words that the voice models read in another spelling (RESPELLINGS), so that they say them right."""
        return Respelling.parse(self.settings.respellings)

    @property
    def call_control_id(self) -> str:
        return self.call_data["call_id"]

    async def _finish_call(self, store: LeadStore, ctx: CallContext, messages: list, *, visited: list[str], transferred: bool) -> CallSummary:
        """The summary model reads the whole call. Its summary is what the next call from this number sees.
        Its note for the owner goes into the end-of-call email: a refused transfer with no message, or a
        warm lead. Either way the owner gets what the caller said, even when no agent saved it."""
        try:
            summary = await summarize_call(self.respell.written_messages(messages), init_chat_model(self.settings.summary_model), config=tracing.run_config(self.call_control_id, "call-summary"))
        except Exception:
            log.exception("call %s: the call summary failed", self.call_control_id)
            summary = CallSummary()
        if ctx.spoken_phrases:
            # The summary becomes a note for the next call, and the owner's note goes into an email. A passphrase is in neither.
            summary = replace(summary, **{key: masked(getattr(summary, key), ctx.spoken_phrases) for key in ("summary", "topic", "owner_note")})
        noted = False
        try:
            if await note_refused_transfer_at_call_end(ctx, lead_saved=message_saved(ctx), visited=visited, summary=summary):
                noted = True
                log.info("call %s: the caller left no message after a refused transfer; the owner got a note with the call details", self.call_control_id)
        except Exception:
            log.exception("call %s: could not save the note for a refused transfer", self.call_control_id)
        try:
            store.finish_call(self.call_control_id, agents=visited, summary=summary.summary, caller_name=ctx.caller_name or summary.caller_name, topic=summary.topic)
            log.info("call %s: summary saved (%d chars, interest: %s)", self.call_control_id, len(summary.summary), summary.interest)
        except Exception:
            log.exception("call %s: could not save the call summary", self.call_control_id)
        if transferred:
            return summary  # the owner is on the line with the caller, or the call comes back to an agent
        try:
            if await send_warm_lead_at_call_end(ctx, summary, lead_saved=noted or message_saved(ctx), visited=visited):
                log.info("call %s: no message was saved; the owner got a warm lead email (%s)", self.call_control_id, summary.interest)
        except Exception:
            log.exception("call %s: could not send the warm lead email", self.call_control_id)
        return summary

    async def _record_analytics(self, ctx: CallContext, stats: CallStats, messages: list, *, visited: list[str], outcome: str, summary_tries: int, summary_faults: tuple[str, ...] = ()) -> None:
        """The grader scores the call, and the figures and the scores go to the analytics store (`analytics.py`).

        A caller who asked to be forgotten gets no record. Nothing here can stop the end of a call: a failure is logged.
        Then the public report is published again (`publish.py`), so the page shows this call.
        Last, each failed rule starts the job on GitHub that proposes a fix (`improve.py`).

        The decision grader hears the call audio (`voice/call_audio.py`). The audio is deleted here, graded or not.
        After a transfer it stays for a short time: the call comes back when nobody answers."""
        audio = call_audio.get(self.call_control_id)
        try:
            if ctx.forgotten:
                audio = None
                forget_call(self.settings.leads_db, self.call_control_id)  # a record from before a transfer that came back
                await publish_report(self.settings)
                return
            try:
                profile = load_profile(self.settings.profile_path).full
            except OSError:
                profile = ""
            dollars, _ = stats.cost()
            call = CallAnalytics(
                call_id=self.call_control_id,
                day=call_day(stats.started_at),
                channel=VOICE,
                version=harness_version(self.settings.harness_version),
                test=ctx.caller_number == self.settings.owner_phone,
                duration_secs=int((datetime.now(UTC) - stats.started_at).total_seconds()),
                agents=visited,
                outcome=outcome,
                failovers=stats.failovers,
                tokens_in=sum(usage.tokens_in for usage in stats.usage.values()),
                tokens_out=sum(usage.tokens_out for usage in stats.usage.values()),
                cost_usd=dollars,
                answers=list(stats.answers),
                handoff_secs=list(stats.handoffs),
            )
            lines = transcript_lines(self.respell.written_messages(messages))
            grader = init_chat_model(self.settings.grader_model) if self.settings.grader_model else None
            decider = default_decision_grader(self.settings)
            record = await record_call(
                self.settings.leads_db,
                call,
                lines,
                summary_tries=summary_tries,
                profile=profile,
                owner=self.settings.owner_name,
                grader=grader,
                config=tracing.run_config(self.call_control_id, "call-grade"),
                decider=decider,
                audio=audio,
            )
            log.info("call %s: grade (version %s): %s", self.call_control_id, record.version, describe_scores(record.scores))
            if record.readings:
                log.info("call %s: how the call sounded: %s", self.call_control_id, ", ".join(f"{key} {value}" for key, value in record.readings.items()))
            if self.settings.grader_compare and decider is not None and grader is not None:
                # One more request, to the grader model. Its verdicts are for the log only; the record keeps the decision grader's.
                read = await grade_call(lines, visited=visited, summary_tries=summary_tries, profile=profile, owner=self.settings.owner_name, model=grader)
                differ = differences(record.scores, read)
                log.info("call %s: the grader model against the decision grader: %s", self.call_control_id, "; ".join(differ) if differ else "the same verdict for each rule that both checked")
            if not record.test:  # a test call is not in the report
                await publish_report(self.settings)

            async def fault_of(request: FixRequest) -> FaultReport:
                """What went wrong, for the job that proposes a fix. Code removes each private word (fault_report.py)."""
                rule = rule_of(request.rule)
                if request.rule == "summary_first_try":
                    return summary_report(summary_faults)  # fixed texts of the summary check; no model reads the call
                if grader is None or rule is None:
                    return FaultReport()
                note = next((score.note for score in record.scores if score.rule == request.rule), "")
                names = [*(name for name in (ctx.caller_name, ctx.returning_name) if name), *private_words(ctx.spoken_phrases)]
                return await write_report(lines, rule=rule, agent=request.agent, note=note, model=grader, owner=self.settings.owner_name, private=names, config=tracing.run_config(self.call_control_id, "fault-report"))

            # A failed rule starts the job that proposes a fix. A test call can start it too: it shows the loop with no caller.
            await request_fixes(self.settings, self.settings.leads_db, record, report=fault_of)
            # After a deploy: did the fix of a failed rule hold at this version (fix_check.py)?
            # A new check goes into the public report now, not after the next call.
            if await check_fixes(self.settings, self.settings.leads_db, record) and not record.test:
                await publish_report(self.settings)
            # What the caller said aloud about the line: a fault or a wish for a feature. It becomes a GitHub issue (feedback.py).
            # A report that the feedback agent filed counts as certain, also with no answer from the decision grader.
            answers = decider.last_answers if decider is not None else {}
            if answers or ctx.feedback_filed:
                await report_feedback(
                    self.settings,
                    self.settings.leads_db,
                    lines,
                    chances=feedback_chances(answers),
                    filed=ctx.feedback_filed,
                    model=grader,
                    call_id=self.call_control_id,
                    caller_number=ctx.caller_number,
                    version=record.version,
                    private=[*(name for name in (ctx.caller_name, ctx.returning_name) if name), *private_words(ctx.spoken_phrases)],
                    config=tracing.run_config(self.call_control_id, "caller-feedback"),
                )
        except Exception:
            log.exception("call %s: could not record the analytics", self.call_control_id)
        finally:
            if outcome == TRANSFER and not ctx.forgotten:
                call_audio.drop_later(self.call_control_id)
            else:
                call_audio.drop(self.call_control_id)

    async def run(self) -> None:
        started = time.monotonic()
        resume = self.resume
        # A new call starts with the classifier. A call that comes back from a transfer starts with the
        # contact agent. A call that changes provider starts with the agent that was speaking.
        entry_agent = resume.entry_agent if resume else ENTRY_AGENT
        failover_resume = bool(resume and resume.reason in FAILOVER_REASONS)
        health = self.health or ProviderHealth()
        # Speed, tokens, and cost of this call. One object for the whole call, also across a stream restart.
        stats = resume.stats if resume and isinstance(resume.stats, CallStats) else CallStats()
        state = VoiceCallState(active_agent=entry_agent, handoffs=list(resume.handoffs) if resume else [], lead_saved=bool(resume and resume.lead_saved))
        transfers = self.transfers or TransferManager(self.telephony, stream_url=self.settings.media_stream_url, ring_secs=self.settings.transfer_ring_secs)
        caller_number = self.call_data.get("from") or "unknown"
        store = LeadStore(self.settings.leads_db)
        # Hidden numbers share one log key, so the per-number call limit covers them too.
        store.start_call(self.call_control_id, HIDDEN_NUMBER if caller_number in UNKNOWN_NUMBERS else caller_number)
        if resume:
            store.reopen_call(self.call_control_id)
        # Every caller hears the same greeting, which asks who they are. No agent knows if the number
        # called before: identify_caller checks the name in code and gives the notes only on a match.
        hello = self.respell.spoken(greeting_words(self.settings.owner_name, self.settings.brand_name))
        # Number lookup started at call.initiated; take the result if it is ready.
        info = await self.lookups.get(caller_number) if self.lookups else None
        lookup_block = lookup_text(info) if info is not None else ""
        if lookup_block:
            log.info("call %s: number lookup: line=%s carrier=%s area=%s name known: %s", self.call_control_id, info.line_type, info.carrier, info.state, bool(info.caller_name))
        # Each spec as it will run here: a leg with no API key, or one marked down, runs on another
        # provider, voice included.
        self.specs = {name: resolve_spec(spec, self.settings, health.down()) for name, spec in self.specs.items()}
        schemas = voice_tool_schemas(self.specs)
        legs: dict[str, VoiceLeg] = {}
        for name, spec in self.specs.items():
            service = build_leg(
                spec,
                instructions=self.respell.instructions(voice_instructions(spec, spec.leg, lookup_block)),
                settings=self.settings,
                tools=tools_for_agent(spec, schemas),
            )
            legs[name] = VoiceLeg(name, spec.leg, service)
        await transfers.settled()  # the start-up check of the Telnyx application; no wait after the first second
        ctx = CallContext(
            call_control_id=self.call_control_id,
            caller_number=caller_number,
            owner_phone=self.settings.owner_phone,
            leads=store,
            telephony=self.telephony,
            notifier=notifier_from_settings(self.settings),
            legs={name: vl.leg for name, vl in legs.items()},
            mode="voice",
            identity_confirmed=bool(resume and resume.identity_confirmed),
            identity_verified=bool(resume and resume.identity_verified),
            open_calls=list(resume.open_calls) if resume else [],
            phrase_closed=bool(resume and resume.phrase_closed),
            phrase_offered=bool(resume and resume.phrase_offered),
            phrase_set=bool(resume and resume.phrase_set),
            spoken_phrases=list(resume.spoken_phrases) if resume else [],
            phrase_offer=self.settings.caller_passphrase_offer,
            returning_name=resume.returning_name if resume else "",
            identified=bool(resume and resume.identified),
            forgotten=bool(resume and resume.forgotten),
            welcome=Welcome(
                owner=self.settings.owner_name,
                companies=companies_phrase({name: vl.leg for name, vl in legs.items()}),
                tip=greeting_tip(self.settings.owner_name, writing=bool(default_library().posts)),
                rag=bool(default_library().posts),
            ),
            earlier_calls=store.call_count_for(caller_number, exclude=self.call_control_id) if caller_number not in UNKNOWN_NUMBERS else 0,
            number_info=info,
            caller_lookup=lookup_block,
            caller_name=resume.caller_name if resume else "",
            transfer_note_id=resume.transfer_note_id if resume else 0,
            refused_transfer=resume.refused_transfer if resume else None,
            transfer_unavailable=transfers.unavailable,
            decider=default_decider(self.settings),
            crm=default_crm(self.settings),
            caller_line_type=info.line_type if info else "",
            texter=self.telephony if self.settings.sms_enabled else None,
            portfolio_url=self.settings.portfolio_url,
            owner_name=self.settings.owner_name,
            website_spoken=site.spoken if (site := web_address(self.settings.website_url, self.settings.brand_name)) else "",
            transfer_hours=TransferHours.parse(self.settings.transfer_hours, self.settings.transfer_days),
            started_at=stats.started_at,
        )
        entry = self.specs[entry_agent]

        serializer = TelnyxFrameSerializer(
            stream_id=self.call_data["stream_id"],
            call_control_id=self.call_control_id,
            outbound_encoding=self.call_data.get("outbound_encoding") or "PCMU",
            inbound_encoding="PCMU",
            api_key=self.settings.telnyx_api_key.get_secret_value(),
            params=TelnyxFrameSerializer.InputParams(auto_hang_up=False),  # a transfer must not be followed by a hang-up
        )
        transport = FastAPIWebsocketTransport(
            websocket=self.websocket,
            params=FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer),
        )

        entry_platform = LEG_PLATFORMS.get(entry.leg, entry.leg)
        caller_spoke = bool(resume) and any(isinstance(m, dict) and m.get("role") == "user" for m in resume.messages)
        if failover_resume:
            log.info("call %s: the %s agent takes over on %s (%s; caller has spoken: %s)", self.call_control_id, entry_agent, entry.leg, resume.reason, caller_spoke)
        if failover_resume and not caller_spoke:
            # The provider failed before the caller said anything: greet as on a new call.
            first_messages = [{"role": "developer", "content": connect_message(hello)}]
        elif failover_resume:
            instruction = handoff_instruction(entry_agent, entry_platform, resume.caller_name) if resume.reason == HANDOFF_REROUTED else failover_instruction(entry_agent, entry_platform)
            first_messages = [*resume.messages, {"role": "developer", "content": instruction}]
        elif resume:
            first_messages = [*after_transfer_attempt(resume.messages), {"role": "developer", "content": self.respell.spoken(resume_message(resume.reason, self.settings.owner_name, state.lead_saved))}]
        else:
            first_messages = [{"role": "developer", "content": connect_message(hello)}]
        context = LLMContext(messages=first_messages, tools=tools_for_agent(entry, schemas))
        caller_words = CallerWords()
        # The transcript is for code and for people (the transfer check, the lead email): it has the written words.
        ctx.transcript = lambda: [self.respell.written(line) for line in caller_words.with_newest(transcript_lines(context.messages))]
        aggregators = LLMContextAggregatorPair(context)
        user_aggregator, assistant_aggregator = aggregators
        # LangSmith: this call's spans and the caller's words go to the call's thread. Off unless tracing is on.
        traced = tracing.follow_call(aggregators, self.call_control_id)
        # The first service in the list is active at start.
        switcher = LLMSwitcher(llms=[legs[entry_agent].service, *(vl.service for name, vl in legs.items() if name != entry_agent)])

        def snapshot(**changes) -> Snapshot:
            return Snapshot(
                **{
                    "messages": list(context.messages),
                    "handoffs": list(state.handoffs),
                    "lead_saved": state.lead_saved,
                    "transfer_note_id": ctx.transfer_note_id,
                    "refused_transfer": ctx.refused_transfer,
                    "identity_confirmed": ctx.identity_confirmed,
                    "identity_verified": ctx.identity_verified,
                    "open_calls": list(ctx.open_calls),
                    "phrase_closed": ctx.phrase_closed,
                    "phrase_offered": ctx.phrase_offered,
                    "phrase_set": ctx.phrase_set,
                    "spoken_phrases": list(ctx.spoken_phrases),
                    "caller_name": ctx.caller_name,
                    "identified": ctx.identified,
                    "returning_name": ctx.returning_name,
                    "forgotten": ctx.forgotten,
                    "failovers": resume.failovers if resume else 0,
                    "stats": stats,
                    **changes,
                }
            )

        async def do_transfer() -> None:
            # The manager stops the media stream, which ends this session, and then transfers the
            # call. It keeps the conversation, so the contact agent can come back if the owner
            # does not pick up.
            log.info("call %s: transferring to the owner", self.call_control_id)
            state.end_reason = "transfer"
            transfers.start(self.call_control_id, to=self.settings.owner_phone, snapshot=snapshot(), announce=state.transfer_announce)

        # --- a voice provider that stops answering ---
        background: set[asyncio.Task] = set()
        timers: list[asyncio.Task] = []

        def spawn(coro) -> None:
            task = asyncio.create_task(coro)
            background.add(task)
            task.add_done_callback(background.discard)

        def spoken_fallback() -> Fallback:
            owner = self.respell.spoken(self.settings.owner_name)
            can_transfer = self.settings.transfer_on_failure and bool(self.settings.owner_phone) and transfer_closed_message(ctx) is None
            return Fallback(
                to=self.settings.owner_phone if can_transfer else "",
                connecting=f"Sorry, the AI agents on this line have a technical problem. I am connecting you to {owner} now.",
                no_answer=f"Sorry, {owner} could not take the call. Please call again later. Goodbye.",
            )

        def start_fail_over(agent: str, kind: str, reason: str) -> bool:
            """Move the call to another voice provider: restart the media stream, so a new session
            builds its agents on a provider that answers. `agent` speaks first there. When no
            provider is left, Telnyx's voice speaks, and the call goes to the owner or ends.
            Returns False when the session is ending already."""
            if state.end_reason or state.transfer_requested:
                return False
            failovers = resume.failovers if resume else 0
            ready = ready_legs(self.settings, health.down())
            if ready and failovers < MAX_FAILOVERS:
                log.error("call %s: moving the %s agent to %s (%s)", self.call_control_id, agent, ready[0], reason[:160])
                state.end_reason = "failover"
                if kind == PROVIDER_FAILED:
                    stats.failovers += 1
                transfers.restart(self.call_control_id, snapshot=snapshot(reason=kind, entry_agent=agent, failovers=failovers + 1), fallback=spoken_fallback())
            else:
                log.error("call %s: no voice provider is answering (%s); using the spoken fallback", self.call_control_id, reason[:160])
                state.end_reason = "fallback"
                transfers.last_resort(self.call_control_id, fallback=spoken_fallback())
            return True

        async def fail_over(agent: str, kind: str, reason: str) -> None:
            if start_fail_over(agent, kind, reason):
                await runner.cancel()  # Telnyx closes the stream too; do not wait for it

        def on_dead(vl: VoiceLeg, reason: str) -> None:
            """A voice service has failed for good. Its provider is avoided for a few minutes."""
            health.mark_down(vl.leg, reason)
            dead_agents.add(vl.agent)
            if vl.agent == state.active_agent:
                spawn(fail_over(vl.agent, PROVIDER_FAILED, reason))

        def reroute(target: str) -> bool:
            """Called at a handoff. True when the target cannot answer and the call moves to another provider instead."""
            vl = legs[target]
            if target not in dead_agents and vl.service.is_usable and not health.is_down(vl.leg):
                return False
            health.mark_down(vl.leg, "a handoff target was not answering")
            spawn(fail_over(target, HANDOFF_REROUTED, f"the {target} agent on {vl.leg} is not answering"))
            return True

        def expect_speech(vl: VoiceLeg) -> None:
            """The agent was just asked to speak. If no audio starts in time, its service has failed."""
            asked = time.monotonic()

            async def check() -> None:
                await asyncio.sleep(ANSWER_DEADLINE_SECS)
                if gate.last_started < asked and not state.end_reason and not state.transfer_requested and state.active_agent == vl.agent:
                    log.error("call %s: the %s agent did not speak within %.0f s", self.call_control_id, vl.agent, ANSWER_DEADLINE_SECS)
                    on_dead(vl, f"no audio within {ANSWER_DEADLINE_SECS:.0f} s of a request to speak")

            timers.append(asyncio.create_task(check()))

        async def watch_services() -> None:
            """Pipecat gives up on a service that cannot connect (`is_usable` goes False). The error
            of a service that is not active does not reach the watchdog, so look once a second."""
            while True:
                await asyncio.sleep(1.0)
                for vl in legs.values():
                    if vl.agent not in dead_agents and not vl.service.is_usable:
                        on_dead(vl, "the service gave up after repeated connection failures")

        dead_agents: set[str] = set()
        gate = TransferGate(do_transfer)
        watchdog = ServiceWatchdog(legs, switcher, context, on_dead=on_dead)

        sound_notes: list = []  # the note about the newest caller sound, while an agent's session holds it

        async def drop_sound_note() -> None:
            """The agent answered the turn that the note was about, or a newer note replaces it."""
            while sound_notes:
                try:
                    await forget_note(sound_notes.pop())
                except Exception:
                    log.exception("call %s: could not take a caller-sound note out of the session", self.call_control_id)

        async def on_sound(note: str, cut: bool) -> None:
            """A caller's turn starts: tell the agent what to say if the turn has no words (a cough, a noise)."""
            await drop_sound_note()
            if state.ending or state.transfer_requested:
                return
            vl = legs[state.active_agent]
            try:
                sent = await tell_now(vl.service, note)
            except Exception:
                log.exception("call %s: could not send the %s agent a note about a caller sound", self.call_control_id, vl.agent)
                return
            if sent:
                sound_notes.append(sent)
            if cut:
                log.info("call %s: a caller sound cut the %s agent off; agent told: %s", self.call_control_id, vl.agent, bool(sent))

        async def on_spoken_tool(target: str) -> None:
            """The agent said a handoff tool's name aloud. If no tool call follows, the server makes the handoff."""
            holder = state.active_agent

            async def hand_off() -> None:
                await asyncio.sleep(SPOKEN_TOOL_WAIT_SECS)
                if state.active_agent != holder or state.ending or state.transfer_requested:
                    return  # the real tool call came, or the call is on its way out
                log.info("call %s: the %s agent said the tool name transfer_to_%s and did not call it; the server makes the handoff", self.call_control_id, holder, target)
                try:
                    await self._handoff(
                        target, NoToolCall(), legs=legs, switcher=switcher, worker=worker, context=context, state=state, schemas=schemas, reroute=reroute, expect_speech=expect_speech, stats=stats,
                        caller_words=caller_words,
                        known_name=lambda: ctx.caller_name,
                    )
                except Exception:
                    log.exception("call %s: the handoff by the server to the %s agent failed", self.call_control_id, target)

            spawn(hand_off())

        async def tell_agent(text: str) -> None:
            # Into the switcher, not the head of the pipeline: the active voice service must see
            # the message first and put it in its session. The assistant aggregator then adds it
            # to the shared context and asks for an answer once the agent has stopped speaking.
            await switcher.queue_frame(LLMMessagesAppendFrame(messages=[{"role": "developer", "content": self.respell.spoken(text)}], run_llm=True))

        async def send_voice_note(text: str):
            if not self.settings.voice_steer or state.ending or state.transfer_requested:
                return None
            return await tell_now(legs[state.active_agent].service, text)

        frustration = FrustrationNote(send_voice_note, forget_note, current=lambda note: note.service is legs[state.active_agent].service)

        async def on_voice_reading(reading: VoiceReading) -> None:
            """The voice model read the caller's turn: keep the reading, and tell the agent when the caller sounds frustrated."""
            keep_reading(ctx.voice_readings, reading)
            log.info(
                "call %s: the caller's voice in turn %d: %s %.2f, arousal %.2f, laughter %.2f, %.1f syllables/s, sarcasm %.2f, upset %.2f (%.1f s read in %d ms)",
                self.call_control_id, reading.turn, reading.tone, reading.confidence, reading.arousal, reading.laughter, reading.speech_rate, reading.sarcasm, reading.upset, reading.seconds, reading.millis,
            )  # fmt: skip
            try:
                if await frustration.on_reading(reading):
                    log.info("call %s: the caller sounds frustrated in turn %d; the %s agent got a note", self.call_control_id, reading.turn, state.active_agent)
            except Exception:
                log.exception("call %s: could not send or remove the note about the caller's voice", self.call_control_id)

        save_nudged: set[int] = set()

        async def on_agent_stopped() -> None:
            """The caller said yes to the message read-back, and the agent spoke without saving it: tell it to save now."""
            if "save_lead" not in self.specs[state.active_agent].tools or state.ending or state.transfer_requested:
                return

            async def check() -> None:
                await asyncio.sleep(MISSED_SAVE_WAIT_SECS)  # a save_lead call right after the words needs time to land
                if gate.speaking or state.ending or state.transfer_requested:
                    return
                key = missed_save(ctx, ctx.transcript())
                if not key or key in save_nudged:
                    return
                save_nudged.add(key)
                log.info("call %s: the caller confirmed the message and the %s agent did not save it; agent told to save now", self.call_control_id, state.active_agent)
                await tell_agent(SAVE_NOW)

            spawn(check())

        caller_audio = CallerAudioFile(caller_audio_path(self.settings.caller_audio_dir, self.call_control_id)) if self.settings.save_caller_audio else None
        # The voice reader hears each caller turn. It is off for a caller who asked to be forgotten.
        voice_model = default_voice_model(self.settings)
        voice_reader = CallerVoiceReader(voice_model, on_voice_reading, active=lambda: not ctx.forgotten) if voice_model else None
        processors = [transport.input(), *([caller_audio] if caller_audio else []), *([voice_reader] if voice_reader else []), OpeningQuiet(), user_aggregator, caller_words, watchdog, switcher, CallerSoundWatch(on_sound, drop_sound_note), SpokenToolWatch(self.specs, on_spoken_tool)]
        if self.settings.voice_level_dbfs:
            # Every voice at one loudness, so a handoff does not sound like a volume change.
            voice_by_service = {id(vl.service): self.specs[name].voice or "" for name, vl in legs.items()}
            processors.append(
                VoiceLeveler(
                    lambda: voice_by_service.get(id(switcher.strategy.active_service), ""),
                    target_dbfs=self.settings.voice_level_dbfs,
                    seeds=VOICE_LEVELS_DBFS,
                )
            )
        # The decision grader hears the call after hangup, so both sides are recorded in memory (voice/call_audio.py).
        hearing = default_decision_grader(self.settings)
        recording = recorder(self.call_control_id, sample_rate=TELNYX_SAMPLE_RATE, forgotten=lambda: ctx.forgotten) if hearing is not None and hearing.hears and not ctx.forgotten else None
        pipeline = Pipeline([*processors, transport.output(), *([recording] if recording else []), gate, AgentStoppedWatch(on_agent_stopped), assistant_aggregator])

        clocks = [(vl.leg, clock) for vl in legs.values() if (clock := session_clock(vl.service)) is not None]
        stats.bind({vl.service.name: vl.leg for vl in legs.values()}, lambda: legs[state.active_agent].leg, lambda: state.active_agent, clocks)
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=TELNYX_SAMPLE_RATE,
                audio_out_sample_rate=TELNYX_SAMPLE_RATE,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[MetricsLogObserver(), stats],
            idle_timeout_secs=300,
            enable_tracing=traced,
            enable_turn_tracking=True,
            conversation_id=self.call_control_id,
            additional_span_attributes={"call.entry_agent": entry_agent, "call.caller_number": caller_number, "call.restart_reason": resume.reason if resume else ""},
        )
        runner = WorkerRunner(handle_sigint=False)

        # --- tool handlers: the voice model requests, this code acts ---
        async def on_save_lead(params: FunctionCallParams) -> None:
            args = {k: str(params.arguments.get(k) or "") for k in ("name", "need", "email", "company", "callback_number", "call_notes", "timeline", "size")}
            # The caller's yes can reach the context a moment after the tool call, as with a transfer.
            waited = 0.0
            while not ctx.message_confirm_asked and not confirmed_the_message(ctx.transcript()) and waited < TRANSCRIPT_WAIT_SECS:
                await asyncio.sleep(0.1)
                waited += 0.1
            if question := message_confirm_question(ctx, ctx.transcript()):
                ctx.held_message = args  # saved at the end of the call if the caller hangs up before a yes
                log.info("call %s: save held back; the caller has not confirmed the message", self.call_control_id)
                await params.result_callback(question)
                return
            result, lead = await save_lead_record(ctx, **args, visited=[ENTRY_AGENT, *state.handoffs])
            state.lead_saved = state.lead_saved or bool(lead)
            await params.result_callback(result)

        async def on_identify_caller(params: FunctionCallParams) -> None:
            # Code compares the name with the names on record. Only on a match do the notes from
            # earlier calls enter the conversation; from there every agent sees them in the shared context.
            result = identify(ctx, str(params.arguments.get("name") or ""))
            log.info("call %s: caller identified (name given: %s, earlier caller: %s)", self.call_control_id, bool(ctx.caller_name), ctx.identity_confirmed)
            await params.result_callback(result)

        async def on_verify_passphrase(params: FunctionCallParams) -> None:
            # Code compares the passphrase. Only after a match do the notes from earlier calls enter the conversation.
            result = verify(ctx, str(params.arguments.get("passphrase") or ""))
            log.info("call %s: passphrase check (try %d, verified: %s)", self.call_control_id, ctx.phrase_tries, ctx.identity_verified)
            await params.result_callback(result)

        async def on_set_passphrase(params: FunctionCallParams) -> None:
            await params.result_callback(set_phrase(ctx, str(params.arguments.get("passphrase") or "")))

        async def on_file_feedback(params: FunctionCallParams) -> None:
            # Kept in the call context only. The GitHub issue is made after hangup (`_record_analytics`).
            await params.result_callback(file_report(ctx, str(params.arguments.get("kind") or ""), str(params.arguments.get("summary") or "")))

        async def on_forget_caller(params: FunctionCallParams) -> None:
            result, _ = forget(ctx)
            if ctx.forgotten:
                # The call audio goes at once, and the rest of the call is not recorded.
                if recording:
                    await recording.stop_recording()
                call_audio.drop(self.call_control_id)
            await params.result_callback(result)

        async def on_call_stats(params: FunctionCallParams) -> None:
            await params.result_callback(
                describe_call(
                    ctx,
                    visited=[ENTRY_AGENT, *state.handoffs],
                    active_agent=state.active_agent,
                    lead_saved=state.lead_saved,
                    transfer_requested=state.transfer_requested,
                    figures=stats.describe(),
                )
            )

        async def on_search_writing(params: FunctionCallParams) -> None:
            query = str(params.arguments.get("query") or "")
            log.info("call %s: search_writing %r", self.call_control_id, query)
            # The query embedding takes a few milliseconds of CPU: keep it off the audio loop.
            await params.result_callback(await asyncio.to_thread(search_result, default_library(), query))

        async def on_crm_lookup(params: FunctionCallParams) -> None:
            # One read of the owner's CRM over MCP. Code allows it only after the passphrase check (crm.request_status).
            started_read = time.monotonic()
            result = await request_status(ctx)
            log.info("call %s: crm_lookup (verified: %s) in %.0f ms", self.call_control_id, ctx.identity_verified, (time.monotonic() - started_read) * 1000)
            await params.result_callback(result)

        async def on_text_link(params: FunctionCallParams) -> None:
            await params.result_callback(await text_portfolio_link_to_caller(ctx, owner_name=self.settings.owner_name))

        async def on_transfer(params: FunctionCallParams) -> None:
            reason = str(params.arguments.get("reason") or "")
            # A transfer rings the owner's phone, so the caller's own words must ask for it.
            # The provider's transcript of the last turn can arrive after the tool call: wait for it briefly.
            waited = 0.0
            if ctx.decider is not None and not ctx.forgotten and not ctx.transfer_question_asked and not asked_for_the_owner(ctx.transcript()):
                # No match in the word list yet, so the loop below waits for the transcript. Open Jev's
                # connection during that wait. On a match there is no wait, and the question opens it.
                warm_task = asyncio.create_task(ctx.decider.warm())
                background.add(warm_task)
                warm_task.add_done_callback(background.discard)
            while not ctx.transfer_question_asked and not asked_for_the_owner(ctx.transcript()) and waited < TRANSCRIPT_WAIT_SECS:
                await asyncio.sleep(0.1)
                waited += 0.1
            if question := await transfer_question(ctx, ctx.transcript()):
                log.info("call %s: transfer held back; the caller did not ask for a live call", self.call_control_id)
                await params.result_callback(question)
                return
            who = {"caller_name": str(params.arguments.get("caller_name") or ""), "company": str(params.arguments.get("company") or ""), "reason": reason}
            if closed := transfer_closed_message(ctx, caller_name=who["caller_name"]):
                # No email now: the agent offers to take a message, and that message is the email.
                remember_refused_transfer(ctx, **who)
                log.info("call %s: transfer refused (%s)", self.call_control_id, ctx.transfer_unavailable or "outside the transfer hours")
                await params.result_callback(closed)
                return
            # The owner's phone rings next. The owner gets a record of who called, even if nobody answers.
            if await note_caller_before_transfer(ctx, lead_saved=state.lead_saved, **who, visited=[ENTRY_AGENT, *state.handoffs]):
                state.lead_saved = True
            state.transfer_requested = True
            state.transfer_announce = screening_prompt(str(params.arguments.get("caller_name") or "") or ctx.caller_name, reason)
            # The owner reads who is calling while the agent says its goodbye line.
            llm_task = asyncio.create_task(text_owner_before_transfer(ctx, str(params.arguments.get("reason", ""))))
            background.add(llm_task)
            llm_task.add_done_callback(background.discard)
            log.info("call %s: owner transfer requested (%s)", self.call_control_id, params.arguments.get("reason", ""))
            gate.arm()
            owner = self.settings.owner_name
            if self.settings.transfer_screening:
                # The caller hears a ring tone, and comes back to this agent when the owner does not accept.
                line, then = f"I am calling {owner} now. Please hold.", "The caller hears a ring tone the moment you stop speaking."
            else:
                line, then = f"Connecting you to {owner} now. Take care.", f"The call moves to {owner}'s phone the moment you stop speaking."
            await params.result_callback(f'Say only this one line now: "{line}" Say nothing before it and nothing after it. {then}')

        async def do_hangup() -> None:
            log.info("call %s: the agent ended the call after the caller said goodbye", self.call_control_id)
            state.end_reason = "agent"
            try:
                await self.telephony.hangup(self.call_control_id)
            except Exception:
                log.exception("call %s: the hang-up for end_call failed", self.call_control_id)
            await runner.cancel()

        async def on_end_call(params: FunctionCallParams) -> None:
            if state.ending or state.transfer_requested:
                await params.result_callback("The call is ending already. Say nothing more.", properties=FunctionCallResultProperties(run_llm=False))
                return
            # The caller's own words must end the call. The transcript of the last turn can arrive
            # after the tool call: wait for it briefly.
            waited = 0.0
            while not said_goodbye(ctx.transcript()) and waited < TRANSCRIPT_WAIT_SECS:
                await asyncio.sleep(0.1)
                waited += 0.1
            if not said_goodbye(ctx.transcript()):
                log.info("call %s: end_call held back; the caller did not say goodbye", self.call_control_id)
                await params.result_callback(CALL_NOT_ENDED)
                return
            state.ending = True
            if gate.speaking:
                # The agent said its goodbye before it called the tool. Do not ask for a second one.
                gate.arm(do_hangup)
                await params.result_callback("The call ends when you stop speaking.", properties=FunctionCallResultProperties(run_llm=False))
                return
            gate.arm(do_hangup)
            await params.result_callback(say_goodbye(ctx.website_spoken))

        def make_handoff(target: str):
            async def on_handoff(params: FunctionCallParams) -> None:
                await self._handoff(
                    target, params, legs=legs, switcher=switcher, worker=worker, context=context, state=state, schemas=schemas, reroute=reroute, expect_speech=expect_speech, stats=stats,
                    caller_words=caller_words,
                    known_name=lambda: ctx.caller_name,
                )

            return on_handoff

        respelled = self.respell.tool  # a tool's result goes to the voice model respelled; its arguments come in as written
        for vl in legs.values():  # any service can become active, so each one knows every tool
            vl.service.register_function("save_lead", respelled(on_save_lead))
            vl.service.register_function("identify_caller", respelled(on_identify_caller))
            vl.service.register_function("verify_passphrase", respelled(on_verify_passphrase))
            vl.service.register_function("set_passphrase", respelled(on_set_passphrase))
            vl.service.register_function("file_feedback", respelled(on_file_feedback))
            vl.service.register_function("forget_caller", respelled(on_forget_caller))
            vl.service.register_function("get_call_stats", respelled(on_call_stats))
            vl.service.register_function("search_writing", respelled(on_search_writing))
            vl.service.register_function("crm_lookup", respelled(on_crm_lookup))
            vl.service.register_function("request_owner_transfer", respelled(on_transfer))
            vl.service.register_function("text_portfolio_link", respelled(on_text_link))
            vl.service.register_function("end_call", respelled(on_end_call))
            for name in self.specs:
                vl.service.register_function(handoff_tool_name(name), respelled(make_handoff(name)))

        async def enforce_time_limit() -> None:
            """Warn one minute before the limit, ask for a goodbye 15 s before it, then end the call."""
            limit = self.settings.max_call_minutes * 60
            if limit <= 0:
                return

            await asyncio.sleep(max(limit - 60, 0))
            log.info("call %s: one minute left", self.call_control_id)
            # On 2026-11-17 the contact agent itself said "the contact agent can help", and the caller asked for it.
            follow_up = (
                "If they have not left a message yet, offer to take one now."
                if "save_lead" in self.specs[state.active_agent].tools
                else "If they want a follow-up, offer the contact agent."
            )
            await tell_agent(f"About one minute is left on this call; the line has a time limit. Tell the caller in one sentence. {follow_up}")
            await asyncio.sleep(min(45, limit))
            await tell_agent("The time limit is here. Say goodbye in one short sentence now, and say they can call again.")
            await asyncio.sleep(15)
            log.info("call %s: time limit reached; hanging up", self.call_control_id)
            state.end_reason = "limit"
            try:
                await self.telephony.hangup(self.call_control_id)
            except Exception:
                log.exception("call %s: hang-up at the time limit failed", self.call_control_id)
            await runner.cancel()

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, _client) -> None:
            timers.append(asyncio.create_task(enforce_time_limit()))
            timers.append(asyncio.create_task(watch_services()))
            log.info("call %s connected after %.0f ms; agent=%s leg=%s voice=%s", self.call_control_id, (time.monotonic() - started) * 1000, entry.name, legs[entry_agent].leg, entry.voice)
            await _answer(legs[entry_agent], context, worker)
            expect_speech(legs[entry_agent])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, _client) -> None:
            log.info("call %s disconnected", self.call_control_id)
            state.end_reason = state.end_reason or "caller"
            await runner.cancel()

        await runner.add_workers(worker)
        summary_tries, message_left = 0, False
        summary_faults: tuple[str, ...] = ()
        try:
            await runner.run()
        finally:
            for timer in timers:
                timer.cancel()
            if caller_audio:
                caller_audio.close()
            if not state.end_reason:
                # Nothing ended this session: not the caller, a transfer, or the time limit. The
                # pipeline stopped by itself, and the caller would sit in silence.
                log.error("call %s: the pipeline ended by itself", self.call_control_id)
                health.mark_down(legs[state.active_agent].leg, "the pipeline ended by itself")
                start_fail_over(state.active_agent, PROVIDER_FAILED, "the pipeline ended by itself")
            if state.end_reason == "failover":
                # The call goes on in a new session, which writes the summary when the call ends.
                log.info("call %s: session ended after %.0f s to change the voice provider", self.call_control_id, time.monotonic() - started)
            else:
                # The call is over. A message that waited for the caller's yes is saved as it is.
                try:
                    if await save_held_message_at_call_end(ctx, visited=[ENTRY_AGENT, *state.handoffs]):
                        state.lead_saved = True
                        log.info("call %s: the call ended before the caller confirmed the message; saved as it is", self.call_control_id)
                except Exception:
                    log.exception("call %s: could not save the held message", self.call_control_id)
                try:
                    message_left = message_saved(ctx)  # read now: the store closes below
                except Exception:
                    log.exception("call %s: could not read if a message was saved", self.call_control_id)
                if ctx.forgotten:
                    # The caller asked to be forgotten: no transcript in the log, no summary, no warm-lead email.
                    # The call row stays, empty, for the per-number call limits.
                    store.finish_call(self.call_control_id, agents=[ENTRY_AGENT, *state.handoffs], summary="")
                    try:  # forget_caller clears a refused transfer, so this one came after it
                        await note_refused_transfer_at_call_end(ctx, lead_saved=state.lead_saved, visited=[ENTRY_AGENT, *state.handoffs])
                    except Exception:
                        log.exception("call %s: could not save the note for a refused transfer", self.call_control_id)
                    if caller_audio:
                        caller_audio.path.unlink(missing_ok=True)
                    log.info("call %s: the caller asked to be forgotten; nothing about the call is kept", self.call_control_id)
                else:
                    # What was said, as each agent heard it. Without it, a call that went wrong cannot be read afterwards.
                    log.info("call %s: what was said:\n%s", self.call_control_id, "\n".join(transcript_lines(self.respell.written_messages(context.messages))) or "(nothing)")
                    summary = await self._finish_call(store, ctx, context.messages, visited=[ENTRY_AGENT, *state.handoffs], transferred=state.end_reason == "transfer")
                    summary_tries, summary_faults = summary.tries, summary.first_faults
                    # The owner's CRM gets what the call left: after hangup, so the caller waits for nothing (crm.py).
                    await sync_call(
                        ctx, summary, visited=[ENTRY_AGENT, *state.handoffs], transferred=state.end_reason == "transfer", test=ctx.caller_number == self.settings.owner_phone
                    )
            store.close()
        if state.end_reason == "failover":
            call_audio.drop_later(self.call_control_id)  # the next media stream keeps it; with no next stream it is deleted
            return
        log.info("call %s: figures: %s", self.call_control_id, stats.describe())
        log.info(
            "call %s finished after %.0f s; agents: %s; lead=%s transfer=%s",
            self.call_control_id,
            time.monotonic() - started,
            " -> ".join([ENTRY_AGENT, *state.handoffs]),
            state.lead_saved,
            state.transfer_requested,
        )
        outcome = TRANSFER if state.end_reason == "transfer" else MESSAGE if message_left else NO_OUTCOME
        await self._record_analytics(ctx, stats, context.messages, visited=[ENTRY_AGENT, *state.handoffs], outcome=outcome, summary_tries=summary_tries, summary_faults=summary_faults)

    async def _handoff(
        self,
        target: str,
        params: FunctionCallParams,
        *,
        legs: dict[str, VoiceLeg],
        switcher: LLMSwitcher,
        worker: PipelineWorker,
        context: LLMContext,
        state: VoiceCallState,
        schemas: dict,
        reroute=None,
        expect_speech=None,
        stats: CallStats | None = None,
        caller_words: CallerWords | None = None,
        known_name=None,  # () -> the caller's name when they gave or confirmed it on this call, else ""
    ) -> None:
        """Switch the active voice service to the target agent and have it answer."""
        if target == state.active_agent:
            # A model can call a tool it only saw in the conversation history. Measured 2026-11-16 on Nova.
            log.info("call %s: handoff %s -> %s refused; the agent has the call already", self.call_control_id, target, target)
            await params.result_callback(f"No handoff was made: you are the {target} agent already. Do not mention this. Answer the caller yourself.")
            return
        if state.took_over_at and caller_words is not None:
            # Nova reports the caller's words and calls the tool in the same millisecond: let the words arrive.
            deadline = time.monotonic() + SPOKE_WAIT_SECS
            while not caller_words.spoke_since(state.took_over_at) and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
        if state.took_over_at and caller_words is not None and not caller_words.spoke_since(state.took_over_at):
            # The agent got this call by a handoff and the caller has said nothing to it. A handoff now
            # is the agent passing the caller on unanswered. Measured 2026-11-16: consultant -> contact
            # -> consultant -> contact in 10 s, with "Let me bring in the right agent" three times.
            holder = legs[state.active_agent]
            log.info("call %s: handoff %s -> %s refused; the caller has not spoken since the last handoff", self.call_control_id, holder.agent, target)
            await params.result_callback(handoff_refused(holder.agent, LEG_PLATFORMS.get(holder.leg, holder.leg)))
            return
        previous = legs[state.active_agent]
        nxt = legs[target]
        spec = self.specs[target]
        state.active_agent = target
        state.handoffs.append(target)
        t0 = time.monotonic()
        platform = LEG_PLATFORMS.get(nxt.leg, nxt.leg)

        async def switch_when_context_has_result() -> None:
            # Runs once the assistant aggregator has written the tool result into the shared
            # context. That happens only after the old agent's sentence has played out, because
            # the result frame rides the audio queue in the output transport. Switching earlier
            # made the new session replay a context without the result; when the result arrived
            # later it was sent to a session that never had that call id, which OpenAI rejects
            # and Pipecat treats as fatal.
            if reroute is not None and reroute(target):
                return  # the target's provider is not answering; a new session on another provider takes the handoff
            # The next agent answers the caller's last words, so they must be in the context it gets.
            # A service can call the handoff tool before Pipecat has committed those words.
            if caller_words is not None and caller_words.missing_from(transcript_lines(context.messages)):
                t_wait = time.monotonic()
                await _wait_for(lambda: not caller_words.missing_from(transcript_lines(context.messages)), CALLER_WORDS_WAIT_SECS, what="the caller's last words")
                log.info("call %s: waited %.0f ms for the caller's last words to reach the context", self.call_control_id, (time.monotonic() - t_wait) * 1000)
                if missing := caller_words.missing_from(transcript_lines(context.messages)):
                    # Pipecat has not committed the caller's turn. Without these words the next agent
                    # said "Ready to keep going. What would you like to do next?" and asked for a name it had.
                    context.add_message({"role": "user", "content": missing})
                    log.info("call %s: added the caller's last words to the context for the %s agent", self.call_control_id, target)
            context.set_tools(tools_for_agent(spec, schemas))
            retire_connect_message(context.messages)
            # The last message of a replayed context decides what the new session does with it.
            # After a bare tool result, a session sometimes only says it is ready to continue.
            # A plain instruction as the last message makes it speak.
            context.add_message({"role": "developer", "content": handoff_instruction(target, platform, known_name() if known_name else "", transcript_lines(context.messages))})
            async with nxt.lock:  # a background reset of this service may still be running
                await ensure_connected(nxt.service)
                await adopt_context(nxt.service, context)
                await worker.queue_frames([ManuallySwitchServiceFrame(service=nxt.service)])
                await _wait_for(lambda: switcher.strategy.active_service is nxt.service, SWITCH_TIMEOUT_SECS)
                await _answer(nxt, context, worker)
            state.took_over_at = time.monotonic()
            if expect_speech is not None:
                expect_speech(nxt)
            if stats is not None:
                stats.note_handoff(time.monotonic() - t0)
            log.info("call %s: handoff %s -> %s (leg=%s voice=%s) switched %.0f ms after the tool call", self.call_control_id, previous.agent, target, nxt.leg, spec.voice, (time.monotonic() - t0) * 1000)
            # The old service keeps a stale server-side conversation. Reset it in the background
            # (about 2.5 s) so a later handoff back to it replays the context cleanly.
            previous.service.create_task(_reset_in_background(previous))

        await params.result_callback(
            f"Handed off to the {target} agent.",
            properties=FunctionCallResultProperties(run_llm=False, on_context_updated=switch_when_context_has_result),
        )


async def _wait_for(condition, timeout: float, *, what: str = "service switch") -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            log.warning("%s did not complete within %.1f s", what, timeout)
            return
        await asyncio.sleep(0.01)


async def _answer(vl: VoiceLeg, context: LLMContext, worker: PipelineWorker) -> None:
    """Have this agent's service answer from the shared context. The service must be the active one."""
    if not vl.used and first_answer_needs_run_frame(vl.service):
        # First use on OpenAI: the context frame from LLMRunFrame sets up the session and answers.
        await worker.queue_frames([LLMRunFrame()])
    else:
        # Its server-side session is fresh (never used, or reset when we left it): replay and answer.
        await replay_and_respond(vl.service, context)
    vl.used = True


async def _reset_in_background(vl: VoiceLeg) -> None:
    async with vl.lock:
        t0 = time.monotonic()
        try:
            await reset_session(vl.service)
            log.info("%s: session reset in %.0f ms, ready for a later handoff", vl.agent, (time.monotonic() - t0) * 1000)
        except Exception:
            log.exception("%s: background session reset failed", vl.agent)
