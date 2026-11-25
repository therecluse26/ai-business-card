"""Voice legs: build the Pipecat speech-to-speech service for one agent spec, and drive it.

Each agent gets its own service, pre-connected at call start, because a provider fixes a session's
voice once it has spoken. A reconnect-and-replay swap on OpenAI was measured at 3-4 s of silence;
switching between pre-connected services takes under 1 s.

Amazon Nova Sonic and Phonic are the exceptions: their service connects when a handoff reaches
its agent, and disconnects when the agent is left. A Bedrock stream that carries no audio is
closed by AWS, so a stream opened at call start would be dead by the time it is needed. Phonic
bills each minute that a conversation is open. Phonic's service is in phonic_leg.py.

The functions at the bottom (`reset_session`, `replay_and_respond`, ...) are the only place that
knows how each provider's Pipecat service resets and replays. They use private members of those
services, pinned by the Pipecat version in `uv.lock`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time

from google.genai.types import EndSensitivity, LiveServerContent, LiveServerMessage, StartSensitivity
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextType,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiVADParams
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime import events as openai_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from ai_call_center.voice.phonic_leg import PhonicLLMService, build_phonic
from ai_call_center.voice.repeat import DROP, HOLD, PASS, RepeatGuard

from ai_call_center.agents.specs import AgentSpec, default_voice
from ai_call_center.config import Settings

log = logging.getLogger(__name__)

FALLBACK_LEG = "openai_realtime"


def turn_detection(settings: Settings) -> TurnDetection | SemanticTurnDetection:
    """Server VAD ends the turn after a fixed silence; semantic VAD lets the model judge."""
    if settings.openai_turn_detection == "semantic":
        return SemanticTurnDetection(eagerness=settings.openai_vad_eagerness, create_response=True, interrupt_response=True)
    return TurnDetection(
        type="server_vad",
        threshold=settings.openai_vad_threshold,
        silence_duration_ms=settings.openai_vad_silence_ms,
        prefix_padding_ms=200,
    )


def noise_reduction(settings: Settings) -> InputAudioNoiseReduction | None:
    if settings.openai_noise_reduction == "off":
        return None
    return InputAudioNoiseReduction(type=settings.openai_noise_reduction)


HELD_AUDIO_MAX_CHARS = 192_000  # base64 text of about 3 s of audio; a held item with no verdict by then plays


class OnceOpenAIRealtime(OpenAIRealtimeLLMService):
    """OpenAI Realtime that drops an audio item which says again what the same answer said already.

    See voice/repeat.py. Audio and text of a later item are held until the item's first letters
    show if it is a copy. The first item is still playing then, so the caller hears no gap.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._repeat = RepeatGuard()
        self._held: list[tuple[str, object]] = []  # ("audio" | "text", event) of the one item that is held

    async def _play(self, kind: str, evt) -> None:
        if kind == "audio":
            await super()._handle_evt_audio_delta(evt)
        else:
            await super()._handle_evt_audio_transcript_delta(evt)

    async def _release(self, verdict: str) -> None:
        held, self._held = self._held, []
        if not held:
            return
        if verdict == DROP:
            log.info("%s: one answer had the same words two times; the second copy was dropped", self)
            return
        for kind, evt in held:
            await self._play(kind, evt)

    async def _route(self, kind: str, evt, verdict: str) -> None:
        if self._held and self._held[0][1].item_id != evt.item_id:
            await self._release(self._repeat.settle(self._held[0][1].item_id))  # the held item ended
        if verdict == HOLD:
            self._held.append((kind, evt))
            if sum(len(e.delta or "") for k, e in self._held if k == "audio") > HELD_AUDIO_MAX_CHARS:
                await self._release(self._repeat.give_up(evt.item_id))
        elif verdict == DROP:
            await self._release(DROP)
        else:
            await self._release(PASS)
            await self._play(kind, evt)

    async def _handle_evt_audio_delta(self, evt) -> None:
        await self._route("audio", evt, self._repeat.audio(evt.response_id, evt.item_id))

    async def _handle_evt_audio_transcript_delta(self, evt) -> None:
        await self._route("text", evt, self._repeat.text(evt.response_id, evt.item_id, evt.delta or ""))

    async def _handle_evt_speech_started(self, evt) -> None:
        self._held = []  # the caller spoke: nothing that was held may play after it
        await super()._handle_evt_speech_started(evt)

    async def _handle_evt_response_done(self, evt) -> None:
        if self._held:
            item_id = self._held[0][1].item_id
            await self._release(self._repeat.settle(item_id) if evt.response.status == "completed" else DROP)
        await super()._handle_evt_response_done(evt)


def build_openai_realtime(spec: AgentSpec, *, instructions: str, settings: Settings, tools: ToolsSchema | None = None) -> LLMService:
    """gpt-realtime-2.1-mini over the Realtime WebSocket. Audio stays PCM 24 kHz inside Pipecat;
    the Telnyx serializer converts to and from 8 kHz PCMU. Tools come from the shared context."""
    return OnceOpenAIRealtime(
        api_key=settings.openai_api_key.get_secret_value(),
        settings=OpenAIRealtimeLLMService.Settings(
            model=settings.openai_realtime_model,
            system_instruction=instructions,
            session_properties=SessionProperties(
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription(language="en"),  # without it, a short "yeah" came back as Japanese
                        turn_detection=turn_detection(settings),
                        noise_reduction=noise_reduction(settings),
                    ),
                    output=AudioOutput(voice=spec.voice or settings.openai_realtime_voice),
                ),
            ),
        ),
    )


def gemini_vad(settings: Settings) -> GeminiVADParams:
    """Gemini's own turn detection. Low start sensitivity ignores more background noise."""
    return GeminiVADParams(
        silence_duration_ms=settings.gemini_vad_silence_ms,
        prefix_padding_ms=200,
        start_sensitivity=(
            StartSensitivity.START_SENSITIVITY_LOW if settings.gemini_vad_start_sensitivity == "low" else StartSensitivity.START_SENSITIVITY_HIGH
        ),
        end_sensitivity=EndSensitivity.END_SENSITIVITY_HIGH,
    )


RECORDING_CHUNK_SECS = 0.04  # the size of the audio frames of a stored recording, about the size Gemini sends
RECORDING_SLACK_SECS = 0.3  # a recording still counts as playing this long after its pushed length; the phone starts it a moment later


class HandoffGeminiLive(GeminiLiveLLMService):
    """Gemini Live, with two changes.

    1. A tool result marked `run_llm=False` is not sent to Gemini. Our handoff tools return such a
       result, because the next agent answers, not this one. Gemini answers every tool result it
       receives, so the old agent would speak the new agent's opening line in the old voice.
       Without the result the model waits, silent, until we reset its session.
    2. It can play a stored recording as its own turn (`play_recording`, for the stored greeting),
       and its session can be seeded with the call so far without an answer (`seed_without_answer`).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quiet_seed = False  # the next seed of a new session asks for no answer; see seed_without_answer
        self._recording_ends = 0.0  # monotonic time when a stored recording that plays now ends; 0.0 when none plays

    async def play_recording(self, audio: bytes, sample_rate: int, text: str) -> None:
        """Push a recording down the pipeline as this agent's turn, in the frames of a model answer.

        So the output transport resamples it for the phone, the voice leveler levels it, an
        interruption stops it, the trace shows `text` as the agent's answer, and the watchdog and the
        call figures see the agent speak. `text` does not go into the shared context from here: the
        caller does that (see `_greet_from_recording` in call_session.py), because the session must
        be seeded with it before the recording has played out.
        """
        step = max(int(sample_rate * RECORDING_CHUNK_SECS), 1) * 2
        chunks = [audio[at : at + step] for at in range(0, len(audio) - len(audio) % 2, step)]
        if not chunks or sample_rate <= 0:
            raise ValueError("the recording has no audio")
        words = TTSTextFrame(text, text_type=TextType.SENTENCE)
        words.append_to_context = False
        self._recording_ends = time.monotonic() + len(audio) / 2 / sample_rate + RECORDING_SLACK_SECS
        await self._set_bot_is_responding(True)
        await self.push_frame(TTSStartedFrame())
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(words)
        for chunk in chunks:
            await self.push_frame(TTSAudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=1))
        # End the turn as a model turn ends: TTSStoppedFrame and LLMFullResponseEndFrame, and the
        # trace's "llm_response" span with the words as its output.
        self._llm_output_buffer = text
        await self._handle_msg_turn_complete(LiveServerMessage(server_content=LiveServerContent(turn_complete=True)))

    async def _handle_msg_input_transcription(self, message: LiveServerMessage) -> None:
        # Gemini reports "interrupted" only when the caller talks over its own answer. It does not
        # know of the recording, so the caller's first words while it plays stop it here.
        content = message.server_content
        heard = content.input_transcription.text if content and content.input_transcription else ""
        if heard and heard.strip() and time.monotonic() < self._recording_ends:
            self._recording_ends = 0.0
            log.info("%s: the caller spoke over the stored greeting; it stops", self)
            await self.broadcast_interruption()
        await super()._handle_msg_input_transcription(message)

    async def _handle_interruption(self) -> None:
        self._recording_ends = 0.0
        await super()._handle_interruption()

    async def _create_initial_response(self, for_reconnect: bool = False) -> None:
        if not self.quiet_seed or for_reconnect:
            await super()._create_initial_response(for_reconnect)
            return
        # Pipecat answers the seed of a first context unless `_inference_on_context_initialization`
        # is off. Off only for this seed: a later replay must answer. If the session is still
        # opening, Pipecat calls this again when it is ready; the flag stays until a seed is sent.
        sends = self._session is not None and not self._disconnecting
        answers = self._inference_on_context_initialization
        self._inference_on_context_initialization = False
        try:
            await super()._create_initial_response()
        finally:
            self._inference_on_context_initialization = answers
        if sends:
            self.quiet_seed = False

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame) -> None:
        if frame.properties is not None and frame.properties.run_llm is False:
            self._completed_tool_calls.add(frame.tool_call_id)  # so a later context frame does not send it either
            if self._bot_is_responding:
                # The model's turn never completes now. End it here, so the sentence the agent
                # just said is stored as its own message in the shared context.
                await self._set_bot_is_responding(False)
                await self.push_frame(TTSStoppedFrame())
                await self.push_frame(LLMFullResponseEndFrame())
            return
        await super()._handle_function_call_result(frame)


def build_gemini_live(spec: AgentSpec, *, instructions: str, settings: Settings, tools: ToolsSchema | None = None) -> LLMService:
    """gemini-3.8-live over the Live API. The agent's tools are fixed when the session opens:
    a Gemini session cannot change tools later, and taking them from the context costs a reconnect."""
    return HandoffGeminiLive(
        api_key=settings.google_api_key.get_secret_value(),
        tools=tools,
        settings=GeminiLiveLLMService.Settings(
            model=settings.gemini_live_model,
            system_instruction=instructions,
            voice=spec.voice or default_voice(spec.name, "gemini_live"),
            vad=gemini_vad(settings),
        ),
    )


NOVA_SETTLE_SECS = 0.3  # between the history and the request to speak; see HandoffNovaSonic


def nova_tool_result(result) -> str:
    """A tool result in the only form Bedrock accepts: a JSON object, as text.

    Our tools return a sentence. Pipecat sends it as a JSON string, and Bedrock answers
    "Unsupported JSON type in Tool Result. Please provide the Tool Result as a JSON object".
    The stream then restarts and the agent says nothing. Measured 2026-11-16.
    """
    try:
        value = json.loads(result) if isinstance(result, str) else result
    except ValueError:
        value = result
    return json.dumps(value if isinstance(value, dict) else {"result": value}, ensure_ascii=False)


class HandoffNovaSonic(AWSNovaSonicLLMService):
    """Amazon Nova Sonic, with five changes.

    1. It does not connect at setup. `ensure_connected` opens the stream when the agent is needed.
    2. A tool result marked `run_llm=False` is not sent to Nova, for the same reason as on Gemini:
       Nova answers every tool result, and after a handoff the next agent answers, not this one.
    3. Token usage keeps speech and text apart, so the cost estimate can price them apart.
    4. It waits `NOVA_SETTLE_SECS` between the conversation history and the request to speak.
       Measured 2026-11-16: with four history messages and no wait, Bedrock answers
       "Invalid input request". A wait of 0.15 s was enough in five of five tries.
    5. A tool result goes to Nova as a JSON object. See `nova_tool_result`.
    """

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super(AWSNovaSonicLLMService, self).setup(setup)  # the base setup, without Nova's connect
        self._wants_connection = True

    async def _send_audio_input_start_event(self) -> None:
        # Pipecat sends the history, then this event, then the request to speak.
        await super()._send_audio_input_start_event()
        await asyncio.sleep(NOVA_SETTLE_SECS)

    async def _send_tool_result(self, tool_call_id, result) -> None:
        await super()._send_tool_result(tool_call_id, nova_tool_result(result))

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame) -> None:
        if frame.properties is not None and frame.properties.run_llm is False:
            self._completed_tool_calls.add(frame.tool_call_id)
            if self._assistant_is_responding:
                self._assistant_is_responding = False
                await self._report_assistant_response_ended()
            return
        await super()._handle_function_call_result(frame)

    async def _handle_usage_event(self, event_json) -> None:
        delta = event_json["usageEvent"].get("details", {}).get("delta", {})
        tokens_in, tokens_out = delta.get("input", {}), delta.get("output", {})
        speech_in, speech_out = tokens_in.get("speechTokens", 0), tokens_out.get("speechTokens", 0)
        prompt = speech_in + tokens_in.get("textTokens", 0)
        completion = speech_out + tokens_out.get("textTokens", 0)
        if prompt or completion:
            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=prompt + completion,
                    input_audio_tokens=speech_in,
                    output_audio_tokens=speech_out,
                )
            )


def build_nova_sonic(spec: AgentSpec, *, instructions: str, settings: Settings, tools: ToolsSchema | None = None) -> LLMService:
    """amazon.nova-2-5-sonic over a Bedrock two-way stream. The tools are fixed when the session opens."""
    return HandoffNovaSonic(
        access_key_id=settings.nova_aws_access_key_id,
        secret_access_key=settings.nova_aws_secret_access_key.get_secret_value(),
        region=settings.nova_aws_region,
        tools=tools,
        settings=AWSNovaSonicLLMService.Settings(
            model=settings.nova_sonic_model,
            system_instruction=instructions,
            voice=spec.voice or default_voice(spec.name, "nova_sonic"),
            endpointing_sensitivity=settings.nova_endpointing,
        ),
    )


# The order is the fallback order: when an agent's own leg cannot run, the first other ready leg runs.
LEG_BUILDERS = {
    "openai_realtime": build_openai_realtime,
    "gemini_live": build_gemini_live,
    "nova_sonic": build_nova_sonic,
    "phonic": build_phonic,
}


def has_credentials(leg: str, settings: Settings) -> bool:
    """True when this server holds the API key that the leg needs."""
    if leg == "openai_realtime":
        return bool(settings.openai_api_key.get_secret_value())
    if leg == "gemini_live":
        return bool(settings.google_api_key.get_secret_value())
    if leg == "nova_sonic":
        return bool(settings.nova_aws_access_key_id and settings.nova_aws_secret_access_key.get_secret_value())
    if leg == "phonic":
        return bool(settings.phonic_api_key.get_secret_value())
    return False


def ready_legs(settings: Settings, down: frozenset[str] = frozenset()) -> list[str]:
    """The legs that can run now: built, with an API key, and not marked down."""
    return [leg for leg in LEG_BUILDERS if has_credentials(leg, settings) and leg not in down]


def available_leg(leg: str, settings: Settings, down: frozenset[str] = frozenset()) -> str:
    """The leg that will run for a configured leg.

    The configured leg runs when it is built, has an API key, and is not marked down. Otherwise
    the first other leg that is ready runs. When no leg is ready, a leg with a key is tried anyway.
    """
    candidates = [leg, *(other for other in LEG_BUILDERS if other != leg)]
    with_key = [c for c in candidates if c in LEG_BUILDERS and has_credentials(c, settings)]
    healthy = [c for c in with_key if c not in down]
    choice = (healthy or with_key or [FALLBACK_LEG])[0]
    if choice != leg:
        why = "is marked down" if leg in down else ("needs an API key" if leg in LEG_BUILDERS else "is not built yet")
        log.warning("voice leg %s %s; using %s", leg, why, choice)
    return choice


def resolve_spec(spec: AgentSpec, settings: Settings, down: frozenset[str] = frozenset()) -> AgentSpec:
    """The spec as it will run on this server. On a fallback the voice changes too, because voice
    names belong to a provider."""
    leg = available_leg(spec.leg, settings, down)
    if leg == spec.leg:
        return spec
    return dataclasses.replace(spec, leg=leg, voice=default_voice(spec.name, leg))


def build_leg(spec: AgentSpec, *, instructions: str, settings: Settings, tools: ToolsSchema | None = None) -> LLMService:
    try:
        builder = LEG_BUILDERS[spec.leg]
    except KeyError as exc:
        raise ValueError(f"voice leg {spec.leg!r} is not available yet (have: {sorted(LEG_BUILDERS)})") from exc
    return builder(spec, instructions=instructions, settings=settings, tools=tools)


# --- driving a service: connect, reset, replay ---


def self_heals(service: LLMService) -> bool:
    """True when the Pipecat service reconnects by itself after an error. Gemini does (with session
    resumption). Nova does: on a stream error it opens a new stream and replays the context.
    OpenAI does not: its receive loop ends on any error event. Phonic does not: its service reports
    a close of the conversation, and the watchdog opens a new one and replays the call."""
    return isinstance(service, GeminiLiveLLMService | AWSNovaSonicLLMService)


@dataclasses.dataclass
class SessionNote:
    """A note that `tell_now` put into a service's session, so that `forget_note` can take it out."""

    service: LLMService
    websocket: object  # the connection that holds the note; a reset or a reconnect makes a new one
    item_id: str


async def tell_now(service: LLMService, text: str) -> SessionNote | None:
    """Put a note into the service's own session at once, with no answer asked for. None when not sent.

    The note does not go into the shared context: it is about one moment, and the next agent must
    not read it. OpenAI only. Gemini and Nova report an interruption after they acted on it.
    Phonic cannot take a note out again, and `forget_note` must: a system message stays, and its
    system prompt can be replaced only once per conversation. Measured 2026-11-24: the second
    `update_system_prompt` got the error "System prompt can only be updated once per conversation",
    and the agent kept following the first note.
    """
    if not isinstance(service, OpenAIRealtimeLLMService):
        return None
    if service._websocket is None or not service._api_session_ready or service._llm_needs_conversation_setup:  # noqa: SLF001
        return None
    item = openai_events.ConversationItem(type="message", role="system", content=[openai_events.ItemContent(type="input_text", text=text)])
    await service.send_client_event(openai_events.ConversationItemCreateEvent(item=item))
    return SessionNote(service, service._websocket, item.id)  # noqa: SLF001


async def forget_note(note: SessionNote) -> bool:
    """Take a note out of its session. True when the session still held it.

    A note about one caller sound must not steer a later answer: on 2026-11-17 an agent read an
    old "say this again" note and said the whole greeting again before a handoff. A session that
    was reset has a new connection and no note: a delete there is an error, and Pipecat restarts
    the session on any error.
    """
    service = note.service
    if not isinstance(service, OpenAIRealtimeLLMService) or service._websocket is not note.websocket:  # noqa: SLF001
        return False
    await service.send_client_event(openai_events.ConversationItemDeleteEvent(item_id=note.item_id))
    return True


def session_clock(service: LLMService):
    """For a provider that bills by the minute, a function that gives the seconds this service has
    held a session open. None for a provider that bills by the token (it reports its tokens)."""
    return (lambda: service.seconds_held) if isinstance(service, PhonicLLMService) else None


def first_answer_needs_run_frame(service: LLMService) -> bool:
    """OpenAI sets up its session from the first context frame, so its first answer must come from
    an LLMRunFrame through the pipeline. Gemini, Nova and Phonic can be seeded directly every time."""
    return isinstance(service, OpenAIRealtimeLLMService)


async def ensure_connected(service: LLMService) -> None:
    if isinstance(service, OpenAIRealtimeLLMService) and service._websocket is None:  # noqa: SLF001
        log.warning("%s was disconnected; reconnecting before the handoff", service.name)
        await service._connect()  # noqa: SLF001
    elif isinstance(service, GeminiLiveLLMService) and service._session is None and service._connection_task is None:  # noqa: SLF001
        log.warning("%s was disconnected; reconnecting before the handoff", service.name)
        await service._connect()  # noqa: SLF001
    elif isinstance(service, AWSNovaSonicLLMService) and service._client is None:  # noqa: SLF001
        await service._start_connecting()  # noqa: SLF001 - the normal case: Nova connects when it is needed
    elif isinstance(service, PhonicLLMService):
        await service.connect()  # the normal case too; Phonic is ready about 0.9 s later, and `replay` waits for that


async def adopt_context(service: LLMService, context: LLMContext) -> None:
    """Prepare a service to become the active one. Call this before the switch.

    Gemini: hand it the shared context now. A context frame can reach the service right after the
    switch; if that frame were its first context, Pipecat would reconnect the session (about 0.5 s)
    and send it tool results that belong to other agents' sessions.

    OpenAI, on a return to an agent that was used before: mark every tool result in the context as
    known. Pipecat sends a result it has not marked to the session, on the next context frame.
    The results written while another agent had the call belong to other sessions. OpenAI rejects
    such a result ("Tool call ID ... not found in conversation"), and the session then restarts.

    Phonic: drop an instruction it kept from the last time it was active. A context frame does
    not make it speak; only an instruction appended while it is active does.
    """
    if isinstance(service, GeminiLiveLLMService):
        service._context = context  # noqa: SLF001
        await service._process_completed_function_calls(send_new_results=False)  # noqa: SLF001
    elif isinstance(service, OpenAIRealtimeLLMService) and service._context is not None:  # noqa: SLF001
        await service._process_completed_function_calls(send_new_results=False)  # noqa: SLF001
    elif isinstance(service, PhonicLLMService):
        await service.adopt(context)


async def reset_session(service: LLMService) -> None:
    """Give the service a fresh server-side session with no conversation in it.

    OpenAI: about 2.4 s. Gemini: about 0.4 s. Nova and Phonic: a close; `ensure_connected` opens
    a new session later.
    """
    if isinstance(service, OpenAIRealtimeLLMService):
        await service.reset_conversation()
        # Pipecat forgets the last audio item only when this service hears that the agent stopped
        # speaking. After a handoff another service hears that. When the caller then talks over
        # this agent's first words, Pipecat cuts the old item, which the new session does not
        # have: "item_truncate_invalid_item_id", and the session restarts. Measured 2026-11-16.
        service._current_audio_response = None  # noqa: SLF001
    elif isinstance(service, GeminiLiveLLMService):
        await service._disconnect()  # noqa: SLF001
        service._session_resumption_handle = None  # noqa: SLF001 - a resumed session would bring the old conversation back
        service._context = None  # noqa: SLF001 - so the new session is not seeded until the agent is active again
        if isinstance(service, HandoffGeminiLive):
            service.quiet_seed = False  # the next seed is a replay, which must answer
        await service._connect()  # noqa: SLF001
    elif isinstance(service, AWSNovaSonicLLMService):
        await service._disconnect()  # noqa: SLF001 - also drops its copy of the context; `ensure_connected` opens a new stream later
    elif isinstance(service, PhonicLLMService):
        await service.reset_conversation()
    else:  # pragma: no cover
        raise NotImplementedError(f"reset for {type(service).__name__}")


async def replay_and_respond(service: LLMService, context: LLMContext) -> None:
    """Replay the shared context into the service's fresh session and have it answer.

    OpenAI: the service already holds the shared context from its first use; ask for a response.
    Nova: open the stream if needed, then hand it the context; it sets up the session and answers.
    Gemini: seed the session with the context, with the turn marked complete. If the session is
    still opening, the service does this as soon as it is ready.
    Phonic: open the conversation if needed and wait until it is ready, give it the call so far as
    text, then ask it to speak: the greeting word for word, any other instruction as a system message.
    """
    if isinstance(service, OpenAIRealtimeLLMService):
        await service._create_response()  # noqa: SLF001
    elif isinstance(service, GeminiLiveLLMService):
        await adopt_context(service, context)
        if isinstance(service, HandoffGeminiLive):
            service.quiet_seed = False  # this replay asks for an answer
        await service._create_initial_response()  # noqa: SLF001
    elif isinstance(service, AWSNovaSonicLLMService):
        # Nova sets up its session from the first context it sees: tools, instructions, the
        # conversation so far. The last message is our instruction to speak; Nova takes a system
        # or developer message as the caller's turn, so it answers. If a context frame from the
        # pipeline got here first, the session is set up already and there is nothing to do.
        await ensure_connected(service)
        if service._context is None:  # noqa: SLF001
            await service._handle_context(context)  # noqa: SLF001
    elif isinstance(service, PhonicLLMService):
        await service.replay(context)
    else:  # pragma: no cover
        raise NotImplementedError(f"replay for {type(service).__name__}")


def plays_recordings(service: LLMService) -> bool:
    """True when the service can play a stored recording as its own turn and be seeded without an
    answer. Gemini only: the stored greeting is a Gemini take (`record-greeting`)."""
    return isinstance(service, HandoffGeminiLive)


async def seed_without_answer(service: LLMService, context: LLMContext) -> None:
    """Give the service's session the call so far, and ask for no answer: it speaks after the caller does.

    For a played recording. Pipecat's Gemini seed is the conversation as client content; with the
    turn not complete, the model takes it in and waits for the caller's audio. That is the use the
    google-genai docs give for `turn_complete=False`: context before a realtime conversation. Pipecat
    seeds a reconnect this way too. If the session is still opening, the seed goes when it is ready.
    The service holds the shared context from here, so a later context frame from the pipeline is
    an update: it neither seeds the session again nor reconnects it.
    """
    if not isinstance(service, HandoffGeminiLive):
        raise NotImplementedError(f"seed without an answer for {type(service).__name__}")
    await adopt_context(service, context)
    service.quiet_seed = True
    await service._create_initial_response()  # noqa: SLF001
