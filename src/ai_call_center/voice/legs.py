"""Voice legs: build the Pipecat speech-to-speech service for one agent spec.

Each agent gets its own service, pre-connected at call start, because OpenAI fixes a session's
voice once it has spoken. A reconnect-and-replay swap was measured at 3-4 s of silence; switching
between pre-connected services takes under 1 s."""

from __future__ import annotations

import logging

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
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from ai_call_center.agents.specs import AgentSpec
from ai_call_center.config import Settings

log = logging.getLogger(__name__)


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


def build_openai_realtime(spec: AgentSpec, *, instructions: str, settings: Settings) -> LLMService:
    """gpt-realtime-2.1-mini over the Realtime WebSocket. Audio stays PCM 24 kHz inside Pipecat;
    the Telnyx serializer converts to and from 8 kHz PCMU."""
    return OpenAIRealtimeLLMService(
        api_key=settings.openai_api_key.get_secret_value(),
        settings=OpenAIRealtimeLLMService.Settings(
            model=settings.openai_realtime_model,
            system_instruction=instructions,
            session_properties=SessionProperties(
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription(),
                        turn_detection=turn_detection(settings),
                        noise_reduction=noise_reduction(settings),
                    ),
                    output=AudioOutput(voice=spec.voice or settings.openai_realtime_voice),
                ),
            ),
        ),
    )


LEG_BUILDERS = {
    "openai_realtime": build_openai_realtime,
}

def available_leg(leg: str) -> str:
    """The leg to use for a configured leg. Legs without a builder fall back to OpenAI (until M3)."""
    if leg in LEG_BUILDERS:
        return leg
    log.warning("voice leg %r is not built yet; using openai_realtime", leg)
    return "openai_realtime"


def build_leg(leg: str, spec: AgentSpec, *, instructions: str, settings: Settings) -> LLMService:
    try:
        builder = LEG_BUILDERS[leg]
    except KeyError as exc:
        raise ValueError(f"voice leg {leg!r} is not available yet (have: {sorted(LEG_BUILDERS)})") from exc
    return builder(spec, instructions=instructions, settings=settings)

