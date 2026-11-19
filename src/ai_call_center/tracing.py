"""LangSmith tracing for calls and for the text mode.

Off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY is set. When it is on, every call is one LangSmith
thread, keyed by the Telnyx call id:

- The voice path sends Pipecat's OpenTelemetry spans through LangSmith's Pipecat integration: one trace per
  media stream, with a span per turn, the agent's words, each tool call, and the timing. A speech-to-speech
  service gives the caller's words through the user aggregator, not a span, so `follow_call` hooks it.
  A restart of the stream (failover, a transfer nobody answered) is a second trace in the same thread.
- The LangGraph text mode and the call summary trace through LangChain's own LangSmith client, with
  `run_config`, which puts the call id in the run's `thread_id` metadata.

Limits: Pipecat 1.16 has no trace spans for Amazon Nova Sonic, so a Nova agent shows turns and timing only.
No audio is sent. Spans are sent from a background thread, so tracing does not delay an answer, and a
LangSmith failure only writes an error to the log.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langsmith.integrations.pipecat import PipecatLangSmithSpanProcessor
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )

    from ai_call_center.config import Settings

log = logging.getLogger(__name__)

_started = False
_processor: PipecatLangSmithSpanProcessor | None = None


def tracing_enabled(settings: Settings) -> bool:
    """Is tracing set on, with a key to send it?"""
    return settings.langsmith_tracing and bool(settings.langsmith_api_key.get_secret_value())


def start(settings: Settings, *, voice: bool = True) -> PipecatLangSmithSpanProcessor | None:
    """Turn tracing on for this process, once. Returns the Pipecat span processor, or None.

    `voice=False` (the text mode) skips the Pipecat part; LangChain traces from the environment alone.
    """
    global _started, _processor
    if _started:
        return _processor
    _started = True
    if not tracing_enabled(settings):
        if settings.langsmith_tracing:
            log.warning("LANGSMITH_TRACING is on, but LANGSMITH_API_KEY is empty: no traces are sent")
        os.environ["LANGSMITH_TRACING"] = "false"  # LangChain reads the environment, not Settings
        return None
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key.get_secret_value())
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    if voice:
        from langsmith.integrations.pipecat import configure_pipecat

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*is in beta", category=UserWarning)  # LangSmithBetaWarning
            _processor = configure_pipecat(service_name="voice-business-card")
        if _processor is None:
            log.error("LangSmith tracing for calls is off: Pipecat tracing could not start (see the warning above)")
    log.info("LangSmith tracing is on: project %s", project)
    return _processor


def processor() -> PipecatLangSmithSpanProcessor | None:
    return _processor


def follow_call(pair: LLMContextAggregatorPair, call_id: str) -> bool:
    """Put this call's spans and the caller's words in the call's thread. Run it in the call's own task.

    Returns True when tracing is on for calls.
    """
    if _processor is None:
        return False
    from langsmith.integrations.pipecat import set_thread_id

    set_thread_id(call_id)  # a ContextVar: each call's task keeps its own id
    _processor.instrument_user_aggregator(pair, call_id)
    return True


def run_config(call_id: str, run_name: str) -> dict[str, Any]:
    """LangChain run config that names the run and puts it in the call's thread."""
    return {"run_name": run_name, "metadata": {"thread_id": call_id}}


def _reset() -> None:
    """For tests."""
    global _started, _processor
    _started = False
    _processor = None
