"""Live figures for one call: how fast the agents answer, how many tokens they use, and what that costs.

The tour agent reads these to the caller through `get_call_stats`. The server logs them when the
call ends.

Answer time is measured here, on the phone side of the pipeline, the same way for every provider:
from the last moment the caller's audio held speech to the moment the agent's audio starts. A
provider's own figure would not do: Gemini Live reports no "caller stopped" event.

The cost is an estimate from list prices. It covers the voice models only: not Telnyx minutes,
not the number lookup, and not the text model that writes the call summary.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from pipecat.frames.frames import BotStartedSpeakingFrame, InputAudioRawFrame, MetricsFrame
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed

from ai_call_center.agents.specs import LEG_PLATFORMS


@dataclass(frozen=True)
class Price:
    """US dollars for 1 million tokens."""

    text_in: float
    text_in_cached: float
    audio_in: float
    audio_in_cached: float
    text_out: float
    audio_out: float


# List prices, read on 2026-11-16. Check them before you rely on the estimate.
PRICES: dict[str, Price] = {
    # gpt-realtime-2.1-mini, from OpenAI's model page.
    "openai_realtime": Price(text_in=0.60, text_in_cached=0.06, audio_in=10.00, audio_in_cached=0.30, text_out=2.40, audio_out=20.00),
    # gemini-3.8-live. The audio prices agree across sources. The text prices do not (0.375 to 0.75
    # in, 3.75 to 4.50 out), so the higher ones are used. No cached price was found, so cached
    # tokens count at the full price. The estimate is therefore on the high side.
    "gemini_live": Price(text_in=0.75, text_in_cached=0.75, audio_in=3.00, audio_in_cached=3.00, text_out=4.50, audio_out=12.00),
}


@dataclass
class Usage:
    """Tokens one provider has reported on this call."""

    text_in: int = 0  # not from the cache
    text_in_cached: int = 0
    audio_in: int = 0  # not from the cache
    audio_in_cached: int = 0
    text_out: int = 0
    audio_out: int = 0

    def add(self, usage: LLMTokenUsage) -> None:
        # Both providers report the input count with the cached tokens and the audio tokens inside it.
        audio_in = usage.input_audio_tokens or 0
        audio_cached = min(usage.cache_read_input_audio_tokens or 0, audio_in)
        text_in = max(usage.prompt_tokens - audio_in, 0)
        text_cached = min(max((usage.cache_read_input_tokens or 0) - audio_cached, 0), text_in)
        audio_out = min(usage.output_audio_tokens or 0, usage.completion_tokens)
        self.text_in += text_in - text_cached
        self.text_in_cached += text_cached
        self.audio_in += audio_in - audio_cached
        self.audio_in_cached += audio_cached
        self.text_out += usage.completion_tokens - audio_out
        self.audio_out += audio_out

    @property
    def tokens_in(self) -> int:
        return self.text_in + self.text_in_cached + self.audio_in + self.audio_in_cached

    @property
    def tokens_out(self) -> int:
        return self.text_out + self.audio_out

    def cost(self, price: Price) -> float:
        """US dollars."""
        return (
            self.text_in * price.text_in
            + self.text_in_cached * price.text_in_cached
            + self.audio_in * price.audio_in
            + self.audio_in_cached * price.audio_in_cached
            + self.text_out * price.text_out
            + self.audio_out * price.audio_out
        ) / 1_000_000


SPEECH_GATE = 10 ** (-45 / 20)  # quieter caller audio is a pause or line noise
MIN_SPEECH_FRAMES = 5  # 100 ms of speech in a row; a click is not the caller talking
MAX_ANSWER_SECS = 15.0  # a longer gap is not an answer to what the caller said


@dataclass
class AnswerTimer:
    """Time from the end of the caller's speech to the start of the agent's audio."""

    spoke: bool = False  # the caller has talked since the agent last started to speak
    last_speech: float = 0.0
    _run: int = 0

    def caller_audio(self, rms: float, now: float) -> None:
        """Feed each 20 ms frame of the caller's audio. `rms` is 0.0 to 1.0."""
        if rms < SPEECH_GATE:
            self._run = 0
            return
        self._run += 1
        if self._run >= MIN_SPEECH_FRAMES:
            self.spoke = True
            self.last_speech = now

    def agent_started(self, now: float) -> float | None:
        """The agent's audio started. Returns the answer time in seconds, or None when this is not an answer."""
        if not self.spoke:
            return None
        self.spoke = False
        self._run = 0
        gap = now - self.last_speech
        return gap if 0.0 <= gap <= MAX_ANSWER_SECS else None


def _seconds(value: float) -> str:
    return f"{value:.1f}"


@dataclass(eq=False)
class CallStats(BaseObserver):
    """Collects the figures for one call. One object lives through the whole call, also when the
    media stream restarts; call `bind` with the services of each new pipeline."""

    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    answers: list[tuple[str, float]] = field(default_factory=list)  # (leg, seconds)
    handoffs: list[float] = field(default_factory=list)  # seconds from the tool call to the switch
    usage: dict[str, Usage] = field(default_factory=dict)  # leg -> tokens
    failovers: int = 0
    timer: AnswerTimer = field(default_factory=AnswerTimer)
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        super().__init__()
        self._leg_of: dict[str, str] = {}  # Pipecat processor name -> leg
        self._active_leg: Callable[[], str] = lambda: ""

    def bind(self, leg_of_service: dict[str, str], active_leg: Callable[[], str]) -> None:
        self._leg_of = dict(leg_of_service)
        self._active_leg = active_leg

    # --- what the pipeline reports ---

    async def on_push_frame(self, data: FramePushed) -> None:
        if not data.first_push:
            return  # every processor on the way pushes the frame again
        frame = data.frame
        if isinstance(frame, InputAudioRawFrame):
            samples = np.frombuffer(frame.audio, dtype=np.int16)
            if len(samples):
                self.timer.caller_audio(float(np.sqrt(np.mean((samples.astype(np.float64) / 32768) ** 2))), self.clock())
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.agent_started()
        elif isinstance(frame, MetricsFrame):
            for item in frame.data:
                if isinstance(item, LLMUsageMetricsData):
                    self.add_usage(self._leg_of.get(item.processor, ""), item.value)

    def agent_started(self) -> None:
        gap = self.timer.agent_started(self.clock())
        if gap is not None:
            self.answers.append((self._active_leg(), gap))

    def add_usage(self, leg: str, usage: LLMTokenUsage) -> None:
        self.usage.setdefault(leg, Usage()).add(usage)

    def note_handoff(self, seconds: float) -> None:
        self.handoffs.append(seconds)
        # The next agent's first words are part of the handoff, not an answer.
        self.timer.spoke = False

    # --- figures ---

    def cost(self) -> tuple[float, list[str]]:
        """(US dollars at list prices, the legs that have no price)."""
        total, unknown = 0.0, []
        for leg, usage in self.usage.items():
            if leg in PRICES:
                total += usage.cost(PRICES[leg])
            elif usage.tokens_in or usage.tokens_out:
                unknown.append(leg)
        return total, unknown

    def describe(self) -> str:
        """The figures as sentences an agent can say."""
        parts: list[str] = []
        if self.answers:
            times = [secs for _, secs in self.answers]
            parts.append(
                f"Speed: over {len(times)} answer{'s' if len(times) != 1 else ''}, an agent started to speak "
                f"{_seconds(statistics.median(times))} seconds after the caller stopped talking (the median); the slowest answer took {_seconds(max(times))} seconds."
            )
            by_leg: dict[str, list[float]] = {}
            for leg, secs in self.answers:
                by_leg.setdefault(leg, []).append(secs)
            if len(by_leg) > 1:
                parts.append(
                    "By provider: "
                    + "; ".join(f"{LEG_PLATFORMS.get(leg, leg or 'unknown')} {_seconds(statistics.median(v))} seconds over {len(v)}" for leg, v in by_leg.items())
                    + "."
                )
        else:
            parts.append("Speed: no answer has been timed yet.")
        if self.handoffs:
            parts.append(
                f"Handoffs: {len(self.handoffs)}; the switch to the next agent took {_seconds(statistics.mean(self.handoffs))} seconds on average."
            )
        if self.failovers:
            parts.append(f"A voice provider stopped answering {self.failovers} time{'s' if self.failovers != 1 else ''}, and the call moved to another provider.")
        tokens_in = sum(u.tokens_in for u in self.usage.values())
        tokens_out = sum(u.tokens_out for u in self.usage.values())
        if tokens_in or tokens_out:
            dollars, unknown = self.cost()
            cents = dollars * 100
            amount = "under one cent" if cents < 1 else f"about {cents:.0f} cent{'s' if round(cents) != 1 else ''}"
            parts.append(
                f"Tokens so far: {tokens_in:,} in and {tokens_out:,} out. Estimated cost of the voice models so far: {amount} in US money, at list prices."
                + (f" No price is known for {', '.join(unknown)}." if unknown else "")
            )
        else:
            parts.append("Tokens: no usage has been reported yet.")
        return " ".join(parts)
