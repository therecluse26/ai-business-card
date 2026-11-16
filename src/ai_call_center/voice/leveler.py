"""One loudness for every agent voice.

Voices differ in level, most of all across providers: measured on the phone side, Gemini's Charon
was 7 dB louder than OpenAI's marin and 10 dB louder than coral. A handoff then sounds like someone
turned the volume up. The leveler sits after the voice services, measures the speech level of each
voice, and scales the audio toward one target level.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pipecat.frames.frames import Frame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

SPEECH_GATE = 10 ** (-45 / 20)  # quieter audio is a pause, not speech; it does not move the measured level
TIME_CONSTANT_SECS = 4.0  # how fast the measured level follows the voice
MIN_SPEECH_SECS = 0.6  # an unmeasured voice plays as it is until this much speech is measured
MIN_GAIN, MAX_GAIN = 10 ** (-14 / 20), 10 ** (8 / 20)


def dbfs_to_amplitude(dbfs: float) -> float:
    return 10 ** (dbfs / 20)


@dataclass
class VoiceLevel:
    """The running speech level of one voice. `seed_dbfs` is a level measured earlier for this voice."""

    target: float  # amplitude (RMS, full scale = 1.0)
    seed_dbfs: float | None = None
    mean_square: float | None = None
    speech_secs: float = 0.0

    def __post_init__(self) -> None:
        if self.seed_dbfs is not None:
            self.mean_square = dbfs_to_amplitude(self.seed_dbfs) ** 2

    def gain_for(self, samples: np.ndarray, secs: float) -> float:
        """Update the level from this audio and return the gain to apply to it."""
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        if rms > SPEECH_GATE:
            mean_square = rms * rms
            if self.mean_square is None:
                self.mean_square = mean_square
            else:
                self.mean_square += min(1.0, secs / TIME_CONSTANT_SECS) * (mean_square - self.mean_square)
            self.speech_secs += secs
        if self.mean_square is None or (self.seed_dbfs is None and self.speech_secs < MIN_SPEECH_SECS):
            return 1.0
        return min(MAX_GAIN, max(MIN_GAIN, self.target / math.sqrt(self.mean_square)))


def scale_pcm16(audio: bytes, gain: float) -> bytes:
    if abs(gain - 1.0) < 0.01:
        return audio
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) * gain
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


class VoiceLeveler(FrameProcessor):
    """Scales the output audio of the active voice toward the target level."""

    def __init__(self, active_voice: Callable[[], str], *, target_dbfs: float, seeds: dict[str, float] | None = None) -> None:
        super().__init__()
        self._active_voice = active_voice
        self._target = dbfs_to_amplitude(target_dbfs)
        self._seeds = seeds or {}
        self._levels: dict[str, VoiceLevel] = {}

    def gain(self, voice: str, audio: bytes, sample_rate: int) -> float:
        level = self._levels.get(voice)
        if level is None:
            level = self._levels[voice] = VoiceLevel(self._target, self._seeds.get(voice))
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768
        return level.gain_for(samples, len(samples) / max(sample_rate, 1))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM and frame.audio:
            frame.audio = scale_pcm16(frame.audio, self.gain(self._active_voice(), frame.audio, frame.sample_rate))
        await self.push_frame(frame, direction)
