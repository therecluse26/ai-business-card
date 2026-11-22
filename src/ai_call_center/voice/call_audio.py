"""The call audio: both sides of a call in one channel, held in memory until the grader has heard it.

The grader that hears a call (`decision_grader.py`) needs the sound, not only the transcript: a
transcript does not show an agent that talks over the caller, or a word that is cut off.

Rules for this audio:

- It is in memory only. It is never written to a file.
- It leaves the server one time: after hangup, in the request to the decision model.
- After the grade it is deleted. After a transfer it stays for a short time, because a call comes
  back when nobody answers the transfer.
- A caller who asks to be forgotten loses it at once, and the rest of that call is not kept.

A call can have more than one media stream: a change of voice provider starts a new one. Each stream
adds its part, so the grader hears the whole call.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

SAMPLE_BYTES = 2  # 16-bit samples, one channel
MAX_SECS = 15 * 60  # more audio than this is not kept; MAX_CALL_MINUTES ends a call before it
KEEP_AFTER_TRANSFER_SECS = 5 * 60  # a transfer that nobody answers comes back well before this


@dataclass(frozen=True)
class CallAudio:
    """The caller and the agents, mixed into one channel, as the phone line carried them."""

    pcm: bytes
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.pcm) / (SAMPLE_BYTES * self.sample_rate) if self.sample_rate else 0.0

    def wav(self) -> bytes:
        """The audio as a WAV file, for a request to a model."""
        out = io.BytesIO()
        with wave.open(out, "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(SAMPLE_BYTES)
            file.setframerate(self.sample_rate)
            file.writeframes(self.pcm)
        return out.getvalue()


def read_wav(data: bytes) -> CallAudio:
    """A WAV file as call audio. Only the first channel of a file with more channels is kept."""
    with wave.open(io.BytesIO(data), "rb") as file:
        if file.getsampwidth() != SAMPLE_BYTES:
            raise ValueError("the WAV file must have 16-bit samples")
        channels, pcm = file.getnchannels(), file.readframes(file.getnframes())
        if channels > 1:
            step = SAMPLE_BYTES * channels
            pcm = b"".join(pcm[at : at + SAMPLE_BYTES] for at in range(0, len(pcm), step))
        return CallAudio(pcm, file.getframerate())


class CallAudioStore:
    """The audio of each call that is in progress, by call id."""

    def __init__(self) -> None:
        self._parts: dict[str, list[bytes]] = {}
        self._rates: dict[str, int] = {}
        self._drops: dict[str, asyncio.TimerHandle] = {}

    def add(self, call_id: str, pcm: bytes, sample_rate: int) -> None:
        """One more part of a call. A part at another sample rate, or past the limit, is not kept."""
        self.keep(call_id)
        rate = self._rates.setdefault(call_id, sample_rate)
        held = sum(len(part) for part in self._parts.get(call_id, []))
        room = MAX_SECS * SAMPLE_BYTES * rate - held
        if not pcm or sample_rate != rate or room <= 0:
            return
        self._parts.setdefault(call_id, []).append(pcm[:room])

    def get(self, call_id: str) -> CallAudio | None:
        parts = self._parts.get(call_id)
        return CallAudio(b"".join(parts), self._rates[call_id]) if parts else None

    def drop(self, call_id: str) -> None:
        self.keep(call_id)
        self._parts.pop(call_id, None)
        self._rates.pop(call_id, None)

    def drop_later(self, call_id: str, seconds: float = KEEP_AFTER_TRANSFER_SECS) -> None:
        """Delete the audio after `seconds`, unless the call adds a part before that."""
        self.keep(call_id)
        if call_id in self._parts:
            self._drops[call_id] = asyncio.get_running_loop().call_later(seconds, self.drop, call_id)

    def keep(self, call_id: str) -> None:
        """Stop a delete that `drop_later` set: the call goes on in a new media stream."""
        timer = self._drops.pop(call_id, None)
        if timer is not None:
            timer.cancel()


call_audio = CallAudioStore()  # one for the process: a call's next media stream finds the earlier parts


def recorder(call_id: str, *, sample_rate: int, forgotten: Callable[[], bool], store: CallAudioStore = call_audio) -> Any:
    """A processor that records both sides of one media stream. It sits after the transport output.

    Pipecat's own recorder keeps the two sides in time with each other. When the stream ends, or when
    `stop_recording` is called, the part goes to the store. For a caller who asked to be forgotten,
    the part is not kept, and the earlier parts of the call are deleted."""
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

    store.keep(call_id)  # the earlier parts of this call stay until this stream adds its part
    buffer = AudioBufferProcessor(sample_rate=sample_rate, num_channels=1, auto_start_recording=True)

    @buffer.event_handler("on_audio_data")
    async def keep(_buffer: Any, audio: bytes, rate: int, _channels: int) -> None:
        if forgotten():
            store.drop(call_id)
        else:
            store.add(call_id, audio, rate)

    return buffer
