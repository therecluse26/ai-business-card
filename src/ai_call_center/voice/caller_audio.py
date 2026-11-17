"""The caller's side of a call, saved to a WAV file. Off unless SAVE_CALLER_AUDIO=true.

A voice provider decides when the caller speaks, and it takes a cough for a word. To test that,
the real sounds of a real phone line are needed: a Telnyx recording mixes both sides into one
channel. The file holds what the server received from the caller, before any processing.
"""

import logging
import re
import wave
from datetime import datetime
from pathlib import Path

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

log = logging.getLogger(__name__)


def caller_audio_path(folder: Path, call_id: str, now: datetime | None = None) -> Path:
    """One file per media stream: the start time, then the end of the call id."""
    tail = re.sub(r"[^A-Za-z0-9]", "", call_id)[-10:] or "call"
    return folder / f"{(now or datetime.now()):%Y%m%d-%H%M%S}-{tail}.wav"


class CallerAudioFile(FrameProcessor):
    """Sits right after the transport input and writes each caller audio frame to `path`.

    A write that fails is logged once and the file is given up: a call never depends on it.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._wav: wave.Wave_write | None = None
        self._failed = False

    def write(self, frame: InputAudioRawFrame) -> None:
        if self._failed:
            return
        try:
            if self._wav is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._wav = wave.open(str(self.path), "wb")
                self._wav.setnchannels(frame.num_channels or 1)
                self._wav.setsampwidth(2)
                self._wav.setframerate(frame.sample_rate)
                log.info("caller audio is saved to %s", self.path)
            self._wav.writeframes(frame.audio)
        except Exception:
            self._failed = True
            log.exception("could not save the caller audio to %s", self.path)

    def close(self) -> None:
        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                log.exception("could not close %s", self.path)
            self._wav = None
            self._failed = True  # nothing is written after the file is closed

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self.write(frame)
        elif isinstance(frame, EndFrame | CancelFrame):
            self.close()
        await self.push_frame(frame, direction)
