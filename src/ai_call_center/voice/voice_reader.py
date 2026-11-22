"""Pipecat Voice Reader on the caller's audio: the part of Pipecat's expressive conversation that fits this server.

Pipecat's expressive conversation (alpha in Pipecat 1.16) has two halves:

- The LLM writes expression tags, and a TTS service performs them. Each agent here is one
  speech-to-speech model. It has no TTS service, and it already chooses its own delivery. So this
  half does not apply.
- A reader hears how the user sounded. Pipecat's `PerceptionReader` puts the reading at the head of
  the user's message, and a text LLM reads it before it answers. A speech-to-speech model gets the
  caller's audio directly, and no such message. So this module runs the same model, Pipecat Voice
  Reader, in a processor of its own.

What the processor does: it keeps the speech of the caller's turn, and each time the caller pauses
it reads the turn, at most its last 8 seconds. The read runs in a thread, about 60 ms on a laptop.
It never holds an audio frame back, so it adds no time to an answer.

What a reading does (`CallSession`): it goes to the log, to the call stats that the tour agent can
say, and to the owner's lead email. When the caller sounds frustrated, the agent gets one note.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import BotStartedSpeakingFrame, Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ai_call_center.caller_voice import TONES, VoiceReading

log = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = 16000  # the model hears 16 kHz; a phone call is 8 kHz, so each clip is resampled
READ_SECS = 8.0  # the model reads at most the last 8 seconds of a turn
MIN_READ_SECS = 0.8  # under this the readings mean little (Pipecat's own limit)
LEAD_SECS = 0.3  # audio kept from before the detector confirmed speech, so the first syllable is not cut
PAUSE_SECS = 0.2  # silence that ends a stretch of speech and starts a read
NOTE_TURNS = 2  # a note about frustration stays for this many caller turns

FRUSTRATED_NOTE = (
    "This is a note from the phone system, not from the caller. "
    "A model that listens to the sound of the caller's voice heard frustration in their last turn. "
    "In your next answer, first show in one short sentence that you understood what they want. Then help them directly. "
    "Ask no question that you do not need. "
    "Do not say that their voice was measured, and do not tell them how they sound."
)


class VoiceModel:
    """Pipecat Voice Reader, loaded once for the process. `read` and `load` block: call them in a thread."""

    def __init__(self, cache_dir: Path, *, variant: Literal["base", "tiny"] = "base", threads: int = 1) -> None:
        self.cache_dir = cache_dir
        self.variant = variant
        self.threads = threads  # one: a call's audio runs on the same CPU
        self.name = ""
        self._model: Any = None
        self._resampler: Any = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model. The first load downloads about 28 MB from Hugging Face into `cache_dir`. A later load uses the network for nothing."""
        from huggingface_hub import hf_hub_download, snapshot_download
        from pipecat.classifiers.voice.model import VOICE_REPO, VOICE_REVISION, PipecatVoiceReaderModel

        def folder(local_only: bool) -> Path:
            found = dict(revision=VOICE_REVISION, cache_dir=str(self.cache_dir), local_files_only=local_only)
            index = json.loads(Path(hf_hub_download(VOICE_REPO, "config.json", **found)).read_text(encoding="utf-8"))
            name = index["variants"][self.variant]["folder"]
            root = snapshot_download(VOICE_REPO, allow_patterns=[f"{name}/config.json", f"{name}/model.int8.onnx"], **found)
            if not (Path(root) / name / "model.int8.onnx").exists():
                raise FileNotFoundError(f"{name}/model.int8.onnx is not in {self.cache_dir}")
            return Path(root) / name

        try:
            path = folder(local_only=True)  # the files are here already: no request to Hugging Face at a server start
        except Exception:
            path = folder(local_only=False)
        model = PipecatVoiceReaderModel(path, precision="int8", threads=self.threads)
        if model.tones != list(TONES):
            raise ValueError(f"the model's tones are {model.tones}, and this code knows {list(TONES)}")
        self.name = f"{VOICE_REPO}@{VOICE_REVISION}/{self.variant}"
        self._model = model

    def read(self, pcm: bytes, sample_rate: int) -> dict[str, Any]:
        """The model's values for one clip of 16-bit mono PCM: `tone` as a probability for each tone, the rest as numbers."""
        if sample_rate != MODEL_SAMPLE_RATE:
            from pipecat_resampler import Resampler

            if self._resampler is None or self._resampler.in_rate != sample_rate:
                self._resampler = Resampler(sample_rate, MODEL_SAMPLE_RATE)
            pcm = self._resampler.resample(pcm)
        return self._model.read(pcm)


def _unit(value: Any) -> float:
    return min(max(float(value), 0.0), 1.0)


def reading_from(values: dict[str, Any], *, turn: int, seconds: float, millis: int) -> VoiceReading:
    """A voice reading from the model's values."""
    tones: dict[str, float] = values["tone"]
    tone = max(tones, key=lambda name: tones[name])
    return VoiceReading(
        turn=turn,
        tone=tone,
        confidence=_unit(tones[tone]),
        arousal=_unit(values.get("arousal", 0.0)),
        laughter=_unit(values.get("laughter", 0.0)),
        speech_rate=min(max(float(values.get("speech_rate", 0.0)), 0.0), 12.0),
        sarcasm=_unit(values.get("sarcasm", 0.0)),
        upset=_unit(values.get("upset", 0.0)),
        seconds=seconds,
        millis=millis,
    )


class TurnAudio:
    """The speech of the caller's current turn, without the silence between its stretches.

    A stretch is speech from its start to the next pause. A turn is every stretch from the agent's
    last words to its next words. The model was trained on speech, so the silence is left out.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.turn = 0  # the turn the kept speech belongs to; 0 before the caller spoke
        self._speech = bytearray()
        self._lead = bytearray()
        self._stretch: bytearray | None = None
        self._confirmed = False  # the detector confirmed the stretch as speech
        self._answered = True  # the agent spoke after the kept speech: the next stretch starts a new turn

    def _bytes(self, secs: float) -> int:
        return int(secs * self.sample_rate) * 2

    def feed(self, audio: bytes, state: VADState) -> bool:
        """Add one frame of 16-bit mono PCM. True when a stretch of speech just ended: the turn can be read."""
        if state != VADState.QUIET:
            if self._stretch is None:
                self._stretch = bytearray(self._lead)
            self._stretch += audio
            self._confirmed = self._confirmed or state in (VADState.SPEAKING, VADState.STOPPING)
            return False
        ended = False
        if self._stretch is not None and self._confirmed:
            if self._answered:
                self.turn += 1
                self._speech.clear()
                self._answered = False
            self._speech += self._stretch
            del self._speech[: max(len(self._speech) - self._bytes(READ_SECS), 0)]
            ended = True
        self._stretch, self._confirmed = None, False
        self._lead += audio
        del self._lead[: max(len(self._lead) - self._bytes(LEAD_SECS), 0)]
        return ended

    def agent_spoke(self) -> None:
        self._answered = True

    @property
    def seconds(self) -> float:
        return len(self._speech) / 2 / self.sample_rate

    def clip(self) -> bytes:
        return bytes(self._speech)


def silero(sample_rate: int) -> Any:
    """Silero, the local speech detector. It only finds the stretches to read: it sends no frame, so it
    changes no turn. The voice services still decide when the caller's turn starts and ends."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    detector = SileroVADAnalyzer(params=VADParams(stop_secs=PAUSE_SECS))
    detector.set_sample_rate(sample_rate)  # this also sets the frame size; without it the detector cannot run
    return detector


class CallerVoiceReader(FrameProcessor):
    """Sits after the transport's input. Calls `on_reading` with a voice reading each time the caller pauses.

    Each frame goes on first, and the work comes after it. One read runs at a time: a pause that
    comes during a read starts one more read after it, of the turn as it is then.
    """

    def __init__(
        self,
        model: VoiceModel,
        on_reading: Callable[[VoiceReading], Awaitable[None]],
        *,
        active: Callable[[], bool] = lambda: True,
        detector: Callable[[int], Any] = silero,
    ) -> None:
        super().__init__()
        self._model = model
        self._on_reading = on_reading
        self._active = active  # False = hear nothing, for example after the caller asked to be forgotten
        self._make_detector = detector
        self._detector: Any = None
        self._audio: TurnAudio | None = None
        self._wanted = False
        self._reads: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        await self.observe(frame)

    async def observe(self, frame: Frame) -> None:
        if isinstance(frame, BotStartedSpeakingFrame):
            if self._audio is not None:
                self._audio.agent_spoke()
        elif isinstance(frame, InputAudioRawFrame) and frame.audio and self._model.ready and self._active():
            try:
                await self._hear(frame)
            except Exception:
                log.exception("the voice reader could not hear a frame of the caller's audio")

    async def _hear(self, frame: InputAudioRawFrame) -> None:
        if self._audio is None or self._audio.sample_rate != frame.sample_rate:
            self._audio = TurnAudio(frame.sample_rate)
            self._detector = self._make_detector(frame.sample_rate)
        if self._audio.feed(frame.audio, await self._detector.analyze_audio(frame.audio)):
            self._wanted = True
            if self._reads is None or self._reads.done():
                self._reads = asyncio.create_task(self._read())

    async def _read(self) -> None:
        while self._wanted and self._audio is not None:
            self._wanted = False
            audio = self._audio
            clip, turn, seconds = audio.clip(), audio.turn, audio.seconds
            if seconds < MIN_READ_SECS:
                continue
            started = time.monotonic()
            try:
                values = await asyncio.to_thread(self._model.read, clip, audio.sample_rate)
                reading = reading_from(values, turn=turn, seconds=seconds, millis=int((time.monotonic() - started) * 1000))
                if self._active():
                    await self._on_reading(reading)
            except Exception:
                log.exception("the voice reader could not read the caller's turn %d", turn)

    async def cleanup(self) -> None:
        await super().cleanup()
        if self._reads is not None:
            self._reads.cancel()
        if self._detector is not None and hasattr(self._detector, "cleanup"):
            await self._detector.cleanup()


class FrustrationNote:
    """When the caller sounds frustrated, the agent that holds the call gets one note in its session.

    The note stays for `NOTE_TURNS` caller turns, and a new frustrated turn starts the count again.
    Then the server takes it out, so it cannot steer a later answer. The note is in one agent's
    session, not in the shared context: after a handoff the next agent does not read it.

    `send(text)` gives the note that the session holds, or None when it sent nothing (only OpenAI
    takes a note). `forget(note)` takes it out. `current(note)` is True while the agent that got the
    note holds the call.
    """

    def __init__(self, send: Callable[[str], Awaitable[Any]], forget: Callable[[Any], Awaitable[Any]], current: Callable[[Any], bool] = lambda note: True) -> None:
        self._send = send
        self._forget = forget
        self._current = current
        self._note: Any = None
        self._until_turn = 0

    @property
    def held(self) -> bool:
        return self._note is not None

    async def drop(self) -> None:
        note, self._note = self._note, None
        if note is not None:
            await self._forget(note)

    async def on_reading(self, reading: VoiceReading) -> bool:
        """True when this reading sent a note."""
        if self._note is not None and not self._current(self._note):
            await self.drop()  # another agent holds the call now
        if reading.frustrated:
            self._until_turn = reading.turn + NOTE_TURNS
            if self._note is None:
                self._note = await self._send(FRUSTRATED_NOTE)
                return self._note is not None
        elif self._note is not None and reading.turn >= self._until_turn:
            await self.drop()
        return False


_default: VoiceModel | None = None


def default_voice_model(settings: Any) -> VoiceModel | None:
    """The process-wide voice model, or None when VOICE_READER is off. It is not loaded yet: see `load_voice_model`."""
    global _default
    if not settings.voice_reader:
        return None
    if _default is None:
        _default = VoiceModel(settings.voice_model_dir, variant=settings.voice_reader_variant)
    return _default


async def load_voice_model(settings: Any) -> None:
    """Load the voice model in a thread, at server start. A model that cannot load leaves the reader off: calls go on."""
    model = default_voice_model(settings)
    if model is None or model.ready:
        return
    started = time.monotonic()
    try:
        await asyncio.to_thread(model.load)
    except Exception as error:
        log.error("the voice reader is off: the model could not load (%s: %s)", type(error).__name__, error)
        return
    log.info("the voice reader is on: %s, loaded in %.1f s", model.name, time.monotonic() - started)
