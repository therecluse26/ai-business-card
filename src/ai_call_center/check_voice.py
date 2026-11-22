"""`uv run check-voice [file.wav ...]`: read saved caller audio with the voice reader.

With no file it reads every WAV in CALLER_AUDIO_DIR (the server writes them when SAVE_CALLER_AUDIO
is true). It uses the same parts as a call: the speech detector, the turn audio, and the model. A
saved file has only the caller's side, so every pause of 2 s or more counts as the agent's answer.
It prints one line for each caller turn, and the time of each read. Nothing leaves this computer.
"""

from __future__ import annotations

import asyncio
import sys
import time
import wave
from pathlib import Path

from ai_call_center.caller_voice import VoiceReading, keep_reading, voice_sentence
from ai_call_center.config import get_settings
from ai_call_center.voice.voice_reader import MIN_READ_SECS, TurnAudio, VoiceModel, reading_from, silero

FRAME_SECS = 0.02  # the size of a frame on a phone call
ANSWER_GAP_SECS = 2.0  # a pause this long stands for the agent's answer, which the file does not hold


async def read_file(model: VoiceModel, path: Path) -> list[VoiceReading]:
    with wave.open(str(path)) as wav:
        rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
        pcm = wav.readframes(wav.getnframes())
    if width != 2 or channels != 1 or rate not in (8000, 16000):
        print(f"{path.name}: skipped. The file must be 16-bit mono at 8 or 16 kHz.")
        return []
    detector, audio = silero(rate), TurnAudio(rate)
    step = int(rate * FRAME_SECS) * 2
    readings: list[VoiceReading] = []
    quiet = 0.0
    print(f"\n{path.name}: {len(pcm) / 2 / rate:.0f} s at {rate} Hz")
    for offset in range(0, len(pcm) - step + 1, step):
        frame = pcm[offset : offset + step]
        ended = audio.feed(frame, await detector.analyze_audio(frame))
        quiet = 0.0 if ended else quiet + FRAME_SECS
        if quiet >= ANSWER_GAP_SECS:
            audio.agent_spoke()
        if not ended or audio.seconds < MIN_READ_SECS:
            continue
        started = time.monotonic()
        values = model.read(audio.clip(), rate)
        keep_reading(readings, reading_from(values, turn=audio.turn, seconds=audio.seconds, millis=int((time.monotonic() - started) * 1000)))
    await detector.cleanup()
    for r in readings:
        flags = " ".join(flag for flag, on in (("FRUSTRATED", r.frustrated), ("LAUGHTER", r.laughed)) if on)
        print(
            f"  turn {r.turn:>2}  {r.seconds:>4.1f} s  {r.tone:<10} {r.confidence:.2f}  arousal {r.arousal:.2f}  laughter {r.laughter:.2f}  "
            f"{r.speech_rate:>4.1f} syl/s  sarcasm {r.sarcasm:.2f}  upset {r.upset:.2f}  {r.millis:>3} ms  {flags}"
        )
    return readings


async def run(paths: list[Path]) -> int:
    settings = get_settings()
    if not paths:
        paths = sorted(settings.caller_audio_dir.glob("*.wav"))
    if not paths:
        print(f"No WAV file in {settings.caller_audio_dir}. Set SAVE_CALLER_AUDIO=true, make a call, and run this again. Or name a file.")
        return 1
    model = VoiceModel(settings.voice_model_dir, variant=settings.voice_reader_variant)
    started = time.monotonic()
    model.load()
    print(f"model {model.name}, loaded in {time.monotonic() - started:.1f} s")
    every: list[VoiceReading] = []
    for path in paths:
        every += await read_file(model, path)
    if not every:
        print("\nThe model read no turn. A turn needs 0.8 s of speech or more.")
        return 1
    millis = sorted(r.millis for r in every)
    print(f"\n{len(every)} turns read. Read time: median {millis[len(millis) // 2]} ms, longest {millis[-1]} ms.")
    print(f"Frustrated turns (the agent gets a note): {sum(r.frustrated for r in every)}. Turns with laughter: {sum(r.laughed for r in every)}.")
    print(f"\nWhat the tour agent can say about a call with these turns:\n  {voice_sentence(every)}")
    return 0


def main() -> None:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    sys.exit(asyncio.run(run([Path(arg) for arg in sys.argv[1:]])))
