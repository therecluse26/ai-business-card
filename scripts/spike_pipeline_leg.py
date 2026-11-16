"""M7 spike: is a local pipeline leg fast enough on this machine?

A pipeline leg is three steps where a speech-to-speech leg has one model: a local speech-to-text
model, Claude, and local Kokoro text-to-speech. This script times each step on the CPU, with
phone-quality caller audio (8 kHz mu-law), and prints one table. It changes nothing in the project.

Run (the extra packages live in a temporary environment; `pyproject.toml` stays as it is):

    uv run --with kokoro-onnx --with moonshine-voice --with faster-whisper python scripts/spike_pipeline_leg.py

The first run downloads the models: Kokoro (about 340 MB) to ~/.cache/pipecat/kokoro-onnx, the
Moonshine and Whisper models to their own caches.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import ai_call_center.config  # noqa: F401 - loads .env, and gives Python a CA bundle on NixOS

KOKORO_DIR = Path.home() / ".cache/pipecat/kokoro-onnx"  # the same place Pipecat's KokoroTTSService uses
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

# What a caller says. Short, medium, long.
CALLER_LINES = [
    "Tell me about his background.",
    "I run a bakery, and we miss a lot of phone orders during the morning rush.",
    "Can you connect me to him now? It's about a project we talked about last week, and I have two questions on the timeline.",
]
# First sentences of an agent's answer: the text-to-speech step only has to start the first one fast.
AGENT_SENTENCES = [
    "Sure.",
    "He is a software architect with over twelve years of experience.",
    "An agent could take those orders for you, so no caller waits while your staff is busy baking.",
]
SYSTEM_PROMPT = (
    "You are the consultant agent on a phone line. This is a phone call. Use short sentences and plain words. "
    "Say one to three sentences, then let the caller talk. "
) + "Background notes follow. " + ("The owner builds voice agents, chat agents, APIs, and cloud platforms for businesses. " * 60)


def note(text: str) -> None:
    print(text, flush=True)


def best(values: list[float]) -> float:
    return min(values)


def words(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() or c == "'" else " " for c in text).split()


def word_errors(reference: str, heard: str) -> int:
    """Word-level edit distance."""
    a, b = words(reference), words(heard)
    row = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, wb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (wa != wb))
    return row[-1]


def resample(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
    n = int(len(samples) * dst / src)
    return np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype(np.float32)


def phone_quality(samples: np.ndarray, rate: int) -> np.ndarray:
    """What the server gets from Telnyx: 8 kHz mu-law. Returned as 16 kHz float audio for the speech-to-text models."""
    from pipecat.audio import g711

    box = np.ones(3) / 3  # a crude low-pass before the downsample
    narrow = resample(np.convolve(samples, box, mode="same"), rate, 8000)
    pcm = (np.clip(narrow, -1, 1) * 32767).astype(np.int16).tobytes()
    back = np.frombuffer(g711.ulaw_to_pcm(g711.pcm_to_ulaw(pcm)), dtype=np.int16).astype(np.float32) / 32768
    return resample(back, 8000, 16000)


async def kokoro_step() -> tuple[list[tuple[str, np.ndarray]], list[str]]:
    from kokoro_onnx import Kokoro

    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in KOKORO_FILES.items():
        target = KOKORO_DIR / name
        if not target.exists():
            note(f"downloading {name} ...")
            urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed https URL
    t0 = time.perf_counter()
    kokoro = Kokoro(str(KOKORO_DIR / "kokoro-v1.0.onnx"), str(KOKORO_DIR / "voices-v1.0.bin"))
    load = time.perf_counter() - t0
    rows = [f"| Kokoro text-to-speech | load the model | {load * 1000:.0f} ms | once, at server start |"]
    for sentence in AGENT_SENTENCES:
        firsts, totals, audio_secs = [], [], 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            first, n, rate = None, 0, 24000
            async for samples, rate in kokoro.create_stream(sentence, voice="af_heart", lang="en-us", speed=1.0):
                if first is None:
                    first = time.perf_counter() - t0
                n += len(samples)
            totals.append(time.perf_counter() - t0)
            firsts.append(first or 0.0)
            audio_secs = n / rate
        rows.append(
            f"| Kokoro text-to-speech | first audio, {len(sentence.split())}-word sentence | {best(firsts) * 1000:.0f} ms | "
            f"{audio_secs:.1f} s of audio made in {best(totals) * 1000:.0f} ms |"
        )
    callers = []
    for line in CALLER_LINES:
        samples, rate = kokoro.create(line, voice="am_michael", lang="en-us", speed=1.0)
        callers.append((line, phone_quality(np.asarray(samples, dtype=np.float32), rate)))
    return callers, rows


def stt_rows(name: str, load, transcribe, callers) -> list[str]:
    try:
        t0 = time.perf_counter()
        model = load()
        load_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 - a model that does not load is a result too
        return [f"| {name} | load the model | failed | {type(exc).__name__}: {str(exc)[:80]} |"]
    rows = []
    errors = total_words = 0
    for line, audio in callers:
        times, text = [], ""
        for _ in range(3):
            t0 = time.perf_counter()
            text = transcribe(model, audio)
            times.append(time.perf_counter() - t0)
        errors += word_errors(line, text)
        total_words += len(words(line))
        rows.append(f"| {name} | {len(audio) / 16000:.1f} s of caller speech | {best(times) * 1000:.0f} ms | heard: {text.strip()[:70]} |")
    rows.append(f"| {name} | word errors | {errors} of {total_words} | model load {load_s * 1000:.0f} ms |")
    return rows


def moonshine(arch: str):
    def load():
        from moonshine_voice import Transcriber, get_model_for_language, string_to_model_arch

        path, model_arch = get_model_for_language("en", string_to_model_arch(arch))
        return Transcriber(path, model_arch, options={})

    def transcribe(model, audio: np.ndarray) -> str:
        return " ".join(line.text for line in model.transcribe_without_streaming(audio.tolist(), 16000).lines)

    return load, transcribe


def whisper(size: str):
    def load():
        from faster_whisper import WhisperModel

        return WhisperModel(size, device="cpu", compute_type="int8")

    def transcribe(model, audio: np.ndarray) -> str:
        segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segments)

    return load, transcribe


async def claude_rows() -> list[str]:
    """Time to the first full sentence of a streamed answer. Uses the project's ANTHROPIC_API_KEY."""
    try:
        from langchain.chat_models import init_chat_model
    except Exception as exc:  # noqa: BLE001
        return [f"| Claude | not measured | - | {type(exc).__name__}: {exc} |"]
    rows = []
    for model_id in ("anthropic:claude-haiku-5-5", "anthropic:claude-sonnet-5-5", "anthropic:claude-opus-5-5"):
        try:
            model = init_chat_model(model_id, max_tokens=120)
            first_tokens, first_sentences = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                first_token = first_sentence = None
                text = ""
                async for chunk in model.astream([("system", SYSTEM_PROMPT), ("user", CALLER_LINES[1])]):
                    piece = chunk.text if isinstance(chunk.text, str) else chunk.text()
                    if piece and first_token is None:
                        first_token = time.perf_counter() - t0
                    text += piece or ""
                    if first_sentence is None and any(mark in text for mark in ".!?"):
                        first_sentence = time.perf_counter() - t0
                        break
                first_tokens.append(first_token or 0.0)
                first_sentences.append(first_sentence or (time.perf_counter() - t0))
            rows.append(
                f"| {model_id.split(':')[1]} | first full sentence of the answer | {statistics.median(first_sentences) * 1000:.0f} ms | "
                f"first token {statistics.median(first_tokens) * 1000:.0f} ms; median of 3 |"
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(f"| {model_id.split(':')[1]} | first full sentence of the answer | failed | {type(exc).__name__}: {str(exc)[:80]} |")
    return rows


async def main() -> None:
    only = set(sys.argv[1:])
    note("Timing the three steps of a pipeline leg on this CPU. Best of 3 runs unless a row says otherwise.\n")
    callers, rows = await kokoro_step()
    for name, (load, transcribe) in {
        "Moonshine tiny": moonshine("tiny"),
        "Moonshine base": moonshine("base"),
        "Moonshine small-streaming": moonshine("small-streaming"),
        "Whisper tiny.en (int8)": whisper("tiny.en"),
        "Whisper base.en (int8)": whisper("base.en"),
        "Whisper small.en (int8)": whisper("small.en"),
    }.items():
        if only and not any(key.lower() in name.lower() for key in only):
            continue
        note(f"timing {name} ...")
        rows += stt_rows(name, load, transcribe, callers)
    if not only or "claude" in only:
        note("timing Claude ...")
        rows += await claude_rows()
    note("\n| Step | What | Time | Notes |\n|---|---|---|---|")
    for row in rows:
        note(row)


if __name__ == "__main__":
    asyncio.run(main())
