"""Make the stored recording of the greeting (voice/stored_greeting.py).

Run: uv run record-greeting takes/                  three takes, saved as takes/take-1.wav ...
     uv run record-greeting --takes 5 takes/        five takes
     uv run record-greeting --install takes/take-2.wav

Each take is one short Gemini Live session in the greeting agent's voice, asked to say exactly the
greeting words of these settings. Listen to the takes, and install one where every word is right.
The install trims the silence at both ends and writes the recording and its sidecar into the package,
so the next deploy carries it. Run both with the settings of the server that plays it: the sidecar
holds what the recording was made for, and a call plays it only when that is still true.

A take costs well under one cent: a few hundred tokens in, a few seconds of audio out.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from ai_call_center.config import Settings, get_settings
from ai_call_center.voice.phone_messages import connect_message
from ai_call_center.voice.stored_greeting import SAMPLE_RATE, StoredGreeting, greeting_target, install, trim_silence, write_wav

TAKE_SECS = 40.0
TAKE_INSTRUCTIONS = "You are a phone agent. Say exactly the words you are given, and nothing else."


class RecordFailed(Exception):
    pass


def take_paths(folder: Path, takes: int) -> list[Path]:
    """The files of the takes: take-1.wav, take-2.wav, ... A number already used is skipped, so a
    second run keeps the takes of the first."""
    paths: list[Path] = []
    number = 1
    while len(paths) < takes:
        path = folder / f"take-{number}.wav"
        if not path.exists():
            paths.append(path)
        number += 1
    return paths


def target_for(settings: Settings) -> dict[str, str]:
    """The words, leg, voice and model of the greeting, as a call on this server says it."""
    from ai_call_center.graph.build import ENTRY_AGENT, default_specs
    from ai_call_center.voice.legs import resolve_spec

    spec = resolve_spec(default_specs()[ENTRY_AGENT], settings)
    if spec.leg != "gemini_live":
        raise RecordFailed(f"the {ENTRY_AGENT} agent runs on {spec.leg} with these settings; only a Gemini Live greeting can be recorded")
    return greeting_target(settings, spec)


async def record_take(settings: Settings, *, words: str, voice: str, model: str) -> tuple[bytes, str]:
    """One session: (24 kHz PCM16 audio, Gemini's transcript of what it said)."""
    from google import genai
    from google.genai import types
    from pipecat.services.google.utils import gemini_api_http_options

    # Without these HTTP options the Live API answers HTTP 404.
    client = genai.Client(api_key=settings.google_api_key.get_secret_value(), http_options=gemini_api_http_options(None))
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=TAKE_INSTRUCTIONS,
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    audio, said = bytearray(), []
    async with client.aio.live.connect(model=model, config=config) as session:
        # The same order to speak that a call's first message gives (phone_messages.py).
        await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=connect_message(words))]), turn_complete=True)
        async with asyncio.timeout(TAKE_SECS):
            async for message in session.receive():
                if message.data:
                    audio.extend(message.data)
                content = message.server_content
                if content and content.output_transcription and content.output_transcription.text:
                    said.append(content.output_transcription.text)
                if content and content.turn_complete:
                    break
    if not audio:
        raise RecordFailed("Gemini sent no audio")
    return bytes(audio), "".join(said).strip()


async def record(settings: Settings, folder: Path, takes: int) -> int:
    if not settings.google_api_key.get_secret_value():
        print("Not recorded. Set GOOGLE_API_KEY in .env first.")
        return 1
    try:
        target = target_for(settings)
    except RecordFailed as exc:
        print(f"Not recorded: {exc}.")
        return 1
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Recording {takes} take(s) in the voice {target['voice']}. The words:\n  {target['words']}")
    failed = 0
    for path in take_paths(folder, takes):
        try:
            audio, said = await record_take(settings, words=target["words"], voice=target["voice"], model=target["model"])
            take = trim_silence(StoredGreeting(audio, SAMPLE_RATE))
        except Exception as exc:  # noqa: BLE001 - one failed take does not stop the others
            # Only the type and the start of the message: never the request, which holds the key.
            print(f"{path.name}: FAILED ({type(exc).__name__}: {str(exc)[:200]})")
            failed += 1
            continue
        write_wav(path, StoredGreeting(audio, SAMPLE_RATE))
        print(f"{path}: {take.seconds:.1f} s of speech. Gemini's transcript: {said or '(none)'}")
    if failed == takes:
        return 1
    print(f"Listen to the takes. Then: uv run record-greeting --install {folder}/take-<n>.wav")
    return 0


def install_take(settings: Settings, take: Path, **paths) -> int:
    try:
        target = target_for(settings)
        greeting = install(take, **target, **paths)
    except (RecordFailed, OSError, ValueError) as exc:
        print(f"Not installed: {exc}.")
        return 1
    print(
        f"Installed: {greeting.seconds:.1f} s at {greeting.sample_rate} Hz, made for these greeting words, the {target['voice']} voice on {target['leg']}."
        " Commit src/ai_call_center/voice/greeting.wav and greeting.json; calls play it after the next deploy."
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="record-greeting", description="Record takes of the greeting, or install one as the stored greeting.")
    parser.add_argument("folder", nargs="?", type=Path, help="where the takes go")
    parser.add_argument("--takes", type=int, default=3, help="how many takes (default 3)")
    parser.add_argument("--install", type=Path, metavar="WAV", help="trim this take and install it as the stored greeting")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.install:
        sys.exit(install_take(settings, args.install))
    if args.folder is None or args.takes < 1:
        parser.error("name a folder for the takes, and at least one take")
    sys.exit(asyncio.run(record(settings, args.folder, args.takes)))
