"""Check the Phonic settings in `.env`: open one short conversation and have it say one sentence.

Run: uv run check-phonic

It uses the same key, model, turn taking and config as a phone call, at the phone's 8 kHz. Phonic
bills by the minute, so the check costs under one cent.
"""

from __future__ import annotations

import asyncio
import base64
import sys
import time

from ai_call_center.config import Settings, get_settings

WAIT_SECS = 15.0
SENTENCE = "The Phonic leg works."


def missing_settings(settings: Settings) -> list[str]:
    """Names of the `.env` settings that the Phonic leg needs and that are empty."""
    return [] if settings.phonic_api_key.get_secret_value() else ["PHONIC_API_KEY"]


def explain(error: Exception) -> str:
    """One plain sentence for the errors a new account gives most often. The key is never in it."""
    text = str(error)
    low = text.lower()
    if "api key" in low or "invalid_api_key" in low or "401" in low:
        return text + '\nPhonic did not accept the key. Check PHONIC_API_KEY: it starts with "ph_".'
    if "model" in low:
        return text + "\nCheck PHONIC_MODEL, or leave it empty for Phonic's default."
    if "voice" in low:
        return text + "\nPhonic does not know one of the voices in LEG_VOICES['phonic'] (agents/specs.py)."
    return text


class CheckFailed(Exception):
    pass


async def ask_once(settings: Settings) -> tuple[float, float, str, str]:
    """Open a conversation and have it say one sentence. Returns (seconds to ready, seconds from the
    request to the first speech audio, the text Phonic sent with the audio, the model Phonic reports)."""
    from phonic import AsyncPhonic
    from phonic.core.api_error import ApiError
    from phonic.types.audio_chunk_payload import AudioChunkPayload
    from phonic.types.say_payload import SayPayload

    from ai_call_center.agents.specs import default_voice
    from ai_call_center.voice.phonic_leg import PHONIC_FORMATS, phonic_config
    from ai_call_center.voice.call_session import TELNYX_SAMPLE_RATE

    key = settings.phonic_api_key.get_secret_value()
    fmt = PHONIC_FORMATS[TELNYX_SAMPLE_RATE]
    client = AsyncPhonic(api_key=key)
    t0 = time.monotonic()
    conversation_id = ""
    try:
        async with client.conversations.connect() as socket:
            await socket.send_config(
                phonic_config(
                    instructions="You are a test voice. Answer in one short sentence.",
                    voice=default_voice("classifier", "phonic"),
                    audio_format=(fmt, fmt),
                    turn_taking=settings.phonic_turn_taking,
                    model=settings.phonic_model,
                )
            )
            silence = base64.b64encode(b"\x00" * (TELNYX_SAMPLE_RATE // 50 * 2)).decode("ascii")

            async def send_silence() -> None:
                # A call sends 20 ms of caller audio at a time, all the time.
                while True:
                    await socket.send_audio_chunk(AudioChunkPayload(audio=silence))
                    await asyncio.sleep(0.02)

            microphone: asyncio.Task | None = None
            ready = asked = first_audio = 0.0
            said: list[str] = []
            speaking = False
            try:
                async with asyncio.timeout(WAIT_SECS):
                    async for message in socket:
                        data = message.dict()
                        kind = data.get("type")
                        if kind == "error":
                            raise CheckFailed(f"Phonic error: {(data.get('error') or {}).get('message')}")
                        if kind == "conversation_created":
                            conversation_id = data.get("conversation_id") or ""
                        elif kind == "ready_to_start_conversation":
                            ready = time.monotonic() - t0
                            microphone = asyncio.create_task(send_silence())
                            asked = time.monotonic()
                            await socket.send_say(SayPayload(text=SENTENCE))
                        elif kind == "assistant_started_speaking":
                            speaking = True
                        elif kind == "audio_chunk" and speaking:
                            audio = base64.b64decode(data.get("audio") or "")
                            if not first_audio and audio.strip(b"\x00"):
                                first_audio = time.monotonic() - asked
                            said.append(data.get("text") or "")
                        elif kind == "assistant_finished_speaking":
                            break
            except TimeoutError:
                raise CheckFailed(f"no answer came in {WAIT_SECS:.0f} s") from None
            finally:
                if microphone is not None:
                    microphone.cancel()
    except ApiError as exc:
        # Never print the exception itself: its text holds the request headers, and so the key.
        raise CheckFailed(f"Phonic refused the connection with HTTP {exc.status_code}") from None
    if not first_audio:
        raise CheckFailed("Phonic answered with no speech audio")
    model = ""
    if conversation_id:
        try:
            model = (await client.conversations.get(conversation_id)).conversation.model or ""
        except Exception:  # noqa: BLE001 - the model name is extra information
            pass
    return ready, first_audio, "".join(said).strip(), model


async def run(settings: Settings) -> int:
    missing = missing_settings(settings)
    if missing:
        print(f"Not checked. Set these in .env first: {', '.join(missing)}")
        return 1
    print(f"Asking Phonic ({settings.phonic_model or 'its default model'}) to say one sentence...")
    try:
        ready, first_audio, said, model = await ask_once(settings)
    except Exception as exc:  # noqa: BLE001 - show the cause; this is a setup check
        message = explain(exc)
        key = settings.phonic_api_key.get_secret_value()
        print("FAILED. " + (message.replace(key, "***") if key else message))
        return 1
    print(
        f"It works. Ready after {ready:.1f} s; first audio {first_audio:.2f} s after the request."
        f" Model: {model or 'not reported'}. Phonic said: {said or '(no text came with the audio)'}"
    )
    print("No agent runs on Phonic by default. To move an agent to it, set for example AGENT_LEGS=classifier=phonic in .env.")
    return 0


def main() -> None:
    sys.exit(asyncio.run(run(get_settings())))
