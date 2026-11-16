"""Check the Amazon Nova Sonic settings in `.env`: open one short session and ask for one sentence.

Run: uv run check-nova

It uses the same service class and the same keys as a phone call. It costs a fraction of a cent.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid

from ai_call_center.config import Settings, get_settings

WAIT_SECS = 20.0


def missing_settings(settings: Settings) -> list[str]:
    """Names of the `.env` settings that the Nova leg needs and that are empty."""
    pairs = (("NOVA_AWS_ACCESS_KEY_ID", settings.nova_aws_access_key_id), ("NOVA_AWS_SECRET_ACCESS_KEY", settings.nova_aws_secret_access_key.get_secret_value()))
    return [name for name, value in pairs if not value]


def explain(error: Exception) -> str:
    """One plain sentence for the AWS errors a new account gives most often."""
    text = f"{type(error).__name__}: {error}"
    low = text.lower()
    if "accessdenied" in low or "not authorized" in low:
        return text + "\nThe key works, but it may not use this model. Give the IAM user the permission bedrock:InvokeModelWithBidirectionalStream, and check that Bedrock offers the model to your account in this region."
    if "unrecognizedclient" in low or "invalidsignature" in low or "security token" in low:
        return text + "\nAWS did not accept the key. Check NOVA_AWS_ACCESS_KEY_ID and NOVA_AWS_SECRET_ACCESS_KEY."
    if "validation" in low or "not found" in low or "model" in low:
        return text + "\nCheck NOVA_SONIC_MODEL and NOVA_AWS_REGION. Nova 2.5 Sonic runs in us-east-1, us-west-2, eu-north-1, and ap-northeast-1."
    return text


async def ask_once(settings: Settings) -> tuple[float, str]:
    """Open a session, ask for one sentence, and return (seconds to the first audio, the text Nova said)."""
    from ai_call_center.agents.specs import AgentSpec  # noqa: F401 - keeps the import order of a real call
    from ai_call_center.voice.legs import HandoffNovaSonic
    from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService

    service = HandoffNovaSonic(
        access_key_id=settings.nova_aws_access_key_id,
        secret_access_key=settings.nova_aws_secret_access_key.get_secret_value(),
        region=settings.nova_aws_region,
        settings=AWSNovaSonicLLMService.Settings(model=settings.nova_sonic_model, voice="matthew", endpointing_sensitivity=settings.nova_endpointing),
    )
    prompt, audio_content = str(uuid.uuid4()), str(uuid.uuid4())
    stream = await service.open_stream(service.create_client())
    try:
        await service.send_event(service.build_session_start_json(), stream)
        await service.send_prompt_start([], prompt, stream)
        await service.send_text("You are a test voice. Answer in one short sentence.", "SYSTEM", prompt, stream, False)
        await service.send_audio_input_start(prompt, audio_content, stream)
        t0 = time.monotonic()
        await service.send_text("Say: the Nova leg works.", "USER", prompt, stream, True)
        said: list[str] = []
        first_audio = 0.0
        while time.monotonic() - t0 < WAIT_SECS:
            output = await asyncio.wait_for(stream.await_output(), WAIT_SECS)
            result = await asyncio.wait_for(output[1].receive(), WAIT_SECS)
            if result is None:
                raise RuntimeError("AWS Bedrock ended the stream before an answer came")
            value = getattr(result, "value", None)
            if value is None or not value.bytes_:
                raise RuntimeError(f"AWS Bedrock stream reported: {result}")
            event = json.loads(value.bytes_.decode("utf-8")).get("event", {})
            if "textOutput" in event and event["textOutput"].get("role") == "ASSISTANT":
                said.append(event["textOutput"].get("content", ""))
            if "audioOutput" in event and not first_audio:
                first_audio = time.monotonic() - t0
            if "completionEnd" in event or (first_audio and "contentEnd" in event and event["contentEnd"].get("type") == "AUDIO"):
                break
        if not first_audio:
            raise RuntimeError(f"no audio came back in {WAIT_SECS:.0f} s")
        return first_audio, " ".join(dict.fromkeys(text.strip() for text in said if text.strip()))
    finally:
        try:
            await stream.close()
        except Exception:  # noqa: BLE001 - the check is over either way
            pass


async def run(settings: Settings) -> int:
    missing = missing_settings(settings)
    if missing:
        print(f"Not checked. Set these in .env first: {', '.join(missing)}")
        return 1
    print(f"Asking {settings.nova_sonic_model} in {settings.nova_aws_region} for one sentence...")
    try:
        seconds, said = await ask_once(settings)
    except Exception as exc:  # noqa: BLE001 - show the cause; this is a setup check
        print("FAILED. " + explain(exc))
        return 1
    print(f"It works. First audio after {seconds:.1f} s. Nova said: {said or '(no text came with the audio)'}")
    print('To use it on calls, set for example AGENT_LEGS=consultant=nova_sonic in .env, then restart the server.')
    return 0


def main() -> None:
    sys.exit(asyncio.run(run(get_settings())))
