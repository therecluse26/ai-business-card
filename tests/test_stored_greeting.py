"""The stored greeting: when a call plays it, what the call holds after it, and how a recording is made."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest
from google.genai.types import LiveServerContent, LiveServerMessage, Transcription
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from ai_call_center import record_greeting
from ai_call_center.agents.specs import VOICE_LEVELS_DBFS
from ai_call_center.config import Settings
from ai_call_center.voice.call_session import TransferGate, VoiceLeg, _greet_from_recording
from ai_call_center.voice.legs import build_leg, plays_recordings, replay_and_respond, seed_without_answer
from ai_call_center.voice.leveler import VoiceLeveler
from ai_call_center.voice.phone_messages import GREETING_DONE, connect_message
from ai_call_center.voice.stats import CallStats
from ai_call_center.voice.stored_greeting import StoredGreeting, choose, greeting_key, greeting_target, install, load, read_wav, trim_silence, write_wav

RATE = 24_000
WORDS = "Hi, you've reached the test line. Who am I speaking with?"
MADE_FOR = {"words": WORDS, "leg": "gemini_live", "voice": "Aoede", "model": "models/gemini-3.8-live"}


def settings(**over) -> Settings:
    base = {"openai_api_key": "sk-test", "google_api_key": "g-test", "nova_aws_access_key_id": "", "nova_aws_secret_access_key": "", "phonic_api_key": "", "respellings": "", "agent_legs": ""}
    return Settings(_env_file=None, **{**base, **over})


def tone(secs: float, *, level: float = 0.1, hz: float = 220.0) -> bytes:
    t = np.arange(int(RATE * secs)) / RATE
    return (np.sin(2 * np.pi * hz * t) * level * 32767).astype(np.int16).tobytes()


def silence(secs: float) -> bytes:
    return bytes(int(RATE * secs) * 2)


def installed(tmp_path, made_for=MADE_FOR, audio: bytes | None = None):
    take = tmp_path / "take-1.wav"
    write_wav(take, StoredGreeting(audio or silence(0.5) + tone(1.0) + silence(1.0)))
    paths = {"wav": tmp_path / "greeting.wav", "sidecar": tmp_path / "greeting.json"}
    install(take, **made_for, **paths)
    return paths


class FakeSession:
    """Stands in for a Gemini Live session: keeps what the service sends."""

    def __init__(self) -> None:
        self.content: list[dict] = []
        self.realtime: list[dict] = []

    async def send_client_content(self, **kwargs) -> None:
        self.content.append(kwargs)

    async def send_realtime_input(self, **kwargs) -> None:
        self.realtime.append(kwargs)


def gemini(specs, pushed: list | None = None):
    service = build_leg(specs["classifier"], instructions="x", settings=settings())
    if pushed is not None:

        async def push(frame, direction=None) -> None:
            pushed.append(frame)

        service.push_frame = push
    return service


def turn_texts(sent: dict) -> list[str]:
    return [part.text if hasattr(part, "text") else part["text"] for turn in sent["turns"] for part in (turn.parts if hasattr(turn, "parts") else turn["parts"])]


# --- what the recording was made for ---


def test_the_sidecar_keeps_hashes_and_no_words(tmp_path):
    paths = installed(tmp_path)
    text = paths["sidecar"].read_text()
    assert WORDS not in text and "Aoede" not in text and "gemini_live" not in text and "3.8" not in text
    made_for = json.loads(text)
    assert made_for["sha256"] == greeting_key(**MADE_FOR)["sha256"] and made_for["sample_rate"] == RATE
    assert greeting_key(**{**MADE_FOR, "voice": "Puck"})["sha256"] != made_for["sha256"]


def test_a_matching_recording_is_loaded(tmp_path):
    greeting, why = load(**MADE_FOR, **installed(tmp_path))
    assert why == "" and greeting is not None and greeting.sample_rate == RATE
    assert 1.0 < greeting.seconds < 1.3  # the silence at both ends is trimmed


@pytest.mark.parametrize(("change", "named"), [({"words": WORDS + " Now."}, "words"), ({"voice": "Puck"}, "voice"), ({"leg": "openai_realtime", "voice": "marin"}, "leg, voice"), ({"model": "models/other"}, "model")])
def test_a_recording_made_for_something_else_is_not_played(tmp_path, change, named):
    greeting, why = load(**{**MADE_FOR, **change}, **installed(tmp_path))
    assert greeting is None and why == f"the recording was made for other {named}; record it again"


def test_a_missing_or_broken_recording_is_not_played(tmp_path):
    paths = {"wav": tmp_path / "greeting.wav", "sidecar": tmp_path / "greeting.json"}
    assert load(**MADE_FOR, **paths) == (None, "no recording is installed (uv run record-greeting)")
    paths["wav"].write_bytes(b"not a wav")
    paths["sidecar"].write_text("{}")
    greeting, why = load(**MADE_FOR, **paths)
    assert greeting is None and why.startswith("the recording cannot be read")
    paths = installed(tmp_path)
    paths["sidecar"].unlink()
    greeting, why = load(**MADE_FOR, **paths)
    assert greeting is None and why.startswith("the recording cannot be read")


def test_the_call_chooses_the_recording_only_for_a_fresh_call_on_a_leg_that_plays_it(tmp_path):
    paths = installed(tmp_path)
    greeting, why = choose(enabled=True, fresh=True, leg_plays=True, **MADE_FOR, **paths)
    assert greeting is not None and why == ""
    assert choose(enabled=False, fresh=True, leg_plays=True, **MADE_FOR, **paths) == (None, "STORED_GREETING is off")
    # After a transfer that did not connect, or a provider change, the first words are not the standard greeting.
    assert choose(enabled=True, fresh=False, leg_plays=True, **MADE_FOR, **paths) == (None, "this media stream continues a call")
    assert choose(enabled=True, fresh=True, leg_plays=False, **MADE_FOR, **paths) == (None, "the greeting agent's leg gemini_live cannot play a recording")
    # After a failover at call start, the greeting agent runs on another leg with another voice.
    greeting, why = choose(enabled=True, fresh=True, leg_plays=True, **{**MADE_FOR, "leg": "openai_realtime", "voice": "marin"}, **paths)
    assert greeting is None and "leg, voice" in why


def test_the_call_and_the_command_agree_on_what_the_greeting_is(specs):
    plain = greeting_target(settings(owner_name="Ada Lovelace"), specs["classifier"])
    assert plain["leg"] == "gemini_live" and plain["voice"] == "Aoede" and plain["model"] == Settings.model_fields["gemini_live_model"].default
    respelled = greeting_target(settings(owner_name="Ada Lovelace", respellings="Lovelace=Luv-layss"), specs["classifier"])
    assert "Luv-layss" in respelled["words"] and "Lovelace" not in respelled["words"]  # the words the voice model reads
    assert plays_recordings(gemini(specs)) and not plays_recordings(build_leg(specs["contact"], instructions="x", settings=settings()))


def test_the_setting_is_on_by_default():
    assert Settings(_env_file=None).stored_greeting is True


# --- making the recording ---


def test_trim_keeps_the_speech_and_a_little_around_it():
    take = StoredGreeting(silence(0.6) + tone(1.0) + silence(2.0))
    trimmed = trim_silence(take)
    assert 1.0 + 0.15 <= trimmed.seconds <= 1.0 + 0.02 + 0.15 + 0.02
    samples = np.frombuffer(trimmed.audio, dtype=np.int16)
    assert np.abs(samples[: int(RATE * 0.03)]).max() > 1000  # the first sound comes within 30 ms
    with pytest.raises(ValueError, match="no speech"):
        trim_silence(StoredGreeting(silence(1.0)))


def test_install_writes_a_mono_24_khz_wav(tmp_path):
    paths = installed(tmp_path)
    greeting = read_wav(paths["wav"])
    assert greeting.sample_rate == RATE and 1.0 < greeting.seconds < 1.3


def test_takes_get_new_numbers(tmp_path):
    (tmp_path / "take-1.wav").write_bytes(b"")
    assert [p.name for p in record_greeting.take_paths(tmp_path, 2)] == ["take-2.wav", "take-3.wav"]


def test_the_command_installs_a_take_for_the_settings(tmp_path, monkeypatch, capsys):
    take = tmp_path / "take-1.wav"
    write_wav(take, StoredGreeting(silence(0.3) + tone(0.8)))
    monkeypatch.setattr(record_greeting, "target_for", lambda _settings: MADE_FOR)
    paths = {"wav": tmp_path / "greeting.wav", "sidecar": tmp_path / "greeting.json"}
    assert record_greeting.install_take(settings(), take, **paths) == 0
    assert load(**MADE_FOR, **paths)[0] is not None
    assert "g-test" not in capsys.readouterr().out  # no key is printed
    assert record_greeting.install_take(settings(), tmp_path / "missing.wav", **paths) == 1


def test_the_command_records_only_a_gemini_greeting(monkeypatch, specs):
    monkeypatch.setattr("ai_call_center.graph.build.default_specs", lambda: specs)
    assert record_greeting.target_for(settings())["voice"] == "Aoede"
    with pytest.raises(record_greeting.RecordFailed, match="only a Gemini Live greeting"):
        record_greeting.target_for(settings(google_api_key=""))  # the greeting agent then runs on OpenAI


# --- playing it ---


async def test_the_recording_goes_down_the_pipeline_as_a_model_answer(specs):
    pushed: list = []
    service = gemini(specs, pushed)
    audio = tone(0.5)
    await service.play_recording(audio, RATE, WORDS)

    kinds = [type(f) for f in pushed]
    assert kinds[:3] == [TTSStartedFrame, LLMFullResponseStartFrame, TTSTextFrame] and kinds[-2:] == [TTSStoppedFrame, LLMFullResponseEndFrame]
    chunks = [f for f in pushed if isinstance(f, TTSAudioRawFrame)]
    assert b"".join(f.audio for f in chunks) == audio and {f.sample_rate for f in chunks} == {RATE}
    words = pushed[2]
    assert words.text == WORDS and words.append_to_context is False  # the call writes the context; not twice
    assert not service._bot_is_responding  # the turn is closed, as after a model answer
    # The voice leveler acts on these frames, with the greeting agent's measured level.
    leveler = VoiceLeveler(lambda: "Aoede", target_dbfs=-22.0, seeds=VOICE_LEVELS_DBFS)
    assert isinstance(chunks[0], OutputAudioRawFrame) and leveler.gain("Aoede", chunks[0].audio, RATE) < 1.0


async def test_the_callers_words_stop_the_recording(specs, monkeypatch):
    service = gemini(specs, [])
    interruptions = []

    async def interrupt() -> None:
        interruptions.append(time.monotonic())

    async def keep(self, message) -> None:  # the base service buffers the words for the transcript
        return None

    service.broadcast_interruption = interrupt
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_msg_input_transcription", keep)

    def heard(text: str) -> LiveServerMessage:
        return LiveServerMessage(server_content=LiveServerContent(input_transcription=Transcription(text=text)))

    await service._handle_msg_input_transcription(heard("hello"))
    assert interruptions == []  # no recording plays: Gemini's own "interrupted" handles its answers

    await service.play_recording(tone(2.0), RATE, WORDS)
    await service._handle_msg_input_transcription(heard(" "))
    assert interruptions == []  # no words yet
    await service._handle_msg_input_transcription(heard("Hi, this is"))
    await service._handle_msg_input_transcription(heard(" Sam"))
    assert len(interruptions) == 1


async def test_the_session_is_seeded_without_an_order_to_answer(specs):
    service = gemini(specs, [])
    session = FakeSession()
    service._session = session
    context = LLMContext(messages=[{"role": "developer", "content": GREETING_DONE}, {"role": "assistant", "content": WORDS}])

    await seed_without_answer(service, context)

    assert len(session.content) == 1 and session.content[0]["turn_complete"] is False
    assert session.realtime == []  # Gemini 3 answers only after a realtime nudge; none is sent
    assert WORDS in turn_texts(session.content[0])
    assert service._context is context and service._ready_for_realtime_input  # caller audio goes to Gemini now
    assert service._inference_on_context_initialization is True  # only that seed was quiet

    # A later replay of this agent (a handoff back to it) answers as before.
    session.content.clear()
    await replay_and_respond(service, context)
    assert session.content[-1]["turn_complete"] is True and session.realtime


async def test_a_session_that_is_still_opening_is_seeded_without_an_answer_when_ready(specs):
    service = gemini(specs, [])
    context = LLMContext(messages=[{"role": "developer", "content": GREETING_DONE}, {"role": "assistant", "content": WORDS}])

    await seed_without_answer(service, context)  # no session yet
    session = FakeSession()
    await service._handle_session_ready(session)

    assert [sent["turn_complete"] for sent in session.content] == [False] and session.realtime == []
    assert not service.quiet_seed

    # A context frame from the pipeline later is an update: no second seed, no reconnect.
    await service._handle_context(context)
    assert len(session.content) == 1


async def test_a_replay_before_the_session_opened_still_answers(specs):
    service = gemini(specs, [])
    context = LLMContext(messages=[{"role": "developer", "content": "Speak."}])
    await seed_without_answer(service, context)
    await replay_and_respond(service, context)  # the quiet seed never went out; this one must answer
    session = FakeSession()
    await service._handle_session_ready(session)
    assert [sent["turn_complete"] for sent in session.content] == [True]


async def test_after_the_recording_the_context_holds_the_greeting_as_the_agents_words(specs):
    service = gemini(specs, [])
    session = FakeSession()
    service._session = session
    leg = VoiceLeg("classifier", "gemini_live", service)
    context = LLMContext(messages=[{"role": "developer", "content": connect_message(WORDS)}])

    assert await _greet_from_recording(leg, context, StoredGreeting(tone(0.3)), WORDS)

    # As after a model greeting and a handoff: no order to greet, and the greeting as the agent's own message.
    assert context.messages == [{"role": "developer", "content": GREETING_DONE}, {"role": "assistant", "content": WORDS}]
    assert leg.used
    # The session knows the greeting was said, and has no order to say it, or to answer.
    texts = " ".join(turn_texts(session.content[0]))
    assert WORDS in texts and "Say exactly these words" not in texts
    assert session.content[0]["turn_complete"] is False and session.realtime == []


async def test_a_recording_that_cannot_start_leaves_the_greeting_to_the_model(specs):
    service = gemini(specs, [])
    leg = VoiceLeg("classifier", "gemini_live", service)
    context = LLMContext(messages=[{"role": "developer", "content": connect_message(WORDS)}])

    assert not await _greet_from_recording(leg, context, StoredGreeting(b""), WORDS)

    assert context.messages == [{"role": "developer", "content": connect_message(WORDS)}]  # the order to greet stays for the model
    assert not leg.used and not service.quiet_seed


class PhoneOut(BaseOutputTransport):
    """The output transport with a phone that takes every frame."""

    def __init__(self) -> None:
        super().__init__(TransportParams(audio_out_enabled=True))
        self.written: list[OutputAudioRawFrame] = []

    async def start(self, frame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)  # as the Telnyx transport does once the socket is open

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        self.written.append(frame)
        return True


async def test_the_recording_plays_on_the_phone_and_the_watchdog_and_figures_see_it(specs, monkeypatch):
    """The recording through the leveler and the output transport, as in a call: the transport says
    the agent speaks (so the watchdog of `expect_speech` does not fire, and the caller's audio opens),
    the audio reaches the phone at 8 kHz, and the call figures count no answer and no tokens."""
    service = gemini(specs)

    async def no_connection(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(service, "_connect", no_connection)
    out = PhoneOut()
    gate = TransferGate(lambda: None)
    stats = CallStats()
    stats.bind({service.name: "gemini_live"}, lambda: "gemini_live", lambda: "classifier")
    pipeline = Pipeline([service, VoiceLeveler(lambda: "Aoede", target_dbfs=-22.0, seeds=VOICE_LEVELS_DBFS), out, gate])
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_out_sample_rate=8000), observers=[stats], cancel_on_idle_timeout=False)
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _started(_worker, _frame) -> None:
        started.set()

    async def play() -> float:
        await asyncio.wait_for(started.wait(), 2.0)
        asked = time.monotonic()
        await service.play_recording(tone(0.3), RATE, WORDS)
        for _ in range(200):
            if gate.last_started >= asked and not gate.speaking:
                break
            await asyncio.sleep(0.01)
        await worker.queue_frame(EndFrame())
        return asked

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    _, asked = await asyncio.gather(runner.run(), play())

    assert gate.last_started >= asked  # what expect_speech checks: the agent spoke after the request
    assert out.written and {f.sample_rate for f in out.written} == {8000}
    assert sum(len(f.audio) for f in out.written) >= int(0.3 * 8000) * 2 * 0.9
    assert stats.answers == [] and stats.cost() == (0.0, [])  # the greeting is no answer, and a recording costs no tokens


def test_a_click_at_the_start_is_kept_out_also_while_the_recording_plays(monkeypatch):
    """The stored greeting starts at once, before a click of the line (measured 2026-11-16: 60 ms
    after the stream opened, 170 ms long) is over. The caller's audio opens after the click."""
    from pipecat.frames.frames import InputAudioRawFrame

    from ai_call_center.voice import call_session
    from ai_call_center.voice.call_session import OPENING_CLICK_SECS, OpeningQuiet

    now = [100.0]
    monkeypatch.setattr(call_session.time, "monotonic", lambda: now[0])

    def audio() -> InputAudioRawFrame:
        return InputAudioRawFrame(audio=b"\x01\x02" * 80, sample_rate=8000, num_channels=1)

    quiet = OpeningQuiet(click_secs=OPENING_CLICK_SECS)
    quiet.screen(audio())
    quiet.screen(BotStartedSpeakingFrame())
    now[0] += 0.23
    click = audio()
    quiet.screen(click)
    assert click.audio == bytes(160) and not quiet.open
    now[0] += OPENING_CLICK_SECS
    speech = audio()
    quiet.screen(speech)
    assert speech.audio == b"\x01\x02" * 80 and quiet.open
