"""The Phonic voice leg (voice/phonic_leg.py), on a fake Phonic socket in a real Pipecat pipeline.

The fake socket stands in for `AsyncPhonic(...).conversations.connect()`. What it sends back follows
what the live API sent on 2026-11-24 (see the module docstring of phonic_leg.py).
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager

import pytest
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    FunctionCallResultProperties,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.utils.errors import ErrorCategory
from pipecat.workers.runner import WorkerRunner
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from ai_call_center.agents.specs import parse_agent_legs
from ai_call_center.config import Settings
from ai_call_center.voice import legs
from ai_call_center.voice.call_session import handoff_instruction, retire_connect_message
from ai_call_center.voice.phone_messages import GREETING_DONE, connect_message
from ai_call_center.voice.phonic_leg import (
    HISTORY_HEAD,
    RESUME_INSTRUCTION,
    PhonicLLMService,
    error_text,
    history_lines,
    phonic_tool_output,
    phonic_tools,
    render_call_so_far,
    replay_plan,
)

KEY = "ph_unit_test_key_0000"
SPEECH = (1000).to_bytes(2, "little", signed=True) * 160  # 20 ms at 8 kHz
SILENCE = b"\x00" * 320


def b64(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")


class FakeSocket:
    """Records what the service sends. `feed` queues a message from Phonic; `drop` closes the socket."""

    def __init__(self, *, ready: bool = True) -> None:
        self.sent: list[tuple[str, dict]] = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.ready = ready
        self.closed = False

    def feed(self, kind: str, **fields) -> None:
        self.incoming.put_nowait({"type": kind, **fields})

    def drop(self, code: int, reason: str = "") -> None:
        self.incoming.put_nowait(ConnectionClosedError(Close(code, reason), None))

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        while True:
            item = await self.incoming.get()
            if isinstance(item, Exception):
                raise item
            yield item

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.sent]

    def of(self, kind: str) -> list[dict]:
        return [payload for k, payload in self.sent if k == kind]

    async def _record(self, kind: str, payload) -> None:
        self.sent.append((kind, payload.dict(exclude_none=True)))

    async def send_config(self, payload) -> None:
        await self._record("config", payload)
        if self.ready:
            self.feed("conversation_created", conversation_id="conv_test")
            self.feed("ready_to_start_conversation")

    async def send_audio_chunk(self, payload) -> None:
        await self._record("audio", payload)

    async def send_add_system_message(self, payload) -> None:
        await self._record("add_system_message", payload)

    async def send_generate_reply(self, payload) -> None:
        await self._record("generate_reply", payload)

    async def send_say(self, payload) -> None:
        await self._record("say", payload)

    async def send_tool_call_output(self, payload) -> None:
        await self._record("tool_call_output", payload)


class FakePhonic:
    """`connect` for the service: a new FakeSocket for each conversation."""

    def __init__(self, **socket_options) -> None:
        self.sockets: list[FakeSocket] = []
        self.options = socket_options

    @property
    def socket(self) -> FakeSocket:
        return self.sockets[-1]

    def connect(self):
        @asynccontextmanager
        async def conversation():
            socket = FakeSocket(**self.options)
            self.sockets.append(socket)
            try:
                yield socket
            finally:
                socket.closed = True

        return conversation()


def lookup_tools() -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(name="lookup_caller", description="Look up the caller.", properties={"name": {"type": "string"}}, required=["name"]),
            FunctionSchema(name="transfer_to_profile", description="Hand the call to the profile agent.", properties={}, required=[]),
        ]
    )


def service(phonic: FakePhonic, **over) -> PhonicLLMService:
    options = {"api_key": KEY, "instructions": "You are the classifier.", "voice": "sabrina", "tools": lookup_tools(), "connect": phonic.connect}
    return PhonicLLMService(**{**options, **over})


class Run:
    def __init__(self, worker: PipelineWorker, down: asyncio.Queue, up: asyncio.Queue) -> None:
        self.worker, self.down, self.up = worker, down, up
        self.down_frames: list = []
        self.up_frames: list = []

    async def settle(self) -> None:
        for _ in range(10):
            await asyncio.sleep(0.01)
        while not self.down.empty():
            self.down_frames.append(self.down.get_nowait())
        while not self.up.empty():
            self.up_frames.append(self.up.get_nowait())

    def down_of(self, cls) -> list:
        return [f for f in self.down_frames if isinstance(f, cls)]

    def up_of(self, cls) -> list:
        return [f for f in self.up_frames if isinstance(f, cls)]

    def take(self) -> list:
        frames, self.down_frames = self.down_frames, []
        return frames


@asynccontextmanager
async def running(svc: PhonicLLMService, rate: int = 8000):
    down, up = asyncio.Queue(), asyncio.Queue()
    source = QueuedFrameProcessor(queue=up, queue_direction=FrameDirection.UPSTREAM)
    sink = QueuedFrameProcessor(queue=down, queue_direction=FrameDirection.DOWNSTREAM)
    worker = PipelineWorker(Pipeline([source, svc, sink]), cancel_on_idle_timeout=False, enable_rtvi=False, params=PipelineParams(audio_in_sample_rate=rate, audio_out_sample_rate=rate))
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _started(_worker, _frame):
        started.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    await asyncio.wait_for(started.wait(), 2)
    run = Run(worker, down, up)
    try:
        yield run
    finally:
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(task, 5)


def greeting_context() -> LLMContext:
    return LLMContext([{"role": "developer", "content": connect_message("Hi, you've reached the line. Who am I speaking with?")}])


# --- the call so far, as text ---


def test_the_call_so_far_is_one_line_for_each_turn_tool_and_note():
    messages = [
        {"role": "developer", "content": connect_message("Hi there.")},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": "I'm Quentin.\n I run a   bakery."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "identify_caller", "arguments": '{"name": "Quentin"}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": json.dumps("Welcome, Quentin.")},
        {"role": "tool", "tool_call_id": "t9", "content": json.dumps({"found": False})},
        {"role": "developer", "content": handoff_instruction("profile", "Google Gemini", "Quentin", ["caller: I'm Quentin."])},
        {"role": "user", "content": [{"type": "text", "text": "Tell me about the owner."}]},
        {"role": "user", "content": "   "},
    ]
    lines = history_lines(messages)
    assert lines[:5] == [
        f"phone system: {GREETING_DONE}",  # the order to say the greeting is not repeated to a later agent
        "agent: Hi there.",
        "caller: I'm Quentin. I run a bakery.",
        'tool call: identify_caller({"name": "Quentin"})',
        "tool result (identify_caller): Welcome, Quentin.",
    ]
    assert lines[5] == 'tool result: {"found": false}'
    assert lines[6].startswith("phone system: ") and "running on Google Gemini" in lines[6]
    assert "The call so far, word for word" not in lines[6]  # the repeated block is cut; the lines above hold it
    assert lines[7:] == ["caller: Tell me about the owner."]

    text = render_call_so_far(messages)
    assert text.startswith(HISTORY_HEAD + "\n") and text.endswith("caller: Tell me about the owner.")
    assert render_call_so_far([]) == "" and render_call_so_far([{"role": "user", "content": " "}]) == ""


def test_a_long_call_keeps_its_newest_part():
    messages = [{"role": "user", "content": f"turn {n} " + "x" * 200} for n in range(300)]
    text = render_call_so_far(messages)
    assert len(text) < 26_000 and "(The start of the call is left out.)" in text
    assert "turn 299 " in text and "turn 0 " not in text


def test_the_replay_plan_says_the_greeting_word_for_word():
    plan = replay_plan(greeting_context().messages)
    assert plan.history == "" and plan.say == "Hi, you've reached the line. Who am I speaking with?" and plan.instruction is None


def test_the_replay_plan_after_a_handoff_sends_the_call_then_the_instruction():
    context = greeting_context()
    context.add_message({"role": "assistant", "content": "Hi, you've reached the line. Who am I speaking with?"})
    context.add_message({"role": "user", "content": "Sam. Tell me about the owner."})
    retire_connect_message(context.messages)
    instruction = handoff_instruction("profile", "Phonic", "Sam", ["caller: Sam. Tell me about the owner."])
    context.add_message({"role": "developer", "content": instruction})

    plan = replay_plan(context.messages)
    assert plan.say is None and plan.instruction == instruction  # the last message, word for word
    assert plan.history.endswith("agent: Hi, you've reached the line. Who am I speaking with?\ncaller: Sam. Tell me about the owner.")

    # A context with no instruction last (a recovery in the middle of a turn) gets the resume instruction.
    assert replay_plan([{"role": "user", "content": "Hello?"}]).instruction == RESUME_INSTRUCTION


def test_tools_are_inline_websocket_tools_and_a_handoff_waits_for_the_sentence():
    tools = phonic_tools(lookup_tools())
    lookup, handoff = tools
    assert lookup == {
        "type": "custom_websocket",
        "tool_schema": {
            "type": "function",
            "function": {"name": "lookup_caller", "description": "Look up the caller.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        },
        "execution_mode": "sync",
        "tool_call_output_timeout_ms": 20000,
        "require_speech_before_tool_call": False,
    }
    assert handoff["wait_for_speech_before_tool_call"] is True and handoff["forbid_speech_after_tool_call"] is True
    assert phonic_tools(None) == []
    assert phonic_tool_output("Done, he called before.") == "Done, he called before."
    assert phonic_tool_output({"found": True}) == {"found": True} and phonic_tool_output(None) == "Done."


def test_a_refused_key_carries_the_mark_that_moves_the_call_at_once():
    from ai_call_center.voice.call_session import PERMANENT_ERROR_MARKS

    text, category = error_text({"message": "Invalid API key"})
    assert "invalid_api_key" in text and any(mark in text.lower() for mark in PERMANENT_ERROR_MARKS)
    assert category.name == "AUTHENTICATION"
    assert error_text({"message": "Something broke", "code": "boom"})[0] == "Phonic error boom: Something broke"


# --- the service ---


async def test_the_config_sent_at_connect():
    phonic = FakePhonic()
    svc = service(phonic, model="phonic-1", turn_taking="patient")
    async with running(svc) as run:
        assert phonic.sockets == []  # no conversation at setup: Phonic bills by the minute
        await legs.ensure_connected(svc)
        await run.settle()
        assert svc.connected and phonic.socket.kinds() == ["config"]
        config = phonic.socket.of("config")[0]
        assert {k: config[k] for k in ("system_prompt", "voice_id", "input_format", "output_format", "turn_taking", "model")} == {
            "system_prompt": "You are the classifier.",
            "voice_id": "sabrina",
            "input_format": "pcm_8000",
            "output_format": "pcm_8000",
            "turn_taking": "patient",
            "model": "phonic-1",
        }
        assert config["no_input_end_conversation_sec"] == 3600 and config["detect_voicemail_tone"] is False
        assert not any(key.startswith("no_input_poke") or key == "generate_no_input_poke_text" for key in config)
        assert [tool["tool_schema"]["function"]["name"] for tool in config["tools"]] == ["lookup_caller", "transfer_to_profile"]
        assert all(tool["type"] == "custom_websocket" for tool in config["tools"])  # no built-in Phonic tool
    assert phonic.socket.closed  # the end of the pipeline closes the conversation


async def test_an_empty_model_setting_leaves_the_model_to_phonic():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.ensure_connected(svc)
        await run.settle()
        assert "model" not in phonic.socket.of("config")[0]


async def test_a_pipeline_rate_with_a_phonic_format_is_used_and_one_without_is_an_error():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc, rate=16000) as run:
        await legs.ensure_connected(svc)
        await run.settle()
        assert phonic.socket.of("config")[0]["input_format"] == "pcm_16000"

    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc, rate=22050) as run:
        await run.settle()
        errors = run.up_of(ErrorFrame)
        assert errors and "no PCM format" in errors[0].error and "22050" in errors[0].error
        await legs.ensure_connected(svc)
        assert phonic.sockets == []  # it never opens a conversation it cannot hear


async def test_the_greeting_is_said_word_for_word_with_nothing_before_it():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        await run.settle()
        assert phonic.socket.kinds() == ["config", "say"]
        assert phonic.socket.of("say") == [{"type": "say", "text": "Hi, you've reached the line. Who am I speaking with?"}]


async def test_a_handoff_replay_sends_the_call_then_asks_with_the_instruction():
    phonic = FakePhonic()
    svc = service(phonic)
    context = LLMContext([{"role": "developer", "content": GREETING_DONE}, {"role": "user", "content": "Sam here."}, {"role": "developer", "content": "Say: Hi, I'm the classifier agent."}])
    async with running(svc) as run:
        await legs.ensure_connected(svc)
        await legs.adopt_context(svc, context)
        await legs.replay_and_respond(svc, context)
        await run.settle()
        assert phonic.socket.kinds() == ["config", "add_system_message", "generate_reply"]
        assert phonic.socket.of("add_system_message")[0]["system_message"].endswith(f"phone system: {GREETING_DONE}\ncaller: Sam here.")
        assert phonic.socket.of("generate_reply")[0]["system_message"] == "Say: Hi, I'm the classifier agent."


async def test_the_agent_speech_becomes_audio_and_text_frames_and_silence_does_not():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        await run.settle()
        run.take()
        socket = phonic.socket
        socket.feed("audio_chunk", audio=b64(SILENCE), text="")  # between turns: no frames, no BotStartedSpeaking
        socket.feed("assistant_started_speaking")
        socket.feed("audio_chunk", audio=b64(SILENCE), text="")  # the silent start of a turn
        socket.feed("audio_chunk", audio=b64(SPEECH), text="Hi, ")
        socket.feed("audio_chunk", audio=b64(SILENCE), text="")  # a pause inside the response keeps its time
        socket.feed("audio_chunk", audio=b64(SPEECH), text="Sam. ")
        socket.feed("assistant_finished_speaking")
        socket.feed("audio_chunk", audio=b64(SILENCE), text="")  # after the turn: nothing
        await run.settle()
        kinds = [type(f) for f in run.take()]
        assert kinds == [
            LLMFullResponseStartFrame,
            TTSStartedFrame,
            TTSAudioRawFrame,
            LLMTextFrame,
            TTSTextFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            LLMTextFrame,
            TTSTextFrame,
            TTSStoppedFrame,
            LLMFullResponseEndFrame,
        ]


async def test_text_frames_carry_the_words_with_their_spaces():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        socket = phonic.socket
        socket.feed("assistant_started_speaking")
        socket.feed("audio_chunk", audio=b64(SPEECH), text="Hi, ")
        socket.feed("audio_chunk", audio=b64(SPEECH), text="Sam.")
        await run.settle()
        llm = run.down_of(LLMTextFrame)
        tts = run.down_of(TTSTextFrame)
        assert [f.text for f in tts] == ["Hi, ", "Sam."] and all(f.includes_inter_frame_spaces for f in tts)
        assert [f.text for f in llm if not isinstance(f, TTSTextFrame)] == ["Hi, ", "Sam."]
        assert all(f.append_to_context is False for f in llm if not isinstance(f, TTSTextFrame))  # the TTS text goes into the context, once
        assert run.down_of(TTSAudioRawFrame)[0].sample_rate == 8000


async def test_the_caller_turn_and_words_go_up_to_the_aggregator():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        socket = phonic.socket
        socket.feed("user_started_speaking")
        socket.feed("input_text", text="Sam")
        socket.feed("input_text", text="here.")
        socket.feed("user_finished_speaking")
        await run.settle()
        assert [f.text for f in run.up_of(TranscriptionFrame)] == ["Sam", "here."]
        assert len(run.up_of(ProposedUserStartedSpeakingFrame)) == 1 and len(run.up_of(ProposedUserStoppedSpeakingFrame)) == 1
        assert run.down_of(TranscriptionFrame) == []


async def test_a_barge_in_stops_the_agent_audio_and_ends_the_response():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        socket = phonic.socket
        socket.feed("assistant_started_speaking")
        socket.feed("audio_chunk", audio=b64(SPEECH), text="Let ")
        await run.settle()
        run.take()
        # Measured: interrupted_response comes 1 ms before user_started_speaking, and no assistant_finished_speaking follows.
        socket.feed("interrupted_response", text="Let me tell you")
        socket.feed("user_started_speaking")
        socket.feed("audio_chunk", audio=b64(SPEECH), text="")  # late audio of the stopped turn is dropped
        await run.settle()
        kinds = [type(f) for f in run.take()]
        assert kinds[:3] == [InterruptionFrame, LLMFullResponseEndFrame, TTSStoppedFrame]
        assert TTSAudioRawFrame not in kinds
        assert len(run.up_of(InterruptionFrame)) == 1


async def test_the_caller_audio_goes_to_phonic_only_once_it_is_ready():
    phonic = FakePhonic(ready=False)
    svc = service(phonic)
    async with running(svc) as run:
        await legs.ensure_connected(svc)
        await run.worker.queue_frame(InputAudioRawFrame(audio=SPEECH, sample_rate=8000, num_channels=1))
        await run.settle()
        assert phonic.socket.kinds() == ["config"]
        phonic.socket.feed("ready_to_start_conversation")
        await run.settle()
        await run.worker.queue_frame(InputAudioRawFrame(audio=SPEECH, sample_rate=8000, num_channels=1))
        await run.settle()
        assert phonic.socket.of("audio") == [{"type": "audio_chunk", "audio": b64(SPEECH)}]


async def test_a_tool_call_runs_and_its_result_goes_back_to_phonic():
    phonic = FakePhonic()
    svc = service(phonic)
    asked = []

    async def lookup(params):
        asked.append(params.arguments)
        await params.result_callback("Quentin called once, about a bakery website.")

    svc.register_function("lookup_caller", lookup)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        phonic.socket.feed("tool_call", tool_call_id="call_1", tool_name="lookup_caller", parameters={"name": "Quentin"})
        await run.settle()
        assert asked == [{"name": "Quentin"}]
        assert phonic.socket.of("tool_call_output") == [{"type": "tool_call_output", "tool_call_id": "call_1", "output": "Quentin called once, about a bakery website."}]


async def test_a_handoff_result_is_not_sent_and_ends_the_agent_turn():
    """Phonic keeps the turn open until the tool returns (measured: no assistant_finished_speaking).
    A result with run_llm=False is for the next agent: it is not sent, and the turn ends here."""
    phonic = FakePhonic()
    svc = service(phonic)

    async def transfer(params):
        await params.result_callback("Handed off to the profile agent.", properties=FunctionCallResultProperties(run_llm=False))

    svc.register_function("transfer_to_profile", transfer)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        socket = phonic.socket
        socket.feed("assistant_started_speaking")
        socket.feed("audio_chunk", audio=b64(SPEECH), text="I'm bringing in the profile agent.")
        socket.feed("tool_call", tool_call_id="call_2", tool_name="transfer_to_profile", parameters={})
        await run.settle()
        assert socket.of("tool_call_output") == []
        frames = run.take()
        kinds = [type(f) for f in frames]
        assert kinds.index(TTSStoppedFrame) < kinds.index(LLMFullResponseEndFrame)
        assert kinds.count(LLMFullResponseEndFrame) == 1
        # The silence Phonic streams while it holds the turn makes no audio frames.
        socket.feed("audio_chunk", audio=b64(SILENCE), text="")
        await run.settle()
        assert run.down_of(TTSAudioRawFrame) == []


async def test_a_context_frame_after_the_switch_does_not_make_the_agent_speak_again():
    phonic = FakePhonic()
    svc = service(phonic)
    context = greeting_context()
    async with running(svc) as run:
        await legs.ensure_connected(svc)
        await legs.adopt_context(svc, context)
        await legs.replay_and_respond(svc, context)
        await run.worker.queue_frame(LLMContextFrame(context), FrameDirection.UPSTREAM)  # from the assistant aggregator
        await run.worker.queue_frame(LLMContextFrame(context))  # a user turn's frame, downstream
        await run.settle()
        assert phonic.socket.kinds() == ["config", "say"]


async def test_a_note_for_the_agent_makes_it_speak_once():
    """`tell_agent`: an appended developer message with run_llm=True. The assistant aggregator asks
    for the answer with a context frame; a second frame must not make a second answer."""
    phonic = FakePhonic()
    svc = service(phonic)
    context = greeting_context()
    note = {"role": "developer", "content": "The caller's transfer did not connect. Tell them."}
    async with running(svc) as run:
        await legs.replay_and_respond(svc, context)
        await run.worker.queue_frame(LLMMessagesAppendFrame(messages=[note], run_llm=True))
        await run.settle()
        assert phonic.socket.kinds() == ["config", "say"]  # nothing yet: the aggregator asks once the agent is silent
        context.add_message(note)
        await run.worker.queue_frame(LLMContextFrame(context), FrameDirection.UPSTREAM)
        await run.worker.queue_frame(LLMContextFrame(context), FrameDirection.UPSTREAM)
        await run.settle()
        assert phonic.socket.of("generate_reply") == [{"type": "generate_reply", "system_message": note["content"]}]


async def test_a_note_that_comes_while_the_agent_speaks_waits_for_the_end_of_its_turn():
    phonic = FakePhonic()
    svc = service(phonic)
    context = greeting_context()
    async with running(svc) as run:
        await legs.replay_and_respond(svc, context)
        phonic.socket.feed("assistant_started_speaking")
        await run.settle()
        await run.worker.queue_frame(LLMMessagesAppendFrame(messages=[{"role": "developer", "content": "Say the time."}], run_llm=True))
        await run.settle()  # on a call the append passes the service before the aggregator asks
        await run.worker.queue_frame(LLMContextFrame(context), FrameDirection.UPSTREAM)
        await run.settle()
        assert phonic.socket.of("generate_reply") == []
        phonic.socket.feed("assistant_finished_speaking")
        await run.settle()
        assert phonic.socket.of("generate_reply") == [{"type": "generate_reply", "system_message": "Say the time."}]


async def test_a_message_without_an_answer_goes_in_as_a_system_message():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        await run.worker.queue_frame(LLMMessagesAppendFrame(messages=[{"role": "developer", "content": "The caller's name is Sam."}], run_llm=False))
        await run.settle()
        assert phonic.socket.of("add_system_message") == [{"type": "add_system_message", "system_message": "The caller's name is Sam."}]
        assert phonic.socket.of("generate_reply") == []


async def test_a_reset_closes_the_conversation_and_the_next_use_opens_a_new_one():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        await run.worker.queue_frame(LLMMessagesAppendFrame(messages=[{"role": "developer", "content": "stale"}], run_llm=True))
        await run.settle()
        first = phonic.socket
        await legs.reset_session(svc)
        await run.settle()
        assert first.closed and not svc.connected
        assert run.up_of(ErrorFrame) == []  # a close we asked for is not an error

        await legs.ensure_connected(svc)
        context = LLMContext([{"role": "user", "content": "Back again."}, {"role": "developer", "content": "Say hello to the caller again."}])
        await legs.adopt_context(svc, context)
        await legs.replay_and_respond(svc, context)
        await run.worker.queue_frame(LLMContextFrame(context), FrameDirection.UPSTREAM)
        await run.settle()
        second = phonic.socket
        assert second is not first and svc.connected
        assert second.kinds() == ["config", "add_system_message", "generate_reply"]  # the stale note is gone
        assert svc.seconds_held > 0


async def test_a_close_we_did_not_ask_for_is_an_error_for_the_watchdog():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        phonic.socket.drop(1011, "internal error")
        await run.settle()
        errors = run.up_of(ErrorFrame)
        assert [e.error for e in errors] == ["Phonic closed the conversation (code 1011: internal error)"]
        assert not svc.connected
        await legs.ensure_connected(svc)  # the watchdog's recovery: a new conversation
        await run.settle()
        assert len(phonic.sockets) == 2 and svc.connected


async def test_a_refused_key_is_one_error_that_names_invalid_api_key_and_never_the_key():
    phonic = FakePhonic(ready=False)
    svc = service(phonic)
    async with running(svc) as run:
        await legs.ensure_connected(svc)
        # Measured with a made-up key: the error message, then a close with 4000 about 1 s later.
        phonic.socket.feed("error", error={"message": "Invalid API key"})
        phonic.socket.drop(4000, "Invalid API key")
        await run.settle()
        errors = run.up_of(ErrorFrame)
        assert len(errors) == 1 and "invalid_api_key" in errors[0].error and KEY not in errors[0].error
        assert "code 4000" in errors[0].error and errors[0].category == ErrorCategory.AUTHENTICATION


async def test_an_error_message_alone_is_no_failure_and_the_conversation_keeps_working():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        await run.settle()
        run.take()
        # Measured 2026-11-24: this error left the conversation open and working.
        phonic.socket.feed("error", error={"message": "System prompt can only be updated once per conversation"})
        phonic.socket.feed("assistant_started_speaking")
        phonic.socket.feed("audio_chunk", audio=b64(SPEECH), text="Hi.")
        await run.settle()
        assert run.up_of(ErrorFrame) == []
        assert svc.connected and len(phonic.sockets) == 1
        assert len(run.down_of(TTSAudioRawFrame)) == 1  # the later audio still plays


async def test_an_error_then_a_close_is_one_error_frame_with_both_texts():
    phonic = FakePhonic()
    svc = service(phonic)
    async with running(svc) as run:
        await legs.replay_and_respond(svc, greeting_context())
        phonic.socket.feed("error", error={"message": "Something broke", "code": "boom"})
        phonic.socket.drop(4501, "internal")
        await run.settle()
        errors = run.up_of(ErrorFrame)
        assert [e.error for e in errors] == ["Phonic closed the conversation (code 4501: internal); last error: Phonic error boom: Something broke"]
        assert errors[0].category == ErrorCategory.CONNECTIVITY  # from the close code, not from the error


async def test_a_close_while_opening_makes_wait_ready_false_with_one_report():
    phonic = FakePhonic(ready=False)
    svc = service(phonic)
    async with running(svc) as run:
        waiting = asyncio.ensure_future(svc.wait_ready())
        await run.settle()
        phonic.socket.drop(4102, "bad config")
        assert await asyncio.wait_for(waiting, 2) is False
        await run.settle()
        errors = run.up_of(ErrorFrame)
        assert [e.error for e in errors] == ["Phonic closed the conversation (code 4102: bad config)"]
        assert errors[0].category == ErrorCategory.INVALID_REQUEST


async def test_an_http_refusal_names_only_the_status():
    from phonic.core.api_error import ApiError

    def refuse():
        @asynccontextmanager
        async def conversation():
            raise ApiError(status_code=401, headers={"Authorization": f"Bearer {KEY}"}, body="no")
            yield  # pragma: no cover

        return conversation()

    svc = PhonicLLMService(api_key=KEY, instructions="x", voice="sabrina", connect=refuse)
    async with running(svc) as run:
        await legs.ensure_connected(svc)
        await run.settle()
        errors = run.up_of(ErrorFrame)
        assert [e.error for e in errors] == ["Phonic refused the connection with HTTP 401 (invalid_api_key)"]


async def test_a_failed_connect_makes_the_replay_give_up_without_hanging():
    phonic = FakePhonic(ready=False)
    svc = service(phonic)
    async with running(svc) as run:
        await legs.ensure_connected(svc)
        phonic.socket.drop(1006)
        await asyncio.wait_for(legs.replay_and_respond(svc, greeting_context()), 2)
        await run.settle()
        assert run.up_of(ErrorFrame) and "say" not in phonic.socket.kinds()


# --- legs.py ---


def settings(**over) -> Settings:
    base = {"openai_api_key": "sk-test", "google_api_key": "g-test", "nova_aws_access_key_id": "", "nova_aws_secret_access_key": "", "phonic_api_key": ""}
    return Settings(_env_file=None, **{**base, **over})


def test_the_classifier_runs_on_phonic_and_falls_back_to_openai_without_a_key(specs):
    classifier = specs["classifier"]
    assert (classifier.leg, classifier.voice) == ("phonic", "sabrina")
    assert legs.resolve_spec(classifier, settings(phonic_api_key=KEY)) is classifier
    moved = legs.resolve_spec(classifier, settings())
    assert (moved.leg, moved.voice) == ("openai_realtime", "marin")
    assert legs.ready_legs(settings(phonic_api_key=KEY)) == ["openai_realtime", "gemini_live", "phonic"]
    assert legs.ready_legs(settings()) == ["openai_realtime", "gemini_live"]


def test_the_phonic_leg_is_built_from_the_settings_and_opens_nothing(specs):
    from ai_call_center.voice.tools import tools_for_agent, voice_tool_schemas

    spec = specs["classifier"]
    tools = tools_for_agent(spec, voice_tool_schemas(specs))
    svc = legs.build_leg(spec, instructions="You are the classifier.", settings=settings(phonic_api_key=KEY, phonic_turn_taking="relaxed"), tools=tools)  # gitleaks:allow - KEY is the made-up key at the top of this file
    assert isinstance(svc, PhonicLLMService) and not svc.connected
    assert svc._voice == "sabrina" and svc._turn_taking == "relaxed" and svc._model == ""
    names = [tool["tool_schema"]["function"]["name"] for tool in svc._phonic_tools]
    assert names == [schema.name for schema in tools.standard_tools] and "transfer_to_profile" in names


async def test_phonic_takes_no_removable_note_and_heals_by_the_watchdog():
    phonic = FakePhonic()
    svc = service(phonic)
    assert not legs.self_heals(svc) and not legs.first_answer_needs_run_frame(svc)
    assert await legs.tell_now(svc, "A note.") is None and phonic.sockets == []
    assert legs.session_clock(svc)() == 0.0


def test_agent_legs_can_name_phonic():
    assert parse_agent_legs("classifier=openai_realtime, feedback=phonic") == {"classifier": "openai_realtime", "feedback": "phonic"}
    with pytest.raises(ValueError, match="phonic"):
        parse_agent_legs("tour=phonik")
