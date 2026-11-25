"""Phonic voice leg: Phonic's speech-to-speech WebSocket as a Pipecat service.

Pipecat has no Phonic service, so this module is one. Phonic keeps the conversation on its side,
like OpenAI Realtime, but it takes no replayed message list. So the call so far goes in as text
(`render_call_so_far`), and the last message of the shared context, the instruction to speak, goes
in as a request to speak (`replay_plan`).

The lifecycle is Nova's (see legs.py): no connection at setup, a new conversation when the agent
is needed (`connect`), and a close when the agent is left (`reset_conversation`). Phonic bills by
the minute of an open conversation, so a session that waits for a handoff would cost money.

Measured live on 2026-11-24 with phonic 0.33.10, model phonic-1, PCM at 8 kHz in and out:
- socket open 180-270 ms; connect start to `ready_to_start_conversation` 650-1080 ms.
- `say` and `generate_reply` both start the agent's speech 270-400 ms after the request.
- Phonic sends audio paced in real time, also between turns: 20 ms chunks of digital silence
  arrive every 20 ms. Only the chunks between `assistant_started_speaking` and the end of the
  turn are the agent's speech, so only those become audio frames here.
- Each `audio_chunk` that starts a word carries the word in `text`, with its spaces ("Hi, ").
- The caller's words come as one `input_text` message for each word.
- When the caller talks over the agent, `interrupted_response` comes 1 ms before
  `user_started_speaking`.
- A tool result goes in as a plain string or as a JSON object; Phonic used either.
- Phonic remembers what it said with `say`: asked "what did you just say", it quoted the greeting.
- A made-up key: the socket opens, Phonic sends `error` "Invalid API key", then closes with 4000.
- `update_system_prompt` works once per conversation: the second got the error "System prompt can
  only be updated once per conversation". So this leg takes no removable note (legs.tell_now).
- The classifier's real instructions (7,700 characters) and its 10 tools were accepted, and `say`
  spoke the greeting word for word.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from typing import Any

from phonic import AsyncPhonic
from phonic.core.api_error import ApiError
from phonic.types.add_system_message_payload import AddSystemMessagePayload
from phonic.types.audio_chunk_payload import AudioChunkPayload
from phonic.types.config_payload import ConfigPayload
from phonic.types.generate_reply_payload import GenerateReplyPayload
from phonic.types.say_payload import SayPayload
from phonic.types.tool_call_output_payload import ToolCallOutputPayload
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TextType,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from websockets.exceptions import ConnectionClosed

from ai_call_center.agents.specs import AgentSpec, default_voice
from ai_call_center.config import Settings
from ai_call_center.graph.handoffs import handoff_tool_name
from ai_call_center.voice.phone_messages import GREETING_DONE, greeting_of, without_call_so_far

log = logging.getLogger(__name__)

# Phonic's PCM formats. Each sample rate of the pipeline that has one needs no resampling.
PHONIC_FORMATS = {8000: "pcm_8000", 16000: "pcm_16000", 24000: "pcm_24000", 44100: "pcm_44100"}
CONNECT_TIMEOUT_SECS = 5.0
READY_TIMEOUT_SECS = 5.0  # measured: ready 0.65-1.1 s after the socket started to open
CLOSE_TIMEOUT_SECS = 2.0  # a reset holds the agent's lock; a slow close handshake must not hold it long
# A sync tool that gets no output in this time is given up by Phonic (its default is 5000 ms).
# Our slowest tools, a CRM read and the RAG search, can take seconds.
TOOL_OUTPUT_TIMEOUT_MS = 20_000
# Phonic ends a conversation after this much silence. A caller who thinks for a minute is still on the call.
NO_INPUT_END_SECS = 3600
HISTORY_MAX_CHARS = 24_000  # the newest part of a long call, if it is longer than this

HISTORY_HEAD = (
    "This is a message from the phone system, not from the caller. It is the call so far, oldest first, up to now. "
    '"caller" lines are the caller. "agent" lines are the agents of this phone line: you, or another agent before a handoff to you. '
    '"tool call" and "tool result" lines are tools that the agents used. "phone system" lines are notes that the agents got. '
    "You know all of this. Never ask the caller to say it again."
)
# The last message of a replayed context is normally an instruction. When it is not (a session that
# failed in the middle of the call and was opened again), the agent gets this one.
RESUME_INSTRUCTION = (
    "This is a message from the phone system, not from the caller. The line dropped for a moment and is back. "
    "Do not greet the caller again. If the caller's last words have no answer yet, answer them now. "
    "If they were answered, ask in one short sentence what else they want to know."
)


# --- the call so far, as text ---


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _tool_value(content: Any) -> str:
    """A tool result as the context holds it (JSON text), as one line: a sentence stays a sentence."""
    text = _text(content)
    try:
        value = json.loads(text)
    except ValueError:
        return _one_line(text)
    return _one_line(value) if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def history_lines(messages: list) -> list[str]:
    """One line for each turn, tool call, tool result, and phone-system note of the shared context."""
    lines: list[str] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue  # a provider-specific message: no other provider can read it
        role = message.get("role")
        text = _text(message.get("content"))
        if role == "user":
            line = f"caller: {_one_line(text)}"
        elif role == "assistant":
            if text.strip():
                lines.append(f"agent: {_one_line(text)}")
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                tool_names[call.get("id", "")] = function.get("name", "")
                lines.append(f"tool call: {function.get('name', '')}({_one_line(str(function.get('arguments') or ''))})")
            continue
        elif role == "tool":
            name = tool_names.get(message.get("tool_call_id", ""), "")
            line = f"tool result ({name}): {_tool_value(message.get('content'))}" if name else f"tool result: {_tool_value(message.get('content'))}"
        elif role in ("developer", "system"):
            # The order to say the greeting is done once the agent has said it, and the repeated
            # "call so far" block of a handoff instruction is in these lines already.
            note = GREETING_DONE if greeting_of(text) is not None else without_call_so_far(text)
            line = f"phone system: {_one_line(note)}"
        else:
            continue
        if line.split(":", 1)[1].strip():
            lines.append(line)
    return lines


def render_call_so_far(messages: list) -> str:
    """The call so far as one text for Phonic, or "" when there is nothing.

    OpenAI gets the shared context as one JSON text, and Gemini and Nova as replayed turns. Phonic
    takes neither, so the agent gets the call as lines of text, tool traffic included. A long call
    keeps its newest part: the text stays under `HISTORY_MAX_CHARS`.
    """
    lines = history_lines(messages)
    if not lines:
        return ""
    cut = False
    while len(lines) > 1 and sum(len(line) + 1 for line in lines) > HISTORY_MAX_CHARS:
        lines.pop(0)
        cut = True
    return HISTORY_HEAD + ("\n(The start of the call is left out.)" if cut else "") + "\n" + "\n".join(lines)


@dataclass(frozen=True)
class ReplayPlan:
    """What a fresh Phonic conversation gets: the call so far, then one request to speak."""

    history: str  # "" when there is nothing before the request
    say: str | None = None  # the exact greeting
    instruction: str | None = None  # an instruction to speak, as a system message


def replay_plan(messages: list) -> ReplayPlan:
    """Everything before the last message is the call so far. The last message is the request to speak.

    The greeting is said with `say`, which speaks exactly the given words. Any other instruction
    goes to the model, which speaks in its own words.
    """
    last = messages[-1] if messages else None
    if isinstance(last, dict) and last.get("role") in ("developer", "system"):
        text = _text(last.get("content"))
        history = render_call_so_far(messages[:-1])
        words = greeting_of(text)
        return ReplayPlan(history, say=words) if words is not None else ReplayPlan(history, instruction=text)
    return ReplayPlan(render_call_so_far(messages), instruction=RESUME_INSTRUCTION)


# --- tools ---


def phonic_tools(tools: ToolsSchema | None) -> list[dict]:
    """The agent's tools as Phonic inline WebSocket tools, run by this server.

    A handoff tool waits for the agent's sentence to end, and the agent says nothing after it:
    the next agent speaks. Measured 2026-11-24 without these two settings: the tool call came
    first and the sentence "I'm bringing in the profile agent" played after it, so it would have
    played over the next agent. With them, the call came 60 ms after the sentence ended.
    """
    handoff = handoff_tool_name("")
    out = []
    for schema in tools.standard_tools if tools else []:
        tool = {
            "type": "custom_websocket",
            "tool_schema": {"type": "function", "function": schema.to_default_dict()},
            "execution_mode": "sync",
            "tool_call_output_timeout_ms": TOOL_OUTPUT_TIMEOUT_MS,
            "require_speech_before_tool_call": False,  # the agents' instructions say when to speak first
        }
        if schema.name.startswith(handoff):
            tool |= {"wait_for_speech_before_tool_call": True, "forbid_speech_after_tool_call": True}
        out.append(tool)
    return out


def phonic_tool_output(result: Any) -> Any:
    """A tool result as Phonic's `output`. Our tools return a sentence, which goes as it is."""
    if result is None:
        return "Done."
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        return str(result)
    return result


def phonic_config(*, instructions: str, voice: str, audio_format: tuple[str, str], turn_taking: str, model: str = "", tools: list[dict] | None = None) -> ConfigPayload:
    """The config that opens a conversation. Phonic's own tools (keypad, transfer, end of call) are
    left out: our graph has its own. No-input pokes are off, and so is the voicemail check."""
    options: dict[str, Any] = {
        "system_prompt": instructions,
        "voice_id": voice,
        "input_format": audio_format[0],
        "output_format": audio_format[1],
        "turn_taking": turn_taking,
        "no_input_end_conversation_sec": NO_INPUT_END_SECS,
        "detect_voicemail_tone": False,
        "tools": tools or [],
    }
    if model:
        options["model"] = model
    return ConfigPayload(**options)


# --- errors ---


def close_category(code: int | None) -> ErrorCategory:
    """Phonic's close codes, from its documentation (read 2026-11-24)."""
    if code is None:
        return ErrorCategory.CONNECTIVITY
    if 4100 <= code <= 4199 or 4300 <= code <= 4399:
        return ErrorCategory.INVALID_REQUEST  # the config was refused: it will be refused again
    if code in (4004, 4010, 4429):
        return ErrorCategory.RATE_LIMIT
    return ErrorCategory.CONNECTIVITY


def error_text(error: dict) -> tuple[str, ErrorCategory]:
    """A Phonic `error` message as error text for the watchdog, and its category.

    A refused key must carry "invalid_api_key", one of `PERMANENT_ERROR_MARKS` in call_session:
    a reconnect cannot help, so the call moves to another provider at once.
    """
    message = str(error.get("message") or "no message")
    code = error.get("code")
    if "api key" in message.lower():
        return f"Phonic refused the API key (invalid_api_key): {message}", ErrorCategory.AUTHENTICATION
    return f"Phonic error{f' {code}' if code else ''}: {message}", ErrorCategory.UNKNOWN


def _as_dict(message: Any) -> dict | None:
    if isinstance(message, dict):
        return message
    if isinstance(message, bytes):
        return None  # Phonic sends JSON text only
    try:
        return message.dict()
    except AttributeError:
        return None


class PhonicLLMService(LLMService):
    """One Phonic conversation at a time, for one agent.

    - `connect` opens a conversation and sends the config. `reset_conversation` closes it.
    - `replay` gives a fresh conversation the call so far and asks the agent to speak.
    - An appended developer message with `run_llm=True` makes the agent speak about it once, when
      the assistant aggregator asks for an answer. Every other context frame changes nothing: Phonic
      answers the caller and its tool results by itself, so an answer here would be a second one.
    - A tool result marked `run_llm=False` (our handoff tools) is not sent: the next agent speaks.
    """

    def __init__(
        self,
        *,
        api_key: str,
        instructions: str,
        voice: str,
        tools: ToolsSchema | None = None,
        model: str = "",
        turn_taking: str = "standard",
        connect: Callable[[], AbstractAsyncContextManager] | None = None,
        **kwargs,
    ) -> None:
        settings = LLMSettings(
            model=model or None,
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)
        self._api_key = api_key
        self._instructions = instructions
        self._voice = voice
        self._tools = tools
        self._phonic_tools = phonic_tools(tools)
        self._model = model
        self._turn_taking = turn_taking
        self._connect_factory = connect or self._sdk_connect
        self._in_format = self._out_format = ""
        self._out_rate = 0
        self._format_error = ""
        self._stack: AsyncExitStack | None = None
        self._socket: Any = None
        self._receive_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._ended = asyncio.Event()  # the receive loop of the current socket is over
        self._closing = False  # a close we asked for: not an error
        self._failed = False  # an error of this conversation was reported already
        self._last_error: tuple[str, ErrorCategory] | None = None  # Phonic's last `error` message here
        self._speaking = False  # between assistant_started_speaking and the end of the turn
        self._responding = False  # LLMFullResponseStartFrame pushed, the end not yet
        self._owed: list[str] = []  # instructions to speak about at the next request for an answer
        self._owed_due = False  # the aggregator asked while the agent was speaking
        self._context: LLMContext | None = None
        self._done_calls: set[str] = set()  # tool calls whose result must not go to Phonic
        self._asked_at: float | None = None  # when the agent was last asked to speak, or the caller stopped
        self._connect_started: float | None = None
        self._opened_at: float | None = None
        self._held_secs = 0.0
        self.conversation_id = ""

    # --- Pipecat service ---

    def can_generate_metrics(self) -> bool:
        return True

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        # Phonic decides when the caller's turn starts and ends. The interruption comes from Phonic
        # too (`interrupted_response`), so it happens only when Phonic really stops the agent.
        return LLMServiceMetadataFrame(
            service_name=self.name, is_realtime_service=True, user_turn_strategies=ExternalUserTurnStrategies(enable_interruptions=False)
        )

    def _service_tools(self) -> ToolsSchema | None:
        return self._tools

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super().setup(setup)
        rate_in, rate_out = setup.audio_in_sample_rate, setup.audio_out_sample_rate
        self._in_format, self._out_format = PHONIC_FORMATS.get(rate_in, ""), PHONIC_FORMATS.get(rate_out, "")
        self._out_rate = rate_out
        if not (self._in_format and self._out_format):
            self._format_error = (
                f"Phonic has no PCM format for the pipeline's audio: {rate_in} Hz in, {rate_out} Hz out. "
                f"Phonic takes {', '.join(str(r) for r in PHONIC_FORMATS)} Hz."
            )

    async def start(self, frame) -> None:
        await super().start(frame)
        if self._format_error:
            await self.push_error(self._format_error, category=ErrorCategory.INVALID_REQUEST)

    async def stop(self, frame) -> None:
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame) -> None:
        await super().cancel(frame)
        await self._disconnect()

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self._send_audio(frame)
        elif isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context, direction)
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_append(frame)
        elif isinstance(frame, InterruptionFrame):
            await self.stop_all_metrics()
            await self._end_response(interrupted=True)
        await self.push_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        # The base service broadcasts tool results and cancellations; the downstream copy goes to Phonic.
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, FunctionCallResultFrame):
                await self._handle_tool_result(frame)
            elif isinstance(frame, FunctionCallCancelFrame):
                self._done_calls.add(frame.tool_call_id)
        await super().push_frame(frame, direction)

    # --- the conversation ---

    @property
    def connected(self) -> bool:
        return self._socket is not None and self._receive_task is not None and not self._receive_task.done()

    @property
    def seconds_held(self) -> float:
        """How long this service has held a Phonic conversation open: what Phonic bills."""
        return self._held_secs + (time.monotonic() - self._opened_at if self._opened_at is not None else 0.0)

    def _sdk_connect(self) -> AbstractAsyncContextManager:
        return AsyncPhonic(api_key=self._api_key).conversations.connect()

    def _config(self) -> ConfigPayload:
        return phonic_config(
            instructions=self._instructions,
            voice=self._voice,
            audio_format=(self._in_format, self._out_format),
            turn_taking=self._turn_taking,
            model=self._model,
            tools=self._phonic_tools,
        )

    async def connect(self) -> None:
        """Open a new Phonic conversation and send its config. Returns before Phonic is ready."""
        if self.connected:
            return
        if self._socket is not None:
            await self._disconnect()  # the last socket was lost
        if self._format_error:
            await self._report(self._format_error, ErrorCategory.INVALID_REQUEST)
            return
        self._closing = self._failed = False
        self._last_error = None
        self._ready.clear()
        self._ended.clear()
        t0 = self._connect_started = time.monotonic()
        stack = AsyncExitStack()
        try:
            socket = await asyncio.wait_for(stack.enter_async_context(self._connect_factory()), CONNECT_TIMEOUT_SECS)
        except ApiError as exc:
            # Never put this exception in a log or an error frame: its text holds the request
            # headers, and so the API key. Only the status code is used.
            category = ErrorCategory.AUTHENTICATION if exc.status_code in (401, 403) else ErrorCategory.CONNECTIVITY
            mark = " (invalid_api_key)" if exc.status_code in (401, 403) else ""
            await self._report(f"Phonic refused the connection with HTTP {exc.status_code}{mark}", category)
            return
        except Exception as exc:  # noqa: BLE001 - any failure to connect is reported the same way
            await self._report(f"Phonic: the connection failed ({type(exc).__name__}: {exc})", ErrorCategory.CONNECTIVITY)
            return
        self._stack, self._socket = stack, socket
        self._opened_at = time.monotonic()
        self._receive_task = self.create_task(self._receive(socket), name=f"{self}::phonic_receive")
        try:
            await socket.send_config(self._config())
        except Exception as exc:  # noqa: BLE001
            await self._report(f"Phonic: the config could not be sent ({type(exc).__name__}: {exc})", ErrorCategory.CONNECTIVITY)
            return
        log.info("%s: Phonic conversation opening (%.0f ms to the socket)", self, (time.monotonic() - t0) * 1000)

    async def reset_conversation(self) -> None:
        """Close the conversation. A later `connect` opens a new one, with no memory of this one."""
        await self._end_response()
        await self._disconnect(wait=False)

    async def _disconnect(self, *, wait: bool = True) -> None:
        """Forget the conversation at once. With `wait=False` the socket closes in the background:
        measured 2026-11-24, a close took about 1 s, and a reset holds the agent's lock."""
        self._closing = True
        task, self._receive_task = self._receive_task, None
        stack, self._stack = self._stack, None
        self._socket = None  # the receive loop of the old socket stops acting on its messages
        if task is not None or stack is not None:
            if wait:
                await self._close(task, stack)
            else:
                self.create_task(self._close(task, stack), name=f"{self}::phonic_close")
        if self._opened_at is not None:
            self._held_secs += time.monotonic() - self._opened_at
            self._opened_at = None
        self._ready.clear()
        self._ended.set()
        self._speaking = self._responding = self._owed_due = False
        self._owed.clear()
        self._done_calls.clear()
        self._asked_at = None

    async def _close(self, task: asyncio.Task | None, stack: AsyncExitStack | None) -> None:
        if task is not None:
            await self.cancel_task(task, timeout=1.0)
        if stack is not None:
            try:
                await asyncio.wait_for(stack.aclose(), CLOSE_TIMEOUT_SECS)
            except Exception:  # noqa: BLE001 - the socket is gone either way
                log.debug("%s: Phonic socket close did not complete", self)

    def _with_last_error(self, text: str, category: ErrorCategory) -> tuple[str, ErrorCategory]:
        """A failure's text with Phonic's last `error` message of this conversation. A refused key
        keeps its category, so the watchdog marks the provider down at once."""
        if self._last_error is None:
            return text, category
        error, error_category = self._last_error
        return f"{text}; last error: {error}", error_category if error_category == ErrorCategory.AUTHENTICATION else category

    async def _report_failure(self, text: str, category: ErrorCategory) -> None:
        await self._report(*self._with_last_error(text, category))

    async def _report(self, text: str, category: ErrorCategory) -> None:
        """One error frame, which the session's watchdog acts on. The key never goes into it."""
        self._failed = True
        if self._api_key:
            text = text.replace(self._api_key, "***")
        log.warning("%s: %s", self, text)
        await self.push_error(text, category=category)

    async def wait_ready(self) -> bool:
        """Connect if needed, and wait until Phonic takes audio and requests. False on a failure (reported)."""
        await self.connect()
        if not self.connected:
            return False
        ready, ended = asyncio.ensure_future(self._ready.wait()), asyncio.ensure_future(self._ended.wait())
        try:
            await asyncio.wait({ready, ended}, timeout=READY_TIMEOUT_SECS, return_when=asyncio.FIRST_COMPLETED)
        finally:
            ready.cancel()
            ended.cancel()
        if self._ready.is_set() and self.connected:
            return True
        if not self._failed:
            await self._report_failure(f"Phonic: the conversation was not ready after {READY_TIMEOUT_SECS:.0f} s", ErrorCategory.CONNECTIVITY)
        return False

    async def replay(self, context: LLMContext) -> None:
        """Give a fresh conversation the call so far, then have the agent speak. See `replay_plan`."""
        self._context = context
        self._owed.clear()
        self._owed_due = False
        if not await self.wait_ready():
            return
        plan = replay_plan(list(context.messages))
        if plan.history:
            await self._send("send_add_system_message", AddSystemMessagePayload(system_message=plan.history))
        await self._ask(say=plan.say, instruction=plan.instruction)

    async def adopt(self, context: LLMContext) -> None:
        """Become the active service. An instruction owed from an earlier time is stale now."""
        self._context = context
        self._owed.clear()
        self._owed_due = False

    async def _ask(self, *, say: str | None = None, instruction: str | None = None) -> None:
        self._asked_at = time.monotonic()
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        if say is not None:
            await self._send("send_say", SayPayload(text=say))
        else:
            await self._send("send_generate_reply", GenerateReplyPayload(system_message=instruction))

    async def _send_audio(self, frame: InputAudioRawFrame) -> None:
        if not (self.connected and self._ready.is_set()):
            return  # Phonic takes audio only once it is ready; the agent speaks first at every start
        await self._send("send_audio_chunk", AudioChunkPayload(audio=base64.b64encode(frame.audio).decode("ascii")))

    async def _send(self, method: str, payload: Any) -> None:
        """Send one message. A socket that closed meanwhile is not an error here: its receive loop reports the close."""
        socket = self._socket
        if socket is None:
            return
        try:
            await getattr(socket, method)(payload)
        except ConnectionClosed:
            log.debug("%s: Phonic socket closed before %s", self, method)

    # --- the shared context ---

    async def _handle_context(self, context: LLMContext, direction: FrameDirection) -> None:
        self._context = context
        # Only the assistant aggregator's request for an answer (upstream) may make the agent speak.
        # A user turn's context frame (downstream) comes while Phonic answers that turn itself.
        if direction != FrameDirection.UPSTREAM or not self._owed:
            return
        if self._speaking:
            self._owed_due = True  # after this turn ends; two requests at once would make two answers
            return
        await self._speak_owed()

    async def _speak_owed(self) -> None:
        owed, self._owed, self._owed_due = "\n\n".join(self._owed), [], False
        if owed and await self.wait_ready():
            await self._ask(instruction=owed)

    async def _handle_append(self, frame: LLMMessagesAppendFrame) -> None:
        texts = [t for t in (_text(m.get("content")) for m in frame.messages if isinstance(m, dict)) if t.strip()]
        if not texts:
            return
        if frame.run_llm:
            self._owed.extend(texts)
        elif self.connected and self._ready.is_set():
            await self._send("send_add_system_message", AddSystemMessagePayload(system_message="\n\n".join(texts)))

    # --- tools ---

    async def _handle_tool_result(self, frame: FunctionCallResultFrame) -> None:
        properties = frame.properties
        if properties is not None and not properties.is_final:
            return
        if frame.tool_call_id in self._done_calls:
            return
        self._done_calls.add(frame.tool_call_id)
        if properties is not None and properties.run_llm is False:
            # Phonic holds the agent's turn open until the tool returns, and sends no end of the
            # turn. End it here, so the sentence the agent said is stored as its own message.
            # Audio that still comes (the end of a goodbye before end_call) starts a new response.
            await self._end_response()
            return
        if not self.connected:
            return
        self._asked_at = time.monotonic()
        await self.start_ttfb_metrics()
        await self._send("send_tool_call_output", ToolCallOutputPayload(tool_call_id=frame.tool_call_id, output=phonic_tool_output(frame.result)))

    # --- what Phonic sends ---

    async def _receive(self, socket) -> None:
        # A socket that was left (a reset) may still deliver a message or its close: both are ignored.
        # A close we did not ask for is the one failure this loop reports: one error frame, with
        # Phonic's last `error` message of the conversation in it.
        failure: tuple[str, ErrorCategory] | None = None
        try:
            async for message in socket:
                data = _as_dict(message)
                if data is not None and socket is self._socket:
                    await self._handle_message(data)
            failure = ("Phonic closed the conversation", ErrorCategory.CONNECTIVITY)
        except ConnectionClosed as exc:
            close = exc.rcvd
            code, reason = (close.code, close.reason) if close is not None else (None, "")
            failure = (f"Phonic closed the conversation (code {code}{f': {reason}' if reason else ''})", close_category(code))
        except Exception as exc:  # noqa: BLE001
            failure = (f"Phonic: the conversation failed ({type(exc).__name__}: {exc})", ErrorCategory.UNKNOWN)
        finally:
            if socket is self._socket and not self._closing and not self._failed and failure is not None:
                await self._report_failure(*failure)
            if socket is self._socket:
                self._ended.set()  # after the report: `wait_ready` must see the failure as reported

    async def _handle_message(self, data: dict) -> None:
        kind = data.get("type")
        if kind == "audio_chunk":
            await self._handle_audio(data)
        elif kind == "assistant_started_speaking":
            self._speaking = True
        elif kind == "assistant_finished_speaking":
            self._speaking = False
            await self._end_response()
            if self._owed_due:
                await self._speak_owed()
        elif kind == "interrupted_response":
            if self._speaking or self._responding:
                self._speaking = False
                await self.broadcast_interruption()  # the output transport drops the agent audio it holds
                await self.stop_all_metrics()
                await self._end_response(interrupted=True)
        elif kind == "user_started_speaking":
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        elif kind == "user_finished_speaking":
            self._asked_at = time.monotonic()
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        elif kind == "input_text":
            text = str(data.get("text") or "")
            if text.strip():
                await self.push_frame(TranscriptionFrame(user_id="", timestamp=time_now_iso8601(), result=text, text=text), FrameDirection.UPSTREAM)
        elif kind == "tool_call":
            await self.run_function_calls(
                [FunctionCallFromLLM(function_name=str(data.get("tool_name")), tool_call_id=str(data.get("tool_call_id")), arguments=data.get("parameters") or {}, context=self._context)]
            )
        elif kind == "tool_call_interrupted":
            self._done_calls.add(str(data.get("tool_call_id")))  # Phonic dropped the call; an output now is for nothing
        elif kind == "ready_to_start_conversation":
            if self._connect_started is not None:
                log.info("%s: Phonic ready %.0f ms after the connect started", self, (time.monotonic() - self._connect_started) * 1000)
            self._ready.set()
        elif kind == "conversation_created":
            self.conversation_id = str(data.get("conversation_id") or "")
            log.info("%s: Phonic conversation %s", self, self.conversation_id)
        elif kind == "error":
            # Not a failure by itself. Measured 2026-11-24: a second update_system_prompt got the error
            # "System prompt can only be updated once per conversation", and the conversation went on
            # working. Phonic closes the socket on a failure (config errors 4100-4108 and 4300-4323,
            # its own failures 4501-4504 and 4206, a bad key 4000 after the error), and that close is
            # reported with this text. A conversation that stays open but silent after a request to
            # speak is caught by `expect_speech` in call_session.py (no audio within 8 s).
            text, category = error_text(data.get("error") or {})
            if self._api_key:
                text = text.replace(self._api_key, "***")
            self._last_error = (text, category)
            log.warning("%s: %s", self, text)
        elif kind == "warning":
            log.warning("%s: Phonic warning: %s", self, (data.get("warning") or {}).get("message"))
        elif kind == "assistant_chose_not_to_respond":
            await self.stop_processing_metrics()
        elif kind == "assistant_ended_conversation":
            log.warning("%s: Phonic ended the conversation by itself", self)

    async def _handle_audio(self, data: dict) -> None:
        if not self._speaking:
            return  # the digital silence that Phonic streams between turns
        audio = base64.b64decode(data.get("audio") or "")
        text = data.get("text")
        silent = audio == bytes(len(audio))
        if silent and not text and not self._responding:
            # Digital silence also comes inside a turn: before its first word, and for as long as
            # Phonic holds a turn open after a handoff tool (measured: 9 s). From a service that
            # is no longer active it would play beside the next agent's voice. A pause between
            # two words of one response stays: it keeps the agent's timing.
            return
        if not self._responding:
            self._responding = True
            await self.stop_ttfb_metrics()
            if self._asked_at is not None:
                log.info("%s: first audio %.0f ms after the request or the caller's last word", self, (time.monotonic() - self._asked_at) * 1000)
                self._asked_at = None
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TTSStartedFrame())
        if audio:
            await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=self._out_rate, num_channels=1))
        if text:
            llm_text = LLMTextFrame(text)
            llm_text.append_to_context = False  # the TTS text frame carries it into the context
            await self.push_frame(llm_text)
            tts_text = TTSTextFrame(text, text_type=TextType.WORD)
            tts_text.includes_inter_frame_spaces = True
            await self.push_frame(tts_text)

    async def _end_response(self, *, interrupted: bool = False) -> None:
        if not self._responding:
            return
        self._responding = False
        await self.stop_processing_metrics()
        if interrupted:
            await self.push_frame(LLMFullResponseEndFrame())
            await self.push_frame(TTSStoppedFrame())
        else:
            await self.push_frame(TTSStoppedFrame())
            await self.push_frame(LLMFullResponseEndFrame())


def build_phonic(spec: AgentSpec, *, instructions: str, settings: Settings, tools: ToolsSchema | None = None) -> LLMService:
    """Phonic over its speech-to-speech WebSocket. The tools are fixed when the conversation opens."""
    return PhonicLLMService(
        api_key=settings.phonic_api_key.get_secret_value(),
        instructions=instructions,
        voice=spec.voice or default_voice(spec.name, "phonic"),
        tools=tools,
        model=settings.phonic_model,
        turn_taking=settings.phonic_turn_taking,
    )
