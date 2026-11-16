"""One live call: Telnyx WebSocket ⇄ Pipecat pipeline ⇄ voice legs.

The speech-to-speech model hears and speaks. It asks for tools; this module runs them.

Every agent has its own voice service, built at call start and connected before the greeting,
behind Pipecat's `LLMSwitcher`. A handoff switches the active service and asks the new one to
answer from the shared context, so the caller hears a new voice within about a second.
`save_lead` and `get_call_stats` call the same code as the LangGraph tools.
`request_owner_transfer` arms a gate that moves the call to the owner's phone once the agent
has finished its goodbye line.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from fastapi import WebSocket
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    FunctionCallResultProperties,
    LLMRunFrame,
    ManuallySwitchServiceFrame,
)
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from ai_call_center.agents.specs import LEG_PLATFORMS, LEGS, AgentSpec
from ai_call_center.config import Settings
from ai_call_center.graph.build import ENTRY_AGENT, default_specs
from ai_call_center.graph.handoffs import handoff_tool_name
from ai_call_center.graph.state import CallContext
from ai_call_center.graph.tools import describe_call, save_lead_record
from ai_call_center.leads import LeadStore
from ai_call_center.notify import notifier_from_settings
from ai_call_center.telephony.telnyx_client import TelnyxTelephony
from ai_call_center.voice.legs import available_leg, build_leg
from ai_call_center.voice.tools import tools_for_agent, voice_tool_schemas

log = logging.getLogger(__name__)

TELNYX_SAMPLE_RATE = 8000
TRANSFER_FALLBACK_SECS = 4.0  # transfer anyway if the agent never says its goodbye line
SWITCH_TIMEOUT_SECS = 2.0


def voice_instructions(spec: AgentSpec, leg: str) -> str:
    return f"{spec.instructions}\n\nIn this call you run on {LEGS.get(leg, leg)}."


@dataclass
class VoiceCallState:
    """What LangGraph's CallState holds in text mode, kept here for the voice session."""

    active_agent: str = ENTRY_AGENT
    handoffs: list[str] = field(default_factory=list)
    lead_saved: bool = False
    transfer_requested: bool = False


class TransferGate(FrameProcessor):
    """Sits after the transport output. When armed, it transfers the call as soon as the bot
    stops speaking, so the caller hears the whole goodbye line first."""

    def __init__(self, on_transfer) -> None:
        super().__init__()
        self._on_transfer = on_transfer
        self._armed = False
        self._fired = False
        self._speaking = False

    def arm(self) -> None:
        if self._armed:
            return
        self._armed = True
        self.create_task(self._fire_after(TRANSFER_FALLBACK_SECS))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking = False
            if self._armed:
                await self._fire()
        await self.push_frame(frame, direction)

    async def _fire_after(self, secs: float) -> None:
        await asyncio.sleep(secs)
        if self._speaking:  # the goodbye line is still playing; BotStoppedSpeakingFrame fires it
            return
        await self._fire()

    async def _fire(self) -> None:
        if self._fired:
            return
        self._fired = True
        await self._on_transfer()


class ServiceWatchdog(FrameProcessor):
    """Sits before the switcher and watches the error frames the voice services push upstream.

    Pipecat ends a realtime service's receive loop on any error event from OpenAI, even one
    that only rejects a single client event. Without this, the active session goes silent for
    the rest of the call. The watchdog reconnects the session; if it is the active one, it
    replays the context and asks for a response.
    """

    def __init__(self, legs: dict[str, VoiceLeg], switcher: LLMSwitcher, *, max_recoveries: int = 3) -> None:
        super().__init__()
        self._legs = legs
        self._switcher = switcher
        self._max = max_recoveries
        self._count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame) and direction == FrameDirection.UPSTREAM and not frame.fatal:
            vl = next((vl for vl in self._legs.values() if vl.service is frame.processor), None)
            if vl is not None:
                self.create_task(self._recover(vl, frame.error))
        await self.push_frame(frame, direction)

    async def _recover(self, vl: VoiceLeg, error: str) -> None:
        if self._count >= self._max:
            log.error("%s: not recovering again (%d recoveries already); error: %s", vl.agent, self._count, error)
            return
        self._count += 1
        t0 = time.monotonic()
        active = self._switcher.strategy.active_service is vl.service
        log.warning("%s: session error (%s); reconnecting%s", vl.agent, error[:200], " and replaying" if active else "")
        async with vl.lock:
            try:
                if isinstance(vl.service, OpenAIRealtimeLLMService):
                    await vl.service.reset_conversation()
                if active:
                    await _replay_and_respond(vl.service)
                log.info("%s: recovered in %.0f ms", vl.agent, (time.monotonic() - t0) * 1000)
            except Exception:
                log.exception("%s: recovery failed", vl.agent)


@dataclass
class VoiceLeg:
    """One agent's voice service and what the session knows about it."""

    agent: str
    leg: str
    service: LLMService
    used: bool = False  # it has answered in this call; its server-side session holds history
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class CallSession:
    websocket: WebSocket
    call_data: dict
    settings: Settings
    telephony: TelnyxTelephony
    specs: dict[str, AgentSpec] = field(default_factory=default_specs)

    @property
    def call_control_id(self) -> str:
        return self.call_data["call_id"]

    async def run(self) -> None:
        started = time.monotonic()
        state = VoiceCallState()
        legs: dict[str, VoiceLeg] = {}
        for name, spec in self.specs.items():
            leg = available_leg(spec.leg)
            legs[name] = VoiceLeg(name, leg, build_leg(leg, spec, instructions=voice_instructions(spec, leg), settings=self.settings))
        ctx = CallContext(
            call_control_id=self.call_control_id,
            caller_number=self.call_data.get("from") or "unknown",
            owner_phone=self.settings.owner_phone,
            leads=LeadStore(self.settings.leads_db),
            telephony=self.telephony,
            notifier=notifier_from_settings(self.settings),
            legs={name: vl.leg for name, vl in legs.items()},
            mode="voice",
        )
        entry = self.specs[ENTRY_AGENT]

        serializer = TelnyxFrameSerializer(
            stream_id=self.call_data["stream_id"],
            call_control_id=self.call_control_id,
            outbound_encoding=self.call_data.get("outbound_encoding") or "PCMU",
            inbound_encoding="PCMU",
            api_key=self.settings.telnyx_api_key.get_secret_value(),
            params=TelnyxFrameSerializer.InputParams(auto_hang_up=False),  # a transfer must not be followed by a hang-up
        )
        transport = FastAPIWebsocketTransport(
            websocket=self.websocket,
            params=FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer),
        )

        schemas = voice_tool_schemas(self.specs)
        context = LLMContext(
            messages=[{"role": "developer", "content": "The call has just connected. Greet the caller now."}],
            tools=tools_for_agent(entry, schemas),
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
        # The first service in the list is active at start.
        switcher = LLMSwitcher(llms=[legs[ENTRY_AGENT].service, *(vl.service for name, vl in legs.items() if name != ENTRY_AGENT)])

        async def do_transfer() -> None:
            log.info("call %s: transferring to the owner", self.call_control_id)
            try:
                await self.telephony.transfer_from_stream(self.call_control_id, to=self.settings.owner_phone)
            except Exception:
                log.exception("call %s: transfer failed", self.call_control_id)

        gate = TransferGate(do_transfer)
        watchdog = ServiceWatchdog(legs, switcher)
        pipeline = Pipeline([transport.input(), user_aggregator, watchdog, switcher, transport.output(), gate, assistant_aggregator])

        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=TELNYX_SAMPLE_RATE,
                audio_out_sample_rate=TELNYX_SAMPLE_RATE,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[MetricsLogObserver()],
            idle_timeout_secs=300,
        )
        runner = WorkerRunner(handle_sigint=False)

        # --- tool handlers: the voice model requests, this code acts ---

        async def on_save_lead(params: FunctionCallParams) -> None:
            args = {k: str(params.arguments.get(k, "")) for k in ("name", "email", "company", "need")}
            result, _lead = await save_lead_record(ctx, **args)
            state.lead_saved = True
            await params.result_callback(result)

        async def on_call_stats(params: FunctionCallParams) -> None:
            await params.result_callback(
                describe_call(
                    ctx,
                    visited=[ENTRY_AGENT, *state.handoffs],
                    active_agent=state.active_agent,
                    lead_saved=state.lead_saved,
                    transfer_requested=state.transfer_requested,
                )
            )

        async def on_transfer(params: FunctionCallParams) -> None:
            state.transfer_requested = True
            log.info("call %s: owner transfer requested (%s)", self.call_control_id, params.arguments.get("reason", ""))
            gate.arm()
            owner = self.settings.owner_name
            await params.result_callback(
                "The transfer is ready. Say one short goodbye line now, for example: "
                f'"Here is {owner}, take care." The call moves to {owner} the moment you stop speaking.'
            )

        def make_handoff(target: str):
            async def on_handoff(params: FunctionCallParams) -> None:
                await self._handoff(target, params, legs=legs, switcher=switcher, worker=worker, context=context, state=state, schemas=schemas)

            return on_handoff

        for vl in legs.values():  # any service can become active, so each one knows every tool
            vl.service.register_function("save_lead", on_save_lead)
            vl.service.register_function("get_call_stats", on_call_stats)
            vl.service.register_function("request_owner_transfer", on_transfer)
            for name in self.specs:
                vl.service.register_function(handoff_tool_name(name), make_handoff(name))

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, _client) -> None:
            log.info("call %s connected after %.0f ms; agent=%s leg=%s voice=%s", self.call_control_id, (time.monotonic() - started) * 1000, entry.name, legs[ENTRY_AGENT].leg, entry.voice)
            legs[ENTRY_AGENT].used = True
            await worker.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, _client) -> None:
            log.info("call %s disconnected", self.call_control_id)
            await runner.cancel()

        await runner.add_workers(worker)
        try:
            await runner.run()
        finally:
            ctx.leads.close()
        log.info(
            "call %s finished after %.0f s; agents: %s; lead=%s transfer=%s",
            self.call_control_id,
            time.monotonic() - started,
            " -> ".join([ENTRY_AGENT, *state.handoffs]),
            state.lead_saved,
            state.transfer_requested,
        )

    async def _handoff(
        self,
        target: str,
        params: FunctionCallParams,
        *,
        legs: dict[str, VoiceLeg],
        switcher: LLMSwitcher,
        worker: PipelineWorker,
        context: LLMContext,
        state: VoiceCallState,
        schemas: dict,
    ) -> None:
        """Switch the active voice service to the target agent and have it answer."""
        previous = legs[state.active_agent]
        nxt = legs[target]
        spec = self.specs[target]
        state.active_agent = target
        state.handoffs.append(target)
        t0 = time.monotonic()
        platform = LEG_PLATFORMS.get(nxt.leg, nxt.leg)

        async def switch_when_context_has_result() -> None:
            # Runs once the assistant aggregator has written the tool result into the shared
            # context. That happens only after the old agent's sentence has played out, because
            # the result frame rides the audio queue in the output transport. Switching earlier
            # made the new session replay a context without the result; when the result arrived
            # later it was sent to a session that never had that call id, which OpenAI rejects
            # and Pipecat treats as fatal.
            context.set_tools(tools_for_agent(spec, schemas))
            async with nxt.lock:  # a background reset of this service may still be running
                await _ensure_connected(nxt.service)
                await worker.queue_frames([ManuallySwitchServiceFrame(service=nxt.service)])
                await _wait_for(lambda: switcher.strategy.active_service is nxt.service, SWITCH_TIMEOUT_SECS)
                if nxt.used:
                    # Its server-side session was reset when we left it; replay the context and answer.
                    await _replay_and_respond(nxt.service)
                else:
                    # First use: the context frame from LLMRunFrame sets up the session and answers.
                    await worker.queue_frames([LLMRunFrame()])
                nxt.used = True
            log.info("call %s: handoff %s -> %s (leg=%s voice=%s) switched %.0f ms after the tool call", self.call_control_id, previous.agent, target, nxt.leg, spec.voice, (time.monotonic() - t0) * 1000)
            # The old service keeps a stale server-side conversation. Reset it in the background
            # (about 2.5 s) so a later handoff back to it replays the context cleanly.
            previous.service.create_task(_reset_in_background(previous))

        await params.result_callback(
            f"Handed off to the {target} agent. You are now the {target} agent, with a new voice. "
            f'Your first sentence must be exactly: "Hi, I\'m the {target} agent, running on {platform}." '
            "Then answer what the caller asked. Do not greet again.",
            properties=FunctionCallResultProperties(run_llm=False, on_context_updated=switch_when_context_has_result),
        )


async def _wait_for(condition, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            log.warning("service switch did not complete within %.1f s", timeout)
            return
        await asyncio.sleep(0.01)


async def _ensure_connected(service: LLMService) -> None:
    if isinstance(service, OpenAIRealtimeLLMService) and service._websocket is None:  # noqa: SLF001
        log.warning("%s was disconnected; reconnecting before the handoff", service.name)
        await service._connect()  # noqa: SLF001


async def _replay_and_respond(service: LLMService) -> None:
    """Ask a used service to replay the shared context into its fresh session and respond.

    Pipecat only creates a response from a context frame on first use. For a later use the
    service already holds the (shared) context object, so we ask for the response directly.
    """
    if isinstance(service, OpenAIRealtimeLLMService):
        await service._create_response()  # noqa: SLF001
    else:  # pragma: no cover - other legs arrive in M3
        raise NotImplementedError(f"replay for {type(service).__name__}")


async def _reset_in_background(vl: VoiceLeg) -> None:
    async with vl.lock:
        t0 = time.monotonic()
        try:
            if isinstance(vl.service, OpenAIRealtimeLLMService):
                await vl.service.reset_conversation()
            log.info("%s: session reset in %.0f ms, ready for a later handoff", vl.agent, (time.monotonic() - t0) * 1000)
        except Exception:
            log.exception("%s: background session reset failed", vl.agent)
