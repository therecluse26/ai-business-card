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
import json
import logging
import time
from dataclasses import dataclass, field

from fastapi import WebSocket
from langchain.chat_models import init_chat_model
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
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
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from ai_call_center.agents.specs import LEG_PLATFORMS, LEGS, VOICE_LEVELS_DBFS, AgentSpec
from ai_call_center.callers import UNKNOWN_NUMBERS, greeting_note, history_text, lookup_text, recall, summarize_call
from ai_call_center.config import Settings
from ai_call_center.graph.build import ENTRY_AGENT, default_specs
from ai_call_center.graph.handoffs import handoff_tool_name
from ai_call_center.graph.state import CallContext
from ai_call_center.graph.tools import (
    describe_call,
    note_caller_before_transfer,
    release_caller_history,
    save_lead_record,
    text_owner_before_transfer,
    text_portfolio_link_to_caller,
    transfer_closed_message,
)
from ai_call_center.guardrails import HIDDEN_NUMBER, TransferHours
from ai_call_center.leads import LeadStore
from ai_call_center.notify import notifier_from_settings
from ai_call_center.telephony.lookup import CallerLookup
from ai_call_center.telephony.telnyx_client import TelnyxTelephony
from ai_call_center.telephony.transfers import Snapshot, TransferManager
from ai_call_center.voice.leveler import VoiceLeveler
from ai_call_center.voice.legs import (
    adopt_context,
    build_leg,
    ensure_connected,
    first_answer_needs_run_frame,
    replay_and_respond,
    reset_session,
    resolve_spec,
    self_heals,
)
from ai_call_center.voice.tools import tools_for_agent, voice_tool_schemas

log = logging.getLogger(__name__)

TELNYX_SAMPLE_RATE = 8000
TRANSFER_FALLBACK_SECS = 4.0  # transfer anyway if the agent never says its goodbye line
SWITCH_TIMEOUT_SECS = 2.0


def voice_instructions(spec: AgentSpec, leg: str, *notes: str) -> str:
    """The agent's instructions for this call, followed by the notes that apply to it (empty notes are dropped)."""
    return "\n\n".join([f"{spec.instructions}\n\nIn this call you run on {LEGS.get(leg, leg)}.", *(n for n in notes if n)])


def handoff_instruction(target: str, platform: str) -> str:
    """What the new agent reads last when it takes over the call."""
    return (
        f"The call is now with the {target} agent. You are the {target} agent, with a new voice. Speak now. "
        f'Your first sentence must be exactly: "Hi, I\'m the {target} agent, running on {platform}." '
        "Then answer what the caller last asked. Do not greet again."
    )


def connect_message(returning: bool) -> str:
    if returning:
        return (
            "The call has just connected. Your instructions end with a returning caller note. Greet the caller now as that note says: "
            "first say you are the AI assistant on this line, then ask if you are speaking with that first name. Say nothing else."
        )
    return "The call has just connected. Greet the caller now."


TRANSFER_DID_NOT_CONNECT = "The transfer was attempted. It did not connect."


def after_transfer_attempt(messages: list) -> list:
    """The conversation as the contact agent reads it when the call comes back to it.

    The transfer tool's result told the agent to say a goodbye line. That instruction is now
    wrong, so it is replaced with what happened.
    """
    transfer_calls = {
        call.get("id")
        for m in messages
        if isinstance(m, dict)
        for call in (m.get("tool_calls") or [])
        if (call.get("function") or {}).get("name") == "request_owner_transfer"
    }
    return [
        {**m, "content": json.dumps(TRANSFER_DID_NOT_CONNECT)} if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id") in transfer_calls else m
        for m in messages
    ]


def resume_message(reason: str, owner: str, lead_saved: bool) -> str:
    """What the contact agent reads when the call comes back to it after a transfer that did not connect."""
    what = f"{owner} did not pick up" if reason == "no_answer" else f"the call to {owner} could not be connected"
    if lead_saved:
        follow = f'Then say to the caller: "Your details are saved, and {owner} will call you back." Then ask if there is anything else you can help with.'
    else:
        follow = f"Then offer to take their name and a message for {owner}."
    return (
        f"The live transfer did not go through: {what}. The caller is back on the line with you, the contact agent. Speak to the caller now. "
        f"Say in one sentence that {what}. {follow} "
        "Do not introduce yourself again. Do not offer another transfer, and do not call save_lead for details that are already saved. "
        "Try the transfer again only if the caller asks for it."
    )


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
    replays the context and asks for a response. A Gemini service reconnects by itself, so the
    watchdog leaves it alone.
    """

    def __init__(self, legs: dict[str, VoiceLeg], switcher: LLMSwitcher, context: LLMContext, *, max_recoveries: int = 3) -> None:
        super().__init__()
        self._legs = legs
        self._switcher = switcher
        self._context = context
        self._max = max_recoveries
        self._count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame) and direction == FrameDirection.UPSTREAM and not frame.fatal:
            vl = next((vl for vl in self._legs.values() if vl.service is frame.processor), None)
            if vl is not None and not self_heals(vl.service):
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
                await reset_session(vl.service)
                if active:
                    await replay_and_respond(vl.service, self._context)
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
    lookups: CallerLookup | None = None
    transfers: TransferManager | None = None
    resume: Snapshot | None = None  # set when the stream restarts after a transfer that did not connect

    @property
    def call_control_id(self) -> str:
        return self.call_data["call_id"]

    async def run(self) -> None:
        started = time.monotonic()
        resume = self.resume
        # A new call starts with the classifier. A call that comes back from a transfer starts with the contact agent.
        entry_agent = "contact" if resume else ENTRY_AGENT
        state = VoiceCallState(active_agent=entry_agent, handoffs=list(resume.handoffs) if resume else [], lead_saved=bool(resume and resume.lead_saved))
        transfers = self.transfers or TransferManager(self.telephony, stream_url=self.settings.media_stream_url, ring_secs=self.settings.transfer_ring_secs)
        caller_number = self.call_data.get("from") or "unknown"
        store = LeadStore(self.settings.leads_db)
        # Hidden numbers share one log key, so the per-number call limit covers them too.
        store.start_call(self.call_control_id, HIDDEN_NUMBER if caller_number in UNKNOWN_NUMBERS else caller_number)
        if resume:
            store.reopen_call(self.call_control_id)
        # Returning-caller memory: look the number up before any agent speaks. The greeting agent
        # gets a first name to confirm. The history stays here until the caller says yes.
        memory = recall(store, caller_number, exclude_call=self.call_control_id)
        greeting = greeting_note(memory) if memory and not resume else ""
        history = history_text(memory) if memory and memory.first_name else ""
        # Number lookup started at call.initiated; take the result if it is ready.
        info = await self.lookups.get(caller_number) if self.lookups else None
        lookup_block = lookup_text(info) if info is not None else ""
        if lookup_block:
            log.info("call %s: number lookup: line=%s carrier=%s area=%s name known: %s", self.call_control_id, info.line_type, info.carrier, info.state, bool(info.caller_name))
        if memory:
            log.info("call %s: returning caller (%d earlier calls, name known: %s)", self.call_control_id, memory.previous_calls, bool(memory.name))
        # Each spec as it will run here: a leg with no API key falls back to OpenAI, voice included.
        self.specs = {name: resolve_spec(spec, self.settings) for name, spec in self.specs.items()}
        schemas = voice_tool_schemas(self.specs)
        legs: dict[str, VoiceLeg] = {}
        for name, spec in self.specs.items():
            service = build_leg(
                spec,
                instructions=voice_instructions(spec, spec.leg, lookup_block, greeting if name == ENTRY_AGENT else ""),
                settings=self.settings,
                tools=tools_for_agent(spec, schemas),
            )
            legs[name] = VoiceLeg(name, spec.leg, service)
        ctx = CallContext(
            call_control_id=self.call_control_id,
            caller_number=caller_number,
            owner_phone=self.settings.owner_phone,
            leads=store,
            telephony=self.telephony,
            notifier=notifier_from_settings(self.settings),
            legs={name: vl.leg for name, vl in legs.items()},
            mode="voice",
            caller_greeting=greeting,
            caller_history=history,
            identity_confirmed=bool(resume and resume.identity_confirmed),
            caller_lookup=lookup_block,
            caller_name=resume.caller_name if resume else "",
            transfer_unavailable=transfers.unavailable,
            caller_line_type=info.line_type if info else "",
            texter=self.telephony if self.settings.sms_enabled else None,
            portfolio_url=self.settings.portfolio_url,
            transfer_hours=TransferHours.parse(self.settings.transfer_hours, self.settings.transfer_days),
        )
        entry = self.specs[entry_agent]

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

        if resume:
            first_messages = [*after_transfer_attempt(resume.messages), {"role": "developer", "content": resume_message(resume.reason, self.settings.owner_name, state.lead_saved)}]
        else:
            first_messages = [{"role": "developer", "content": connect_message(bool(greeting))}]
        context = LLMContext(messages=first_messages, tools=tools_for_agent(entry, schemas))
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
        # The first service in the list is active at start.
        switcher = LLMSwitcher(llms=[legs[entry_agent].service, *(vl.service for name, vl in legs.items() if name != entry_agent)])

        async def do_transfer() -> None:
            # The manager stops the media stream, which ends this session, and then transfers the
            # call. It keeps the conversation, so the contact agent can come back if the owner
            # does not pick up.
            log.info("call %s: transferring to the owner", self.call_control_id)
            transfers.start(
                self.call_control_id,
                to=self.settings.owner_phone,
                snapshot=Snapshot(
                    messages=list(context.messages),
                    handoffs=list(state.handoffs),
                    lead_saved=state.lead_saved,
                    identity_confirmed=ctx.identity_confirmed,
                    caller_name=ctx.caller_name,
                ),
            )

        gate = TransferGate(do_transfer)
        watchdog = ServiceWatchdog(legs, switcher, context)
        processors = [transport.input(), user_aggregator, watchdog, switcher]
        if self.settings.voice_level_dbfs:
            # Every voice at one loudness, so a handoff does not sound like a volume change.
            voice_by_service = {id(vl.service): self.specs[name].voice or "" for name, vl in legs.items()}
            processors.append(
                VoiceLeveler(
                    lambda: voice_by_service.get(id(switcher.strategy.active_service), ""),
                    target_dbfs=self.settings.voice_level_dbfs,
                    seeds=VOICE_LEVELS_DBFS,
                )
            )
        pipeline = Pipeline([*processors, transport.output(), gate, assistant_aggregator])

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
        background: set[asyncio.Task] = set()
        timers: list[asyncio.Task] = []

        async def on_save_lead(params: FunctionCallParams) -> None:
            args = {k: str(params.arguments.get(k) or "") for k in ("name", "need", "email", "company", "callback_number")}
            result, lead = await save_lead_record(ctx, **args)
            state.lead_saved = state.lead_saved or bool(lead)
            await params.result_callback(result)

        async def on_confirm_caller(params: FunctionCallParams) -> None:
            # The caller said yes to the first name. Only now do the notes from earlier calls
            # enter the conversation; from here every agent can see them in the shared context.
            if ctx.caller_history:
                log.info("call %s: caller confirmed their identity; history released", self.call_control_id)
                ctx.caller_name = memory.name if memory else ""
            await params.result_callback(release_caller_history(ctx))

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

        async def on_text_link(params: FunctionCallParams) -> None:
            await params.result_callback(await text_portfolio_link_to_caller(ctx, owner_name=self.settings.owner_name))

        async def on_transfer(params: FunctionCallParams) -> None:
            reason = str(params.arguments.get("reason") or "")
            # The owner gets a record of who called, even if the transfer is refused or not answered.
            if await note_caller_before_transfer(
                ctx, lead_saved=state.lead_saved, caller_name=str(params.arguments.get("caller_name") or ""), company=str(params.arguments.get("company") or ""), reason=reason
            ):
                state.lead_saved = True
            if closed := transfer_closed_message(ctx):
                log.info("call %s: transfer refused (%s)", self.call_control_id, ctx.transfer_unavailable or "outside the transfer hours")
                await params.result_callback(closed)
                return
            state.transfer_requested = True
            # The owner reads who is calling while the agent says its goodbye line.
            llm_task = asyncio.create_task(text_owner_before_transfer(ctx, str(params.arguments.get("reason", ""))))
            background.add(llm_task)
            llm_task.add_done_callback(background.discard)
            log.info("call %s: owner transfer requested (%s)", self.call_control_id, params.arguments.get("reason", ""))
            gate.arm()
            owner = self.settings.owner_name
            await params.result_callback(
                f'Say only this one line now: "Connecting you to {owner} now. Take care." Say nothing before it and nothing after it. '
                f"The call moves to {owner}'s phone the moment you stop speaking."
            )

        def make_handoff(target: str):
            async def on_handoff(params: FunctionCallParams) -> None:
                await self._handoff(target, params, legs=legs, switcher=switcher, worker=worker, context=context, state=state, schemas=schemas)

            return on_handoff

        for vl in legs.values():  # any service can become active, so each one knows every tool
            vl.service.register_function("save_lead", on_save_lead)
            vl.service.register_function("confirm_returning_caller", on_confirm_caller)
            vl.service.register_function("get_call_stats", on_call_stats)
            vl.service.register_function("request_owner_transfer", on_transfer)
            vl.service.register_function("text_portfolio_link", on_text_link)
            for name in self.specs:
                vl.service.register_function(handoff_tool_name(name), make_handoff(name))

        async def enforce_time_limit() -> None:
            """Warn one minute before the limit, ask for a goodbye 15 s before it, then end the call."""
            limit = self.settings.max_call_minutes * 60
            if limit <= 0:
                return

            async def tell_agent(text: str) -> None:
                # Into the switcher, not the head of the pipeline: the active voice service must see
                # the message first and put it in its session. The assistant aggregator then adds it
                # to the shared context and asks for an answer once the agent has stopped speaking.
                await switcher.queue_frame(LLMMessagesAppendFrame(messages=[{"role": "developer", "content": text}], run_llm=True))

            await asyncio.sleep(max(limit - 60, 0))
            log.info("call %s: one minute left", self.call_control_id)
            await tell_agent("About one minute is left on this call; the line has a time limit. Tell the caller in one sentence. If they want a follow-up, offer the contact agent.")
            await asyncio.sleep(min(45, limit))
            await tell_agent("The time limit is here. Say goodbye in one short sentence now, and say they can call again.")
            await asyncio.sleep(15)
            log.info("call %s: time limit reached; hanging up", self.call_control_id)
            try:
                await self.telephony.hangup(self.call_control_id)
            except Exception:
                log.exception("call %s: hang-up at the time limit failed", self.call_control_id)
            await runner.cancel()

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, _client) -> None:
            timers.append(asyncio.create_task(enforce_time_limit()))
            log.info("call %s connected after %.0f ms; agent=%s leg=%s voice=%s", self.call_control_id, (time.monotonic() - started) * 1000, entry.name, legs[entry_agent].leg, entry.voice)
            await _answer(legs[entry_agent], context, worker)

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, _client) -> None:
            log.info("call %s disconnected", self.call_control_id)
            await runner.cancel()

        await runner.add_workers(worker)
        try:
            await runner.run()
        finally:
            for timer in timers:
                timer.cancel()
            # The call is over. Write the short summary that the next call from this number will see.
            try:
                summary, caller_name = await summarize_call(context.messages, init_chat_model(self.settings.summary_model))
                store.finish_call(self.call_control_id, agents=[ENTRY_AGENT, *state.handoffs], summary=summary, caller_name=caller_name)
                log.info("call %s: summary saved (%d chars)", self.call_control_id, len(summary))
            except Exception:
                log.exception("call %s: could not save the call summary", self.call_control_id)
            store.close()
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
            # The last message of a replayed context decides what the new session does with it.
            # After a bare tool result, a session sometimes only says it is ready to continue.
            # A plain instruction as the last message makes it speak.
            context.add_message({"role": "developer", "content": handoff_instruction(target, platform)})
            async with nxt.lock:  # a background reset of this service may still be running
                await ensure_connected(nxt.service)
                await adopt_context(nxt.service, context)
                await worker.queue_frames([ManuallySwitchServiceFrame(service=nxt.service)])
                await _wait_for(lambda: switcher.strategy.active_service is nxt.service, SWITCH_TIMEOUT_SECS)
                await _answer(nxt, context, worker)
            log.info("call %s: handoff %s -> %s (leg=%s voice=%s) switched %.0f ms after the tool call", self.call_control_id, previous.agent, target, nxt.leg, spec.voice, (time.monotonic() - t0) * 1000)
            # The old service keeps a stale server-side conversation. Reset it in the background
            # (about 2.5 s) so a later handoff back to it replays the context cleanly.
            previous.service.create_task(_reset_in_background(previous))

        await params.result_callback(
            f"Handed off to the {target} agent.",
            properties=FunctionCallResultProperties(run_llm=False, on_context_updated=switch_when_context_has_result),
        )


async def _wait_for(condition, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            log.warning("service switch did not complete within %.1f s", timeout)
            return
        await asyncio.sleep(0.01)


async def _answer(vl: VoiceLeg, context: LLMContext, worker: PipelineWorker) -> None:
    """Have this agent's service answer from the shared context. The service must be the active one."""
    if not vl.used and first_answer_needs_run_frame(vl.service):
        # First use on OpenAI: the context frame from LLMRunFrame sets up the session and answers.
        await worker.queue_frames([LLMRunFrame()])
    else:
        # Its server-side session is fresh (never used, or reset when we left it): replay and answer.
        await replay_and_respond(vl.service, context)
    vl.used = True


async def _reset_in_background(vl: VoiceLeg) -> None:
    async with vl.lock:
        t0 = time.monotonic()
        try:
            await reset_session(vl.service)
            log.info("%s: session reset in %.0f ms, ready for a later handoff", vl.agent, (time.monotonic() - t0) * 1000)
        except Exception:
            log.exception("%s: background session reset failed", vl.agent)
