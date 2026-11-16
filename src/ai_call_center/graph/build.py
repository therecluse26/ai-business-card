"""Build the call graph: one agent loop whose prompt and tools follow the active agent.

Every run enters at the model node. Middleware reads `active_agent` from the
checkpointed state and swaps the system prompt and the allowed tools. A handoff
tool changes `active_agent`; the loop then calls the model again as the new agent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from ai_call_center.agents.specs import AgentSpec, LEGS, build_specs, load_profile, parse_agent_legs
from ai_call_center.config import get_settings
from ai_call_center.graph.handoffs import handoff_tool_name, make_handoff_tool
from ai_call_center.graph.state import CallContext, CallState
from ai_call_center.graph.tools import confirm_returning_caller, end_call, get_call_stats, request_owner_transfer, save_lead, text_portfolio_link

ENTRY_AGENT = "classifier"


def all_tools(specs: dict[str, AgentSpec]) -> dict[str, BaseTool]:
    tools: list[BaseTool] = [save_lead, request_owner_transfer, get_call_stats, text_portfolio_link, confirm_returning_caller, end_call]
    tools += [make_handoff_tool(spec) for spec in specs.values()]
    return {t.name: t for t in tools}


def tools_for(spec: AgentSpec, tools: dict[str, BaseTool]) -> list[BaseTool]:
    names = [*spec.tools, *(handoff_tool_name(h) for h in spec.handoffs)]
    return [tools[n] for n in names]


def mode_line(spec: AgentSpec, ctx: CallContext | None) -> str:
    leg = (ctx.legs.get(spec.name) if ctx else None) or spec.leg
    leg_label = LEGS.get(leg, leg)
    if ctx is not None and ctx.mode == "voice":
        return f"In this call you run on {leg_label}."
    return f"This conversation is in text mode for testing. On a phone call you run on {leg_label}."


def build_graph(
    model: BaseChatModel,
    specs: dict[str, AgentSpec],
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    tools = all_tools(specs)

    @wrap_model_call
    async def route_to_active_agent(request: ModelRequest, handler: Callable) -> Any:
        name = request.state.get("active_agent") or ENTRY_AGENT
        spec = specs[name]
        ctx = request.runtime.context if request.runtime is not None else None
        prompt = f"{spec.instructions}\n\n{mode_line(spec, ctx)}"
        if ctx is not None and ctx.caller_lookup:
            prompt = f"{prompt}\n\n{ctx.caller_lookup}"
        # Only the greeting agent learns that the number has called before, and only a first name.
        if ctx is not None and ctx.caller_greeting and name == ENTRY_AGENT:
            prompt = f"{prompt}\n\n{ctx.caller_greeting}"
        return await handler(request.override(system_message=SystemMessage(content=prompt), tools=tools_for(spec, tools)))

    return create_agent(
        model,
        tools=list(tools.values()),
        state_schema=CallState,
        context_schema=CallContext,
        middleware=[route_to_active_agent],
        checkpointer=checkpointer,
        name="voice-business-card",
    )


def default_specs() -> dict[str, AgentSpec]:
    settings = get_settings()
    return build_specs(
        load_profile(settings.profile_path),
        owner_name=settings.owner_name,
        texting=settings.sms_enabled and bool(settings.portfolio_url),
        emailing=bool(settings.owner_email and settings.smtp_host),
        legs=parse_agent_legs(settings.agent_legs),
    )


# Claude binds each thinking block to the system prompt and the tool list that produced it.
# A handoff changes both on purpose, so the blocks from before the handoff no longer match.
# With this setting the API drops those blocks and continues, instead of returning a 400.
ANTHROPIC_MODEL_KWARGS: dict[str, Any] = {
    "thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
    "betas": ["thinking-binding-controls-2026-08-01"],
}


def default_model() -> BaseChatModel:
    name = get_settings().llm_model
    kwargs = ANTHROPIC_MODEL_KWARGS if name.startswith("anthropic") else {}
    return init_chat_model(name, **kwargs)
