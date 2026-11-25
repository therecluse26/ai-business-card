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
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from ai_call_center.agents.specs import (
    LEGS,
    AgentSpec,
    build_specs,
    load_profile,
    parse_agent_legs,
    web_address,
)
from ai_call_center.config import get_settings
from ai_call_center.crm import crm_configured
from ai_call_center.graph.handoffs import handoff_tool_name, make_handoff_tool
from ai_call_center.graph.state import CallContext, CallState
from ai_call_center.graph.tools import (
    crm_lookup,
    end_call,
    file_feedback,
    forget_caller,
    get_call_stats,
    identify_caller,
    set_passphrase,
    request_owner_transfer,
    save_lead,
    verify_passphrase,
    search_writing,
    text_portfolio_link,
)
from ai_call_center.tracing import tracing_enabled
from ai_call_center.writing import about_line, default_library

ENTRY_AGENT = "classifier"


def all_tools(specs: dict[str, AgentSpec]) -> dict[str, BaseTool]:
    tools: list[BaseTool] = [save_lead, request_owner_transfer, get_call_stats, search_writing, text_portfolio_link, identify_caller, verify_passphrase, set_passphrase, file_feedback, crm_lookup, forget_caller, end_call]
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


# Prompt caching: a request that starts with the same text as an earlier request reads that part from the cache.
# A cache read costs about a tenth of the input price, and a cache write about 1.25 times the input price.
# A cache entry lives for 5 minutes after the last request that used it.
CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


def caches_prompts(model: Any) -> bool:
    """True for a Claude model. Only a Claude model gets cache markers. Another model gets the same request as before."""
    return isinstance(model, ChatAnthropic)


def system_message(spec: AgentSpec, ctx: CallContext | None, *, cache: bool = False) -> SystemMessage:
    """The system prompt of one agent: its instructions, the mode line, and the caller lookup of the call.

    With `cache`, the prompt is two text blocks. The first block holds the part that is the same for each
    call, and it has the cache marker. The tools come before the system prompt, so the marker covers them too.
    The second block holds the caller lookup, which is different for each call. Joined, the blocks are the
    same text as the plain prompt."""
    stable = f"{spec.instructions}\n\n{mode_line(spec, ctx)}"
    per_call = f"\n\n{ctx.caller_lookup}" if ctx is not None and ctx.caller_lookup else ""
    if not cache:
        return SystemMessage(content=stable + per_call)
    blocks: list[str | dict[str, Any]] = [{"type": "text", "text": stable, "cache_control": CACHE_CONTROL}]
    if per_call:
        blocks.append({"type": "text", "text": per_call})
    return SystemMessage(content=blocks)


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
        cache = caches_prompts(request.model)
        changes: dict[str, Any] = {"system_message": system_message(spec, ctx, cache=cache), "tools": tools_for(spec, tools)}
        if cache:
            # A second cache marker, which the API puts on the last block of the conversation.
            # So the next request reads the turns before it from the cache. The marker adds no text.
            changes["model_settings"] = {**request.model_settings, "cache_control": CACHE_CONTROL}
        return await handler(request.override(**changes))

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
        brand_name=settings.brand_name,
        name_hint=settings.owner_name_pronunciation,
        writing_about=about_line(default_library(), settings.owner_name),
        website=web_address(settings.website_url, settings.brand_name),
        texting=settings.sms_enabled and bool(settings.portfolio_url),
        emailing=bool(settings.owner_email and settings.smtp_host),
        legs=parse_agent_legs(settings.agent_legs),
        tracing=tracing_enabled(settings),
        crm=crm_configured(settings),
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
