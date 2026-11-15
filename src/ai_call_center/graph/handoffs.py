"""Handoff tools: one transfer_to_<agent> tool per agent."""

from __future__ import annotations

from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ai_call_center.agents.specs import AgentSpec
from ai_call_center.graph.state import CallContext, CallState


def handoff_tool_name(agent_name: str) -> str:
    return f"transfer_to_{agent_name}"


def make_handoff_tool(target: AgentSpec) -> BaseTool:
    name = handoff_tool_name(target.name)

    @tool(name, description=f"Hand the conversation to the {target.name} agent. Use when the caller {target.role}.")
    def handoff(runtime: ToolRuntime[CallContext, CallState]) -> Command:
        return Command(
            update={
                "active_agent": target.name,
                "handoffs": [target.name],
                "messages": [
                    ToolMessage(
                        content=f"Handed off to the {target.name} agent. You are now the {target.name} agent; continue the conversation.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return handoff
