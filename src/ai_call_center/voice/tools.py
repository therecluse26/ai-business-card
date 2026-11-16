"""Tool schemas for the speech-to-speech session.

The voice model only requests a tool. The handlers in `call_session.py` run the same code as
the LangGraph tools, so a lead or a transfer behaves the same in text and voice mode.
"""

from __future__ import annotations

from langchain.tools import BaseTool
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema

from ai_call_center.agents.specs import AgentSpec
from ai_call_center.graph.build import all_tools
from ai_call_center.graph.handoffs import handoff_tool_name


def function_schema(tool: BaseTool) -> FunctionSchema:
    """One Pipecat FunctionSchema from a LangChain tool: same name, description, and arguments."""
    schema = tool.tool_call_schema.model_json_schema()
    properties = {k: {kk: vv for kk, vv in v.items() if kk != "title"} for k, v in schema.get("properties", {}).items()}
    return FunctionSchema(name=tool.name, description=tool.description, properties=properties, required=schema.get("required", []))


def voice_tool_schemas(specs: dict[str, AgentSpec]) -> dict[str, FunctionSchema]:
    return {name: function_schema(t) for name, t in all_tools(specs).items()}


def tools_for_agent(spec: AgentSpec, schemas: dict[str, FunctionSchema]) -> ToolsSchema:
    names = [*spec.tools, *(handoff_tool_name(h) for h in spec.handoffs)]
    return ToolsSchema(standard_tools=[schemas[n] for n in names])
