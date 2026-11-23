"""Routing: the classifier hands off, the specialist keeps the next turn, tools stay scoped."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from ai_call_center.agents.specs import SPECIALISTS
from ai_call_center.graph.handoffs import handoff_tool_name
from tests.conftest import scripted, tool_call


@pytest.mark.parametrize("target", SPECIALISTS)
async def test_classifier_routes_to_each_specialist(make_graph, ctx, config, specs, target):
    model = scripted(tool_call(handoff_tool_name(target)), f"Hello from {target}.")
    graph = make_graph(model)

    result = await graph.ainvoke({"messages": [HumanMessage("Hi, I have a question.")]}, config=config, context=ctx)

    assert result["active_agent"] == target
    assert result["handoffs"] == [target]
    assert result["messages"][-1].content == f"Hello from {target}."
    # The first model call ran as the classifier; the second as the target.
    assert model.system_prompt_of_call(0).startswith(specs["classifier"].instructions)
    assert model.system_prompt_of_call(1).startswith(specs[target].instructions)


async def test_second_turn_goes_straight_to_the_specialist(make_graph, ctx, config, specs):
    model = scripted(tool_call(handoff_tool_name("profile")), "I can tell you about the owner.", "They build AI systems.")
    graph = make_graph(model)

    await graph.ainvoke({"messages": [HumanMessage("Who is the owner?")]}, config=config, context=ctx)
    result = await graph.ainvoke({"messages": [HumanMessage("What do they do?")]}, config=config, context=ctx)

    assert result["active_agent"] == "profile"
    assert len(model.calls) == 3
    assert model.system_prompt_of_call(2).startswith(specs["profile"].instructions)
    assert result["messages"][-1].content == "They build AI systems."


async def test_classifier_sees_only_its_handoff_tools(make_graph, ctx, config):
    model = scripted("Hello, what brings you to call today?")
    graph = make_graph(model)

    await graph.ainvoke({"messages": [HumanMessage("Hi.")]}, config=config, context=ctx)

    assert model.bound_tool_names, "the model was never bound to tools"
    assert set(model.bound_tool_names[-1]) == {"identify_caller", "verify_passphrase", "forget_caller", "end_call", *(handoff_tool_name(s) for s in SPECIALISTS)}


async def test_contact_agent_sees_its_tools_and_not_the_classifier(make_graph, ctx, config):
    model = scripted("What is your name?")
    graph = make_graph(model)

    await graph.ainvoke(
        {"messages": [HumanMessage("I want to leave my details.")], "active_agent": "contact"},
        config=config,
        context=ctx,
    )

    names = set(model.bound_tool_names[-1])
    assert {"save_lead", "request_owner_transfer"} <= names
    assert handoff_tool_name("classifier") not in names
    assert handoff_tool_name("contact") not in names


async def test_text_mode_line_is_appended(make_graph, ctx, config):
    model = scripted("Hello.")
    graph = make_graph(model)

    await graph.ainvoke({"messages": [HumanMessage("Hi.")]}, config=config, context=ctx)

    assert "text mode for testing" in model.system_prompt_of_call(0)
    assert "gpt-realtime-2.1-mini" in model.system_prompt_of_call(0)
