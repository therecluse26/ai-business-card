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
    assert set(model.bound_tool_names[-1]) == {"identify_caller", "verify_passphrase", "set_passphrase", "forget_caller", "end_call", *(handoff_tool_name(s) for s in SPECIALISTS)}


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


@pytest.mark.parametrize("source", [name for name in SPECIALISTS if name != "feedback"])
async def test_each_specialist_can_hand_a_report_about_the_line_to_the_feedback_agent(make_graph, ctx, config, specs, source):
    model = scripted(tool_call(handoff_tool_name("feedback")), "Hi, I'm the feedback agent, running on OpenAI. What happened?")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Your answer just cut off. That is a bug.")], "active_agent": source}, config=config, context=ctx)
    assert result["active_agent"] == "feedback"
    assert model.system_prompt_of_call(1).startswith(specs["feedback"].instructions)


async def test_the_feedback_agent_hands_the_call_back_to_the_classifier(make_graph, ctx, config, specs):
    model = scripted(tool_call(handoff_tool_name("classifier")), "Hi, I'm the classifier agent, running on OpenAI. What else can I do for you?")
    graph = make_graph(model)

    result = await graph.ainvoke({"messages": [HumanMessage("No more feedback. I have another question.")], "active_agent": "feedback"}, config=config, context=ctx)

    assert result["active_agent"] == "classifier" and result["handoffs"] == ["classifier"]
    assert set(model.bound_tool_names[0]) == {"file_feedback", "forget_caller", "end_call", *(handoff_tool_name(name) for name in ("classifier", "profile", "tour", "consultant", "contact"))}


async def test_text_mode_line_is_appended(make_graph, ctx, config):
    model = scripted("Hello.")
    graph = make_graph(model)

    await graph.ainvoke({"messages": [HumanMessage("Hi.")]}, config=config, context=ctx)

    assert "text mode for testing" in model.system_prompt_of_call(0)
    assert "On a phone call you run on Phonic phonic-1" in model.system_prompt_of_call(0)  # the classifier's default leg
