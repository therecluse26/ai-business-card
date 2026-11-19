"""Jev reads a transfer request for its meaning; the word list is the fallback. No test uses the
network: a fake decider, or the real client on a mock transport."""

from __future__ import annotations

import asyncio
import json

import httpx2
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import SecretStr
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from ai_call_center.config import Settings
from ai_call_center.decisions import Decision, JevDecider, conversation_state, default_decider
from ai_call_center.graph.tools import ASK_WHICH, NOT_ASKED, asked_for_the_owner, describe_call, transfer_question
from tests.conftest import scripted, tool_call

MISSED = "Is he around? I'd love a quick word with him."  # a request for a live call that the word list does not hold


class FakeDecider:
    def __init__(self, probability: float | None) -> None:
        self.probability = probability
        self.asked: list[list[str]] = []

    async def asked_for_live_call(self, lines: list[str], *, owner_name: str = "") -> Decision | None:
        self.asked.append(list(lines))
        return None if self.probability is None else Decision(probability=self.probability, millis=212, model="jev-test")

    async def warm(self) -> None:
        pass


def test_the_sample_request_is_one_the_word_list_misses():
    assert not asked_for_the_owner([f"caller: {MISSED}"])


# --- the three bands ---


async def test_jev_decides_when_the_word_list_has_a_match(ctx):
    # "talk" is in the word list, but this caller wants a call back. The meaning decides, not the word.
    said = "caller: Just have him call me back. I can't talk right now."
    assert asked_for_the_owner([said])
    ctx.decider = FakeDecider(0.05)
    assert await transfer_question(ctx, [said]) == NOT_ASKED
    assert ctx.decider.asked == [[said]] and ctx.transfer_question_asked

    ctx.transfer_question_asked, ctx.decider = False, FakeDecider(0.97)
    assert await transfer_question(ctx, ["caller: Can I speak to him?"]) is None


async def test_without_a_decision_a_match_in_the_word_list_lets_the_transfer_go_ahead(ctx):
    ctx.decider = FakeDecider(None)  # a timeout or an error: the word list is the fallback
    assert await transfer_question(ctx, ["caller: Can I speak to him?"]) is None
    assert not ctx.transfer_question_asked and ctx.decisions == []


async def test_a_high_probability_lets_the_transfer_go_ahead(make_graph, ctx, config, telephony):
    ctx.decider = FakeDecider(0.93)
    model = scripted(tool_call("request_owner_transfer", reason="A project.", caller_name="Ada"), "Connecting you now.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage(MISSED)], "active_agent": "contact"}, config=config, context=ctx)

    assert len(telephony.transfers) == 1 and result.get("transfer_requested")
    assert ctx.decider.asked == [[f"caller: {MISSED}"]]
    assert not ctx.transfer_question_asked  # no question was needed


async def test_a_middle_probability_makes_the_agent_ask_and_the_answer_decides(make_graph, ctx, config, telephony):
    ctx.decider = FakeDecider(0.55)
    model = scripted(
        tool_call("request_owner_transfer", reason="A project.", caller_name="Ada"),
        "Do you want me to take a message, or connect you to him now?",
        tool_call("request_owner_transfer", call_id="call_2", reason="A project.", caller_name="Ada"),
        "Connecting you now.",
    )
    graph = make_graph(model)
    result = await graph.ainvoke({"messages": [HumanMessage(MISSED)], "active_agent": "contact"}, config=config, context=ctx)
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == ASK_WHICH and telephony.transfers == []

    await graph.ainvoke({"messages": [HumanMessage("The second one, please.")]}, config=config, context=ctx)
    assert len(telephony.transfers) == 1
    assert len(ctx.decider.asked) == 1  # the caller's answer decides; Jev is not asked again


async def test_a_low_probability_is_the_word_lists_answer(ctx, leads, telephony):
    ctx.decider = FakeDecider(0.04)
    assert await transfer_question(ctx, ["caller: Tell him I called about the invoice."]) == NOT_ASKED
    assert ctx.transfer_question_asked and telephony.transfers == [] and leads.all() == []


async def test_the_limits_are_inclusive_on_the_safe_side(ctx):
    for probability, expected in ((0.8, None), (0.79, ASK_WHICH), (0.31, ASK_WHICH), (0.3, NOT_ASKED)):
        ctx.transfer_question_asked = False
        ctx.decider = FakeDecider(probability)
        assert await transfer_question(ctx, [f"caller: {MISSED}"]) == expected, probability


async def test_no_decision_is_the_word_lists_answer(ctx):
    ctx.decider = FakeDecider(None)  # a timeout or an error
    assert await transfer_question(ctx, [f"caller: {MISSED}"]) == NOT_ASKED
    assert ctx.transfer_question_asked and ctx.decisions == []


async def test_without_jev_the_check_is_the_word_list_alone(ctx):
    assert ctx.decider is None
    assert await transfer_question(ctx, [f"caller: {MISSED}"]) == NOT_ASKED


async def test_a_forgotten_callers_words_are_not_sent(ctx):
    ctx.decider, ctx.forgotten = FakeDecider(0.99), True
    assert await transfer_question(ctx, [f"caller: {MISSED}"]) == NOT_ASKED
    assert ctx.decider.asked == []


async def test_the_call_stats_name_the_decision(ctx):
    facts = dict(visited=["classifier"], active_agent="contact", lead_saved=False, transfer_requested=True)
    assert "Jev" not in describe_call(ctx, **facts)
    ctx.decider = FakeDecider(0.93)
    await transfer_question(ctx, [f"caller: {MISSED}"])
    assert "Jev from TypeSafe AI" in (text := describe_call(ctx, **facts)) and "93 percent yes, in 212 milliseconds" in text


# --- what is sent ---


def test_only_the_end_of_the_call_is_sent():
    lines = ["caller: one", "agent: a", "caller: two", "caller: three", "agent: b", "caller: four", "agent: c", "caller: " + "x" * 900]
    state = conversation_state(lines, owner_name="Test Owner")
    assert state["owner"] == "Test Owner"
    assert "Test Owner" in state["call"] and "'him'" in state["call"]  # the greeting can be cut off; this line says who "him" is
    assert [turn["text"] for turn in state["conversation"]][:5] == ["two", "three", "b", "four", "c"]  # from the fourth-last caller turn on
    assert state["conversation"][0]["speaker"] == "caller" and state["conversation"][2]["speaker"] == "agent"
    assert len(state["conversation"][-1]["text"]) == 400  # a long line is cut
    assert conversation_state(["caller: hi"])["conversation"] == [{"speaker": "caller", "text": "hi"}]


# --- the real client, on a mock transport ---


def _jev(handler, *, time_limit: float = 0.5) -> JevDecider:
    client = AsyncTypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=0), timeout=time_limit, compression=False)
    return JevDecider(api_key="test-key", time_limit=time_limit, client=client)


def _answer(noul: float) -> httpx2.Response:
    return httpx2.Response(200, json={"model": "jev-1.14.0", "answers": {"live_call": {"type": "noul", "noul": noul}}, "usage": {"input_tokens": 300, "output_tokens": 20}})


async def test_the_client_asks_one_yes_or_no_question_and_reads_the_probability():
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return _answer(0.91)

    decision = await _jev(handler).asked_for_live_call(["agent: How can I help?", f"caller: {MISSED}"], owner_name="Test Owner")

    assert decision is not None and decision.probability == 0.91 and decision.model == "jev-1.14.0" and decision.millis >= 0
    assert seen["path"] == "/v1/systemone"
    question = seen["body"]["questions"]["live_call"]
    assert question["type"] == "noul" and "in their own words" in question["instructions"] and set(question["criteria"]) == {"true", "false"}
    assert seen["body"]["state"]["conversation"][-1] == {"speaker": "caller", "text": MISSED}


async def test_a_slow_answer_gives_no_decision_inside_the_time_limit():
    async def handler(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(2)
        return _answer(0.99)

    started = asyncio.get_running_loop().time()
    assert await _jev(handler, time_limit=0.05).asked_for_live_call([f"caller: {MISSED}"]) is None
    assert asyncio.get_running_loop().time() - started < 0.5


async def test_an_error_or_a_bad_answer_gives_no_decision():
    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route", request=request)

    responses = (
        lambda request: httpx2.Response(500, json={"error": "down"}),
        lambda request: httpx2.Response(401, json={"error": "bad key"}),
        lambda request: httpx2.Response(200, json={"model": "jev-1.14.0", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 1}}),
        lambda request: _answer(1.7),
        boom,
    )
    for handler in responses:
        assert await _jev(handler).asked_for_live_call([f"caller: {MISSED}"]) is None


async def test_a_failed_warm_up_is_silent():
    await _jev(lambda request: httpx2.Response(500, json={})).warm()


# --- settings ---


def test_jev_is_off_without_a_key_and_a_bad_key_does_not_stop_the_server(monkeypatch):
    from ai_call_center import decisions

    monkeypatch.setattr(decisions, "_default", None)
    assert default_decider(Settings(typesafe_api_key=SecretStr(""))) is None
    assert default_decider(Settings(typesafe_api_key=SecretStr("a key with spaces"))) is None
    made = default_decider(Settings(typesafe_api_key=SecretStr("test-key"), jev_timeout_ms=300))
    assert isinstance(made, JevDecider) and made.time_limit == 0.3
    assert default_decider(Settings(typesafe_api_key=SecretStr("test-key"))) is made  # one client for the process
