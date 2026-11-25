"""The scorer of the made-up test set: the files load, a made-up call runs with no network, and two results compare."""

from __future__ import annotations

import argparse
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_call_center.config import Settings
from ai_call_center.grading import FAIL, NA, PASS, RULES, Score
from ai_call_center.graph.build import build_graph
from ai_call_center.graph.handoffs import handoff_tool_name
from ai_call_center.grading import grade_call
from ai_call_center.score import (
    CALLS,
    TEST_SET,
    CallRun,
    LiveCallSample,
    MadeUpCall,
    Usage,
    call_checks,
    compare,
    describe,
    load_calls,
    load_live_call,
    result_of,
    run,
    run_call,
    sample_lines,
    usage_lines,
    vote,
    word_list_checks,
)
from tests.conftest import OWNER, scripted, tool_call
from tests.test_prompt_cache import claude


def settings() -> Settings:
    return Settings(_env_file=None, owner_name=OWNER)


def test_the_test_set_in_the_repository_loads():
    samples, calls = load_live_call(), load_calls()
    assert len(samples) >= 16
    assert {True, False} == {sample.wants_live_call for sample in samples}
    assert len({call.id for call in calls}) == len(calls) >= 6
    assert all(call.caller and call.about for call in calls)


def test_a_sample_starts_with_the_greeting_and_marks_the_callers_lines():
    lines = sample_lines(("agent: Message, or a live call?", "Connect me."), "Ann Lee")
    assert lines == ["agent: This is Ann Lee's AI business card. How can I help?", "agent: Message, or a live call?", "caller: Connect me."]


def test_the_word_list_gets_one_check_for_each_sample():
    samples = [
        LiveCallSample(("Can I speak to him?",), True),
        LiveCallSample(("Just have him call me back.",), False),
        LiveCallSample(("I would love a quick word with him.",), True),  # the word list has no word for this
        LiveCallSample(("agent: Would you like me to connect you?", "Sure."), False),  # the agent's words do not count
    ]
    assert word_list_checks(samples, OWNER) == {"live_call/01": True, "live_call/02": True, "live_call/03": False, "live_call/04": True}


def write_set(tmp_path, *, live=None, calls=None):
    """A test set in a folder. Each call goes in a file named by its id, or by "file" when the call has that key."""
    (tmp_path / "live_call.json").write_text(json.dumps(live if live is not None else [{"lines": ["Hi."], "wants_live_call": False}]))
    (tmp_path / "calls").mkdir()
    for call in calls or []:
        call = dict(call)
        (tmp_path / "calls" / call.pop("file", f"{call.get('id', 'x')}.json")).write_text(json.dumps(call))
    return tmp_path


@pytest.mark.parametrize(
    "call",
    [
        {"id": "", "caller": ["Hi."], "file": "empty.json"},
        {"id": "a b", "caller": ["Hi."], "file": "a b.json"},
        {"id": "A", "caller": ["Hi."]},
        {"id": "a--b", "caller": ["Hi."]},
        {"id": "-a", "caller": ["Hi."]},
        {"id": "a" * 61, "caller": ["Hi."]},
        {"id": "a", "caller": ["Hi."], "file": "b.json"},  # the file name is not the id
        {"id": "a", "caller": ["Hi."], "file": "a.txt"},
        {"id": "a", "caller": []},
        {"id": "a", "caller": "Hi."},
        {"id": "a", "caller": ["Hi."], "expect": {"visit": ["contact"]}},
        {"id": "a", "caller": ["Hi."], "expect": {"visits": ["sales"]}},
        {"id": "a", "caller": ["Hi."], "expect": {"outcome": "sold"}},
    ],
)
def test_a_call_with_a_fault_in_the_test_set_is_an_error(tmp_path, call):
    with pytest.raises(ValueError):
        load_calls(write_set(tmp_path, calls=[call]))


def test_a_file_that_is_not_one_call_is_an_error(tmp_path):
    folder = write_set(tmp_path)
    (folder / "calls" / "a.json").write_text(json.dumps([{"id": "a", "caller": ["Hi."]}]))
    with pytest.raises(ValueError):
        load_calls(folder)


def test_the_calls_load_in_the_order_of_their_file_names(tmp_path):
    folder = write_set(tmp_path, calls=[{"id": name, "caller": ["Hi."]} for name in ("b-2", "a", "b-10", "c1")])
    (folder / "calls" / ".DS_Store").write_text("")  # a file that the system keeps there is not a call
    assert [call.id for call in load_calls(folder)] == ["a", "b-10", "b-2", "c1"]


def test_each_file_of_the_test_set_in_the_repository_is_named_by_its_id():
    files = sorted(path.name for path in (TEST_SET / CALLS).iterdir())
    assert files == [f"{call.id}.json" for call in load_calls()] and not (TEST_SET / "calls.json").exists()


def test_a_sample_with_no_caller_line_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        load_live_call(write_set(tmp_path, live=[{"lines": ["agent: Hello."], "wants_live_call": False}]))


async def test_a_made_up_call_runs_one_turn_for_each_caller_line(specs):
    model = scripted("Hello. Who am I speaking with?", "Thanks, Dana. How can I help?", "He builds AI systems.")
    graph = build_graph(model, specs, checkpointer=InMemorySaver())
    call = MadeUpCall(id="plain", about="", caller=("This is Dana.", "What does he do?"))

    run = await run_call(call, graph, settings())

    assert run.error == ""
    assert run.lines == [
        "agent: Hello. Who am I speaking with?",
        "caller: This is Dana.",
        "agent: Thanks, Dana. How can I help?",
        "caller: What does he do?",
        "agent: He builds AI systems.",
    ]
    assert run.visited == ["classifier"]
    assert (run.outcome, run.forgotten) == ("none", False)
    assert len(model.calls) == 3  # the greeting, and one answer for each caller line


async def test_a_handoff_in_a_made_up_call_is_in_the_visited_agents(specs):
    model = scripted("Hello. Who am I speaking with?", tool_call(handoff_tool_name("profile")), "Hi, I'm the profile agent. He builds AI systems.")
    graph = build_graph(model, specs, checkpointer=InMemorySaver())

    run = await run_call(MadeUpCall(id="handoff", about="", caller=("What does he do?",)), graph, settings())

    assert run.visited == ["classifier", "profile"]
    assert run.lines[-1] == "agent: Hi, I'm the profile agent. He builds AI systems."


async def test_a_model_error_stops_the_call_and_is_named(specs):
    model = scripted("Hello. Who am I speaking with?")  # no answer for the caller's line
    graph = build_graph(model, specs, checkpointer=InMemorySaver())
    call = MadeUpCall(id="broken", about="", caller=("Hi.",), outcome="none")

    run = await run_call(call, graph, settings())

    assert run.error
    assert call_checks(call, run, [], graded=False)["call/broken/ran"] is False


def test_the_checks_of_a_call_hold_what_the_test_set_expects():
    call = MadeUpCall(id="msg", about="", caller=("Hi.",), visits=("contact", "profile"), outcome="message", forgotten=False)
    run = CallRun(lines=[], visited=["classifier", "contact"], outcome="message", forgotten=False)

    assert call_checks(call, run, [], graded=False) == {
        "call/msg/ran": True,
        "call/msg/visits/contact": True,
        "call/msg/visits/profile": False,
        "call/msg/outcome": True,
        "call/msg/forgotten": True,
    }


def test_a_rule_that_does_not_apply_gives_no_check_and_a_model_rule_with_no_verdict_fails():
    call = MadeUpCall(id="c", about="", caller=("Hi.",))
    scores = [Score("tool_name_not_spoken", PASS), Score("handoff_intro", NA), Score("summary_first_try", NA), Score("name_asked_once", FAIL, "Asked two times.")]

    checks = call_checks(call, CallRun(), scores, graded=True)

    assert checks["call/c/rule/tool_name_not_spoken"] is True
    assert checks["call/c/rule/name_asked_once"] is False
    assert "call/c/rule/handoff_intro" not in checks
    missing = [rule.key for rule in RULES if rule.check and not rule.hears and rule.key != "name_asked_once"]
    assert missing and all(checks[f"call/c/rule/{key}"] is False for key in missing)
    # A made-up call has no audio, so an audio rule has no verdict, and that is not a failure.
    assert all(f"call/c/rule/{rule.key}" not in checks for rule in RULES if rule.hears)
    # With no grader, a model rule has no verdict and gives no check.
    assert all(f"call/c/rule/{key}" not in call_checks(call, CallRun(), scores[:3], graded=False) for key in missing)


def test_two_results_compare_by_the_name_of_each_check():
    old = result_of({"a": True, "b": False, "c": True, "only_old": True}, version="v1", calls=False)
    new = result_of({"a": False, "b": True, "c": True, "only_new": False}, version="v2", calls=False)

    assert compare(old, new) == (["a"], ["b"])
    assert (new["passed"], new["total"]) == (2, 4)
    assert describe(new).startswith("2 of 4 checks passed (version v2)")
    assert "  only_new" in describe(new)


def test_the_message_type_of_the_fake_model_is_an_ai_message():
    # A guard for the fakes of this file: a scripted string is an agent's answer.
    assert isinstance(tool_call("x"), AIMessage)


async def test_the_usage_of_the_agents_and_of_the_grader_is_summed_by_model(specs):
    usage = Usage()
    model, grader = claude("Hello. Who am I speaking with?", "Thanks, Dana."), claude('{"name_asked_once": {"verdict": "pass"}}')
    grader.model = "claude-sonnet-5-5"
    graph = build_graph(model, specs, checkpointer=InMemorySaver())

    run_ = await run_call(MadeUpCall(id="usage", about="", caller=("This is Dana.",)), graph, settings(), callbacks=[usage])
    await grade_call(run_.lines, visited=run_.visited, profile="", owner=OWNER, model=grader, config={"callbacks": [usage]})

    one = {"input_tokens": 1000, "cache_read": 700, "cache_creation": 200, "output_tokens": 20}  # what each request of the fake model gives
    assert usage.by_model == {
        "claude-haiku-5-5": {"requests": 2, **{key: 2 * value for key, value in one.items()}},
        "claude-sonnet-5-5": {"requests": 1, **one},
    }
    assert usage_lines(usage.by_model) == [
        "claude-haiku-5-5: 2 requests, 2,000 tokens in (1,400 from the cache, 400 written to the cache), 40 tokens out",
        "claude-sonnet-5-5: 1 request, 1,000 tokens in (700 from the cache, 200 written to the cache), 20 tokens out",
    ]


async def test_a_model_with_no_usage_gives_no_usage_line(specs):
    usage = Usage()
    graph = build_graph(scripted("Hello. Who am I speaking with?", "Hi."), specs, checkpointer=InMemorySaver())
    await run_call(MadeUpCall(id="fake", about="", caller=("Hi.",)), graph, settings(), callbacks=[usage])
    assert usage.by_model == {} and usage_lines(usage.by_model) == []


def test_the_usage_is_in_the_result_of_a_run_of_the_calls_and_a_vote_adds_it_up():
    used = {"claude-haiku-5-5": {"requests": 3, "input_tokens": 900, "cache_read": 600, "cache_creation": 100, "output_tokens": 30}}
    first = json.loads(json.dumps(result_of({"call/a/ran": True}, version="v1", calls=True, usage=used)))
    assert first["usage"] == used
    assert "usage" not in result_of({"live_call/01": True}, version="v1", calls=False)

    voted = vote([first, first, result_of({"call/a/ran": False}, version="v1", calls=True)], ["a"])
    assert voted["usage"] == {"claude-haiku-5-5": {key: 2 * value for key, value in used["claude-haiku-5-5"].items()}}
    assert "usage" not in vote([result_of({"call/a/ran": True}, version="v1", calls=True)], ["a"])


async def test_a_run_with_no_made_up_calls_has_no_usage(tmp_path, capsys):
    out = tmp_path / "result.json"
    args = argparse.Namespace(set=str(TEST_SET), calls=False, only=[], limit=0, show=False, out=str(out), compare=None, json=False)
    assert await run(args) == 0
    assert "usage" not in json.loads(out.read_text()) and "requests" not in capsys.readouterr().err
