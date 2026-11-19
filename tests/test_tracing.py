"""LangSmith tracing: off unless set and keyed, set up once, and every run carries the call's thread id."""

from __future__ import annotations

import logging
import os

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tracers.context import collect_runs
from pydantic import SecretStr

from ai_call_center import tracing
from ai_call_center.agents.specs import Profile, build_specs
from ai_call_center.callers import summarize_call
from ai_call_center.config import Settings
from tests.conftest import OWNER, ScriptedModel


def settings(*, on: bool, key: str = "") -> Settings:
    return Settings(_env_file=None, langsmith_tracing=on, langsmith_api_key=SecretStr(key))


@pytest.fixture
def configure_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Stand in for LangSmith's Pipecat setup, which would install a real exporter."""
    import langsmith.integrations.pipecat as ls_pipecat

    calls: list[dict] = []
    marker = object()

    def fake_configure(**kwargs):
        calls.append(kwargs)
        return marker

    monkeypatch.setattr(ls_pipecat, "configure_pipecat", fake_configure)
    return calls


def test_off_by_default(configure_calls):
    assert tracing.start(settings(on=False, key="lsv2_x")) is None
    assert configure_calls == []
    assert tracing.processor() is None
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_on_without_a_key_stays_off_and_warns(configure_calls, caplog):
    with caplog.at_level(logging.WARNING, logger="ai_call_center.tracing"):
        assert tracing.start(settings(on=True)) is None
    assert configure_calls == []
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY is empty" in caplog.text


def test_on_with_a_key_sets_up_pipecat_once(configure_calls):
    s = settings(on=True, key="lsv2_x")
    first = tracing.start(s)
    second = tracing.start(s)
    assert first is not None and first is second
    assert len(configure_calls) == 1
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_text_mode_skips_pipecat(configure_calls):
    assert tracing.start(settings(on=True, key="lsv2_x"), voice=False) is None
    assert configure_calls == []
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_follow_call_does_nothing_when_off():
    assert tracing.follow_call(object(), "call-1") is False  # type: ignore[arg-type]


def test_run_config_names_the_run_and_the_thread():
    assert tracing.run_config("call-1", "call-summary") == {"run_name": "call-summary", "metadata": {"thread_id": "call-1"}}


async def test_the_call_summary_sits_in_the_call_thread():
    model = ScriptedModel(messages=iter([AIMessage('{"summary": "Steve called.", "interest": "none"}')]))
    with collect_runs() as runs:
        await summarize_call([{"role": "user", "content": "Hi, this is Steve."}], model, config=tracing.run_config("call-1", "call-summary"))
    run = runs.traced_runs[0]
    assert run.name == "call-summary"
    assert run.extra["metadata"]["thread_id"] == "call-1"


def test_the_tour_agent_names_langsmith_only_when_tracing_is_on():
    profile = Profile(summary="x", full="x")
    assert "LangSmith" in build_specs(profile, owner_name=OWNER, tracing=True)["tour"].instructions
    for spec in build_specs(profile, owner_name=OWNER).values():
        assert "LangSmith" not in spec.instructions
