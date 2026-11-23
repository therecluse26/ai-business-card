"""Shared fixtures: a scripted fake model, an in-memory lead store, a recording telephony fake."""

from __future__ import annotations

import os

# No test sends a LangSmith trace. This runs before ai_call_center imports config, whose load_dotenv()
# never overrides a variable that is already set, so the owner's .env cannot turn tracing on here.
os.environ["LANGSMITH_TRACING"] = "false"
# No test asks Jev over the network, for the same reason. A test that needs Jev passes a fake decider.
os.environ["TYPESAFE_API_KEY"] = ""
# No test uploads the public report. A test that needs the upload passes its own settings and a fake sender.
os.environ["REPORT_R2_BUCKET"] = ""
# No test starts a job on GitHub, for the same reason. A test that needs a fix request passes its own settings and a fake sender.
os.environ["IMPROVE_GITHUB_TOKEN"] = ""
os.environ["GRADER_DECISION_MODEL"] = ""  # no test sends a call to the decision grader
# No test writes to the owner's CRM or reads it. A test that needs a CRM passes a fake one in the call context.
os.environ["CRM_MCP_URL"] = ""
# The fix job runs these tests with its own request in the environment (improve.yml: FIX_ISSUE, FIX_RULE, ...).
# No test reads the request of the job that runs it, and no test writes an output of that job. A test that
# needs a request sets it. On 2026-11-22 the first fix job for an issue failed here: its FIX_ISSUE reached the
# dry run of the job for a failed rule (test_fix_job.py).
for _name in [*(name for name in os.environ if name.startswith("FIX_")), "QUEUE_ISSUE", "AGENT_ANSWER", "RUN_URL", "GITHUB_OUTPUT"]:
    os.environ.pop(_name, None)

from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from ai_call_center.agents.specs import DEFAULT_LEGS, Profile, build_specs
from ai_call_center.graph.build import build_graph
from ai_call_center.graph.state import CallContext
from ai_call_center.leads import LeadStore
from ai_call_center.notify import RecordingNotifier
from ai_call_center.telephony import RecordingTelephony

OWNER = "Test Owner"
OWNER_PHONE = "+15555550199"


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from ai_call_center import tracing

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    tracing._reset()
    yield
    tracing._reset()


class ScriptedModel(GenericFakeChatModel):
    """Replays a script of AIMessages and records what it was asked."""

    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedModel:
        names = []
        for t in tools:
            name = getattr(t, "name", None)
            if name is None and isinstance(t, dict):
                name = t.get("name") or t.get("function", {}).get("name")
            names.append(name)
        self.bound_tool_names.append(names)
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.calls.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def system_prompt_of_call(self, index: int) -> str:
        first = self.calls[index][0]
        return first.content if isinstance(first.content, str) else str(first.content)


def tool_call(tool_name: str, call_id: str = "call_1", **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}])


def scripted(*messages: AIMessage | str) -> ScriptedModel:
    script: Iterator[AIMessage | str] = iter(messages)
    return ScriptedModel(messages=script)


@pytest.fixture
def specs():
    profile = Profile(summary=f"{OWNER} builds AI systems.", full=f"# Summary\n{OWNER} builds AI systems.\n# Skills\n- Python")
    return build_specs(profile, owner_name=OWNER)


@pytest.fixture
def telephony() -> RecordingTelephony:
    return RecordingTelephony()


@pytest.fixture
def leads() -> LeadStore:
    store = LeadStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier(to="owner@example.test")


@pytest.fixture
def ctx(leads: LeadStore, telephony: RecordingTelephony, notifier: RecordingNotifier) -> CallContext:
    return CallContext(
        call_control_id="v2:test-call",
        caller_number="+15555550123",
        owner_phone=OWNER_PHONE,
        leads=leads,
        telephony=telephony,
        notifier=notifier,
        legs=dict(DEFAULT_LEGS),
        mode="text",
    )


@pytest.fixture
def make_graph(specs):
    def _make(model: ScriptedModel):
        return build_graph(model, specs, checkpointer=InMemorySaver())

    return _make


@pytest.fixture
def config() -> dict:
    return {"configurable": {"thread_id": "thread-1"}}
