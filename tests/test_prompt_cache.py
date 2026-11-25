"""The prompt cache: a Claude model gets cache markers, the text that the model reads does not change, and another model gets the same request as before."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from ai_call_center.grading import GRADER_PROMPT, NOTE_WORDS, grade, grader_message, grader_prompt, model_rules
from ai_call_center.graph.build import ANTHROPIC_MODEL_KWARGS, CACHE_CONTROL, build_graph, caches_prompts, mode_line, system_message
from ai_call_center.graph.handoffs import handoff_tool_name
from tests.conftest import OWNER, scripted, tool_call

LOOKUP = "Caller lookup: a mobile number in Ohio."  # made up
CALL = ["agent: Hello. Who am I speaking with?", "caller: This is Dana.", "agent: Thanks, Dana. How can I help?"]
PROFILE = f"# Summary\n{OWNER} builds AI systems."


class RecordingClaude(ChatAnthropic):
    """A Claude model that sends nothing. It keeps the request body that it would send, and gives the next reply of a script."""

    replies: list[AIMessage] = Field(default_factory=list)
    payloads: list[dict[str, Any]] = Field(default_factory=list)

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.payloads.append(self._get_request_payload(messages, stop=stop, **kwargs))
        reply = self.replies.pop(0)
        usage = {"input_tokens": 1000, "output_tokens": 20, "total_tokens": 1020, "input_token_details": {"cache_read": 700, "cache_creation": 200}}
        message = reply.model_copy(update={"usage_metadata": usage, "response_metadata": {"model_name": self.model}})
        return ChatResult(generations=[ChatGeneration(message=message)])


def claude(*replies: AIMessage | str) -> RecordingClaude:
    return RecordingClaude(
        model="claude-haiku-5-5",
        api_key="not-a-key",
        replies=[AIMessage(content=reply) if isinstance(reply, str) else reply for reply in replies],
        **ANTHROPIC_MODEL_KWARGS,
    )


def old_prompt(spec: Any, ctx: Any) -> str:
    """The system prompt of an agent, made as before the prompt cache."""
    prompt = f"{spec.instructions}\n\n{mode_line(spec, ctx)}"
    return f"{prompt}\n\n{ctx.caller_lookup}" if ctx.caller_lookup else prompt


def joined(content: str | list[Any]) -> str:
    return content if isinstance(content, str) else "".join(block["text"] for block in content)


def test_only_a_claude_model_gets_cache_markers():
    assert caches_prompts(claude()) and not caches_prompts(scripted("Hi."))


def test_the_system_blocks_join_to_the_text_of_before_with_and_without_a_caller_lookup(specs, ctx):
    for lookup in ("", LOOKUP):
        ctx.caller_lookup = lookup
        for spec in specs.values():
            plain, cached = system_message(spec, ctx), system_message(spec, ctx, cache=True)
            assert plain.content == old_prompt(spec, ctx)
            assert joined(cached.content) == old_prompt(spec, ctx)
            assert cached.content[0]["cache_control"] == CACHE_CONTROL
            assert all("cache_control" not in block for block in cached.content[1:])
            assert len(cached.content) == (2 if lookup else 1)


def test_the_cached_block_is_the_same_with_and_without_a_caller_lookup(specs, ctx):
    spec = specs["classifier"]
    first = system_message(spec, ctx, cache=True).content[0]
    ctx.caller_lookup = LOOKUP
    assert system_message(spec, ctx, cache=True).content[0] == first


async def test_a_fake_model_gets_the_system_prompt_as_one_string(specs, ctx, config):
    ctx.caller_lookup = LOOKUP
    model = scripted("Hello. Who am I speaking with?")
    await build_graph(model, specs, checkpointer=InMemorySaver()).ainvoke({"messages": [HumanMessage("Hi.")]}, config=config, context=ctx)
    assert model.calls[0][0].content == old_prompt(specs["classifier"], ctx)


async def test_a_claude_request_has_a_cache_marker_on_the_system_prompt_and_on_the_conversation(specs, ctx, config):
    ctx.caller_lookup = LOOKUP
    model = claude(tool_call(handoff_tool_name("profile")), "Hi, I'm the profile agent.")
    await build_graph(model, specs, checkpointer=InMemorySaver()).ainvoke({"messages": [HumanMessage("What does he do?")]}, config=config, context=ctx)

    first, second = model.payloads
    for payload, name in ((first, "classifier"), (second, "profile")):
        assert joined(payload["system"]) == old_prompt(specs[name], ctx)
        assert payload["system"][0]["cache_control"] == CACHE_CONTROL and "cache_control" not in payload["system"][1]
        assert payload["cache_control"] == CACHE_CONTROL  # the API puts this marker on the last block of the conversation
        assert payload["thinking"] == ANTHROPIC_MODEL_KWARGS["thinking"]
    assert handoff_tool_name("profile") in [tool["name"] for tool in first["tools"]]
    # The text of the conversation has no marker of its own.
    assert all(isinstance(message["content"], str) or all("cache_control" not in block for block in message["content"]) for message in second["messages"])


def test_the_grader_prompt_is_the_text_of_before():
    rules = "\n".join(
        f"- {rule.key}: {rule.title}. {rule.check.format(owner=OWNER)}\n  Cause codes: " + "; ".join(f"{code} = {meaning}" for code, meaning in rule.causes)
        for rule in model_rules()
    )
    before = GRADER_PROMPT.format(owner=OWNER, rules=rules, note_words=NOTE_WORDS, profile=PROFILE, agents="classifier -> tour", transcript="\n".join(CALL))
    assert grader_prompt(CALL, visited=["classifier", "tour"], profile=PROFILE, owner=OWNER) == before


def test_the_grader_blocks_join_to_the_grader_prompt_and_only_the_first_has_the_marker():
    prompt = grader_prompt(CALL, visited=["classifier", "tour"], profile=PROFILE, owner=OWNER)
    message = grader_message(CALL, visited=["classifier", "tour"], profile=PROFILE, owner=OWNER, cache=True)
    first, rest = message.content
    assert first["text"] + rest["text"] == prompt
    assert first["cache_control"] == CACHE_CONTROL and "cache_control" not in rest
    assert first["text"].endswith(PROFILE) and rest["text"].startswith("\n\nAgents on the call, in order: classifier -> tour")
    assert grader_message(CALL, visited=["classifier", "tour"], profile=PROFILE, owner=OWNER).content == prompt


def test_the_first_grader_block_is_the_same_for_two_calls():
    one = grader_message(CALL, visited=["classifier", "tour"], profile=PROFILE, owner=OWNER, cache=True).content[0]
    two = grader_message(["caller: Bye."], visited=["classifier"], profile=PROFILE, owner=OWNER, cache=True).content[0]
    assert one == two


async def test_a_claude_grader_gets_two_blocks_and_another_grader_gets_one_string():
    reply = '{"name_asked_once": {"verdict": "pass"}}'
    model = claude(reply)
    await grade(CALL, visited=["classifier"], profile=PROFILE, owner=OWNER, model=model)
    (message,) = model.payloads[0]["messages"]
    assert [block.get("cache_control") for block in message["content"]] == [CACHE_CONTROL, None]
    assert joined(message["content"]) == grader_prompt(CALL, visited=["classifier"], profile=PROFILE, owner=OWNER)
    assert "cache_control" not in model.payloads[0]  # the grader has one request for each call, so it needs no marker on the end

    fake = scripted(reply)
    await grade(CALL, visited=["classifier"], profile=PROFILE, owner=OWNER, model=fake)
    assert fake.calls[0][0].content == grader_prompt(CALL, visited=["classifier"], profile=PROFILE, owner=OWNER)
