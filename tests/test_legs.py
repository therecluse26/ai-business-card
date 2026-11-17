"""Voice legs: which provider and voice each agent gets, and how a Gemini agent treats a handoff result."""

from pipecat.frames.frames import FunctionCallResultFrame, FunctionCallResultProperties

import pytest

from ai_call_center.agents.specs import LEG_VOICES, parse_agent_legs
from ai_call_center.config import Settings
from ai_call_center.voice.call_session import failover_instruction, handoff_instruction
from ai_call_center.voice.legs import HandoffGeminiLive, HandoffNovaSonic, build_leg, ensure_connected, ready_legs, replay_and_respond, reset_session, resolve_spec, self_heals
from ai_call_center.voice.tools import tools_for_agent, voice_tool_schemas


def settings(**over) -> Settings:
    # Every provider key is set here, so a real key in the developer's environment changes no test.
    base = {"openai_api_key": "sk-test", "google_api_key": "g-test", "nova_aws_access_key_id": "", "nova_aws_secret_access_key": ""}
    return Settings(_env_file=None, **{**base, **over})


def test_each_agent_gets_a_voice_from_its_own_provider(specs):
    for name, spec in specs.items():
        assert spec.voice == LEG_VOICES[spec.leg][name]
    assert specs["profile"].leg == "gemini_live" and specs["profile"].voice == "Charon"
    assert len({spec.voice for spec in specs.values()}) == len(specs)


def test_a_gemini_agent_falls_back_to_openai_with_an_openai_voice_when_there_is_no_key(specs):
    spec = resolve_spec(specs["profile"], settings(google_api_key=""))
    assert (spec.leg, spec.voice) == ("openai_realtime", LEG_VOICES["openai_realtime"]["profile"])
    assert resolve_spec(specs["profile"], settings()) is specs["profile"]


def test_gemini_leg_is_built_with_the_agent_tools_and_voice(specs):
    spec = specs["consultant"]
    tools = tools_for_agent(spec, voice_tool_schemas(specs))
    service = build_leg(spec, instructions="You are the consultant.", settings=settings(), tools=tools)
    assert isinstance(service, HandoffGeminiLive)
    assert service._settings.voice == "Puck"
    assert service._tools_from_init is tools


async def test_a_gemini_agent_is_not_given_a_handoff_result(specs):
    service = build_leg(specs["profile"], instructions="x", settings=settings())
    sent = []

    async def record(tool_call_id, *_args, **_kwargs):
        sent.append(tool_call_id)

    service._tool_result = record
    handoff = FunctionCallResultFrame(
        function_name="transfer_to_tour", tool_call_id="fc_1", arguments={}, result="Handed off.", properties=FunctionCallResultProperties(run_llm=False)
    )
    normal = FunctionCallResultFrame(function_name="get_call_stats", tool_call_id="fc_2", arguments={}, result="Two agents so far.")

    await service._handle_function_call_result(handoff)
    await service._handle_function_call_result(normal)

    assert sent == ["fc_2"]  # Gemini answers every result it receives; the handoff result must not reach it
    assert {"fc_1", "fc_2"} <= service._completed_tool_calls


def test_handoff_instruction_carries_the_exact_intro_line():
    text = handoff_instruction("tour", "OpenAI")
    assert '"Hi, I\'m the tour agent, running on OpenAI."' in text
    assert "Do not greet again" in text
    assert "not from the caller" in text and "do not call a transfer_to tool in this answer" in text
    assert "name" not in text  # no name is known

    # A caller who gave or confirmed their name is not asked for it by the next agent.
    known = handoff_instruction("contact", "OpenAI", "Steve Stevens")
    assert "The caller's name is Steve Stevens." in known and "Never ask for their name, never ask who is calling" in known


async def test_an_openai_agent_is_not_sent_results_of_tool_calls_other_agents_made(specs):
    """Measured 2026-11-16: back on the contact agent, OpenAI got the result of a Nova handoff call,
    answered "Tool call ID ... not found in conversation", and the session restarted mid-call."""
    from pipecat.processors.aggregators.llm_context import LLMContext

    from ai_call_center.voice.legs import adopt_context

    service = build_leg(specs["contact"], instructions="x", settings=settings())
    sent = []

    async def record(tool_call_id, *_args, **_kwargs):
        sent.append(tool_call_id)

    service._send_tool_result = record
    context = LLMContext([{"role": "user", "content": "Hello."}])
    service._context = context  # the agent was used before, and its session was reset when it was left
    # While another agent had the call, that agent handed the call back.
    context.add_message({"role": "assistant", "tool_calls": [{"id": "chatcmpl-tool-1", "type": "function", "function": {"name": "transfer_to_contact", "arguments": "{}"}}]})
    context.add_message({"role": "tool", "content": '"Handed off to the contact agent."', "tool_call_id": "chatcmpl-tool-1"})

    await adopt_context(service, context)
    await service._process_completed_function_calls(send_new_results=True)  # what the next context frame does

    assert sent == [] and "chatcmpl-tool-1" in service._completed_tool_calls


def test_an_agent_moves_to_the_other_provider_when_its_own_is_marked_down(specs):
    down_openai = frozenset({"openai_realtime"})
    spec = resolve_spec(specs["classifier"], settings(), down_openai)
    assert (spec.leg, spec.voice) == ("gemini_live", LEG_VOICES["gemini_live"]["classifier"])
    assert resolve_spec(specs["profile"], settings(), down_openai) is specs["profile"]  # already on Gemini

    spec = resolve_spec(specs["profile"], settings(), frozenset({"gemini_live"}))
    assert (spec.leg, spec.voice) == ("openai_realtime", LEG_VOICES["openai_realtime"]["profile"])


def test_a_provider_without_a_key_is_never_the_fallback(specs):
    spec = resolve_spec(specs["classifier"], settings(google_api_key=""), frozenset({"openai_realtime"}))
    assert spec is specs["classifier"]  # nothing else can run; the configured leg is tried anyway
    assert ready_legs(settings(google_api_key=""), frozenset({"openai_realtime"})) == []
    assert ready_legs(settings(), frozenset({"gemini_live"})) == ["openai_realtime"]


def test_every_agent_keeps_its_own_voice_after_a_move(specs):
    for down in ("openai_realtime", "gemini_live"):
        moved = [resolve_spec(spec, settings(), frozenset({down})) for spec in specs.values()]
        assert {spec.leg for spec in moved} == {"gemini_live" if down == "openai_realtime" else "openai_realtime"}
        assert len({spec.voice for spec in moved}) == len(moved)


def test_failover_instruction_carries_the_exact_apology_line():
    text = failover_instruction("profile", "OpenAI")
    assert '"Sorry about that pause. I\'m the profile agent, now running on OpenAI."' in text
    assert "Do not greet again" in text


# --- Amazon Nova Sonic ---

NOVA_KEYS = {"nova_aws_access_key_id": "AKIATEST", "nova_aws_secret_access_key": "secret"}


def test_agent_legs_setting_moves_an_agent_and_rejects_a_typing_mistake():
    assert parse_agent_legs("") == {}
    assert parse_agent_legs("consultant=nova_sonic, tour = gemini_live") == {"consultant": "nova_sonic", "tour": "gemini_live"}
    with pytest.raises(ValueError, match="unknown voice leg"):
        parse_agent_legs("consultant=nova")
    with pytest.raises(ValueError, match="unknown agent"):
        parse_agent_legs("sales=nova_sonic")


def on_nova(specs, agent="consultant"):
    import dataclasses

    return dataclasses.replace(specs[agent], leg="nova_sonic", voice=LEG_VOICES["nova_sonic"][agent])


def test_a_nova_agent_runs_on_nova_only_when_the_server_has_the_aws_keys(specs):
    spec = on_nova(specs)
    assert resolve_spec(spec, settings(**NOVA_KEYS)) is spec
    moved = resolve_spec(spec, settings())  # no keys: the first other ready leg
    assert (moved.leg, moved.voice) == ("openai_realtime", LEG_VOICES["openai_realtime"]["consultant"])
    # Nova is the last fallback: an agent whose leg is down moves to OpenAI or Gemini first.
    assert ready_legs(settings(**NOVA_KEYS)) == ["openai_realtime", "gemini_live", "nova_sonic"]
    assert ready_legs(settings()) == ["openai_realtime", "gemini_live"]


def test_nova_leg_is_built_with_the_agent_tools_voice_and_model(specs):
    spec = on_nova(specs)
    tools = tools_for_agent(spec, voice_tool_schemas(specs))
    service = build_leg(spec, instructions="You are the consultant.", settings=settings(**NOVA_KEYS), tools=tools)
    assert isinstance(service, HandoffNovaSonic) and self_heals(service)
    assert service._settings.voice == "matthew" and service._settings.model == "amazon.nova-2-5-sonic"
    assert service._settings.system_instruction == "You are the consultant." and service._settings.endpointing_sensitivity == "HIGH"
    assert service._tools is tools and service._region == "us-east-1"


async def test_a_nova_agent_is_not_given_a_handoff_result(specs):
    service = build_leg(on_nova(specs), instructions="x", settings=settings(**NOVA_KEYS))
    sent = []

    async def record(tool_call_id, *_args, **_kwargs):
        sent.append(tool_call_id)

    service._send_tool_result = record
    handoff = FunctionCallResultFrame(
        function_name="transfer_to_tour", tool_call_id="fc_1", arguments={}, result="Handed off.", properties=FunctionCallResultProperties(run_llm=False)
    )
    normal = FunctionCallResultFrame(function_name="end_call", tool_call_id="fc_2", arguments={}, result="Say goodbye.")

    await service._handle_function_call_result(handoff)
    await service._handle_function_call_result(normal)

    assert sent == ["fc_2"] and {"fc_1", "fc_2"} <= service._completed_tool_calls


async def test_a_reset_openai_session_forgets_the_audio_item_of_the_old_session(specs):
    """Measured 2026-11-16: the caller talked over the first words of an agent that had the call
    before. Pipecat cut an audio item of the old session, OpenAI answered
    "item_truncate_invalid_item_id", and the session restarted."""
    service = build_leg(specs["tour"], instructions="x", settings=settings())
    service._current_audio_response = object()  # the agent's last sentence before the handoff

    async def no_network() -> None:
        return None

    service.reset_conversation = no_network
    await reset_session(service)

    assert service._current_audio_response is None


def test_a_nova_tool_result_is_a_json_object():
    import json

    from ai_call_center.voice.legs import nova_tool_result

    assert json.loads(nova_tool_result('"Say goodbye now."')) == {"result": "Say goodbye now."}  # what Pipecat hands over for a sentence
    assert json.loads(nova_tool_result("COMPLETED")) == {"result": "COMPLETED"}
    assert json.loads(nova_tool_result('{"calls": 3}')) == {"calls": 3}
    assert json.loads(nova_tool_result({"calls": 3})) == {"calls": 3}


async def test_nova_connects_when_the_agent_is_needed_and_disconnects_when_it_is_left(specs):
    service = build_leg(on_nova(specs), instructions="x", settings=settings(**NOVA_KEYS))
    steps = []

    async def start_connecting():
        steps.append("connect")
        service._client = object()

    async def handle_context(context):
        steps.append("context")
        service._context = context

    async def disconnect():
        steps.append("disconnect")
        service._client = service._context = None

    service._start_connecting, service._handle_context, service._disconnect = start_connecting, handle_context, disconnect

    assert service._client is None  # nothing is open before the first handoff
    await replay_and_respond(service, context := object())
    assert steps == ["connect", "context"]
    await replay_and_respond(service, context)  # the session is set up already: nothing is sent twice
    await ensure_connected(service)
    assert steps == ["connect", "context"]
    await reset_session(service)
    await replay_and_respond(service, context)
    assert steps == ["connect", "context", "disconnect", "connect", "context"]


async def test_nova_usage_keeps_speech_and_text_tokens_apart(specs):
    from ai_call_center.voice.stats import PRICES, Usage

    service = build_leg(on_nova(specs), instructions="x", settings=settings(**NOVA_KEYS))
    reported = []

    async def record(tokens):
        reported.append(tokens)

    service.start_llm_usage_metrics = record
    await service._handle_usage_event(
        {"usageEvent": {"details": {"delta": {"input": {"speechTokens": 1000, "textTokens": 200}, "output": {"speechTokens": 500, "textTokens": 50}}}}}
    )
    usage = Usage()
    usage.add(reported[0])
    assert (usage.audio_in, usage.text_in, usage.audio_out, usage.text_out) == (1000, 200, 500, 50)
    assert usage.cost(PRICES["nova_sonic"]) == pytest.approx((1000 * 3.00 + 200 * 0.33 + 500 * 12.00 + 50 * 2.75) / 1_000_000)
