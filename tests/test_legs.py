"""Voice legs: which provider and voice each agent gets, and how a Gemini agent treats a handoff result."""

from pipecat.frames.frames import FunctionCallResultFrame, FunctionCallResultProperties

from ai_call_center.agents.specs import LEG_VOICES
from ai_call_center.config import Settings
from ai_call_center.voice.call_session import failover_instruction, handoff_instruction
from ai_call_center.voice.legs import HandoffGeminiLive, build_leg, ready_legs, resolve_spec
from ai_call_center.voice.tools import tools_for_agent, voice_tool_schemas


def settings(**over) -> Settings:
    return Settings(_env_file=None, **{"openai_api_key": "sk-test", "google_api_key": "g-test", **over})


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
