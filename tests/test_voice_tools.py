"""Voice-mode tool schemas mirror the LangGraph tools, agent by agent."""

from ai_call_center.agents.specs import DEFAULT_VOICES
from ai_call_center.graph.build import all_tools
from ai_call_center.voice.tools import tools_for_agent, voice_tool_schemas


def test_every_graph_tool_has_a_voice_schema(specs):
    schemas = voice_tool_schemas(specs)
    assert set(schemas) == set(all_tools(specs))
    lead = schemas["save_lead"]
    assert lead.required == ["name", "email", "company", "need"]
    assert lead.properties["email"] == {"type": "string"}
    assert schemas["get_call_stats"].properties == {}


def test_agent_tool_sets_match_text_mode(specs):
    schemas = voice_tool_schemas(specs)
    names = {name: [t.name for t in tools_for_agent(spec, schemas).standard_tools] for name, spec in specs.items()}
    assert names["classifier"] == ["transfer_to_profile", "transfer_to_tour", "transfer_to_consultant", "transfer_to_contact"]
    assert names["contact"][:2] == ["save_lead", "request_owner_transfer"]
    assert "transfer_to_contact" not in names["contact"]


def test_each_agent_has_its_own_voice(specs):
    voices = [spec.voice for spec in specs.values()]
    assert len(set(voices)) == len(voices)
    assert specs["classifier"].voice == DEFAULT_VOICES["classifier"]
