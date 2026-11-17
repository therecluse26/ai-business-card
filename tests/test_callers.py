"""Returning-caller memory: the call log, what a prompt may see, and the end-of-call summary."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_call_center.callers import greeting_note, history_text, recall, summarize_call, transcript_lines
from ai_call_center.leads import Lead, now_iso
from tests.conftest import scripted, tool_call

NUMBER = "+15555550123"


def _lead(leads, **over):
    lead = Lead(name="Steve Park", email="steve@acme.test", company="Acme Roofing", need="An agent for after-hours calls.", caller_number=NUMBER, call_control_id="call-1", created_at=now_iso())
    leads.save(Lead(**{**lead.__dict__, **over}))


def test_first_time_and_hidden_numbers_have_no_memory(leads):
    assert recall(leads, NUMBER) is None
    leads.start_call("c1", "anonymous")
    leads.finish_call("c1", agents=["classifier"], summary="Asked about pricing.")
    assert recall(leads, "anonymous") is None


def test_recall_combines_lead_and_finished_calls(leads):
    _lead(leads)
    leads.start_call("call-1", NUMBER)
    leads.finish_call("call-1", agents=["classifier", "contact"], summary="Left details; wants a call back.", caller_name="Steve")
    leads.start_call("call-2", NUMBER)  # the call in progress is not history yet

    memory = recall(leads, NUMBER, exclude_call="call-2")

    assert memory.first_name == "Steve"
    assert memory.company == "Acme Roofing"
    assert memory.previous_calls == 1
    assert [c.summary for c in memory.recent] == ["Left details; wants a call back."]


def test_name_comes_from_the_call_log_when_no_lead_exists(leads):
    leads.start_call("c1", NUMBER)
    leads.finish_call("c1", agents=["classifier"], summary="Asked how the system works.", caller_name="Dana")
    assert recall(leads, NUMBER).name == "Dana"


def test_the_newest_name_wins_and_the_history_holds_only_that_person(leads):
    # Steve left a lead with an email address. Later, Bill called from the same number and left no lead.
    _lead(leads, created_at="2026-11-16T03:10:45+00:00")
    leads.start_call("call-1", NUMBER)
    leads.finish_call("call-1", agents=["classifier", "contact"], summary="Steve asked about a shared project.", caller_name="Steve")
    leads._conn.execute("UPDATE calls SET started_at = '2026-11-16T03:08:30+00:00' WHERE call_control_id = 'call-1'")
    leads.start_call("call-2", NUMBER)
    leads.finish_call("call-2", agents=["classifier", "profile"], summary="Bill asked about the owner.", caller_name="Bill")
    leads._conn.execute("UPDATE calls SET started_at = '2026-11-16T14:32:29+00:00' WHERE call_control_id = 'call-2'")

    memory = recall(leads, NUMBER)

    assert memory.first_name == "Bill"  # the next greeting asks for Bill, not for Steve
    assert 'Am I speaking with Bill?' in greeting_note(memory)
    assert [c.summary for c in memory.recent] == ["Bill asked about the owner."] and memory.previous_calls == 1
    assert (memory.company, memory.need, memory.contact_on_file) == ("", "", False)
    text = history_text(memory)
    assert "Steve" not in text and "Acme" not in text and "on file" not in text

    # Steve calls again: his name is the newest, and his own history comes back.
    leads.start_call("call-3", NUMBER)
    leads.finish_call("call-3", agents=["classifier"], summary="Steve asked for an update.", caller_name="Steve Park")
    leads._conn.execute("UPDATE calls SET started_at = '2026-11-16T15:00:00+00:00' WHERE call_control_id = 'call-3'")
    memory = recall(leads, NUMBER)
    assert memory.name == "Steve Park" and memory.company == "Acme Roofing" and memory.contact_on_file
    assert [c.call_control_id for c in memory.recent] == ["call-3", "call-1"]


def test_a_lead_from_the_call_in_progress_is_not_history(leads):
    leads.start_call("call-2", NUMBER)
    _lead(leads, name="Bill Williams", call_control_id="call-2")
    assert recall(leads, NUMBER, exclude_call="call-2") is None


def test_greeting_note_holds_the_first_name_and_nothing_else(leads):
    _lead(leads)
    leads.start_call("call-1", NUMBER)
    leads.finish_call("call-1", agents=["classifier"], summary="Wants an after-hours agent.")
    note = greeting_note(recall(leads, NUMBER))
    assert 'ask only: "Am I speaking with Steve?"' in note
    assert "Say your opening line." in note and "Do not say what you can help with yet." in note
    # Nothing a stranger on the same phone could learn about the earlier call.
    for secret in ("Park", "Acme", "after-hours", "steve@acme.test", NUMBER):
        assert secret not in note


def test_history_is_separate_and_never_holds_the_email_or_the_number(leads):
    _lead(leads)
    leads.start_call("call-1", NUMBER)
    leads.finish_call("call-1", agents=["classifier"], summary="Wants an after-hours agent.")
    text = history_text(recall(leads, NUMBER))
    assert "Steve Park from Acme Roofing" in text and "Wants an after-hours agent." in text
    assert "steve@acme.test" not in text and NUMBER not in text
    assert 'call save_lead with email "on file"' in text


def test_no_name_means_no_greeting_note(leads):
    leads.start_call("c1", NUMBER)
    leads.finish_call("c1", agents=["classifier"], summary="Asked about prices.")
    assert greeting_note(recall(leads, NUMBER)) == ""  # nothing to confirm, so the caller is greeted as new


def test_transcript_lines_skip_tools_and_developer_messages():
    messages = [
        {"role": "developer", "content": "Greet the caller now."},
        {"role": "assistant", "content": "Hi, what brings you to call?"},
        {"role": "user", "content": [{"type": "text", "text": "I run a roofing company."}]},
        {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "transfer_to_consultant", "arguments": "{}"}}]},
        {"role": "tool", "content": "Handed off.", "tool_call_id": "x"},
        HumanMessage("[The call connects. Greet the caller.]"),
        AIMessage("Welcome."),
    ]
    assert transcript_lines(messages) == ["agent: Hi, what brings you to call?", "caller: I run a roofing company.", "agent: Welcome."]


async def test_summarize_call_parses_the_model_json():
    model = scripted('{"summary": "Steve asked about an after-hours agent and left his details.", "caller_name": "Steve"}')
    summary, name = await summarize_call([{"role": "user", "content": "Hi, this is Steve."}], model)
    assert summary.startswith("Steve asked") and name == "Steve"


async def test_summarize_call_skips_a_call_with_no_caller_speech():
    model = scripted("should not be used")
    assert await summarize_call([{"role": "assistant", "content": "Hello?"}], model) == ("", "")
    assert model.calls == []


async def test_only_the_greeting_agent_sees_the_note_and_the_history_stays_out_of_every_prompt(make_graph, ctx, config):
    ctx.caller_greeting = "Returning caller note:\n- The earlier caller gave the first name Steve."
    ctx.caller_history = "Notes from their earlier calls:\n- Known as Steve Park from Acme Roofing."
    model = scripted("Hi, this is the AI assistant. Am I speaking with Steve?", tool_call("transfer_to_profile"), "Hi, I'm the profile agent.")
    graph = make_graph(model)
    await graph.ainvoke({"messages": [HumanMessage("[The call connects.]")]}, config=config, context=ctx)
    await graph.ainvoke({"messages": [HumanMessage("No. Tell me about the owner.")]}, config=config, context=ctx)

    assert "first name Steve" in model.system_prompt_of_call(0)  # the classifier
    assert "first name Steve" not in model.system_prompt_of_call(2)  # the profile agent
    seen = " ".join(str(m.content) for call in model.calls for m in call)
    assert "Acme Roofing" not in seen  # the caller said no, so the history never reached a model
    assert not ctx.identity_confirmed


async def test_history_is_released_only_by_the_confirm_tool(make_graph, ctx, config):
    ctx.caller_greeting = "Returning caller note:\n- The earlier caller gave the first name Steve."
    ctx.caller_history = "Notes from their earlier calls:\n- Known as Steve Park from Acme Roofing."
    model = scripted(tool_call("confirm_returning_caller"), "Welcome back, Steve. Is this about the roofing line again?")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes, this is Steve.")]}, config=config, context=ctx)

    assert ctx.identity_confirmed
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Acme Roofing" in tool_msg.content


async def test_confirm_tool_gives_nothing_for_a_new_caller(make_graph, ctx, config):
    model = scripted(tool_call("confirm_returning_caller"), "What brings you to call today?")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Hi.")]}, config=config, context=ctx)
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "No earlier calls are on record" in tool_msg.content and not ctx.identity_confirmed
