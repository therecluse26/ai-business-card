"""Returning-caller memory: the call log, the name check, what a prompt may see, "forget me", and the end-of-call summary."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_call_center.agents.specs import Welcome, companies_phrase, greeting_words
from ai_call_center.callers import CallSummary, history_text, recall, summarize_call, transcript_lines
from ai_call_center.graph.tools import forget, identify, send_warm_lead_at_call_end
from ai_call_center.leads import Lead, now_iso
from ai_call_center.names import same_person
from tests.conftest import OWNER, scripted, tool_call

NUMBER = "+15555550123"  # the number of the `ctx` fixture


def _lead(leads, **over):
    lead = Lead(name="Steve Park", email="steve@acme.test", company="Acme Roofing", need="An agent for after-hours calls.", caller_number=NUMBER, call_control_id="call-1", created_at=now_iso())
    leads.save(Lead(**{**lead.__dict__, **over}))


def _call(leads, call_id, name, summary, topic="", started=""):
    leads.start_call(call_id, NUMBER)
    leads.finish_call(call_id, agents=["classifier"], summary=summary, caller_name=name, topic=topic)
    if started:
        leads._conn.execute("UPDATE calls SET started_at = ? WHERE call_control_id = ?", (started, call_id))


def _welcome():
    return Welcome(owner=OWNER, companies="on models from three companies", tip="what this call has cost so far", rag=True)


# --- the name check ---


def test_names_match_by_spelling_short_form_and_sound():
    for spoken, stored in [
        ("steve", "Steve Park"),
        ("Stephen", "Steven"),  # sound
        ("Jon", "John"),  # sound and short form
        ("Bob", "Robert"),  # short form
        ("Katherine", "Catherine"),  # a short-form group
        ("José", "Jose"),  # accents
        ("Sean", "Shawn"),
    ]:
        assert same_person(spoken, stored), (spoken, stored)


def test_different_names_and_empty_names_do_not_match():
    for spoken, stored in [("Bill", "Steve"), ("Phil", "Bill"), ("Dave", "Steve"), ("Sam", ""), ("", "Sam"), ("Al", "Ali")]:
        assert not same_person(spoken, stored), (spoken, stored)


# --- recall: one person's records ---


def test_first_time_and_hidden_numbers_have_no_memory(leads):
    assert recall(leads, NUMBER, "Steve") is None
    leads.start_call("c1", "anonymous")
    leads.finish_call("c1", agents=["classifier"], summary="Asked about pricing.", caller_name="Steve")
    assert recall(leads, "anonymous", "Steve") is None
    assert recall(leads, NUMBER, "") is None


def test_recall_combines_lead_and_finished_calls(leads):
    _lead(leads)
    _call(leads, "call-1", "Steve", "Left details; wants a call back.", topic="an after-hours phone agent")
    leads.start_call("call-2", NUMBER)  # the call in progress is not history yet

    memory = recall(leads, NUMBER, "Stephen", exclude_call="call-2")

    assert memory.name == "Steve Park"  # the lead holds the name as the caller confirmed it
    assert memory.company == "Acme Roofing" and memory.contact_on_file
    assert memory.previous_calls == 1 and memory.topic == "an after-hours phone agent"
    assert [c.summary for c in memory.recent] == ["Left details; wants a call back."]


def test_each_name_on_a_shared_number_gets_only_its_own_records(leads):
    # Steve left a lead with an email address. Later, Bill called from the same number and left no lead.
    _lead(leads)
    _call(leads, "call-1", "Steve", "Steve asked about a shared project.", started="2026-11-16T03:08:30+00:00")
    _call(leads, "call-2", "Bill", "Bill asked about the owner.", topic="the owner's background", started="2026-11-16T14:32:29+00:00")

    bill = recall(leads, NUMBER, "Bill")
    assert [c.summary for c in bill.recent] == ["Bill asked about the owner."]
    assert (bill.company, bill.need, bill.contact_on_file) == ("", "", False)
    text = history_text(bill)
    assert "Steve" not in text and "Acme" not in text and "on file" not in text

    steve = recall(leads, NUMBER, "Steve")
    assert steve.company == "Acme Roofing" and [c.call_control_id for c in steve.recent] == ["call-1"]
    assert recall(leads, NUMBER, "Maria") is None


def test_a_lead_from_the_call_in_progress_is_not_history(leads):
    leads.start_call("call-2", NUMBER)
    _lead(leads, name="Bill Williams", call_control_id="call-2")
    assert recall(leads, NUMBER, "Bill", exclude_call="call-2") is None


def test_history_never_holds_the_email_or_the_number(leads):
    _lead(leads)
    _call(leads, "call-1", "", "Wants an after-hours agent.")  # no name in the log; the lead ties the call to Steve
    text = history_text(recall(leads, NUMBER, "Steve"))
    assert "Steve Park from Acme Roofing" in text and "Wants an after-hours agent." in text
    assert "steve@acme.test" not in text and NUMBER not in text
    assert 'call save_lead with email "on file"' in text


# --- the greeting and identify_caller ---


def test_every_caller_hears_the_same_first_words():
    words = greeting_words(OWNER, "Digital Reformer")
    assert words == (
        f"Hi, you've reached Digital Reformer. I'm one of six AI agents {OWNER} built, and this call is a live sample of the work "
        f"{OWNER.split()[0]} does for clients. This call is recorded and reviewed by AI to improve the line. "
        "I remember callers, so first: who am I speaking with?"
    )


def test_the_company_count_comes_from_the_legs():
    assert companies_phrase({"a": "openai_realtime", "b": "gemini_live", "c": "nova_sonic", "d": "openai_realtime"}) == "on models from three companies"
    assert companies_phrase({"a": "openai_realtime"}) == "on OpenAI models"


def test_a_new_caller_hears_the_offer_and_a_tip(ctx):
    ctx.welcome = _welcome()
    result = identify(ctx, "Sam Lee")
    first = OWNER.split()[0]
    assert f"Nice to meet you, Sam. {first} is a software engineer and architect, available for contract work or a full-time role." in result
    assert "Try asking me what this call has cost so far." in result and "RAG search" in result
    assert ctx.caller_name == "Sam Lee" and not ctx.identity_confirmed and ctx.identified


def test_a_matching_name_gets_a_welcome_back_and_no_notes(ctx, leads):
    _lead(leads)
    _call(leads, "call-1", "Steve", "Asked about an after-hours agent.", topic="an after-hours phone agent")
    ctx.welcome = _welcome()
    result = identify(ctx, "Stephen")
    assert '"Welcome back, Stephen. Good to hear from you again. First, so I can verify you next time:' in result  # no passphrase yet: asked to set one
    # A phone number can be faked: a match of number and name opens no notes. They need the passphrase (test_passphrase.py).
    for detail in ("Acme Roofing", "after-hours", "Last time"):
        assert detail not in result
    assert ctx.identity_confirmed and not ctx.identity_verified and ctx.caller_history == ""
    # The agents get the name that the caller said on this call. The stored name waits for the passphrase check.
    assert ctx.returning_name == "Steve Park" and ctx.caller_name == "Stephen"


def test_another_name_on_the_same_number_hears_nothing_about_the_earlier_caller(ctx, leads):
    _lead(leads)
    _call(leads, "call-1", "Steve", "Asked about an after-hours agent.", topic="an after-hours phone agent")
    ctx.welcome = _welcome()
    result = identify(ctx, "Maria")
    for secret in ("Steve", "Acme", "after-hours", "Welcome back", "steve@acme.test"):
        assert secret not in result
    assert "Nice to meet you, Maria." in result and not ctx.identity_confirmed and ctx.caller_history == ""


def test_no_name_and_a_hidden_number_are_new_callers(ctx, leads):
    _lead(leads)
    _call(leads, "call-1", "Steve", "Asked about an after-hours agent.")
    ctx.welcome = _welcome()
    assert "Welcome back" not in identify(ctx, "")
    assert ctx.caller_name == "" and not ctx.identity_confirmed
    ctx.identified, ctx.caller_number = False, "anonymous"
    assert "Welcome back" not in identify(ctx, "Steve")


def test_identify_runs_once_per_call(ctx):
    ctx.welcome = _welcome()
    identify(ctx, "Sam")
    again = identify(ctx, "Steve")
    assert "already identified as Sam" in again and ctx.caller_name == "Sam"


async def test_the_history_reaches_a_model_only_through_identify_caller(make_graph, ctx, config, leads):
    _lead(leads)
    _call(leads, "call-1", "Steve", "Asked about an after-hours agent.")
    model = scripted(tool_call("identify_caller", name="Maria"), "Nice to meet you, Maria.", tool_call("transfer_to_profile"), "Hi, I'm the profile agent.")
    graph = make_graph(model)
    await graph.ainvoke({"messages": [HumanMessage("This is Maria.")]}, config=config, context=ctx)
    await graph.ainvoke({"messages": [HumanMessage("Tell me about the owner.")]}, config=config, context=ctx)
    seen = " ".join(str(m.content) for call in model.calls for m in call)
    assert "Acme Roofing" not in seen and "Steve" not in seen
    assert "Returning caller" not in model.system_prompt_of_call(0)


async def test_identify_caller_runs_in_the_graph(make_graph, ctx, config, leads):
    _lead(leads)
    model = scripted(tool_call("identify_caller", name="Steve"), "Welcome back, Steve.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("It's Steve.")]}, config=config, context=ctx)
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Welcome Steve back" in tool_msg.content and "Acme Roofing" not in tool_msg.content and ctx.identity_confirmed


# --- forget me ---


def test_forget_deletes_that_persons_records_and_keeps_the_call_rows(ctx, leads):
    _lead(leads)
    _call(leads, "call-1", "Steve", "Asked about an after-hours agent.", topic="an after-hours phone agent")
    _call(leads, "call-2", "Bill", "Bill asked about the owner.")
    leads.start_call(ctx.call_control_id, NUMBER)
    identify(ctx, "Steve")

    result, _ = forget(ctx)

    assert "This line won't remember you." in result
    assert recall(leads, NUMBER, "Steve") is None
    assert recall(leads, NUMBER, "Bill") is not None  # another person on the number keeps their records
    assert leads.call_count_for(NUMBER, exclude=ctx.call_control_id) == 2  # the call limits still count Steve's call
    assert ctx.forgotten and not ctx.identity_confirmed and ctx.caller_history == ""


async def test_a_forgotten_caller_gets_no_warm_lead_email(ctx, notifier):
    ctx.forgotten = True
    summary = CallSummary(summary="Wants a booking platform.", topic="a booking platform", interest="project")
    assert not await send_warm_lead_at_call_end(ctx, summary, lead_saved=False)
    assert notifier.sent == []


# --- the warm lead ---


async def test_a_project_with_no_message_reaches_the_owner_as_a_warm_lead(ctx, notifier):
    from ai_call_center.notify import plain_text

    ctx.caller_name = "Sam Lee"
    summary = CallSummary(summary="Sam wants a booking platform for a clinic.", topic="a booking platform", interest="project")
    assert await send_warm_lead_at_call_end(ctx, summary, lead_saved=False, visited=["classifier", "consultant"])
    email = notifier.sent[0]
    assert email["Subject"] == "Voice card: Warm lead: Sam Lee talked about a project, no message left"
    assert "Sam wants a booking platform for a clinic." in plain_text(email) and "Lead record" not in plain_text(email)


async def test_no_warm_lead_without_interest_with_a_message_or_from_a_hidden_number(ctx, notifier):
    project = CallSummary(summary="Wants a booking platform.", interest="project")
    assert not await send_warm_lead_at_call_end(ctx, CallSummary(summary="Asked how it works.", interest="none"), lead_saved=False)
    assert not await send_warm_lead_at_call_end(ctx, project, lead_saved=True)
    ctx.caller_number = "anonymous"
    assert not await send_warm_lead_at_call_end(ctx, project, lead_saved=False)
    assert notifier.sent == []


# --- the summary ---


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
    model = scripted(
        '{"summary": "Steve asked about an after-hours agent and left his details.", "caller_name": "Steve",'
        ' "topic": "an after-hours phone agent for a roofing company in Ohio today", "interest": "Project"}'
    )
    summary = await summarize_call([{"role": "user", "content": "Hi, this is Steve."}], model)
    assert summary.summary.startswith("Steve asked") and summary.caller_name == "Steve"
    assert summary.topic == "an after-hours phone agent for a roofing company"  # code keeps it to 8 words
    assert summary.interest == "project"


async def test_summarize_call_without_a_topic_or_with_a_bad_interest():
    model = scripted('{"summary": "Asked how the system works.", "caller_name": "", "interest": "maybe"}')
    summary = await summarize_call([{"role": "user", "content": "How does this work?"}], model)
    assert summary.topic == "" and summary.interest == "none"


async def test_summarize_call_skips_a_call_with_no_caller_speech():
    model = scripted("should not be used")
    assert await summarize_call([{"role": "assistant", "content": "Hello?"}], model) == CallSummary()
    assert model.calls == []


# --- the 2026-11-17 call: details given, no message saved ---

STEVE_SUMMARY = CallSummary(
    summary="Steve from Steve Corp wants to hire Brad for a consulting role on refinery safety software.",
    caller_name="Steve Stevens",
    topic="a consulting role on safety software",
    interest="role",
    owner_note="Steve runs a consulting agency that builds safety software for oil refineries and paper mills. "
    "He wants Brad to join the team on the site-safety analysis app. Start in two weeks, ongoing, at 100 dollars an hour.",
    timeline="in two weeks",
    size="ongoing, no end date",
)


async def test_summarize_call_reads_the_note_for_the_owner():
    model = scripted(
        '{"summary": "x", "caller_name": "Steve", "topic": "a role", "interest": "role",'
        ' "owner_note": "Builds safety software. Start in two weeks at 100 dollars an hour.", "timeline": "in two weeks", "size": "ongoing"}'
    )
    summary = await summarize_call([{"role": "user", "content": "Hi."}], model)
    assert summary.owner_note.startswith("Builds safety software.") and (summary.timeline, summary.size) == ("in two weeks", "ongoing")


async def test_a_refused_transfer_note_holds_what_the_caller_said(ctx, notifier):
    from ai_call_center.graph.tools import note_refused_transfer_at_call_end, remember_refused_transfer
    from ai_call_center.notify import plain_text

    remember_refused_transfer(ctx, caller_name="Steve Stevens", company="Steve Corp.", reason="The caller wants to connect with Brad now.")
    lead = await note_refused_transfer_at_call_end(ctx, lead_saved=False, summary=STEVE_SUMMARY)
    assert lead
    text = plain_text(notifier.sent[-1])
    assert "oil refineries and paper mills" in text and "100 dollars an hour" in text
    assert "Wants to start: in two weeks" in text and "Size of the work: ongoing, no end date" in text
    assert "connect with Brad now" not in text


async def test_a_warm_lead_follows_a_transfer_note_and_keeps_the_email_on_file(ctx, notifier, leads):
    from ai_call_center.graph.tools import message_saved, note_caller_before_transfer
    from ai_call_center.notify import plain_text

    # The owner's phone rang before the caller left a message: the note says only who called.
    await note_caller_before_transfer(ctx, lead_saved=False, caller_name="Steve Stevens", company="Steve Corp.", reason="Wants to talk.")
    leads._conn.execute("UPDATE leads SET email = 'steve@test.example'")
    assert not message_saved(ctx)
    assert await send_warm_lead_at_call_end(ctx, STEVE_SUMMARY, lead_saved=message_saved(ctx))
    text = plain_text(notifier.sent[-1])
    assert "oil refineries" in text and "steve@test.example" in text and "Wants to start: in two weeks" in text


def test_a_yes_to_the_read_back_with_no_save_is_caught_once(ctx):
    from ai_call_center.graph.tools import missed_save

    lines = [
        "agent: You're looking to hire Brad for a consulting role starting in two weeks, at 100 dollars per hour. Is that right, Steve?",
        "caller: Yeah.",
    ]
    assert missed_save(ctx, lines) == 0  # the agent has not answered yet
    answered = [*lines, "agent: Perfect, thanks. I'll take that as confirming the summary."]
    assert missed_save(ctx, answered) == 2
    assert missed_save(ctx, [*lines[:1], "caller: No, the rate is wrong.", answered[-1]]) == 0
    ctx.leads.save(Lead(name="Steve", email="", company="", need="A role.", caller_number=NUMBER, call_control_id=ctx.call_control_id, created_at=now_iso()))
    assert missed_save(ctx, answered) == 0  # saved


# --- the summary check, with one more try for the model ---


async def test_a_summary_with_a_fault_goes_back_to_the_model_one_time_with_the_fault_named():
    from ai_call_center.callers import summary_faults

    model = scripted(
        '{"summary": "Steve wants a call back on 555-555-0123 about a booking platform.", "topic": "a booking platform", "interest": "project"}',
        '{"summary": "Steve wants a call back about a booking platform.", "topic": "a booking platform", "interest": "project"}',
    )
    summary = await summarize_call([{"role": "user", "content": "Hi, this is Steve."}], model)
    assert summary.tries == 2 and summary.summary == "Steve wants a call back about a booking platform." and not summary_faults(summary)
    second_request = model.calls[1]
    assert len(second_request) == 3 and "555-555-0123" in second_request[1].content  # its own first answer
    assert 'The "summary" holds an email address or a phone number.' in second_request[2].content


async def test_a_summary_with_no_fault_is_one_request():
    model = scripted('{"summary": "Asked how the system works.", "interest": "none"}', "should not be used")
    summary = await summarize_call([{"role": "user", "content": "How does this work?"}], model)
    assert summary.tries == 1 and len(model.calls) == 1


async def test_code_removes_what_is_still_wrong_after_the_second_try():
    long_summary = "Steve asked about a platform. " * 12  # 60 words
    bad = '{"summary": "%s", "topic": "call 555 555 0123", "owner_note": "Write to steve@acme.test or +1 (555) 555-0123.", "interest": "project"}' % long_summary.strip()
    model = scripted(bad, bad)
    summary = await summarize_call([{"role": "user", "content": "Hi, this is Steve."}], model)
    assert summary.tries == 2 and len(model.calls) == 2  # two tries, never a third
    assert len(summary.summary.split()) <= 40 and summary.summary.endswith("platform.")  # whole sentences
    assert summary.topic == ""  # a topic is spoken to the caller: a cut one is worse than none
    assert "steve@acme.test" not in summary.owner_note and "555" not in summary.owner_note and "[removed]" in summary.owner_note


async def test_an_answer_that_is_not_json_gets_a_second_try():
    model = scripted("I cannot do that.", '{"summary": "Asked about a role.", "interest": "role"}')
    summary = await summarize_call([{"role": "user", "content": "I have a role."}], model)
    assert (summary.summary, summary.interest, summary.tries) == ("Asked about a role.", "role", 2)


async def test_when_the_second_try_fails_the_first_answer_stands_after_the_repair():
    model = scripted('{"summary": "Steve left 555-555-0123 for a call back.", "interest": "project"}')  # the model has no second answer
    summary = await summarize_call([{"role": "user", "content": "Hi."}], model)
    assert summary.summary == "Steve left [removed] for a call back." and summary.interest == "project" and summary.tries == 1


def test_contact_details_finds_addresses_and_phone_numbers_and_leaves_dates_and_prices():
    from ai_call_center.callers import contact_details

    assert contact_details("Mail dana.b+work@acme-roofing.co.uk or ring +1 (555) 555-0123.") == ["dana.b+work@acme-roofing.co.uk", "+1 (555) 555-0123"]
    assert contact_details("Start on 2026-12-01, at 150 dollars an hour, for 10,000,000 users, about 40 hours.") == []
