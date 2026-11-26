"""Side-effect tools run inside the graph with the call context."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage

from ai_call_center.graph.tools import note_caller_before_transfer, note_refused_transfer_at_call_end, save_lead_record
from ai_call_center.telephony import TransferRequest
from ai_call_center.notify import plain_text
from tests.conftest import OWNER_PHONE, scripted, tool_call


async def test_save_lead_writes_one_row(make_graph, ctx, config, leads):
    model = scripted(
        tool_call("save_lead", name="Ada Lovelace", email="ADA@Example.com ", company="Analytical Engines", need="A voice agent for inbound sales calls."),
        "Saved. Shall I connect you to the owner now?",
    )
    graph = make_graph(model)

    result = await graph.ainvoke(
        {"messages": [HumanMessage("Yes, that is all correct.")], "active_agent": "contact"},
        config=config,
        context=ctx,
    )

    rows = leads.all()
    assert len(rows) == 1
    assert rows[0].name == "Ada Lovelace"
    assert rows[0].email == "ada@example.com"
    assert rows[0].caller_number == ctx.caller_number
    assert rows[0].call_control_id == ctx.call_control_id
    assert result["lead"]["email"] == "ada@example.com"
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[-1].content.startswith("Lead saved as record")


async def test_request_owner_transfer_calls_telephony_once(make_graph, ctx, config, telephony):
    model = scripted(tool_call("request_owner_transfer", reason="Wants to discuss a project."), "Connecting you now.")
    graph = make_graph(model)

    result = await graph.ainvoke(
        {"messages": [HumanMessage("Yes, please put me through.")], "active_agent": "contact"},
        config=config,
        context=ctx,
    )

    assert telephony.transfers == [TransferRequest(call_control_id=ctx.call_control_id, to=OWNER_PHONE)]
    assert result["transfer_requested"] is True


async def test_get_call_stats_reports_agents_and_legs(make_graph, ctx, config):
    model = scripted(tool_call("get_call_stats"), "You have been talking for a few seconds.")
    graph = make_graph(model)

    result = await graph.ainvoke(
        {"messages": [HumanMessage("What has happened so far?")], "active_agent": "tour", "handoffs": ["tour"]},
        config=config,
        context=ctx,
    )

    stats = next(m.content for m in result["messages"] if isinstance(m, ToolMessage))
    assert "text conversation for testing" in stats
    assert "classifier on Google gemini-3.8-live" in stats
    assert "tour on OpenAI gpt-realtime-2.1-mini" in stats
    assert "Current agent: tour" in stats


async def test_save_lead_emails_owner_without_leaking_address(make_graph, ctx, config, notifier):
    model = scripted(
        tool_call("save_lead", name="Ada Lovelace", email="ada@example.com", company="Analytical Engines", need="A voice agent for inbound sales calls."),
        "Done. The owner has your details.",
    )
    graph = make_graph(model)

    result = await graph.ainvoke(
        {"messages": [HumanMessage("Yes, that is correct.")], "active_agent": "contact"},
        config=config,
        context=ctx,
    )

    assert len(notifier.sent) == 1
    email = notifier.sent[0]
    assert email["To"] == "owner@example.test"
    assert email["Reply-To"] == "ada@example.com"
    assert email["Subject"] == "Voice card: Message from Ada Lovelace (Analytical Engines)"
    body = plain_text(email)
    assert "A voice agent for inbound sales calls." in body
    assert "Call back: +1 (555) 555-0123" in body and "Spoke with: classifier agent" in body
    assert "tel:+15555550123" in email.get_body(preferencelist=("html",)).get_content()
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "emailed to the owner." in tool_msg.content and "by email" in tool_msg.content
    assert "owner@example.test" not in str(result["messages"])


async def test_save_lead_keeps_row_when_email_fails(make_graph, ctx, config, leads):
    class BrokenNotifier:
        async def send_lead(self, lead, *, row_id, details=None):
            raise ConnectionError("SMTP down")

    ctx.notifier = BrokenNotifier()
    model = scripted(
        tool_call("save_lead", name="Ada", email="ada@example.com", company="none", need="A chat agent."),
        "Thanks, the owner will see your details.",
    )
    graph = make_graph(model)

    result = await graph.ainvoke(
        {"messages": [HumanMessage("Yes.")], "active_agent": "contact"},
        config=config,
        context=ctx,
    )

    assert len(leads.all()) == 1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "email to the owner failed" in tool_msg.content


def test_lead_email_drops_bad_reply_to_and_newlines():
    from ai_call_center.leads import Lead
    from ai_call_center.notify import lead_email

    lead = Lead(name="Eve\r\nBcc: x@evil.test", email="not an email", company="none", need="n",
                caller_number="+1", call_control_id="c", created_at="2026-11-15T17:30:00+00:00")
    msg = lead_email(lead, row_id=1, sender="from@example.test", to="owner@example.test")

    assert msg["Reply-To"] is None
    assert "\n" not in msg["Subject"] and "Bcc" not in msg.keys()
    assert "12:30 PM EST" in plain_text(msg)
    # A "none" field is left out, not printed. A misheard email address is still shown, with no link.
    assert "Company:" not in plain_text(msg) and "Email: not an email" in plain_text(msg)
    assert "mailto:" not in msg.get_body(preferencelist=("html",)).get_content()


def test_lead_email_escapes_caller_speech_and_shows_the_call_details():
    from ai_call_center.leads import Lead
    from ai_call_center.notify import CallDetails, lead_email

    lead = Lead(name="Bill Williams", email="", company="", need='Wants to hire the owner. <script>alert("x")</script>',
                caller_number="+13305550123", call_control_id="c", created_at="2026-11-16T18:01:52+00:00")
    details = CallDetails(notes="Asked the tour agent what the call costs.", agents=("classifier", "tour", "contact"), seconds=169,
                          earlier_calls=7, line="mobile on Example Wireless", area="Akron, OH", registered_name="WILLIAMS BILL")
    msg = lead_email(lead, row_id=6, sender="from@example.test", to="owner@example.test", details=details)

    html = msg.get_body(preferencelist=("html",)).get_content()
    assert "<script>" not in html and "&lt;script&gt;" in html
    text = plain_text(msg)
    assert text.startswith("Bill Williams left you a message on your voice business card.")
    for line in (
        "Call length so far: 2 min 49 s",
        "Also on this call: Asked the tour agent what the call costs.",
        "Earlier calls from this number: 7",
        "Phone line: mobile on Example Wireless, Akron, OH",
        "Name on the phone account: Williams Bill",
    ):
        assert line in text

    transfer = lead_email(lead, row_id=6, sender="from@example.test", to="owner@example.test", details=CallDetails(kind="transfer"))
    assert transfer["Subject"] == "Voice card: Bill Williams asked to speak to you"


async def test_save_lead_does_not_claim_an_email_when_smtp_is_not_set_up(make_graph, ctx, config):
    from ai_call_center.notify import RecordingNotifier

    ctx.notifier = RecordingNotifier(delivers=False)
    model = scripted(tool_call("save_lead", name="Ada", email="ada@example.com", company="none", need="A demo."), "Saved.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config=config, context=ctx)
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "do not say an email was sent" in tool_msg.content


async def test_text_portfolio_link_goes_only_to_the_calling_number_once(make_graph, ctx, config, telephony):
    ctx.texter, ctx.portfolio_url = telephony, "https://example.test/work"
    model = scripted(tool_call("text_portfolio_link"), "Sent.", tool_call("text_portfolio_link", call_id="call_2"), "Already sent.")
    graph = make_graph(model)
    await graph.ainvoke({"messages": [HumanMessage("Yes, text me.")], "active_agent": "profile"}, config=config, context=ctx)
    result = await graph.ainvoke({"messages": [HumanMessage("Text it again.")]}, config=config, context=ctx)

    assert len(telephony.texts) == 1
    to, body = telephony.texts[0]
    assert to == ctx.caller_number and body.endswith("https://example.test/work")
    assert "already texted" in [m for m in result["messages"] if isinstance(m, ToolMessage)][-1].content


async def test_text_portfolio_link_refuses_when_texting_is_off_or_a_landline(make_graph, ctx, config, telephony):
    from ai_call_center.graph.tools import text_portfolio_link_to_caller

    assert "not available" in await text_portfolio_link_to_caller(ctx, owner_name="Test Owner")  # no texter
    ctx.texter, ctx.portfolio_url, ctx.caller_line_type = telephony, "https://example.test/work", "fixed line"
    assert "landline" in await text_portfolio_link_to_caller(ctx, owner_name="Test Owner")
    assert telephony.texts == []


async def test_owner_gets_a_text_before_a_transfer_when_texting_is_on(make_graph, ctx, config, telephony):
    ctx.texter, ctx.caller_name = telephony, "Steve Park"
    model = scripted(tool_call("request_owner_transfer", reason="Wants a quote."), "Connecting you now.")
    await make_graph(model).ainvoke({"messages": [HumanMessage("Yes, connect me.")], "active_agent": "contact"}, config=config, context=ctx)
    assert telephony.texts == [(OWNER_PHONE, "AI line: connecting Steve Park to you now. Reason: Wants a quote.")]
    assert len(telephony.transfers) == 1


def test_text_tool_is_offered_only_when_texting_is_set_up(specs):
    from ai_call_center.agents.specs import Profile, build_specs

    assert all("text_portfolio_link" not in s.tools for s in specs.values())
    on = build_specs(Profile(summary="x", full="x"), owner_name="Test Owner", texting=True)
    assert [n for n, s in on.items() if "text_portfolio_link" in s.tools] == ["profile", "tour", "consultant", "contact"]
    assert "text_portfolio_link" in on["profile"].instructions and "text_portfolio_link" not in on["classifier"].instructions


# --- a short contact flow: name and need are enough ---


async def test_save_lead_needs_only_a_name_and_a_need(make_graph, ctx, config, leads):
    model = scripted(tool_call("save_lead", name="Ada Lovelace", need="Wants to talk about a project."), "Done.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config=config, context=ctx)

    (lead,) = leads.all()
    assert (lead.name, lead.email, lead.company) == ("Ada Lovelace", "", "")
    assert lead.caller_number == ctx.caller_number  # how the owner reaches them
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "on the number they are calling from" in tool_msg.content


async def test_a_hidden_number_needs_an_email_or_a_callback_number(make_graph, ctx, config, leads):
    ctx.caller_number = "anonymous"
    model = scripted(
        tool_call("save_lead", name="Ada", need="A demo."),
        tool_call("save_lead", call_id="call_2", name="Ada", need="A demo.", callback_number="555-0142"),
        "Done.",
    )
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config=config, context=ctx)

    first, second = (m for m in result["messages"] if isinstance(m, ToolMessage))
    assert first.content.startswith("Not saved yet") and "callback number" in first.content
    (lead,) = leads.all()
    assert "555-0142" in lead.need and "on the callback number they gave" in second.content


async def test_email_on_file_is_reused_only_after_the_passphrase_check(make_graph, ctx, config, leads):
    from ai_call_center.leads import Lead, now_iso

    leads.save(Lead(name="Steve Park", email="steve@acme.test", company="Acme Roofing", need="Old.", caller_number=ctx.caller_number, call_control_id="old", created_at=now_iso()))

    # Not confirmed: a model that asks for the address on file gets none.
    model = scripted(tool_call("save_lead", name="Someone Else", need="A question.", email="on file"), "Done.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config=config, context=ctx)
    assert leads.all()[-1].email == "" and leads.all()[-1].company == ""
    assert "steve@acme.test" not in str(result["messages"])

    # Recognised by number and name only: a phone number can be faked, so still no address.
    ctx.identity_confirmed, ctx.returning_name = True, "Steve Park"
    model = scripted(tool_call("save_lead", name="Steve Park", need="A first project.", email="on file"), "Done.")
    await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config={"configurable": {"thread_id": "t1"}}, context=ctx)
    assert leads.all()[-1].email == ""

    # Verified by passphrase: the address goes from the store to the new row, and never into a message.
    ctx.identity_verified, ctx.open_calls = True, ["old"]  # "old" is a call that the passphrase opens
    model = scripted(tool_call("save_lead", name="Steve Park", need="A shared project.", email="on file"), "Done.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config={"configurable": {"thread_id": "t2"}}, context=ctx)
    assert leads.all()[-1].email == "steve@acme.test" and leads.all()[-1].company == "Acme Roofing"
    assert "steve@acme.test" not in str(result["messages"])
    assert "by email" in next(m for m in result["messages"] if isinstance(m, ToolMessage)).content


async def test_details_on_file_are_not_given_to_another_person_on_the_same_number(make_graph, ctx, config, leads):
    from ai_call_center.leads import Lead, now_iso

    leads.save(Lead(name="Steve Park", email="steve@acme.test", company="Acme Roofing", need="Old.", caller_number=ctx.caller_number, call_control_id="old", created_at=now_iso()))

    # Bill confirmed that he is Bill. Steve's address and company belong to Steve.
    ctx.identity_confirmed, ctx.returning_name = True, "Bill"
    model = scripted(tool_call("save_lead", name="Bill Williams", need="Wants to hire the owner.", email="on file"), "Done.")
    await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config=config, context=ctx)
    assert (leads.all()[-1].email, leads.all()[-1].company) == ("", "")

    # A caller who confirmed as Steve, and then gives another name, gets nothing from the file either.
    ctx.returning_name = "Steve Park"
    model = scripted(tool_call("save_lead", name="Dana Cole", need="A question.", email="on file"), "Done.")
    await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config={"configurable": {"thread_id": "t2"}}, context=ctx)
    assert (leads.all()[-1].email, leads.all()[-1].company) == ("", "")


# --- the agent hangs up after the caller says goodbye ---


async def test_end_call_hangs_up_after_a_goodbye(make_graph, ctx, config, telephony):
    model = scripted(tool_call("end_call"), "Thanks for calling. Goodbye.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("No, that is all. Thank you.")], "active_agent": "contact"}, config=config, context=ctx)

    assert telephony.hangups == [ctx.call_control_id] and result["call_ended"] is True
    assert "goodbye line" in next(m for m in result["messages"] if isinstance(m, ToolMessage)).content


async def test_end_call_does_not_hang_up_on_a_caller_who_is_still_talking(make_graph, ctx, config, telephony):
    model = scripted(tool_call("end_call"), "Sure. What else can I help with?")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Thanks. And what does he charge?")], "active_agent": "profile"}, config=config, context=ctx)

    assert telephony.hangups == [] and not result.get("call_ended")
    assert next(m for m in result["messages"] if isinstance(m, ToolMessage)).content.startswith("The call was not ended")


def test_only_the_callers_own_goodbye_ends_the_call():
    from ai_call_center.graph.tools import said_goodbye

    for said in ("Bye.", "Okay, goodbye!", "No, that's all.", "That is everything, thanks.", "No thanks.", "Nope.", "No, I don't.", "Not really.", "I'm good.", "Have a great day.", "I have to go.", "You can hang up now."):
        assert said_goodbye([f"caller: {said}"]), said
    for said in ("Thanks. What else can you do?", "No, this is Bill.", "Tell me about his background.", "Yes.", "I would like to leave a message."):
        assert not said_goodbye([f"caller: {said}"]), said
    # The agent's goodbye does not count, and neither does an old one from the caller.
    assert not said_goodbye(["agent: Goodbye!", "caller: Wait, one more question."])
    assert not said_goodbye(["caller: Bye.", "caller: Actually, wait.", "caller: How does this work?"])


# --- a message is not a transfer ---


async def test_a_caller_who_wants_to_leave_a_message_is_not_transferred(make_graph, ctx, config, leads, telephony):
    model = scripted(
        tool_call("request_owner_transfer", reason="Interested in hiring the owner.", caller_name="Bill Williams"),
        "Sure. What should I tell him?",
    )
    result = await make_graph(model).ainvoke(
        {"messages": [HumanMessage("I would like to leave a message with the owner.")], "active_agent": "contact"}, config=config, context=ctx
    )

    assert telephony.transfers == [] and not result.get("transfer_requested")
    assert leads.all() == []  # a held-back transfer saves nothing and emails nothing
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content.startswith("No transfer was made") and "save_lead" in tool_msg.content


async def test_the_answer_to_the_agents_question_allows_the_transfer(make_graph, ctx, config, telephony):
    model = scripted(
        tool_call("request_owner_transfer", reason="A project.", caller_name="Ada"),
        "Do you want me to take a message, or connect you to him now?",
        tool_call("request_owner_transfer", call_id="call_2", reason="A project.", caller_name="Ada"),
        "Connecting you now.",
    )
    graph = make_graph(model)
    await graph.ainvoke({"messages": [HumanMessage("I need to reach him about a project.")], "active_agent": "contact"}, config=config, context=ctx)
    assert telephony.transfers == []
    await graph.ainvoke({"messages": [HumanMessage("The second one, please.")]}, config=config, context=ctx)
    assert len(telephony.transfers) == 1


def test_only_the_callers_own_words_count_as_a_request_for_a_live_call():
    from ai_call_center.graph.tools import asked_for_the_owner

    for said in ("Can I speak to him?", "I'd like to talk to the owner.", "Please transfer me.", "Connect me.", "Put me through.", "Is there a real person?"):
        assert asked_for_the_owner([f"caller: {said}"]), said
    for said in ("Can you call Brad?", "could you please call him now", "Would you ring the owner?"):  # the caller asks the agent to ring someone now
        assert asked_for_the_owner([f"caller: {said}"]), said
    for said in ("I would like to leave a message with him.", "Can you email him?", "Have him call me back.", "I want to hire him.", "Yes."):
        assert not asked_for_the_owner([f"caller: {said}"]), said
    for said in ("Can you call me back?", "Can you have Brad call me?", "Could you call my office tomorrow?", "Can you call back later?"):
        assert not asked_for_the_owner([f"caller: {said}"]), said
    # What the agent said does not count, and neither does an old turn.
    assert not asked_for_the_owner(["agent: I can connect you to the owner.", "caller: No, a message."])
    assert not asked_for_the_owner(["caller: Can I talk to him?", "caller: a", "caller: b", "caller: c", "caller: Just a message."])


def test_call_notes_that_repeat_the_message_are_left_out_of_the_email():
    from ai_call_center.graph.tools import says_the_same

    message = "Bill is interested in hiring Brad to build a voice agent for his farm equipment company starting in January."
    assert says_the_same("Caller wants to discuss hiring Brad for a voice agent project.", message)
    assert not says_the_same("Asked the tour agent how the system works and what a call costs.", message)
    assert not says_the_same("", message)


async def test_a_saved_message_does_not_lead_to_a_transfer(make_graph, ctx, config):
    model = scripted(tool_call("save_lead", name="Bill Williams", need="Wants to hire the owner for a voice agent project."), "He has your message.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes, that is right.")], "active_agent": "contact"}, config=config, context=ctx)
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Do not call request_owner_transfer" in tool_msg.content and "offer once to connect" not in tool_msg.content


def test_the_contact_agent_can_say_it_emails_only_when_email_is_set_up(specs):
    from ai_call_center.agents.specs import Profile, build_specs

    assert "say you can pass their message" in specs["contact"].instructions
    on = build_specs(Profile(summary="x", full="x"), owner_name="Test Owner", emailing=True)
    assert "Yes, you can email Test Owner for the caller" in on["contact"].instructions


async def test_no_transfer_is_attempted_when_the_server_cannot_make_one(make_graph, ctx, config, telephony):
    ctx.transfer_unavailable = "the Telnyx application has no outbound voice profile"
    model = scripted(tool_call("request_owner_transfer", reason="Wants a quote."), "He cannot take the call right now. He will follow up.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Connect me.")], "active_agent": "contact"}, config=config, context=ctx)
    assert telephony.transfers == [] and not result.get("transfer_requested")
    assert "No transfer was made" in next(m for m in result["messages"] if isinstance(m, ToolMessage)).content


# --- "connect me": one tool call transfers the call and leaves a lead ---


async def test_a_transfer_saves_who_called_so_an_unanswered_call_still_leaves_a_lead(make_graph, ctx, config, leads, telephony):
    model = scripted(tool_call("request_owner_transfer", reason="Wants to talk about a project.", caller_name="Ada Lovelace", company="Analytical Engines"), "Connecting you now.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Connect me.")], "active_agent": "contact"}, config=config, context=ctx)

    (lead,) = leads.all()
    assert (lead.name, lead.company, lead.need, lead.caller_number) == ("Ada Lovelace", "Analytical Engines", "Wants to talk about a project.", ctx.caller_number)
    assert len(telephony.transfers) == 1 and result["transfer_requested"] and result["lead"]["name"] == "Ada Lovelace"


async def test_a_transfer_does_not_save_a_second_lead_or_a_nameless_one(make_graph, ctx, config, leads, telephony):
    # No name given and none known: the call is transferred, and nothing is stored.
    model = scripted(tool_call("request_owner_transfer", reason="Wants a quote."), "Connecting you now.")
    await make_graph(model).ainvoke({"messages": [HumanMessage("Connect me.")], "active_agent": "contact"}, config=config, context=ctx)
    assert leads.all() == [] and len(telephony.transfers) == 1

    # A lead saved earlier on the call is not saved again.
    model = scripted(
        tool_call("save_lead", name="Ada", need="A demo."),
        tool_call("request_owner_transfer", call_id="call_2", reason="A demo.", caller_name="Ada"),
        "Connecting you now.",
    )
    await make_graph(model).ainvoke({"messages": [HumanMessage("Yes. Now connect me.")], "active_agent": "contact"}, config={"configurable": {"thread_id": "t2"}}, context=ctx)
    assert len(leads.all()) == 1


async def test_a_refused_transfer_sends_no_email_until_the_call_ends_without_a_message(make_graph, ctx, config, leads, telephony, notifier):
    ctx.transfer_unavailable = "the Telnyx application has no outbound voice profile"
    model = scripted(tool_call("request_owner_transfer", reason="Wants a quote.", caller_name="Ada"), "He will call you back.")
    await make_graph(model).ainvoke({"messages": [HumanMessage("Connect me.")], "active_agent": "contact"}, config=config, context=ctx)
    # Nothing goes out at the refusal: the agent offers to take a message next.
    assert leads.all() == [] and notifier.sent == [] and telephony.transfers == []

    # The caller hangs up without a message. The owner still learns who asked.
    lead = await note_refused_transfer_at_call_end(ctx, lead_saved=False)
    assert lead["name"] == "Ada" and [(saved.name, saved.need) for saved in leads.all()] == [("Ada", "Wants a quote.")]
    assert len(notifier.sent) == 1
    assert await note_refused_transfer_at_call_end(ctx, lead_saved=True) == {} and len(notifier.sent) == 1  # once only


async def test_a_message_after_a_refused_transfer_is_the_only_email(make_graph, ctx, config, leads, telephony, notifier):
    # The real call of 2026-11-16: "Can you call Brad?" outside the transfer hours, then a message.
    ctx.transfer_unavailable = "the Telnyx application has no outbound voice profile"
    model = scripted(
        tool_call("request_owner_transfer", reason="Caller Steve Stevens wants to speak to the owner now.", caller_name="Steve Stevens"),
        tool_call("save_lead", call_id="call_2", name="Steve Stevens", need="I like the cut of your jib."),
        "He has your message.",
    )
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Can you call him?")], "active_agent": "contact"}, config=config, context=ctx)

    _refusal, saved = (m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Lead saved as record 1" in saved.content
    assert [lead.need for lead in leads.all()] == ["I like the cut of your jib."] and len(notifier.sent) == 1
    assert await note_refused_transfer_at_call_end(ctx, lead_saved=True) == {} and len(notifier.sent) == 1


async def test_a_message_after_a_transfer_that_nobody_answered_is_saved_and_emailed(ctx, leads, notifier):
    # The owner's phone rang, so a record of who called exists. The message is still new.
    note = await note_caller_before_transfer(ctx, lead_saved=False, caller_name="Ada Lovelace", company="", reason="Ada Lovelace wants to speak to the owner now.")
    assert note["name"] == "Ada Lovelace" and ctx.transfer_note_id == 1

    again, _ = await save_lead_record(ctx, name="Ada Lovelace", need="Ada wants to speak to the owner.")
    assert again.startswith("Already saved on this call as record 1")  # the same request in other words is not a message

    text, lead = await save_lead_record(ctx, name="Ada Lovelace", need="Please send me a quote for a booking agent.")
    assert "Lead saved as record 2" in text and lead["need"] == "Please send me a quote for a booking agent."
    assert [saved.need for saved in leads.all()][-1] == "Please send me a quote for a booking agent." and len(notifier.sent) == 2

    third, _ = await save_lead_record(ctx, name="Ada Lovelace", need="One more thought about the quote.")
    assert third.startswith("Already saved on this call as record 2")  # one message per call, as before


async def test_the_same_caller_is_not_saved_twice_on_one_call(make_graph, ctx, config, leads, notifier):
    model = scripted(
        tool_call("save_lead", name="Ada Lovelace", need="A demo.", company="Analytical Engines"),
        tool_call("save_lead", call_id="call_2", name="ada lovelace", need="Follow-up after the transfer failed."),
        tool_call("save_lead", call_id="call_3", name="Ada Lovelace", need="A demo.", email="ada@example.test"),
        "Done.",
    )
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config=config, context=ctx)

    _first, second, third = (m for m in result["messages"] if isinstance(m, ToolMessage))
    assert second.content.startswith("Already saved on this call as record 1")
    assert "Lead saved as record 2" in third.content  # a new email address is new information
    assert [lead.email for lead in leads.all()] == ["", "ada@example.test"]


async def test_a_refused_transfer_tells_the_agent_that_it_has_the_name(make_graph, ctx, config):
    ctx.transfer_unavailable = "the Telnyx application has no outbound voice profile"
    model = scripted(tool_call("request_owner_transfer", reason="Wants a quote.", caller_name="Ada Lovelace"), "He will call you back.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Connect me.")], "active_agent": "contact"}, config=config, context=ctx)

    (refusal,) = (m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "You have the caller's name (Ada Lovelace). Do not ask for it again: ask only for the message." in refusal.content
    assert ctx.caller_name == "Ada Lovelace"  # the next agent on this call reads it in the handoff message


def test_the_greeting_says_whose_line_this_is_and_asks_who_is_calling(specs):
    from ai_call_center.agents.specs import NAME_QUESTION, RECORDING_NOTICE, Profile, build_specs, greeting_words, opening_line
    from ai_call_center.voice.call_session import connect_message

    assert opening_line("Ada Lovelace") == (
        "Hi, you've reached Ada Lovelace's AI business card. I'm one of six AI agents Ada built, and this call is a live sample of the work Ada does for clients."
    )
    # A brand name comes first, and the owner's full name stays in the greeting.
    assert opening_line("Ada Lovelace", "Analytical Engines").startswith("Hi, you've reached Analytical Engines. I'm one of six AI agents Ada Lovelace built,")
    branded = build_specs(Profile(summary="x", full="x"), owner_name="Ada Lovelace", brand_name="Analytical Engines")
    assert all("Analytical Engines is the name that Ada Lovelace works under" in s.instructions for s in branded.values())
    assert greeting_words("Ada Lovelace", "Analytical Engines") in branded["classifier"].instructions
    assert "Analytical Engines" not in specs["classifier"].instructions
    # A hint on how to say the owner's name reaches every agent, and only when it is set.
    hinted = build_specs(Profile(summary="x", full="x"), owner_name="Ada Lovelace", name_hint='Lovelace is said "LUV-layss".')
    assert all("How to say Ada Lovelace's name, every time you say it: Lovelace is said \"LUV-layss\"." in s.instructions for s in hinted.values())
    assert "How to say" not in specs["classifier"].instructions
    # Every caller hears the same words: they say nothing about an earlier call.
    hello = greeting_words("Ada Lovelace")
    assert hello == f"{opening_line('Ada Lovelace')} {RECORDING_NOTICE} {NAME_QUESTION}"
    assert "recorded" in RECORDING_NOTICE  # the caller is told before the first question
    assert f'"{hello}"' in connect_message(hello) and "nothing before or after them" in connect_message(hello)
    text = specs["classifier"].instructions
    assert "call identify_caller with the name they said" in text and "Am I speaking with" not in text
    assert "contact" not in [name for name, spec in specs.items() if "Your opening line" in spec.instructions]  # only the greeting agent


def test_the_welcome_offers_contract_or_full_time_work_beyond_ai():
    from ai_call_center.agents.specs import Welcome

    welcome = Welcome(owner="Ada Lovelace", companies="on models from three companies", tip="to forget you")
    words = welcome.new_caller("Sam")
    assert "available for contract work or a full-time role" in words
    assert "new platforms and APIs, cloud systems, modernizing old code, and AI agents" in words
    assert "a graph of AI agents on models from three companies, with caller memory." in words  # no RAG search without a feed
    assert words.endswith("Try asking me to forget you. So, do you have a project or a role in mind, or would you like to look around first?")
    assert welcome.new_caller().startswith("No problem. Ada is")
    assert welcome.returning_caller("Sam Lee") == "Welcome back, Sam. Good to hear from you again. What can I do for you today?"


def test_every_agent_can_forget_the_caller_and_the_consultant_qualifies_a_lead(specs):
    assert all("forget_caller" in spec.tools for spec in specs.values())
    consultant = specs["consultant"].instructions
    assert "When would you want this to start?" in consultant and "Is this a few weeks of work, or a longer engagement?" in consultant
    assert "Code, not a model, compares the name" in specs["tour"].instructions


async def test_the_qualifying_answers_reach_the_lead_and_the_top_of_the_email(make_graph, ctx, config, leads, notifier):
    from ai_call_center.notify import plain_text

    model = scripted(tool_call("save_lead", name="Sam Lee", need="A booking platform for a clinic.", timeline="next month", size="a few months"), "Saved.")
    await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config=config, context=ctx)
    lead = leads.all()[-1]
    assert (lead.timeline, lead.size) == ("next month", "a few months")
    text = plain_text(notifier.sent[-1])
    assert text.index("Wants to start: next month") < text.index("Size of the work: a few months") < text.index("When:")


def test_a_greeting_tip_starts_a_real_feature_and_essay_tips_need_the_rag_search():
    import random

    from ai_call_center.agents.specs import CALL_TIPS, greeting_tip

    no_writing = {greeting_tip("Ada Lovelace", rng=random.Random(seed)) for seed in range(50)}
    assert no_writing == set(CALL_TIPS)
    with_writing = {greeting_tip("Ada Lovelace", writing=True, rng=random.Random(seed)) for seed in range(50)}
    assert "what Ada thinks about microservices" in with_writing and set(CALL_TIPS) < with_writing


def test_the_consultant_offers_software_engineering_not_only_ai(specs):
    consultant = specs["consultant"].instructions
    assert "not only an AI builder" in consultant
    assert "When plain software solves it better, say so." in consultant
    assert "a software project" in specs["classifier"].instructions


def test_a_message_is_saved_only_after_the_caller_says_yes_to_it(ctx):
    from ai_call_center.graph.tools import confirmed_the_message, message_confirm_question

    asked = ["agent: I'll tell him you want a quote, and he can call you back on this number. Is that right?"]
    for yes in ("Yes, that is right.", "Yep.", "That's correct, thanks.", "Sounds good."):
        assert confirmed_the_message([*asked, f"caller: {yes}"]), yes
    assert confirmed_the_message([*asked, "caller: Yes.", "caller: That is right."])  # one turn in two parts
    # The message itself, a change, and silence are not a yes.
    told = "caller: Okay. Then take a message. Tell him I like the cut of his jib and I want to give him a million dollars."
    not_asked = ["agent: Sure, what's your name and what should I tell him?", "caller: Yes, that is right."]  # a yes to another question
    for lines in ([told], [*asked, "caller: No, the number is wrong."], [*asked, "caller: Yes, but wait, change the name."], asked, [], not_asked):
        assert not confirmed_the_message(lines), lines

    # The save is held back once. The caller's answer decides, so the second call passes.
    first = message_confirm_question(ctx, [told])
    assert first.startswith("Not saved yet") and 'ask: "Is that right?"' in first and "Do not ask for their name again." in first
    assert message_confirm_question(ctx, [told]) is None

    ctx.message_confirm_asked = False
    assert message_confirm_question(ctx, [*asked, "caller: Yes."]) is None and not ctx.message_confirm_asked


async def test_a_message_that_waited_for_a_yes_is_saved_when_the_call_ends(ctx, leads, notifier):
    from ai_call_center.graph.tools import NOT_CONFIRMED_NOTE, confirmed_the_message, save_held_message_at_call_end

    # Other words for the same question count too.
    assert confirmed_the_message(["agent: Please confirm, should I save that for him?", "caller: Yes, that is right."])

    assert await save_held_message_at_call_end(ctx) == {}  # nothing waits
    ctx.held_message = {"name": "Ada Lovelace", "need": "Please send me a quote.", "call_notes": ""}
    lead = await save_held_message_at_call_end(ctx)
    assert lead["need"] == "Please send me a quote." and [saved.name for saved in leads.all()] == ["Ada Lovelace"]
    assert NOT_CONFIRMED_NOTE in plain_text(notifier.sent[0]) and ctx.held_message is None

    # A message that the caller did confirm leaves nothing to save at the end.
    ctx.held_message = {"name": "Ada Lovelace", "need": "Please send me a quote.", "call_notes": ""}
    await save_lead_record(ctx, name="Ada Lovelace", need="Please send me a quote.", email="ada@example.test")
    assert ctx.held_message is None and await save_held_message_at_call_end(ctx) == {}


def test_an_agent_asked_for_by_its_own_name_stays_on_the_call(specs):
    assert "If the caller asks for the contact agent, that is you" in specs["contact"].instructions
    assert "If the caller asks for the tour agent, that is you" in specs["tour"].instructions


# --- the feedback agent ---


async def test_file_feedback_records_the_report_in_the_call_and_sends_nothing(make_graph, ctx, config, notifier, telephony):
    from ai_call_center.agents.specs import feedback_filed

    ctx.owner_name = "Ada Lovelace"
    summary = "The tour agent's answer about the cost stopped in the middle."
    model = scripted(tool_call("file_feedback", kind="fault", summary=summary), "Thanks, that's filed.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes, that is right.")], "active_agent": "feedback"}, config=config, context=ctx)

    assert ctx.feedback_filed == [("fault", summary)]
    words = [m.content for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    assert words.startswith(f'Say exactly these words to the caller, and nothing else: "{feedback_filed("Ada Lovelace")}"')
    assert "Then ask if there is anything else." in words
    assert notifier.sent == [] and telephony.transfers == []  # no email, no call, no request during the call


def test_the_words_after_a_filed_report_are_true_and_promise_no_fix():
    from ai_call_center.agents.specs import feedback_filed

    words = feedback_filed("Ada Lovelace")
    assert words == "Thanks, that's filed. When this call ends, it goes to Ada's issue list on GitHub, and Ada decides what to work on."
    assert not any(word in words.lower() for word in ("fix", "will be", "today", "tomorrow", "soon"))
    assert "the owner's issue list" in feedback_filed("")


def test_file_feedback_checks_the_kind_and_keeps_at_most_the_items_of_one_call(ctx):
    from ai_call_center.feedback import MAX_ITEMS
    from ai_call_center.graph.tools import file_report

    assert file_report(ctx, "complaint", "The agent was rude.").startswith("Not filed: kind must be")
    assert file_report(ctx, "idea", "   ").startswith("Not filed: the summary is empty.")
    assert ctx.feedback_filed == []
    assert file_report(ctx, " Idea ", "The line  should speak Spanish.").startswith("Say exactly these words")
    assert file_report(ctx, "idea", "The line should speak Spanish.").startswith("Filed already on this call.")
    for n in range(MAX_ITEMS):
        file_report(ctx, "fault", f"Fault number {n} of the line.")
    assert len(ctx.feedback_filed) == MAX_ITEMS and ctx.feedback_filed[0] == ("idea", "The line should speak Spanish.")
    assert file_report(ctx, "fault", "One more fault.").startswith(f"Not filed: one call can file {MAX_ITEMS} reports at most")


def test_a_caller_who_asked_to_be_forgotten_files_nothing(ctx):
    from ai_call_center.graph.tools import file_report, forget

    file_report(ctx, "fault", "The agent went silent after a handoff.")
    forget(ctx)
    assert ctx.feedback_filed == []  # "forget me" removes what was filed before it
    assert file_report(ctx, "idea", "The line should speak Spanish.").startswith("Not filed: the caller asked to be forgotten")
    assert ctx.feedback_filed == []


def test_each_other_agent_routes_feedback_in_one_line_and_the_tour_agent_explains_it(specs):
    routing = "If the caller reports a fault of this phone line or asks for a feature of it, hand off to the feedback agent."
    assert all(spec.instructions.count(routing) == 1 for name, spec in specs.items() if name != "feedback")
    assert routing not in specs["feedback"].instructions
    assert "a report becomes a GitHub issue, a triage agent sorts it, and a coding agent can open a pull request" in specs["tour"].instructions
    assert all("one of six AI voice agents" in spec.instructions for spec in specs.values())


def test_the_feedback_agent_asks_two_questions_at_most_and_reads_the_report_back(specs):
    text = specs["feedback"].instructions
    assert "two questions at most" in text and "Ask only for what the caller has not said yet" in text
    assert "what happened, and what the caller expected" in text and "what the line should do, and why" in text
    assert 'say the report back in one sentence, and ask: "Is that right?"' in text and "After a yes, call file_feedback, in silence." in text
    assert "Never promise a fix, a date, or a call from" in text
    assert "request_owner_transfer" not in specs["feedback"].tools
