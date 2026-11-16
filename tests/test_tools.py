"""Side-effect tools run inside the graph with the call context."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage

from ai_call_center.telephony import TransferRequest
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
    assert "classifier on OpenAI gpt-realtime-2.1-mini" in stats
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
    assert "Ada Lovelace" in email["Subject"]
    body = email.get_content()
    assert "A voice agent for inbound sales calls." in body
    assert ctx.caller_number in body
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "emailed to the owner." in tool_msg.content and "by email" in tool_msg.content
    assert "owner@example.test" not in str(result["messages"])


async def test_save_lead_keeps_row_when_email_fails(make_graph, ctx, config, leads):
    class BrokenNotifier:
        async def send_lead(self, lead, *, row_id):
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
    assert "12:30 PM EST" in msg.get_content()


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
    await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config=config, context=ctx)
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


async def test_email_on_file_is_reused_only_after_the_caller_confirms_who_they_are(make_graph, ctx, config, leads):
    from ai_call_center.leads import Lead, now_iso

    leads.save(Lead(name="Steve Park", email="steve@acme.test", company="Acme Roofing", need="Old.", caller_number=ctx.caller_number, call_control_id="old", created_at=now_iso()))

    # Not confirmed: a model that asks for the address on file gets none.
    model = scripted(tool_call("save_lead", name="Someone Else", need="A question.", email="on file"), "Done.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config=config, context=ctx)
    assert leads.all()[-1].email == "" and leads.all()[-1].company == ""
    assert "steve@acme.test" not in str(result["messages"])

    # Confirmed: the address goes from the store to the new row, and never into a message.
    ctx.identity_confirmed = True
    model = scripted(tool_call("save_lead", name="Steve Park", need="A shared project.", email="on file"), "Done.")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config={"configurable": {"thread_id": "t2"}}, context=ctx)
    assert leads.all()[-1].email == "steve@acme.test" and leads.all()[-1].company == "Acme Roofing"
    assert "steve@acme.test" not in str(result["messages"])
    assert "by email" in next(m for m in result["messages"] if isinstance(m, ToolMessage)).content


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
    await make_graph(model).ainvoke({"messages": [HumanMessage("Yes.")], "active_agent": "contact"}, config={"configurable": {"thread_id": "t2"}}, context=ctx)
    assert len(leads.all()) == 1


async def test_a_refused_transfer_still_keeps_the_caller_details(make_graph, ctx, config, leads, telephony):
    ctx.transfer_unavailable = "the Telnyx application has no outbound voice profile"
    model = scripted(tool_call("request_owner_transfer", reason="Wants a quote.", caller_name="Ada"), "He will call you back.")
    await make_graph(model).ainvoke({"messages": [HumanMessage("Connect me.")], "active_agent": "contact"}, config=config, context=ctx)
    assert [lead.name for lead in leads.all()] == ["Ada"] and telephony.transfers == []


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
