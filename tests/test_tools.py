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
    assert tool_msg.content.endswith("emailed to the owner.")
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
