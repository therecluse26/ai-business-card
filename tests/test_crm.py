"""The owner's CRM over MCP: what a call writes after hangup, what waits when the CRM is down, and the one read behind the passphrase."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from ai_call_center.agents.specs import Profile, build_specs
from ai_call_center.callers import CallSummary
from ai_call_center.crm import NOT_CONNECTED, NOT_VERIFIED, REFUSED_STATE, WAITING, WRITTEN, CrmError, CrmStore, McpCrm, entry_for_call, request_status, sync_call
from ai_call_center.graph.state import CallContext
from ai_call_center.graph.tools import forget, identify, save_lead_record, set_phrase, verify
from tests.conftest import OWNER, scripted, tool_call

PROJECT = CallSummary(summary="Ada asked for a booking app for her clinic.", topic="a booking app", interest="project", owner_note="A booking app for a clinic, to start in January.")
CHAT = CallSummary(summary="The caller asked how the line works.", topic="how the line works", interest="none")
STAGES = ["New lead", "Contacted", "Proposal sent"]


class FakeCrm:
    """A CRM in memory with the rules of zepto-crm that crm.py depends on: a search by whole words from their
    start, one person for each email address, and a note that needs its record."""

    def __init__(self) -> None:
        self.people: dict[str, dict] = {}
        self.companies: dict[str, dict] = {}
        self.deals: dict[str, dict] = {}
        self.notes: list[dict] = []
        self.tasks: list[dict] = []
        self.calls: list[tuple[str, dict]] = []
        self.down = False
        self.refuse: set[str] = set()

    async def tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if self.down:
            raise CrmError("unreachable", "ConnectError")
        if name in self.refuse:
            raise CrmError("validation_failed", "The input is not valid", [{"path": "title", "message": "Too long"}])
        return getattr(self, name)(**arguments)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def search(self, query: str, types: list[str], limit: int = 6) -> dict:
        words = re.findall(r"[a-z0-9]+", query.lower())
        hits = []
        for kind, rows, fields in (("person", self.people, ("firstName", "lastName", "email", "phone")), ("company", self.companies, ("name",))):
            for row in rows.values() if kind in types else []:
                tokens = re.findall(r"[a-z0-9]+", " ".join(str(row.get(field) or "") for field in fields).lower())
                if words and all(any(token.startswith(word) for token in tokens) for word in words):
                    title = row["name"] if kind == "company" else f"{row['firstName']} {row['lastName']}".strip()
                    hits.append({"type": kind, "entity": kind, "id": row["id"], "title": title, "subtitle": None})
        return {"hits": hits[:limit]}

    def get_record(self, type: str, id: str, timeline: int = 10) -> dict:
        rows = {"person": self.people, "company": self.companies, "deal": self.deals}[type]
        if id not in rows:
            raise CrmError("not_found", f"The {type} was not found")
        return {"record": rows[id], "timeline": []}

    def create_person(self, firstName: str = "", lastName: str = "", email: str | None = None, phone: str | None = None, companyId: str | None = None) -> dict:
        if email and "@" not in email:
            raise CrmError("validation_failed", "The input is not valid", [{"path": "email", "message": "Not an email address"}])
        taken = next((person["id"] for person in self.people.values() if email and person["email"] == email), None)
        if taken:
            raise CrmError("conflict", "A person with this email exists", {"field": "email", "existingId": taken})
        person = {"id": f"person-{len(self.people) + 1}", "firstName": firstName, "lastName": lastName, "email": email, "phone": phone, "companyId": companyId}
        self.people[person["id"]] = person
        return person

    def create_company(self, name: str) -> dict:
        company = {"id": f"company-{len(self.companies) + 1}", "name": name}
        self.companies[company["id"]] = company
        return company

    def create_deal(self, name: str, personIds: list[str], companyId: str | None = None) -> dict:
        deal = {"id": f"deal-{len(self.deals) + 1}", "name": name, "status": "open", "stageId": "stage-1", "stageEnteredAt": 1795190400000, "personIds": personIds, "companyId": companyId}
        self.deals[deal["id"]] = deal
        return deal

    def add_note(self, type: str, recordId: str, kind: str, body: str, title: str = "") -> dict:
        if recordId not in self.people:
            raise CrmError("validation_failed", "The input is not valid", [{"path": "recordId", "message": "This person does not exist"}])
        self.notes.append({"id": f"note-{len(self.notes) + 1}", "recordId": recordId, "kind": kind, "title": title, "body": body})
        return self.notes[-1]

    def create_task(self, title: str, notes: str | None = None, dueDate: str | None = None, type: str | None = None, recordId: str | None = None) -> dict:
        self.tasks.append({"id": f"task-{len(self.tasks) + 1}", "title": title, "notes": notes, "dueDate": dueDate, "type": type, "recordId": recordId})
        return self.tasks[-1]

    def list_pipelines(self) -> dict:
        return {"pipelines": [{"id": "pipeline-1", "name": "Client work", "stages": [{"id": f"stage-{n}", "name": name} for n, name in enumerate(STAGES, 1)]}]}


@pytest.fixture
def crm() -> FakeCrm:
    return FakeCrm()


def _call(ctx: CallContext, call_id: str, crm: FakeCrm | None, number: str = "") -> CallContext:
    """A new call from the same number: the same stores, and nothing else from the call before."""
    number = number or ctx.caller_number
    ctx.leads.start_call(call_id, number)
    return CallContext(
        call_control_id=call_id, caller_number=number, owner_phone=ctx.owner_phone, leads=ctx.leads, telephony=ctx.telephony, notifier=ctx.notifier, legs=ctx.legs, mode="text", crm=crm, owner_name=OWNER
    )


async def _message(call: CallContext, name: str = "Ada Lovelace", need: str = "A booking app for my clinic.", **details: str) -> None:
    await save_lead_record(call, name=name, need=need, **details)


async def _end(call: CallContext, summary: CallSummary = PROJECT, **options: bool) -> None:
    """The end of a call, as `call_session` and `chat` do it: the summary is stored, then the CRM gets the call."""
    call.leads.finish_call(call.call_control_id, agents=["classifier", "contact"], summary=summary.summary, caller_name=call.caller_name, topic=summary.topic)
    await sync_call(call, summary, visited=["classifier", "contact"], **options)


def _set(call: CallContext, phrase: str) -> None:
    set_phrase(call, phrase)
    set_phrase(call, phrase)


# --- the MCP client ---


def _server(handler) -> McpCrm:
    return McpCrm(url="https://crm.example.test/ext/mcp", token="crm_test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _event(result: dict) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': 1, 'result': result})}\n\n")


async def test_a_tool_call_is_one_post_and_the_result_is_read_from_the_event_stream():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _event({"content": [{"type": "text", "text": '{"hits": []}'}]})

    assert await _server(handler).tool("search", {"query": "ada"}) == {"hits": []}
    assert seen == [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "search", "arguments": {"query": "ada"}}}]


async def test_a_refused_tool_call_gives_the_code_and_the_details_of_the_crm():
    fault = {"error": {"code": "conflict", "message": "A person with this email exists", "details": {"existingId": "p1"}}}
    with pytest.raises(CrmError) as caught:
        await _server(lambda request: _event({"isError": True, "content": [{"type": "text", "text": json.dumps(fault)}]})).tool("create_person", {})
    assert (caught.value.code, caught.value.details) == ("conflict", {"existingId": "p1"})


@pytest.mark.parametrize(("status", "code"), [(302, "unreachable"), (401, "unauthenticated"), (500, "unreachable")])
async def test_a_redirect_of_cloudflare_access_and_a_wrong_token_are_named(status, code):
    with pytest.raises(CrmError) as caught:
        await _server(lambda request: httpx.Response(status, text="<html>")).tool("search", {})
    assert caught.value.code == code


async def test_a_server_that_cannot_be_reached_is_a_crm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(CrmError) as caught:
        await _server(handler).tool("search", {})
    assert caught.value.code == "unreachable"


# --- what a call leaves ---


async def test_a_call_with_no_message_and_no_project_leaves_nothing(ctx, crm):
    call = _call(ctx, "call-1", crm)
    await _end(call, CHAT)
    assert crm.calls == []
    assert CrmStore(ctx.leads.connection).get("call-1") is None


async def test_a_message_about_a_project_becomes_a_person_a_timeline_entry_a_deal_and_a_task(ctx, crm):
    call = _call(ctx, "call-1", crm)
    await _message(call, email="ada@example.test", company="Analytical Engines", timeline="in January")
    await _end(call)

    (person,) = crm.people.values()
    assert (person["firstName"], person["lastName"], person["email"], person["phone"]) == ("Ada", "Lovelace", "ada@example.test", call.caller_number)
    assert crm.companies[person["companyId"]]["name"] == "Analytical Engines"
    (note,) = crm.notes
    assert (note["recordId"], note["kind"], note["title"]) == (person["id"], "call", "Call to the AI line: a booking app")
    assert "Message: A booking app for my clinic." in note["body"]
    assert "Start: in January" in note["body"]
    assert "Caller: new on this number." in note["body"]
    assert f"email ada@example.test, the number {call.caller_number}" in note["body"]
    (deal,) = crm.deals.values()
    assert (deal["name"], deal["personIds"], deal["companyId"]) == ("Ada Lovelace: a booking app", [person["id"]], person["companyId"])
    (task,) = crm.tasks
    assert (task["title"], task["type"], task["recordId"]) == ("Follow up with Ada Lovelace: a booking app", "deal", deal["id"])
    assert task["notes"] == "A booking app for my clinic."
    row = CrmStore(ctx.leads.connection).get("call-1")
    assert (row.state, row.entry, row.person_id, row.deal_id, row.deal_made, row.task_id) == (WRITTEN, "", person["id"], deal["id"], True, task["id"])


async def test_a_message_that_is_not_about_work_makes_no_deal(ctx, crm):
    call = _call(ctx, "call-1", crm)
    await _message(call, need="Tell the owner that Grace says hello.", name="Grace Hopper")
    await _end(call, CallSummary(summary="Grace left a greeting.", topic="a greeting", interest="none"))
    assert crm.deals == {}
    assert [(task["type"], task["recordId"]) for task in crm.tasks] == [("person", "person-1")]


async def test_a_talk_about_a_project_with_no_message_still_reaches_the_crm(ctx, crm):
    call = _call(ctx, "call-1", crm)
    call.caller_name = "Ada Lovelace"
    await _end(call)
    assert "The caller left no message. What they talked about: A booking app for a clinic, to start in January." in crm.notes[0]["body"]
    assert len(crm.deals) == len(crm.tasks) == 1


async def test_a_caller_who_asked_to_be_forgotten_leaves_nothing(ctx, crm):
    call = _call(ctx, "call-1", crm)
    await _message(call)
    forget(call)
    await _end(call)
    assert crm.calls == []


async def test_a_call_from_the_owners_phone_leaves_the_person_and_the_entry_only(ctx, crm):
    call = _call(ctx, "call-1", crm)
    await _message(call)
    await _end(call, test=True)
    assert crm.notes[0]["title"] == "Test call to the AI line: a booking app"
    assert (crm.deals, crm.tasks) == ({}, [])


async def test_a_hidden_number_is_not_written_as_a_phone_number(ctx, crm):
    call = _call(ctx, "call-1", crm, number="anonymous")
    await _message(call, email="ada@example.test")
    await _end(call)
    assert crm.people["person-1"]["phone"] is None
    assert "a hidden number" in crm.notes[0]["body"]
    assert "search" not in crm.names()[:1]  # no search by a phone number that is not one


async def test_an_email_address_that_is_no_address_does_not_stop_the_person(ctx, crm):
    call = _call(ctx, "call-1", crm)
    await _message(call, email="ada at example")
    await _end(call)
    assert crm.people["person-1"]["email"] is None
    assert "email ada at example" in crm.notes[0]["body"]


async def test_a_long_topic_of_the_summary_model_is_cut_to_what_the_crm_takes(ctx, crm):
    call = _call(ctx, "call-1", crm)
    await _message(call)
    await _end(call, CallSummary(summary="s", topic="a booking app " * 40, interest="project"))
    assert (len(crm.notes[0]["title"]), len(next(iter(crm.deals.values()))["name"]), len(crm.tasks[0]["title"])) == (200, 200, 300)


# --- one person, many calls ---


async def test_a_second_call_of_the_same_person_uses_their_person_and_their_open_deal(ctx, crm):
    first = _call(ctx, "call-1", crm)
    await _message(first)
    await _end(first)
    second = _call(ctx, "call-2", crm)
    await _message(second, name="Ada", need="One more detail: it must send text reminders.")
    await _end(second)

    assert len(crm.people) == len(crm.deals) == 1
    assert [note["recordId"] for note in crm.notes] == ["person-1", "person-1"]
    assert len(crm.tasks) == 2
    row = CrmStore(ctx.leads.connection).get("call-2")
    assert (row.person_id, row.deal_id, row.deal_made) == ("person-1", "deal-1", False)


async def test_a_closed_deal_is_not_used_again(ctx, crm):
    first = _call(ctx, "call-1", crm)
    await _message(first)
    await _end(first)
    crm.deals["deal-1"]["status"] = "won"
    second = _call(ctx, "call-2", crm)
    await _message(second, need="A second project: a patient portal.")
    await _end(second)
    assert len(crm.deals) == 2
    assert CrmStore(ctx.leads.connection).get("call-2").deal_made is True


async def test_two_people_on_one_number_are_two_persons(ctx, crm):
    # The line matches a returning caller by the first name. The CRM link also needs that the surnames do not differ.
    for call_id, name in (("call-1", "Ada Lovelace"), ("call-2", "Grace Hopper"), ("call-3", "Ada Byron"), ("call-4", "Ada")):
        call = _call(ctx, call_id, crm)
        await _message(call, name=name)
        await _end(call)
    assert [f"{person['firstName']} {person['lastName']}" for person in crm.people.values()] == ["Ada Lovelace", "Grace Hopper", "Ada Byron"]
    assert crm.notes[-1]["recordId"] == "person-3"  # "Ada" alone goes to the newest Ada of this number


async def test_a_person_that_the_owner_added_by_hand_is_found_by_number_and_name(ctx, crm):
    crm.create_person(firstName="Ada", lastName="Lovelace", phone="+1 (555) 555-0123")  # the owner typed it with spaces: not one word, so no match
    crm.create_person(firstName="Ada", lastName="Lovelace", phone=ctx.caller_number)
    call = _call(ctx, "call-1", crm)
    await _message(call)
    await _end(call)
    assert len(crm.people) == 2
    assert crm.notes[0]["recordId"] == "person-2"


async def test_an_email_address_that_the_crm_has_gives_that_person(ctx, crm):
    crm.create_person(firstName="A.", lastName="Lovelace", email="ada@example.test")
    call = _call(ctx, "call-1", crm)
    await _message(call, email="ada@example.test")
    await _end(call)
    assert len(crm.people) == 1
    assert crm.notes[0]["recordId"] == "person-1"


async def test_a_person_that_the_owner_deleted_is_made_again(ctx, crm):
    first = _call(ctx, "call-1", crm)
    await _message(first)
    await _end(first)
    crm.people.clear()
    second = _call(ctx, "call-2", crm)
    await _message(second, need="Still interested in the booking app.")
    await _end(second)
    assert list(crm.people) == ["person-1"] and crm.notes[-1]["recordId"] == "person-1"
    assert CrmStore(ctx.leads.connection).get("call-2").state == WRITTEN
    assert CrmStore(ctx.leads.connection).get("call-1").person_id == ""  # the old link is gone with the person


async def test_a_call_that_came_back_after_a_transfer_adds_what_is_new(ctx, crm):
    call = _call(ctx, "call-1", crm)
    await _message(call, need="Asked to speak to the owner.")
    await _end(call, transferred=True)
    assert len(crm.notes) == 1 and crm.tasks == []  # the owner's phone rings: nothing to follow up yet
    await _message(call, need="Nobody answered. Please call me back about the booking app.", email="ada@example.test")
    await _end(call)
    assert len(crm.people) == len(crm.deals) == 1
    assert len(crm.notes) == 2 and "Please call me back" in crm.notes[1]["body"]
    assert len(crm.tasks) == 1


# --- a CRM that is down, and an entry that it refuses ---


async def test_an_entry_waits_when_the_crm_is_down_and_the_next_call_end_sends_it(ctx, crm):
    crm.down = True
    first = _call(ctx, "call-1", crm)
    await _message(first)
    await _end(first)
    store = CrmStore(ctx.leads.connection)
    assert (store.get("call-1").state, store.get("call-1").tries) == (WAITING, 1)
    assert len(crm.calls) == 1  # one request showed that the CRM is down; the rest was not tried

    crm.down = False
    second = _call(ctx, "call-2", crm)
    await _message(second, name="Grace Hopper", need="A role: a staff engineer for our platform team.")
    await _end(second, CallSummary(summary="Grace has a role to fill.", topic="a staff engineer role", interest="role"))
    assert [store.get(call_id).state for call_id in ("call-1", "call-2")] == [WRITTEN, WRITTEN]
    assert [note["title"] for note in crm.notes] == ["Call to the AI line: a booking app", "Call to the AI line: a staff engineer role"]


async def test_a_step_that_worked_is_not_done_again_on_the_next_try(ctx, crm):
    crm.refuse = {"create_task"}
    call = _call(ctx, "call-1", crm)
    await _message(call)
    await _end(call)
    store = CrmStore(ctx.leads.connection)
    assert store.get("call-1").state == REFUSED_STATE  # the CRM refused what the entry holds: the same entry is not sent again
    assert store.waiting() == []
    assert (len(crm.people), len(crm.notes), len(crm.deals), crm.tasks) == (1, 1, 1, [])


async def test_a_failure_of_the_crm_never_stops_the_end_of_a_call(ctx):
    class Broken:
        async def tool(self, name: str, arguments: dict) -> Any:
            raise RuntimeError("an answer in a shape that nobody expected")

    call = _call(ctx, "call-1", Broken())
    await _message(call)
    await _end(call)
    assert CrmStore(ctx.leads.connection).get("call-1").state == WAITING


# --- the read: only the caller's own request, only after the passphrase ---


async def test_no_crm_record_is_read_before_the_passphrase_check(ctx, crm):
    first = _call(ctx, "call-1", crm)
    await _message(first)
    await _end(first)
    crm.calls.clear()

    second = _call(ctx, "call-2", crm)
    identify(second, "Ada Lovelace")  # the name matches: recognised, not verified
    assert await request_status(second) == NOT_VERIFIED
    assert crm.calls == []
    assert await request_status(_call(ctx, "call-3", None)) == NOT_CONNECTED


async def test_a_verified_caller_gets_the_stage_of_their_own_deal_and_nothing_else(ctx, crm):
    first = _call(ctx, "call-1", crm)
    await _message(first, email="ada@example.test")
    _set(first, "purple lantern")
    await _end(first)
    crm.deals["deal-1"].update(stageId="stage-2", name="Ada - difficult client, quote high")
    crm.calls.clear()

    second = _call(ctx, "call-2", crm)
    identify(second, "Ada Lovelace")
    verify(second, "purple lantern")
    status = await request_status(second)

    assert 'open, in the stage "Contacted" (stage 2 of 3) since Nov 2' in status
    assert f"Read from {OWNER.split()[0]}'s CRM over MCP" in status
    for private in ("difficult client", "ada@example.test", second.caller_number, "Lovelace"):
        assert private not in status
    assert sorted(crm.names()) == ["get_record", "list_pipelines"]


async def test_a_message_with_no_deal_is_reported_as_received(ctx, crm):
    first = _call(ctx, "call-1", crm)
    await _message(first, need="Tell the owner that Ada says hello.")
    _set(first, "purple lantern")
    await _end(first, CHAT)
    second = _call(ctx, "call-2", crm)
    identify(second, "Ada Lovelace")
    verify(second, "purple lantern")
    crm.calls.clear()
    assert "CRM has the caller's message" in await request_status(second)
    assert crm.calls == []  # the answer is in this server's own table


async def test_a_person_with_a_faked_number_and_the_real_name_never_reads_the_real_callers_deal(ctx, crm):
    real = _call(ctx, "call-1", crm)
    await _message(real)
    _set(real, "purple lantern")
    await _end(real)

    # The second person gives the real caller's name, has no passphrase, and leaves a message about a project.
    # Their entry goes on the real caller's person and uses the real caller's open deal: the owner sees both.
    faker = _call(ctx, "call-2", crm)
    identify(faker, "Ada Lovelace")
    verify(faker, "")
    await _message(faker, need="Another project.")
    _set(faker, "copper kettle")  # a passphrase of their own, which opens only this call
    await _end(faker)
    assert CrmStore(ctx.leads.connection).get("call-2").deal_id == "deal-1"
    assert "No passphrase check, so it is not proof" in crm.notes[1]["body"]

    again = _call(ctx, "call-3", crm)
    identify(again, "Ada Lovelace")
    verify(again, "copper kettle")
    assert again.identity_verified and again.open_calls == ["call-2"]
    crm.calls.clear()
    status = await request_status(again)
    assert "stage" not in status.split("It has no stage")[0]
    assert crm.calls == []  # the real caller's deal was not read


async def test_a_crm_that_does_not_answer_gives_the_agent_words_and_no_error(ctx, crm):
    first = _call(ctx, "call-1", crm)
    await _message(first)
    _set(first, "purple lantern")
    await _end(first)
    second = _call(ctx, "call-2", crm)
    identify(second, "Ada Lovelace")
    verify(second, "purple lantern")
    crm.down = True
    assert (await request_status(second)).startswith("The CRM did not answer.")


async def test_forget_me_removes_the_links_and_an_entry_that_waits(ctx, crm):
    first = _call(ctx, "call-1", crm)
    await _message(first)
    _set(first, "purple lantern")
    await _end(first)
    crm.down = True
    second = _call(ctx, "call-2", crm)
    identify(second, "Ada Lovelace")
    verify(second, "purple lantern")
    await _message(second, need="One more thing about the booking app.")
    await _end(second)
    store = CrmStore(ctx.leads.connection)
    assert store.waiting() == ["call-2"]

    third = _call(ctx, "call-3", crm)
    identify(third, "Ada Lovelace")
    verify(third, "purple lantern")
    forget(third)
    assert [store.get(call_id) for call_id in ("call-1", "call-2", "call-3")] == [None, None, None]
    assert await request_status(third) == NOT_VERIFIED
    assert len(crm.people) == 1  # what the CRM has stays with the owner, as the email does


# --- the tool, and which agent has it ---


def test_only_the_contact_agent_has_the_lookup_and_only_with_a_crm():
    profile = Profile(summary="s", full="f")
    off = build_specs(profile, owner_name=OWNER)
    on = build_specs(profile, owner_name=OWNER, crm=True)
    assert all("crm_lookup" not in spec.tools and "CRM" not in spec.instructions for spec in off.values())
    assert [name for name, spec in on.items() if "crm_lookup" in spec.tools] == ["contact"]
    assert "asks where an earlier message or request stands" in on["contact"].role
    assert "CRM over MCP" in on["tour"].instructions


async def test_the_lookup_runs_as_a_tool_of_the_text_graph(make_graph, ctx, config):
    graph = make_graph(scripted(tool_call("crm_lookup"), "I can't check that here."))
    result = await graph.ainvoke({"messages": [HumanMessage("Did he get my message?")], "active_agent": "contact"}, config=config, context=ctx)
    assert next(m.content for m in result["messages"] if isinstance(m, ToolMessage)) == NOT_CONNECTED


def test_the_entry_names_the_day_in_eastern_time(ctx):
    call = _call(ctx, "call-1", None)
    call.caller_name = "Ada Lovelace"
    entry = entry_for_call(call, PROJECT, visited=["classifier"], now=datetime(2026, 11, 24, 2, 30, tzinfo=UTC))
    assert entry.due == "2026-11-23"  # 9:30 PM Eastern on the day before
    assert "on Nov 23 at 9:30 PM Eastern" in entry.body
