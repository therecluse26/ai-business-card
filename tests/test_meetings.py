"""A meeting booked on Cal.com during the call (meetings.py). Every test uses a fake scheduler or a fake HTTP answer:
a real booking emails a real invitation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from ai_call_center.agents.specs import Profile, build_specs
from ai_call_center.callers import CallSummary
from ai_call_center.crm import entry_for_call
from ai_call_center.graph.tools import forget, missed_save
from ai_call_center.meetings import (
    ACCEPTED,
    API,
    NO_ANSWER,
    PENDING,
    Booking,
    CalCom,
    CalcomError,
    MeetingStore,
    Slots,
    book,
    choose_options,
    find_times,
    invitation_notes,
    meeting_booked,
    normal_email,
    spelled_email,
    spoken_time,
)
from ai_call_center.notify import plain_text
from tests.conftest import OWNER, OWNER_PHONE, scripted, tool_call

# Thursday, November 26, 2026, 9 AM Eastern (EST, UTC-5).
NOW = datetime(2026, 11, 26, 14, 0, tzinfo=UTC)


def eastern(day: int, hour: int, minute: int = 0) -> datetime:
    """A time in late November 2026, Eastern standard time."""
    return datetime(2026, 11, day, hour + 5, minute, tzinfo=UTC)


def workday(day: int) -> list[datetime]:
    """Every 15 minutes from 9 AM to 5 PM Eastern, as Cal.com gives them."""
    return [eastern(day, 9) + timedelta(minutes=15 * n) for n in range(32)]


# Friday, November 27; Monday, November 30; Tuesday, December 1. No time on the weekend, as on the owner's calendar.
THREE_DAYS = [*workday(27), *workday(30), *(start + timedelta(days=5) for start in workday(26))]


class FakeScheduler:
    def __init__(self, starts: list[datetime] | None = None, *, status: str = ACCEPTED, fails: Exception | None = None, reads_fail: bool = False) -> None:
        self.starts = starts if starts is not None else THREE_DAYS
        self.status, self.fails, self.reads_fail = status, fails, reads_fail
        self.booked: list[dict] = []

    async def free_slots(self, start: datetime, end: datetime) -> Slots:
        if self.reads_fail:
            raise CalcomError("GET /slots: ConnectTimeout", answered=False)
        return Slots(starts=[s for s in self.starts if start <= s < end], minutes=15)

    async def book(self, start: datetime, *, name: str, email: str, time_zone: str, notes: str) -> Booking:
        self.booked.append({"start": start, "name": name, "email": email, "time_zone": time_zone, "notes": notes})
        if self.fails:
            raise self.fails
        return Booking(uid="uid-1", status=self.status)


@pytest.fixture
def cal(ctx) -> FakeScheduler:
    scheduler = FakeScheduler()
    ctx.scheduler, ctx.owner_name = scheduler, OWNER
    return scheduler


def read_back_and_yes(result: str, yes: str = "Yes, that's right.") -> list[str]:
    """The transcript after the agent said the read-back words of `result` and the caller answered."""
    words = result.split('"')[1]
    return ["caller: Can I book a call with the owner?", "agent: Sure.", f"agent: {words}", f"caller: {yes}"]


async def booked(ctx, *, option: str = "1", email: str = "ada@example.com", topic: str = "A booking app for a clinic.") -> str:
    """The two book_meeting calls of a normal booking: the read-back, then the booking after the caller's yes."""
    first = await book(ctx, option=option, email=email, name="Ada Lovelace", topic=topic, lines=[], now=NOW)
    if not first.startswith("Not booked yet."):
        return first
    return await book(ctx, option=option, email=email, name="Ada Lovelace", topic=topic, lines=read_back_and_yes(first), now=NOW)


# --- the spoken times ---


def test_a_time_is_spoken_in_eastern_time_in_winter_and_in_summer():
    assert spoken_time(datetime(2026, 11, 27, 19, 30, tzinfo=UTC)) == "Friday, November 27, at 2:30 PM Eastern"  # EST, UTC-5
    assert spoken_time(datetime(2026, 7, 1, 18, 0, tzinfo=UTC)) == "Wednesday, July 1, at 2 PM Eastern"  # EDT, UTC-4


def test_the_same_utc_hour_is_a_different_eastern_hour_on_each_side_of_the_clock_change():
    # The clocks went back on Sunday, November 1, 2026.
    assert spoken_time(datetime(2026, 10, 30, 14, 0, tzinfo=UTC)) == "Friday, October 30, at 10 AM Eastern"
    assert spoken_time(datetime(2026, 11, 2, 14, 0, tzinfo=UTC)) == "Monday, November 2, at 9 AM Eastern"


def test_three_times_on_three_days_near_the_target_hours():
    picks = choose_options(THREE_DAYS, now=NOW)
    assert [spoken_time(p) for p in picks] == [
        "Friday, November 27, at 10 AM Eastern",
        "Monday, November 30, at 2 PM Eastern",
        "Tuesday, December 1, at 11 AM Eastern",
    ]


def test_with_one_free_day_the_times_on_it_are_two_hours_apart():
    picks = choose_options(workday(27), now=NOW)
    assert len(picks) == 3 and len({p.date() for p in picks}) == 1
    assert all(b - a >= timedelta(hours=2) for a, b in pairwise(picks))


def test_a_time_that_starts_too_soon_is_not_offered():
    picks = choose_options([NOW + timedelta(minutes=15), NOW + timedelta(hours=1)], now=NOW)
    assert picks == [NOW + timedelta(hours=1)]


def test_the_day_the_caller_asked_for():
    starts = [*workday(27), *workday(30)]
    assert {p.astimezone(UTC).day for p in choose_options(starts, now=NOW, day="Monday")} == {30}
    assert {p.astimezone(UTC).day for p in choose_options(starts, now=NOW, day="tomorrow")} == {27}
    assert len(choose_options(starts, now=NOW, day="sometime soon")) == 3  # words that name no day: no filter


# --- the option ids ---


async def test_find_times_gives_ids_and_spoken_words_and_no_utc_time(ctx, cal):
    result = await find_times(ctx, now=NOW)
    assert "option 1: Friday, November 27, at 10 AM Eastern" in result
    assert "option 3: Tuesday, December 1, at 11 AM Eastern" in result
    assert "2026" not in result and "UTC" not in result  # the model never sees a time it could send
    assert ctx.meeting_options["2"].start == eastern(30, 14)


async def test_book_sends_the_exact_start_of_the_option_and_the_eastern_zone(ctx, cal):
    await find_times(ctx, now=NOW)
    result = await booked(ctx, option="option two")
    assert result.startswith("Booked: Monday, November 30, at 2 PM Eastern")
    assert cal.booked == [
        {"start": eastern(30, 14), "name": "Ada Lovelace", "email": "ada@example.com", "time_zone": "America/New_York", "notes": cal.booked[0]["notes"]}
    ]


async def test_a_second_search_gets_new_ids_and_an_id_from_nowhere_is_refused(ctx, cal):
    await find_times(ctx, now=NOW)
    second = await find_times(ctx, "Monday", now=NOW)
    assert "option 4:" in second and "option 1:" not in second
    assert ctx.meeting_options["1"].start == eastern(27, 10)  # the first ids still name their times
    refused = await book(ctx, option="9", email="ada@example.com", name="Ada", lines=[], now=NOW)
    assert refused.startswith("Not booked: that option is not one") and cal.booked == []


# --- the email address ---


@pytest.mark.parametrize(
    ("heard", "address"),
    [
        ("Ada.Lovelace@Example.com", "ada.lovelace@example.com"),
        ("ada at example dot com", "ada@example.com"),
        ("a d a underscore l at mail dot example dot co dot uk", "ada_l@mail.example.co.uk"),
        ("ada dash l at example dot io.", "ada-l@example.io"),
        ("ada at example", ""),
        ("ada..l@example.com", ""),
        ("ada@@example.com", ""),
        ("ada@example.c", ""),
        ("ada@example.com.", "ada@example.com"),  # the end of the sentence
        ("", ""),
    ],
)
def test_the_email_check(heard, address):
    assert normal_email(heard) == address


def test_the_read_back_spells_the_address():
    assert spelled_email("ada.l@gmail.com") == "A, D, A, dot, L, at, gmail dot com"
    assert spelled_email("bo@ex.io") == "B, O, at, E, X, dot, I, O"


async def test_a_bad_address_is_not_booked_and_not_read_back(ctx, cal):
    await find_times(ctx, now=NOW)
    result = await book(ctx, option="1", email="ada at example", name="Ada", lines=[], now=NOW)
    assert result.startswith("Not booked: that is not a full email address") and ctx.held_meeting == "" and cal.booked == []


# --- the yes before the booking ---


async def test_the_first_call_reads_back_and_books_nothing(ctx, cal):
    await find_times(ctx, now=NOW)
    result = await book(ctx, option="1", email="ada@example.com", name="Ada", lines=[], now=NOW)
    assert "So that's Friday, November 27, at 10 AM Eastern, and the invitation goes to A, D, A, at, E, X, A, M, P, L, E, dot, C, O, M. Is that right?" in result
    assert cal.booked == [] and MeetingStore(ctx.leads.connection).for_call(ctx.call_control_id) is None


async def test_a_second_call_with_no_yes_reads_back_again(ctx, cal):
    await find_times(ctx, now=NOW)
    first = await book(ctx, option="1", email="ada@example.com", name="Ada", lines=[], now=NOW)
    again = await book(ctx, option="1", email="ada@example.com", name="Ada", lines=read_back_and_yes(first, "No, it's a different address."), now=NOW)
    assert again.startswith("Not booked yet.") and cal.booked == []


async def test_a_yes_to_another_address_does_not_book_the_new_one(ctx, cal):
    await find_times(ctx, now=NOW)
    first = await book(ctx, option="1", email="ada@example.com", name="Ada", lines=[], now=NOW)
    changed = await book(ctx, option="1", email="ada@example.org", name="Ada", lines=read_back_and_yes(first), now=NOW)
    assert changed.startswith("Not booked yet.") and "O, R, G" in changed and cal.booked == []


async def test_one_moment_after_the_yes_still_counts(ctx, cal):
    """On a voice call the agent says "One moment." before the second call, as the read-back asks."""
    await find_times(ctx, now=NOW)
    first = await book(ctx, option="1", email="ada@example.com", name="Ada", lines=[], now=NOW)
    lines = [*read_back_and_yes(first), "agent: One moment."]
    result = await book(ctx, option="1", email="ada@example.com", name="Ada", lines=lines, now=NOW)
    assert result.startswith("Booked:") and len(cal.booked) == 1


def test_a_yes_to_the_meeting_read_back_is_not_a_missed_message(ctx):
    lines = ["agent: So that's Friday, November 27, at 10 AM Eastern, and the invitation goes to A, D, A, at, gmail dot com. Is that right?", "caller: Yes.", "agent: One moment."]
    assert missed_save(ctx, lines) == 0
    assert missed_save(ctx, ["agent: So the message is that you want a booking app. Is that right?", "caller: Yes.", "agent: Perfect."]) == 2


async def test_book_meeting_in_the_graph_books_after_the_callers_yes(ctx, cal, config):
    """The tool in the text call, with the transcript from the graph's own messages."""
    from langgraph.checkpoint.memory import InMemorySaver

    from ai_call_center.graph.build import build_graph

    specs = build_specs(Profile(summary="S", full="# Summary\nS"), owner_name=OWNER, booking=True)
    await find_times(ctx, now=NOW)
    model = scripted(
        tool_call("book_meeting", option="1", email="ada@example.com", name="Ada Lovelace", topic="A booking app."),
        "So that's Friday, November 27, at 10 AM Eastern, and the invitation goes to A, D, A, at, E, X, A, M, P, L, E, dot, C, O, M. Is that right?",
        tool_call("book_meeting", call_id="call_2", option="1", email="ada@example.com", name="Ada Lovelace", topic="A booking app."),
        "You're booked for Friday at 10 AM Eastern. The invitation is on its way.",
    )
    graph = build_graph(model, specs, checkpointer=InMemorySaver())
    await graph.ainvoke({"messages": [HumanMessage("Friday at ten. It's ada at example dot com.")], "active_agent": "contact"}, config=config, context=ctx)
    assert cal.booked == []
    result = await graph.ainvoke({"messages": [HumanMessage("Yes, that's right.")]}, config=config, context=ctx)
    tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_results[0].startswith("Not booked yet.") and tool_results[-1].startswith("Booked:")
    assert len(cal.booked) == 1 and cal.booked[0]["start"] == eastern(27, 10)


# --- the two limits ---


async def test_one_call_books_one_meeting(ctx, cal):
    await find_times(ctx, now=NOW)
    await booked(ctx)
    assert "This call has a booking already" in await find_times(ctx, now=NOW)
    assert "This call has a booking already" in await book(ctx, option="2", email="bo@example.com", name="Bo", lines=[], now=NOW)
    assert len(cal.booked) == 1


async def test_the_daily_limit_of_the_line(ctx, cal):
    store = MeetingStore(ctx.leads.connection)
    for n in range(3):
        store.reserve(f"other-{n}", start=eastern(30, 10), now=NOW - timedelta(hours=n))
    assert (await find_times(ctx, now=NOW)).startswith("No meeting can be booked")
    assert "No meeting can be booked" in await booked(ctx) and cal.booked == []
    # A day later the oldest two are out of the 24 hours.
    assert (await find_times(ctx, now=NOW + timedelta(hours=23))).startswith("Free times")


async def test_a_limit_of_zero_books_nothing(ctx, cal):
    ctx.meeting_cap = 0
    assert (await find_times(ctx, now=NOW)).startswith("No meeting can be booked")


async def test_the_owners_phone_counts_too(ctx, cal):
    ctx.caller_number = OWNER_PHONE
    await find_times(ctx, now=NOW)
    await booked(ctx)
    assert "A test booking from the owner's own phone." in cal.booked[0]["notes"]
    assert MeetingStore(ctx.leads.connection).limit_reached("another-call", per_day=1, now=NOW) == "day"


# --- each way it can fail ---


async def test_no_key_means_no_booking_and_no_time(ctx):
    assert ctx.scheduler is None
    for result in (await find_times(ctx, now=NOW), await book(ctx, option="1", email="ada@example.com", lines=[], now=NOW)):
        assert result.startswith("No meeting can be booked") and "offer to take a message" in result and "AM" not in result


async def test_no_answer_to_the_free_times_offers_a_message(ctx, cal):
    cal.reads_fail = True
    result = await find_times(ctx, now=NOW)
    assert result.startswith("No meeting can be booked") and ctx.meeting_options == {}


async def test_no_free_time(ctx, cal):
    cal.starts = []
    assert "There is no free time in the next 7 days" in await find_times(ctx, now=NOW)
    assert 'There is no free time on "Saturday"' in await find_times(ctx, "Saturday", now=NOW)


async def test_a_refused_booking_frees_the_call_for_another_time(ctx, cal):
    cal.fails = CalcomError("POST /bookings: status 400", answered=True)
    await find_times(ctx, now=NOW)
    result = await booked(ctx)
    assert result.startswith("Not booked: the calendar did not accept that time") and "other times" in result
    assert MeetingStore(ctx.leads.connection).limit_reached(ctx.call_control_id, per_day=3, now=NOW) == ""
    assert ctx.notifier.sent == [] and not meeting_booked(ctx)
    cal.fails = None
    assert (await booked(ctx, option="2")).startswith("Booked:")


async def test_no_answer_to_the_booking_is_not_retried_and_the_owner_is_told_in_the_record(ctx, cal):
    cal.fails = CalcomError("POST /bookings: ReadTimeout", answered=False)
    await find_times(ctx, now=NOW)
    result = await booked(ctx)
    assert result.startswith("Not booked: the calendar gave no answer") and "do not try again" in result
    assert MeetingStore(ctx.leads.connection).for_call(ctx.call_control_id).status == NO_ANSWER
    assert "This call has a booking already" in await booked(ctx, option="2")  # a second try could book two
    assert not meeting_booked(ctx)


# --- accepted or pending, and what the owner gets ---


async def test_an_accepted_booking_tells_the_owner(ctx, cal, leads):
    await find_times(ctx, now=NOW)
    result = await booked(ctx)
    assert result.startswith("Booked: Friday, November 27, at 10 AM Eastern. Cal.com emails the invitation")
    assert "do not call save_lead for the meeting" in result
    assert meeting_booked(ctx) and ctx.meeting_lead_id
    (mail,) = ctx.notifier.sent
    assert "Ada Lovelace booked a meeting with you" in mail["Subject"]
    body = plain_text(mail)
    assert "Fri Nov 27 at 10:00 AM EST. Booked in Cal.com" in body and "ada@example.com" in body
    assert leads.all()[0].need == "A booking app for a clinic."


async def test_a_pending_booking_says_the_owner_must_confirm(ctx, cal):
    cal.status = PENDING
    await find_times(ctx, now=NOW)
    result = await booked(ctx)
    assert result.startswith("Requested: Friday, November 27, at 10 AM Eastern.") and "must confirm it first" in result
    assert "Do not say it is confirmed" in result and meeting_booked(ctx)
    assert "It waits for your confirmation in Cal.com." in plain_text(ctx.notifier.sent[0])


async def test_the_crm_entry_names_the_meeting(ctx, cal):
    await find_times(ctx, now=NOW)
    await booked(ctx)
    entry = entry_for_call(ctx, CallSummary(summary="Ada booked a call.", topic="a booking app", interest="project"), visited=["contact"])
    assert "Meeting: Fri Nov 27 at 10:00 AM EST. Booked in Cal.com" in str(entry)


async def test_a_message_after_the_booking_is_saved_as_new(ctx, cal, leads):
    from ai_call_center.graph.tools import save_lead_record

    await find_times(ctx, now=NOW)
    await booked(ctx)
    await save_lead_record(ctx, name="Ada Lovelace", email="ada@example.com", company="", need="Please send the rates before the call.", visited=["contact"])
    assert [lead.need for lead in leads.all()] == ["A booking app for a clinic.", "Please send the rates before the call."]


# --- what the invitation says ---


def test_the_notes_hold_no_passphrase_and_no_note_of_an_earlier_call(ctx):
    ctx.owner_name, ctx.spoken_phrases = OWNER, ["blue harbor"]
    notes = invitation_notes(ctx, "My phrase is blue harbor. A booking app.", ["caller: I want a booking app."])
    assert "blue harbor" not in notes and notes.startswith("Booked by phone on Test Owner's AI line.")
    ctx.caller_history = "Last time: wants a clinic app, budget fifty thousand."
    lines = ["caller: I want to talk about the booking app for my clinic."]
    notes = invitation_notes(ctx, "The booking app for the clinic. Budget fifty thousand from the last call.", lines)
    assert "booking app for the clinic" in notes and "fifty" not in notes


# --- forget me ---


async def test_forget_keeps_only_the_time_of_the_request(ctx, cal, leads):
    await find_times(ctx, now=NOW)
    await booked(ctx)
    forget(ctx)
    row = ctx.leads.connection.execute("SELECT call_control_id, booked_at, start, uid FROM meetings").fetchone()
    assert row[0] == ctx.call_control_id and row[1] and row[2] == "" and row[3] == ""
    assert leads.all() == []
    assert MeetingStore(ctx.leads.connection).limit_reached("another-call", per_day=1, now=NOW) == "day"  # it still counts


async def test_a_booking_after_forget_keeps_no_record_on_the_line(ctx, cal, leads):
    forget(ctx)
    await find_times(ctx, now=NOW)
    assert (await booked(ctx)).startswith("Booked:")
    assert leads.all() == [] and ctx.meeting_lead_id == 0
    assert ctx.leads.connection.execute("SELECT start, uid FROM meetings").fetchone() == ("", "")
    assert len(ctx.notifier.sent) == 1  # the owner still hears of it: Cal.com has the booking


# --- the instructions ---


def test_no_word_of_a_meeting_without_a_key():
    profile = Profile(summary="S", full="# Summary\nS")
    off = build_specs(profile, owner_name=OWNER)
    on = build_specs(profile, owner_name=OWNER, booking=True)
    for spec in off.values():
        text = f"{spec.instructions} {spec.role}".casefold()
        assert "meeting" not in text and "calendar" not in text and "cal.com" not in text
        assert "find_meeting_times" not in spec.tools and "book_meeting" not in spec.tools
    assert {"find_meeting_times", "book_meeting"} <= set(on["contact"].tools)
    assert "book a meeting with Test Owner. Hand off to the contact agent." in on["classifier"].instructions
    assert "Let me check Test's calendar." in on["contact"].instructions
    assert "wants to book a meeting with the owner" in on["contact"].role


# --- the Cal.com client, against fake HTTP answers ---

EVENT_TYPES = {"status": "success", "data": [{"id": 11, "slug": "30min", "lengthInMinutes": 30}, {"id": 12, "slug": "15min", "lengthInMinutes": 15}]}
SLOTS = {"status": "success", "data": {"2026-11-27": [{"start": "2026-11-27T15:00:00.000Z"}, {"start": "2026-11-27T15:15:00.000Z"}]}}


def client_with(handler) -> CalCom:
    return CalCom(api_key="test-key", event_type_slug="15min", client=httpx.AsyncClient(base_url=API, transport=httpx.MockTransport(handler)))


async def test_the_client_finds_the_event_type_by_slug_once_and_reads_the_slots():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=EVENT_TYPES if request.url.path.endswith("/event-types") else SLOTS)

    calcom = client_with(handler)
    slots = await calcom.free_slots(NOW, NOW + timedelta(days=7))
    await calcom.free_slots(NOW, NOW + timedelta(days=7))
    assert slots == Slots(starts=[datetime(2026, 11, 27, 15, 0, tzinfo=UTC), datetime(2026, 11, 27, 15, 15, tzinfo=UTC)], minutes=15)
    assert [r.url.path for r in seen] == ["/v2/event-types", "/v2/slots", "/v2/slots"]
    assert seen[1].url.params["eventTypeId"] == "12" and seen[1].url.params["start"] == "2026-11-26T14:00:00Z"
    assert seen[1].headers["cal-api-version"] == "2024-09-04"


async def test_the_client_books_with_the_attendee_zone_and_the_notes():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            sent.append(json.loads(request.content))
            assert request.headers["cal-api-version"] == "2026-02-25"
            return httpx.Response(201, json={"status": "success", "data": {"id": 1, "uid": "abc", "status": "accepted"}})
        return httpx.Response(200, json=EVENT_TYPES)

    booking = await client_with(handler).book(eastern(27, 10), name="Ada", email="ada@example.com", time_zone="America/New_York", notes="A booking app.")
    assert booking == Booking(uid="abc", status="accepted")
    assert sent == [
        {
            "start": "2026-11-27T15:00:00Z",
            "eventTypeId": 12,
            "attendee": {"name": "Ada", "email": "ada@example.com", "timeZone": "America/New_York", "language": "en"},
            "bookingFieldsResponses": {"notes": "A booking app."},
        }
    ]


@pytest.mark.parametrize(
    ("answer", "answered"),
    [
        (httpx.Response(400, json={"status": "error", "error": {"code": "BadRequestException"}}), True),
        (httpx.Response(503, text="busy"), False),
        (httpx.ConnectTimeout("slow"), False),
    ],
)
async def test_the_client_tells_a_refusal_from_no_answer(answer, answered):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=EVENT_TYPES)
        if isinstance(answer, Exception):
            raise answer
        return answer

    with pytest.raises(CalcomError) as error:
        await client_with(handler).book(eastern(27, 10), name="Ada", email="ada@example.com", time_zone="America/New_York", notes="")
    assert error.value.answered is answered


async def test_an_unknown_slug_is_a_refusal():
    calcom = CalCom(api_key="k", event_type_slug="45min", client=httpx.AsyncClient(base_url=API, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=EVENT_TYPES))))
    with pytest.raises(CalcomError, match="45min"):
        await calcom.free_slots(NOW, NOW + timedelta(days=1))
