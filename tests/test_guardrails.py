"""Guardrails: who may call, when a live transfer is allowed, and how a call is turned away."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, ToolMessage

from ai_call_center.config import Settings
from ai_call_center.guardrails import CallGuard, TransferHours
from ai_call_center.telephony.telnyx_client import decode_state, encode_state
from ai_call_center.telephony.webhooks import handle_event
from tests.conftest import scripted, tool_call

EASTERN = ZoneInfo("America/New_York")
NUMBER = "+15555550123"
OWNER = "+15555550100"


def eastern(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=EASTERN)


# --- transfer hours ---


def test_transfer_hours_follow_the_eastern_clock_and_the_weekday():
    hours = TransferHours.parse("09:00-18:00", "mon-fri")
    assert hours.is_open(eastern(2026, 11, 16, 9, 0))  # Monday 9:00 AM
    assert hours.is_open(eastern(2026, 11, 16, 17, 59))
    assert not hours.is_open(eastern(2026, 11, 16, 18, 0))  # the end is exclusive
    assert not hours.is_open(eastern(2026, 11, 16, 8, 59))
    assert not hours.is_open(eastern(2026, 11, 15, 12, 0))  # Sunday
    # 14:30 UTC on a Monday in November is 9:30 AM Eastern (EST, UTC-5)
    assert hours.is_open(datetime(2026, 11, 16, 14, 30, tzinfo=UTC))
    # the same UTC time in July is 10:30 AM Eastern (EDT, UTC-4)
    assert hours.is_open(datetime(2026, 7, 13, 14, 30, tzinfo=UTC))


def test_transfer_hours_text_for_the_caller():
    assert TransferHours.parse("09:00-18:00", "mon-fri").describe() == "9 AM to 6 PM Eastern, Monday to Friday"
    assert TransferHours.parse("08:30-20:00").describe() == "8:30 AM to 8 PM Eastern, every day"
    assert TransferHours.parse("10:00-14:00", "sat,sun").describe() == "10 AM to 2 PM Eastern, Saturday to Sunday"
    assert TransferHours.parse("10:00-14:00", "mon,wed").describe() == "10 AM to 2 PM Eastern, Monday, Wednesday"


def test_empty_transfer_hours_mean_any_time():
    assert TransferHours.parse("") is None


async def test_transfer_is_refused_outside_the_hours_and_no_transfer_is_made(make_graph, ctx, config, telephony):
    ctx.transfer_hours = TransferHours.parse("00:00-00:01", "mon")  # closed at any time a test runs, bar one minute a week
    if ctx.transfer_hours.is_open():
        ctx.transfer_hours = TransferHours.parse("00:02-00:03", "mon")
    model = scripted(tool_call("request_owner_transfer", reason="Wants a quote."), "He is not available now. May I take a message?")
    result = await make_graph(model).ainvoke({"messages": [HumanMessage("Connect me.")], "active_agent": "contact"}, config=config, context=ctx)

    assert telephony.transfers == []
    assert not result.get("transfer_requested")
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "No transfer was made" in tool_msg.content and "Eastern" in tool_msg.content


# --- admission ---


def guard(leads, **over) -> CallGuard:
    return CallGuard(leads, exempt_numbers=frozenset({OWNER}), **over)


def log_call(leads, call_id, number, *, started: datetime, minutes: float | None):
    leads._conn.execute(
        "INSERT INTO calls (call_control_id, caller_number, started_at, ended_at) VALUES (?, ?, ?, ?)",
        (
            call_id,
            number,
            started.isoformat(timespec="seconds"),
            (started + timedelta(minutes=minutes)).isoformat(timespec="seconds") if minutes is not None else None,
        ),
    )
    leads._conn.commit()


def test_sixth_call_in_a_day_from_one_number_is_turned_away(leads):
    now = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
    for i in range(5):
        log_call(leads, f"c{i}", NUMBER, started=now - timedelta(hours=i + 1), minutes=2)
    g = guard(leads)
    verdict = g.admit(NUMBER, now)
    assert not verdict.allowed and verdict.reason == "per-number daily limit" and "5 calls a day" in verdict.message
    assert g.admit("+15555550999", now).allowed  # another number is not affected
    assert g.admit(NUMBER, now + timedelta(hours=20)).allowed  # the oldest calls are now more than 24 hours old


def test_the_owner_phone_has_no_per_number_limit(leads):
    now = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
    for i in range(9):
        log_call(leads, f"c{i}", OWNER, started=now - timedelta(hours=1, minutes=i), minutes=1)
    assert guard(leads).admit(OWNER, now).allowed


def test_hidden_numbers_share_one_allowance(leads):
    now = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
    g = guard(leads)
    for i in range(5):
        log_call(leads, f"c{i}", g.caller_key("anonymous"), started=now - timedelta(hours=1, minutes=i), minutes=1)
    assert not g.admit("", now).allowed and not g.admit("anonymous", now).allowed


def test_calls_in_progress_are_capped(leads):
    now = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
    for i in range(3):
        log_call(leads, f"c{i}", f"+1555555020{i}", started=now - timedelta(minutes=2), minutes=None)
    log_call(leads, "stale", "+15555550300", started=now - timedelta(hours=3), minutes=None)  # a crashed call does not hold a line
    verdict = guard(leads).admit(NUMBER, now)
    assert not verdict.allowed and verdict.reason == "all lines busy"
    assert guard(leads, max_concurrent_calls=4).admit(NUMBER, now).allowed


def test_daily_minutes_cap_turns_every_caller_away(leads):
    now = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
    for i in range(12):
        log_call(leads, f"c{i}", f"+15555551{i:03d}", started=now - timedelta(hours=2, minutes=11 * i), minutes=10)
    verdict = guard(leads).admit(NUMBER, now)  # 120 minutes used
    assert not verdict.allowed and verdict.reason == "daily minutes cap"
    assert guard(leads, daily_minutes_cap=121).admit(NUMBER, now).allowed
    assert guard(leads, daily_minutes_cap=0).admit(NUMBER, now).allowed  # 0 turns the limit off


# --- a turned-away call: answered with no AI session, told why, hung up ---


@dataclass
class FakeTelephony:
    log: list[tuple] = field(default_factory=list)

    async def answer_with_stream(self, call_control_id, *, stream_url):
        self.log.append(("answer_with_stream", call_control_id))

    async def answer_and_say(self, call_control_id, text):
        self.log.append(("answer_and_say", call_control_id, text))

    async def speak(self, call_control_id, text, *, then_hangup=False):
        self.log.append(("speak", call_control_id, text, then_hangup))

    async def hangup(self, call_control_id):
        self.log.append(("hangup", call_control_id))


def settings() -> Settings:
    return Settings(_env_file=None, public_url="https://example.ts.net", telnyx_app_id="app-1")


async def test_a_call_over_the_limit_never_reaches_the_agents(leads):
    now = datetime.now(UTC)
    for i in range(5):
        log_call(leads, f"c{i}", NUMBER, started=now - timedelta(minutes=30 + i), minutes=1)
    telephony = FakeTelephony()
    incoming = {"event_type": "call.initiated", "payload": {"call_control_id": "v2:x", "direction": "incoming", "from": NUMBER, "to": "+15550002"}}

    await handle_event(incoming, settings(), telephony, guard=guard(leads))

    (kind, ccid, message), = telephony.log
    assert (kind, ccid) == ("answer_and_say", "v2:x") and "call again tomorrow" in message

    # Telnyx then reports the answer, with our state; we speak. It reports the speech ended; we hang up.
    await handle_event({"event_type": "call.answered", "payload": {"call_control_id": "v2:x", "client_state": encode_state({"say": message})}}, settings(), telephony)
    await handle_event({"event_type": "call.speak.ended", "payload": {"call_control_id": "v2:x", "client_state": encode_state({"then": "hangup"})}}, settings(), telephony)
    assert telephony.log[1:] == [("speak", "v2:x", message, True), ("hangup", "v2:x")]


async def test_a_normal_call_is_answered_and_its_events_trigger_no_speech_or_hangup(leads):
    telephony = FakeTelephony()
    incoming = {"event_type": "call.initiated", "payload": {"call_control_id": "v2:y", "direction": "incoming", "from": NUMBER, "to": "+15550002"}}
    await handle_event(incoming, settings(), telephony, guard=guard(leads))
    await handle_event({"event_type": "call.answered", "payload": {"call_control_id": "v2:y"}}, settings(), telephony)
    await handle_event({"event_type": "call.speak.ended", "payload": {"call_control_id": "v2:y"}}, settings(), telephony)
    assert telephony.log == [("answer_with_stream", "v2:y")]


def test_client_state_round_trip_and_bad_input():
    assert decode_state(encode_state({"then": "hangup"})) == {"then": "hangup"}
    assert decode_state(None) == {} and decode_state("not base64 json") == {}
