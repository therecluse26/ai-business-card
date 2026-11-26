"""A short meeting with the owner, booked during the call on Cal.com.

The rules:

- Code reads the free times and makes the booking. The voice model never sends a time: it passes the
  option id of a time that `find_times` gave, and code maps the id back to the exact start in UTC.
- A booking makes Cal.com email an invitation, so it waits for the caller's yes, as a message does. The
  first `book` call gives the agent the read-back words (the time, and the email address letter by letter);
  only a second call with the same time and address, after the caller's yes, books.
- One booking for each call, and CALCOM_MAX_BOOKINGS_PER_DAY for the whole line in 24 hours, counted in the
  table `meetings`. A bad actor could make Cal.com email invitations to a stranger's address; the limits
  bound that. The owner's phone counts too: a phone number can be faked.
- With no key, with no answer from Cal.com, or with a limit reached, the agent offers to take a message.
- "Forget me" keeps only the time of the request in the table, for the daily limit. What Cal.com has stays
  with the owner, as the lead email does.

Each Cal.com endpoint wants its own `cal-api-version`. The versions below are the ones its documents
named on 2026-11-26.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from ai_call_center.leads import Lead, now_iso

if TYPE_CHECKING:
    from ai_call_center.graph.state import CallContext

log = logging.getLogger(__name__)

API = "https://api.cal.com/v2"
VERSIONS = {"/event-types": "2024-06-14", "/slots": "2024-09-04", "/bookings": "2026-02-25"}
EASTERN_ZONE = "America/New_York"
EASTERN = ZoneInfo(EASTERN_ZONE)
DAYS_AHEAD = 7  # the free times come from this many days
MAX_OPTIONS = 3
SOON = timedelta(minutes=30)  # a time that starts sooner is not offered: the call itself takes minutes
SAME_DAY_GAP = timedelta(hours=2)  # two times offered on one day are at least this far apart
TARGETS = (time(10, 0), time(14, 0), time(11, 0))  # Eastern; each pick is the free time nearest the next one
NOTES_MAX = 400  # the notes go into the invitation, to the address the caller gave
DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

ASKING, ACCEPTED, PENDING, NO_ANSWER = "asking", "accepted", "pending", "no answer"
BOOKED = (ACCEPTED, PENDING)
# A word of the read-back that a voice model keeps when it says the words a little differently. A yes after
# a turn with it is a yes to the meeting, and the check for a missed save_lead (graph/tools.py) leaves it alone.
READ_BACK_MARK = "invitation"


class CalcomError(Exception):
    """A request that did not work. `answered`: Cal.com gave an answer (it refused); False = no answer, so a booking may exist."""

    def __init__(self, message: str, *, answered: bool) -> None:
        super().__init__(message)
        self.answered = answered


@dataclass(frozen=True)
class Slots:
    starts: list[datetime]  # UTC
    minutes: int  # the length of the meeting


@dataclass(frozen=True)
class Booking:
    uid: str
    status: str  # Cal.com's: "accepted", or "pending" when the owner must confirm it first


@runtime_checkable  # the tool schema of a LangChain tool checks the call context's fields with isinstance
class Scheduler(Protocol):
    async def free_slots(self, start: datetime, end: datetime) -> Slots:
        """The free start times between two moments. Raises CalcomError."""
        ...

    async def book(self, start: datetime, *, name: str, email: str, time_zone: str, notes: str) -> Booking:
        """Book the meeting. Raises CalcomError."""
        ...


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class CalCom:
    """Cal.com's API v2 with an API key. One client for the process: a new client pays for a new connection.

    The event type is found by its slug on first use, so the setting holds no id of one account.
    Measured on 2026-11-26 from the laptop: the event types in 188 ms, the free times of 7 days in 471 ms,
    then 84 to 88 ms on the open connection."""

    def __init__(self, *, api_key: str, event_type_slug: str, time_limit: float = 5.0, client: Any = None) -> None:
        import httpx

        self.slug = event_type_slug
        self._event: tuple[int, int] | None = None  # (id, minutes)
        self._client = client or httpx.AsyncClient(base_url=API, timeout=time_limit, headers={"Authorization": f"Bearer {api_key}"})

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import httpx

        version = next(v for prefix, v in VERSIONS.items() if path.startswith(prefix))
        try:
            response = await self._client.request(method, path, headers={"cal-api-version": version}, **kwargs)
        except httpx.HTTPError as error:
            raise CalcomError(f"{method} {path}: {type(error).__name__}", answered=False) from error
        if response.status_code >= 500:
            raise CalcomError(f"{method} {path}: status {response.status_code}", answered=False)
        try:
            body = response.json()
        except ValueError as error:
            raise CalcomError(f"{method} {path}: the answer is not JSON", answered=response.status_code < 300) from error
        if response.status_code >= 400 or not isinstance(body, dict) or body.get("status") != "success":
            fault = body.get("error", {}) if isinstance(body, dict) else {}
            raise CalcomError(f"{method} {path}: status {response.status_code} {fault.get('code', '') if isinstance(fault, dict) else ''}".strip(), answered=True)
        return body.get("data")

    async def _event_type(self) -> tuple[int, int]:
        if self._event is None:
            data = await self._request("GET", "/event-types")
            found = next((e for e in data or [] if isinstance(e, dict) and e.get("slug") == self.slug), None)
            if found is None:
                raise CalcomError(f"no event type with the slug {self.slug!r}", answered=True)
            self._event = (int(found["id"]), int(found.get("lengthInMinutes") or 15))
        return self._event

    async def free_slots(self, start: datetime, end: datetime) -> Slots:
        event_id, minutes = await self._event_type()
        data = await self._request("GET", "/slots", params={"eventTypeId": event_id, "start": _utc(start), "end": _utc(end)})
        # {"2026-11-27": [{"start": "2026-11-27T14:00:00.000Z"}, ...], ...}: the days are UTC dates when no timeZone is sent.
        starts = [datetime.fromisoformat(slot["start"]) for day in (data or {}).values() for slot in day if isinstance(slot, dict) and slot.get("start")]
        return Slots(starts=starts, minutes=minutes)

    async def book(self, start: datetime, *, name: str, email: str, time_zone: str, notes: str) -> Booking:
        event_id, _ = await self._event_type()
        data = await self._request(
            "POST",
            "/bookings",
            json={
                "start": _utc(start),
                "eventTypeId": event_id,
                "attendee": {"name": name, "email": email, "timeZone": time_zone, "language": "en"},
                "bookingFieldsResponses": {"notes": notes},
            },
        )
        if not isinstance(data, dict):
            raise CalcomError("POST /bookings: no booking in the answer", answered=True)
        return Booking(uid=str(data.get("uid") or ""), status=str(data.get("status") or ""))


def calcom_configured(settings: Any) -> bool:
    return bool(settings.calcom_api_key.get_secret_value())


_default: CalCom | None = None


def default_scheduler(settings: Any) -> Scheduler | None:
    """The process-wide Cal.com client, or None when CALCOM_API_KEY is not set (the agents say nothing about a meeting)."""
    global _default
    if not calcom_configured(settings):
        return None
    if _default is None:
        _default = CalCom(api_key=settings.calcom_api_key.get_secret_value(), event_type_slug=settings.calcom_event_type_slug, time_limit=settings.calcom_timeout_secs)
        log.info("Meetings are on: Cal.com event type %r, at most %d bookings in 24 hours", settings.calcom_event_type_slug, settings.calcom_max_bookings_per_day)
    return _default


# --- the times ---


@dataclass(frozen=True)
class MeetingOption:
    id: str
    start: datetime  # UTC
    spoken: str  # "Thursday, November 27, at 2:30 PM Eastern"


def spoken_time(start: datetime) -> str:
    local = start.astimezone(EASTERN)
    clock = local.strftime("%-I:%M %p").replace(":00 ", " ")
    return f"{local:%A}, {local:%B} {local.day}, at {clock} Eastern"


def _day_wanted(day: str, now: datetime) -> date | int | None:
    """The day the caller asked for: a date, a weekday (0 = Monday), or None when the words name no day."""
    words = day.casefold()
    today = now.astimezone(EASTERN).date()
    if "tomorrow" in words:
        return today + timedelta(days=1)
    if "today" in words:
        return today
    return next((n for n, name in enumerate(DAY_NAMES) if name in words or re.search(rf"\b{name[:3]}\b", words)), None)


def choose_options(starts: list[datetime], *, now: datetime, day: str = "", count: int = MAX_OPTIONS) -> list[datetime]:
    """At most `count` free times, on different days where possible, sorted. Each day in turn gets the free time
    nearest the next target hour; a second pass over the same days fills up the list."""
    soon = now + SOON
    times = sorted({start.astimezone(UTC) for start in starts if start >= soon})
    wanted = _day_wanted(day, now) if day.strip() else None
    if wanted is not None:
        days = [t.astimezone(EASTERN).date() for t in times]
        times = [t for t, d in zip(times, days) if (d == wanted if isinstance(wanted, date) else d.weekday() == wanted)]
    by_day: dict[date, list[datetime]] = {}
    for moment in times:
        by_day.setdefault(moment.astimezone(EASTERN).date(), []).append(moment)
    picks: list[datetime] = []
    target = 0
    while len(picks) < count:
        added = False
        for slots in by_day.values():
            if len(picks) == count:
                break
            free = [s for s in slots if all(abs(s - p) >= SAME_DAY_GAP for p in picks)]
            if not free:
                continue
            aim = TARGETS[target % len(TARGETS)]
            target += 1
            minute = aim.hour * 60 + aim.minute
            picks.append(min(free, key=lambda s: abs(s.astimezone(EASTERN).hour * 60 + s.astimezone(EASTERN).minute - minute)))
            added = True
        if not added:
            break
    return sorted(picks)


def option_key(text: str) -> str:
    """ "2", "option 2", "Option two" as "2"."""
    words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
    found = re.search(r"\d+", text) or re.search(r"\b(" + "|".join(words) + r")\b", text.casefold())
    if not found:
        return ""
    return found.group(0) if found.group(0).isdigit() else words[found.group(1)]


# --- the email address ---

EMAIL = re.compile(r"^[a-z0-9](?:[a-z0-9._%+-]*[a-z0-9])?@[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\.[a-z]{2,}$")
# Words of a spoken address, when the transcript gives the words and not the signs.
SPOKEN_SIGNS = {"at": "@", "dot": ".", "period": ".", "point": ".", "underscore": "_", "dash": "-", "hyphen": "-", "plus": "+"}
SAID_SIGNS = {"@": "at", ".": "dot", "_": "underscore", "-": "dash", "+": "plus", "%": "percent"}
# Domains that people say as words. Any other domain is read back letter by letter, as the name before the @ is.
SAID_DOMAINS = {"gmail.com": "gmail dot com", "outlook.com": "outlook dot com", "hotmail.com": "hotmail dot com", "yahoo.com": "yahoo dot com", "icloud.com": "icloud dot com", "aol.com": "a o l dot com", "proton.me": "proton dot me", "protonmail.com": "protonmail dot com"}


def normal_email(text: str) -> str:
    """The address in its written form, or "" when it is not one. A strict, simple check: it refuses some odd
    addresses that are valid, and that is fine for a spoken address."""
    text = text.strip().casefold().strip(".,;:!?\"'<>()[]")
    if "@" not in text:
        tokens = re.split(r"[\s,]+", text)
        tokens = [SPOKEN_SIGNS.get(token, token) for token in tokens if token not in ("sign", "symbol")]
        text = "".join(tokens)
    text = re.sub(r"\s+", "", text)
    return text if EMAIL.match(text) and ".." not in text and text.count("@") == 1 else ""


def _letters(part: str) -> str:
    return ", ".join(SAID_SIGNS.get(ch, ch.upper()) for ch in part)


def spelled_email(email: str) -> str:
    """The address as the agent reads it back: "J, O, H, N, at, gmail dot com"."""
    local, _, domain = email.partition("@")
    said = SAID_DOMAINS.get(domain) or _letters(domain)
    return f"{_letters(local)}, at, {said}"


# --- the record of each booking ---


@dataclass(frozen=True)
class Meeting:
    call_control_id: str
    booked_at: str
    start: str  # UTC ISO; "" after "forget me"
    status: str


class MeetingStore:
    """One row for each booking request. It keeps its table in the lead store's database, as `PhraseStore` does."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_control_id TEXT NOT NULL,
                booked_at TEXT NOT NULL,
                start TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '',
                uid TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._conn.commit()

    def limit_reached(self, call_control_id: str, *, per_day: int, now: datetime) -> str:
        """ "call" when this call has a booking or a request with no answer, "day" when the line has `per_day` in 24 hours, else ""."""
        if self._conn.execute("SELECT 1 FROM meetings WHERE call_control_id = ?", (call_control_id,)).fetchone():
            return "call"
        since = (now - timedelta(hours=24)).astimezone(UTC).isoformat(timespec="seconds")
        (count,) = self._conn.execute("SELECT COUNT(*) FROM meetings WHERE booked_at >= ?", (since,)).fetchone()
        return "day" if count >= per_day else ""  # 0 = no booking at all: this limit is never off

    def reserve(self, call_control_id: str, *, start: datetime, now: datetime) -> int:
        """A row before the request, so that two calls at the same moment cannot pass the daily limit together.
        The check and this write run with no await between them."""
        cur = self._conn.execute(
            "INSERT INTO meetings (call_control_id, booked_at, start, status) VALUES (?, ?, ?, ?)",
            (call_control_id, now.astimezone(UTC).isoformat(timespec="seconds"), start.astimezone(UTC).isoformat(timespec="seconds"), ASKING),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def settle(self, row_id: int, *, status: str, uid: str = "", keep: bool = True) -> None:
        """Cal.com's answer. `keep` False: the caller asked to be forgotten, so only the time of the request stays."""
        if keep:
            self._conn.execute("UPDATE meetings SET status = ?, uid = ? WHERE id = ?", (status, uid, row_id))
        else:
            self._conn.execute("UPDATE meetings SET status = ?, uid = '', start = '' WHERE id = ?", (status, row_id))
        self._conn.commit()

    def release(self, row_id: int) -> None:
        """Cal.com refused the booking: nothing was booked, and the caller may pick another time."""
        self._conn.execute("DELETE FROM meetings WHERE id = ?", (row_id,))
        self._conn.commit()

    def for_call(self, call_control_id: str) -> Meeting | None:
        row = self._conn.execute(
            "SELECT call_control_id, booked_at, start, status FROM meetings WHERE call_control_id = ? AND status != ? ORDER BY id DESC LIMIT 1", (call_control_id, ASKING)
        ).fetchone()
        return Meeting(*row) if row else None

    def forget(self, call_ids: list[str]) -> None:
        """The time of each request stays, for the daily limit. The meeting's time and Cal.com's id go."""
        self._conn.executemany("UPDATE meetings SET start = '', uid = '' WHERE call_control_id = ?", [(call_id,) for call_id in call_ids])
        self._conn.commit()


def _when(start: datetime) -> str:
    return start.astimezone(EASTERN).strftime("%a %b %-d at %-I:%M %p %Z")


def meeting_line(status: str, start: datetime) -> str:
    """The booking, for the owner: the lead email and the CRM entry."""
    when = _when(start)
    if status == ACCEPTED:
        return f"{when}. Booked in Cal.com; Cal.com emailed the invitation to the caller."
    if status == PENDING:
        return f"{when}. It waits for your confirmation in Cal.com."
    if status == NO_ANSWER:
        return f"{when} was asked for, and Cal.com gave no answer. Check Cal.com: the booking may exist."
    return f"{when}. Cal.com answered: {status}."


def booked_line(ctx: CallContext) -> str:
    """The booking of this call for the owner, or ""."""
    meeting = MeetingStore(ctx.leads.connection).for_call(ctx.call_control_id)
    if meeting is None or not meeting.start:
        return ""
    return meeting_line(meeting.status, datetime.fromisoformat(meeting.start))


def meeting_booked(ctx: CallContext) -> bool:
    """True when this call booked a meeting: the outcome of the call in the analytics."""
    meeting = MeetingStore(ctx.leads.connection).for_call(ctx.call_control_id)
    return meeting is not None and meeting.status in BOOKED


# --- the two tools ---


def _say(words: str) -> str:
    return f'Say exactly these words to the caller, and nothing else: "{words}"'


def _first(owner: str) -> str:
    return owner.split()[0] if owner.strip() else "the owner"


def _no_booking(owner: str) -> str:
    return (
        "No meeting can be booked on this call right now. Do not offer a time, and do not say that anything is booked. "
        f"Say in one sentence that you cannot book it right now, and offer to take a message so {owner} can set up a time. Do not give a technical reason."
    )


def _limit_text(reason: str, ctx: CallContext) -> str:
    owner = _first(ctx.owner_name)
    if reason == "call":
        meeting = MeetingStore(ctx.leads.connection).for_call(ctx.call_control_id)
        when = f" ({spoken_time(datetime.fromisoformat(meeting.start))})" if meeting and meeting.status in BOOKED and meeting.start else ""
        return (
            f"This call has a booking already{when}. One call books one meeting. Do not book another one. "
            f"If the caller wants something else, offer to take a message for {owner}."
        )
    return _no_booking(owner)


async def find_times(ctx: CallContext, day: str = "", *, now: datetime | None = None) -> str:
    """The free times for the caller, as an option id and the spoken words of each. Shared by text and voice modes."""
    now = now or datetime.now(UTC)
    owner = _first(ctx.owner_name)
    if ctx.scheduler is None:
        return _no_booking(owner)
    if reason := MeetingStore(ctx.leads.connection).limit_reached(ctx.call_control_id, per_day=ctx.meeting_cap, now=now):
        return _limit_text(reason, ctx)
    try:
        slots = await ctx.scheduler.free_slots(now, now + timedelta(days=DAYS_AHEAD))
    except Exception:
        log.exception("call %s: could not read the free times from Cal.com", ctx.call_control_id)
        return _no_booking(owner)
    picks = choose_options(slots.starts, now=now, day=day)
    if not picks:
        if day.strip():
            return f'There is no free time on "{day.strip()}" in the next {DAYS_AHEAD} days. Say so in one sentence, and ask if another day suits them. Do not offer a time before a new find_meeting_times result.'
        return f"There is no free time in the next {DAYS_AHEAD} days. Say so in one sentence, and offer to take a message so {owner} can set up a time."
    first_id = max((int(key) for key in ctx.meeting_options), default=0) + 1
    options = [MeetingOption(id=str(first_id + n), start=start, spoken=spoken_time(start)) for n, start in enumerate(picks)]
    ctx.meeting_options.update({option.id: option for option in options})
    listing = "; ".join(f"option {option.id}: {option.spoken}" for option in options)
    log.info("call %s: offered %d meeting times", ctx.call_control_id, len(options))
    return (
        f"Free times for a {slots.minutes}-minute video meeting with {owner}: {listing}. "
        "Offer them in one or two sentences, in these words. Never say the option numbers, and never offer a time that is not in this list. "
        "When the caller picks one, ask for their email address for the invitation, and ask them to spell it. "
        "If none fits, ask which day suits them, and call find_meeting_times with that day."
    )


def said_yes_to_read_back(lines: list[str]) -> bool:
    """True when the agent's last turn was the read-back and the caller answered with a short yes. A short
    turn of the agent after the yes ("One moment.") does not count against it."""
    from ai_call_center.graph.tools import (
        confirmed_the_message,  # here: graph/tools.py imports this module
    )

    rest = list(lines)
    while rest and rest[-1].startswith("agent:") and READ_BACK_MARK not in rest[-1].casefold():
        rest.pop()
    return is_read_back(rest) and confirmed_the_message(rest)


def is_read_back(lines: list[str]) -> bool:
    """True when the agent's turn before the caller's last words is the meeting read-back."""
    rest = list(lines)
    while rest and rest[-1].startswith("caller:"):
        rest.pop()
    asked: list[str] = []
    while rest and rest[-1].startswith("agent:"):
        asked.insert(0, rest.pop())
    return any(READ_BACK_MARK in line.casefold() for line in asked)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.casefold()) if len(w) > 3}


def invitation_notes(ctx: CallContext, topic: str, lines: list[str]) -> str:
    """The notes that go into the invitation. They reach the address the caller gave, so they hold no passphrase
    and, for a verified caller, no sentence that is not from the caller's own words on this call: the notes of
    earlier calls are in that agent's conversation, and the address can be anyone's."""
    from ai_call_center.passphrase import masked

    topic = masked(" ".join(topic.split()), ctx.spoken_phrases)
    if ctx.caller_history:
        said = _words(" ".join(line for line in lines if line.startswith("caller:")))
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", topic) if s]
        topic = " ".join(s for s in sentences if (w := _words(s)) and len(w & said) / len(w) >= 0.5)
    owner = ctx.owner_name.strip() or "the owner"
    test = " A test booking from the owner's own phone." if ctx.caller_number == ctx.owner_phone else ""
    return f"Booked by phone on {owner}'s AI line.{test} {topic}".strip()[:NOTES_MAX]


async def book(
    ctx: CallContext, *, option: str, email: str, name: str = "", topic: str = "", lines: list[str], visited: list[str] | None = None, now: datetime | None = None
) -> str:
    """Book the time the caller chose. The first call returns the read-back; a call with the same time and
    address after the caller's yes books it. Returns the text for the model. Shared by text and voice modes."""
    now = now or datetime.now(UTC)
    owner = _first(ctx.owner_name)
    if ctx.scheduler is None:
        return _no_booking(owner)
    store = MeetingStore(ctx.leads.connection)
    if reason := store.limit_reached(ctx.call_control_id, per_day=ctx.meeting_cap, now=now):
        ctx.held_meeting = ""
        return _limit_text(reason, ctx)
    chosen = ctx.meeting_options.get(option_key(option))
    if chosen is None:
        return "Not booked: that option is not one that find_meeting_times gave on this call. Call find_meeting_times, and offer only the times it gives."
    address = normal_email(email)
    if not address:
        return "Not booked: that is not a full email address. Ask the caller to spell it again, one letter at a time, then call book_meeting again. Do not say anything is booked."
    name = " ".join(name.split()) or ctx.caller_name
    if not name:
        return "Not booked: the caller's name is needed for the invitation. Ask for it, then call book_meeting again."
    key = f"{chosen.id}|{address}"
    if key != ctx.held_meeting or not said_yes_to_read_back(lines):
        ctx.held_meeting = key
        words = f"So that's {chosen.spoken}, and the invitation goes to {spelled_email(address)}. Is that right?"
        return (
            f"Not booked yet. {_say(words)} "
            'After a yes, say only "One moment." and call book_meeting again with the same option and email address. '
            "If the caller corrects the address, ask them to spell it again, then call book_meeting with the new one."
        )
    row = store.reserve(ctx.call_control_id, start=chosen.start, now=now)
    ctx.held_meeting = ""
    try:
        booking = await ctx.scheduler.book(chosen.start, name=name, email=address, time_zone=EASTERN_ZONE, notes=invitation_notes(ctx, topic, lines))
    except CalcomError as error:
        if error.answered:
            store.release(row)
            log.warning("call %s: Cal.com refused the booking: %s", ctx.call_control_id, error)
            return (
                "Not booked: the calendar did not accept that time; it may have been taken a moment ago. Say so in one sentence. "
                f"Offer to look for other times, or to take a message for {owner}."
            )
        store.settle(row, status=NO_ANSWER, keep=not ctx.forgotten)
        log.error("call %s: Cal.com gave no answer to the booking: %s", ctx.call_control_id, error)
        return (
            "Not booked: the calendar gave no answer, so nobody knows yet if the time is booked. Do not say it is booked, and do not try again. "
            f"Say in one sentence that you could not finish the booking and that {owner} will follow up. Then offer to take a message."
        )
    except Exception:
        store.settle(row, status=NO_ANSWER, keep=not ctx.forgotten)
        log.exception("call %s: the booking failed", ctx.call_control_id)
        return f"Not booked: the booking failed. Do not say it is booked, and do not try again. Say in one sentence that you could not finish it, and offer to take a message for {owner}."
    store.settle(row, status=booking.status, uid=booking.uid, keep=not ctx.forgotten)
    log.info("call %s: meeting booked in Cal.com (status %s)", ctx.call_control_id, booking.status)
    await _tell_owner(ctx, name=name, email=address, topic=topic, status=booking.status, start=chosen.start, visited=visited or [])
    if booking.status == ACCEPTED:
        then = f"Booked: {chosen.spoken}. Cal.com emails the invitation to the address the caller gave. Tell the caller both in one or two sentences, without the address."
    elif booking.status == PENDING:
        then = (
            f"Requested: {chosen.spoken}. {owner} must confirm it first; Cal.com emails the caller when {owner} confirms. "
            "Tell the caller that in one or two sentences, without the address. Do not say it is confirmed."
        )
    else:
        return f"Not booked: the calendar answered \"{booking.status}\". Do not say it is booked. Say in one sentence that you could not finish the booking, and offer to take a message for {owner}."
    return f"{then} Then ask if there is anything else. {owner} has the caller's details now, so do not call save_lead for the meeting."


async def _tell_owner(ctx: CallContext, *, name: str, email: str, topic: str, status: str, start: datetime, visited: list[str]) -> None:
    """The lead email for the booking, and the lead record that the CRM sync reads. A caller who asked to be
    forgotten gets no record on this line; the owner still gets the email, as Cal.com still has the booking."""
    from ai_call_center.graph.tools import (
        call_details,  # here: graph/tools.py imports this module
    )
    from ai_call_center.passphrase import masked

    lead = Lead(
        name=name,
        email=email,
        company="",
        need=masked(" ".join(topic.split()), ctx.spoken_phrases) or f"Booked a meeting with {_first(ctx.owner_name)}.",
        caller_number=ctx.caller_number,
        call_control_id=ctx.call_control_id,
        created_at=now_iso(),
    )
    row_id = 0 if ctx.forgotten else ctx.leads.save(lead)
    if row_id:
        ctx.meeting_lead_id = row_id
        ctx.caller_name = ctx.caller_name or name
    try:
        details = replace(call_details(ctx, kind="meeting", notes="", visited=visited), meeting=meeting_line(status, start))
        await ctx.notifier.send_lead(lead, row_id=row_id, details=details)
    except Exception:
        log.exception("call %s: the meeting is booked, but the owner email failed", ctx.call_control_id)
