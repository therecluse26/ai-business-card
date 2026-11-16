"""Guardrails for a public phone number: who may call, for how long, and when a live transfer is allowed.

Every limit is a setting. The checks read the call log in SQLite, so they survive a server restart.
Nothing here records audio; the server never records a call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from ai_call_center.callers import UNKNOWN_NUMBERS
from ai_call_center.leads import LeadStore

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
HIDDEN_NUMBER = "hidden"  # every caller with no caller ID shares one allowance


def _clock(value: time) -> str:
    return value.strftime("%-I:%M %p").replace(":00 ", " ")


@dataclass(frozen=True)
class TransferHours:
    """When a live transfer to the owner is allowed, in US Eastern time."""

    start: time
    end: time
    days: frozenset[int]  # 0 = Monday

    @classmethod
    def parse(cls, hours: str, days: str = "mon-sun") -> TransferHours | None:
        """`hours` like "09:00-18:00"; `days` like "mon-fri" or "mon,wed,sat". Empty hours mean any time (None)."""
        if not hours.strip():
            return None
        start_text, end_text = (part.strip() for part in hours.split("-", 1))
        chosen: set[int] = set()
        for part in (days or "mon-sun").lower().replace(" ", "").split(","):
            if "-" in part:
                first, last = (DAYS.index(d[:3]) for d in part.split("-", 1))
                chosen.update(range(first, last + 1) if first <= last else [*range(first, 7), *range(0, last + 1)])
            elif part:
                chosen.add(DAYS.index(part[:3]))
        return cls(start=time.fromisoformat(start_text), end=time.fromisoformat(end_text), days=frozenset(chosen))

    def is_open(self, now: datetime | None = None) -> bool:
        local = (now or datetime.now(UTC)).astimezone(EASTERN)
        return local.weekday() in self.days and self.start <= local.time() < self.end

    def describe(self) -> str:
        days = sorted(self.days)
        if len(days) == 7:
            day_text = "every day"
        elif days == list(range(days[0], days[-1] + 1)) and len(days) > 1:
            day_text = f"{DAY_NAMES[days[0]]} to {DAY_NAMES[days[-1]]}"
        else:
            day_text = ", ".join(DAY_NAMES[d] for d in days)
        return f"{_clock(self.start)} to {_clock(self.end)} Eastern, {day_text}"


@dataclass(frozen=True)
class Admission:
    allowed: bool
    reason: str = ""  # for the log
    message: str = ""  # spoken to the caller when the call is turned away


@dataclass
class CallGuard:
    """Decides, when a call arrives, whether the agents take it."""

    store: LeadStore
    max_calls_per_number_per_day: int = 5
    max_concurrent_calls: int = 3
    daily_minutes_cap: int = 120
    max_call_minutes: int = 10
    exempt_numbers: frozenset[str] = frozenset()  # the owner's own phone: no per-number limit

    def admit(self, caller_number: str, now: datetime | None = None) -> Admission:
        now = now or datetime.now(UTC)
        day_ago = (now - timedelta(hours=24)).isoformat(timespec="seconds")
        call_secs = self.max_call_minutes * 60
        live_since = (now - timedelta(seconds=call_secs + 60)).isoformat(timespec="seconds")
        number = HIDDEN_NUMBER if caller_number in UNKNOWN_NUMBERS else caller_number

        if self.max_concurrent_calls and self.store.open_calls_since(live_since) >= self.max_concurrent_calls:
            return Admission(False, "all lines busy", "All lines are busy right now. Please call back in a few minutes. Goodbye.")
        if (
            self.max_calls_per_number_per_day
            and number not in self.exempt_numbers
            and self.store.calls_started_since(number, day_ago) >= self.max_calls_per_number_per_day
        ):
            return Admission(
                False,
                "per-number daily limit",
                f"This demo line takes {self.max_calls_per_number_per_day} calls a day from one number, and you have reached that. Please call again tomorrow. Goodbye.",
            )
        if self.daily_minutes_cap and self.store.call_seconds_since(day_ago, now=now, cap_secs=call_secs) >= self.daily_minutes_cap * 60:
            return Admission(False, "daily minutes cap", "This demo line has reached its limit for today. Please call again tomorrow. Goodbye.")
        return Admission(True)

    def caller_key(self, caller_number: str) -> str:
        """The number a call is logged under: hidden numbers share one key."""
        return HIDDEN_NUMBER if caller_number in UNKNOWN_NUMBERS else caller_number
