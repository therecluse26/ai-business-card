"""SQLite store for leads captured by the contact agent, and a call log with one summary per call.

The call log is what makes a returning caller recognisable. It holds no audio and no transcript:
one or two sentences written by a text model when the call ends, plus the caller's name if given.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Lead:
    name: str
    email: str
    company: str
    need: str
    caller_number: str
    call_control_id: str
    created_at: str


@dataclass(frozen=True)
class CallRecord:
    call_control_id: str
    caller_number: str
    started_at: str
    ended_at: str | None
    agents: str
    summary: str
    caller_name: str


@dataclass(frozen=True)
class NumberInfo:
    """Public carrier data for a phone number, from a number lookup. Any field can be empty."""

    caller_name: str = ""
    carrier: str = ""
    line_type: str = ""
    city: str = ""
    state: str = ""

    @property
    def empty(self) -> bool:
        return not any((self.caller_name, self.carrier, self.line_type, self.city, self.state))


class LeadStore:
    """Three tables: one row per confirmed lead, one row per call, one row per looked-up number."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                company TEXT NOT NULL,
                need TEXT NOT NULL,
                caller_number TEXT NOT NULL,
                call_control_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_control_id TEXT PRIMARY KEY,
                caller_number TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                agents TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                caller_name TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS number_info (
                number TEXT PRIMARY KEY,
                caller_name TEXT NOT NULL DEFAULT '',
                carrier TEXT NOT NULL DEFAULT '',
                line_type TEXT NOT NULL DEFAULT '',
                city TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL DEFAULT '',
                looked_up_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # --- leads ---

    def save(self, lead: Lead) -> int:
        cur = self._conn.execute(
            "INSERT INTO leads (name, email, company, need, caller_number, call_control_id, created_at)"
            " VALUES (:name, :email, :company, :need, :caller_number, :call_control_id, :created_at)",
            asdict(lead),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def all(self) -> list[Lead]:
        rows = self._conn.execute(
            "SELECT name, email, company, need, caller_number, call_control_id, created_at FROM leads ORDER BY id"
        ).fetchall()
        return [Lead(*row) for row in rows]

    def lead_for_call(self, call_control_id: str) -> tuple[int, Lead] | None:
        """The newest lead saved during this call, with its record number."""
        row = self._conn.execute(
            "SELECT id, name, email, company, need, caller_number, call_control_id, created_at FROM leads"
            " WHERE call_control_id = ? ORDER BY id DESC LIMIT 1",
            (call_control_id,),
        ).fetchone()
        return (row[0], Lead(*row[1:])) if row else None

    def latest_lead_for(self, caller_number: str, *, with_email: bool = False) -> Lead | None:
        """The newest lead from this number. With `with_email`, the newest one that holds an email address."""
        row = self._conn.execute(
            "SELECT name, email, company, need, caller_number, call_control_id, created_at FROM leads"
            f" WHERE caller_number = ?{" AND email != ''" if with_email else ''} ORDER BY id DESC LIMIT 1",
            (caller_number,),
        ).fetchone()
        return Lead(*row) if row else None

    # --- calls ---

    def start_call(self, call_control_id: str, caller_number: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO calls (call_control_id, caller_number, started_at) VALUES (?, ?, ?)",
            (call_control_id, caller_number, now_iso()),
        )
        self._conn.commit()

    def reopen_call(self, call_control_id: str) -> None:
        """The AI session is back on a call it had handed over: the call counts as in progress again."""
        self._conn.execute("UPDATE calls SET ended_at = NULL WHERE call_control_id = ?", (call_control_id,))
        self._conn.commit()

    def finish_call(self, call_control_id: str, *, agents: list[str], summary: str, caller_name: str = "") -> None:
        self._conn.execute(
            "UPDATE calls SET ended_at = ?, agents = ?, summary = ?, caller_name = ? WHERE call_control_id = ?",
            (now_iso(), " -> ".join(agents), summary.strip(), caller_name.strip(), call_control_id),
        )
        self._conn.commit()

    def calls_for(self, caller_number: str, *, exclude: str = "", limit: int = 3) -> list[CallRecord]:
        """Finished calls from this number, newest first."""
        rows = self._conn.execute(
            "SELECT call_control_id, caller_number, started_at, ended_at, agents, summary, caller_name FROM calls"
            " WHERE caller_number = ? AND call_control_id != ? AND ended_at IS NOT NULL ORDER BY started_at DESC LIMIT ?",
            (caller_number, exclude, limit),
        ).fetchall()
        return [CallRecord(*row) for row in rows]

    def call_count_for(self, caller_number: str, *, exclude: str = "") -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM calls WHERE caller_number = ? AND call_control_id != ? AND ended_at IS NOT NULL",
            (caller_number, exclude),
        ).fetchone()
        return int(row[0]) if row else 0

    # --- guardrail counts (ISO timestamps in UTC compare as text) ---

    def calls_started_since(self, caller_number: str, since_iso: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM calls WHERE caller_number = ? AND started_at >= ?", (caller_number, since_iso)
        ).fetchone()
        return int(row[0])

    def open_calls_since(self, since_iso: str) -> int:
        """Calls that started after `since_iso` and have not ended: the calls in progress."""
        row = self._conn.execute("SELECT COUNT(*) FROM calls WHERE ended_at IS NULL AND started_at >= ?", (since_iso,)).fetchone()
        return int(row[0])

    def call_seconds_since(self, since_iso: str, *, now: datetime, cap_secs: int) -> float:
        """Total call time since `since_iso`. A call with no end time counts up to now, at most `cap_secs`."""
        total = 0.0
        for started, ended in self._conn.execute("SELECT started_at, ended_at FROM calls WHERE started_at >= ?", (since_iso,)):
            start = datetime.fromisoformat(started)
            end = datetime.fromisoformat(ended) if ended else now
            total += min(max((end - start).total_seconds(), 0.0), cap_secs)
        return total

    # --- number lookups ---

    def save_number_info(self, number: str, info: NumberInfo) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO number_info (number, caller_name, carrier, line_type, city, state, looked_up_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (number, info.caller_name, info.carrier, info.line_type, info.city, info.state, now_iso()),
        )
        self._conn.commit()

    def number_info_for(self, number: str) -> NumberInfo | None:
        row = self._conn.execute(
            "SELECT caller_name, carrier, line_type, city, state FROM number_info WHERE number = ?", (number,)
        ).fetchone()
        return NumberInfo(*row) if row else None

    def close(self) -> None:
        self._conn.close()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
