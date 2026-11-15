"""SQLite store for leads captured by the contact agent."""

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


class LeadStore:
    """One table, one row per confirmed lead."""

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
        self._conn.commit()

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

    def close(self) -> None:
        self._conn.close()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
