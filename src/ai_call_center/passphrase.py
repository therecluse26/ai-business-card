"""A passphrase for a returning caller: the check before an agent may use the notes of an earlier call.

A phone number can be faked, and a name is no secret. So a match of number and name makes a caller
"recognised", and nothing more: the agent says "Welcome back" and gets no notes. A caller who gave a
passphrase on an earlier call can say it again. Only then is the caller "verified", and only then do
the notes of earlier calls reach an agent (`graph/tools.py`: `identify`, `verify`, `set_phrase`).

The rules:

- Code compares the passphrase. The voice model passes on what it heard, and gets a yes or a no. It
  never gets the stored passphrase, and the server does not keep it: the table holds a salted hash.
- Speech is not exact. The comparison ignores capital letters, spaces and punctuation, and it uses how
  the words sound (Metaphone), so "lantern" and "lanturn" are the same passphrase.
- `TRIES` tries in one call. After `LOCK_AT` wrong tries in a row, over any number of calls, the line
  stops asking: that passphrase is locked for good.
- A passphrase opens only the calls that were made with it: the call where it was set, and each later
  call where the caller said it (`phrase_calls`). A call where the caller was only recognised is opened
  by no passphrase. So a person with a faked number who sets a passphrase gets nothing of the real
  caller's calls, before or after.
- A caller who forgot the passphrase can set a new one. It takes the place of the old one, and it opens
  only its own calls. So "I forgot it" never opens the old notes. The owner still has them.
- "Forget me" deletes the caller's row.

What this cannot promise: a short spoken phrase has few possible values, so the hash does not protect it
from someone who has the database file. That person has the notes too. The passphrase is in the private
records of the call (the transcript in the server log, the traces). It is removed from the two public
texts: a fault report and a GitHub issue (`private_words`).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import jellyfish

from ai_call_center.names import same_person

TRIES = 2  # in one call
LOCK_AT = 5  # wrong tries in a row, over all calls; then the line stops asking
MIN_LETTERS = 4  # a passphrase with fewer letters and digits is refused
MASK = "[passphrase]"

_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}
# Words of a passphrase that stay in a public text: without them a report would lose "the" and "of".
_SMALL_WORDS = frozenset("a an and are at be for from i in is it me my of on or our the to we with you your".split())


def plain(phrase: str) -> str:
    """The passphrase as letters and digits only, in lower case."""
    return re.sub(r"[^a-z0-9]+", "", phrase.lower())


def sound_key(phrase: str) -> str:
    """What the comparison uses: how the passphrase sounds. Digits stay as they are."""
    letters = plain(phrase)
    parts = [part if part.isdigit() else jellyfish.metaphone(part) or part for part in re.findall(r"[a-z]+|[0-9]+", letters)]
    return "".join(parts).lower()


def usable(phrase: str, *, name: str = "") -> bool:
    """True when the passphrase is long enough and is not the caller's own name."""
    letters = plain(phrase)
    return len(letters) >= MIN_LETTERS and letters not in {plain(name), *(plain(part) for part in name.split())}


def _hash(phrase: str, salt: bytes) -> bytes:
    return hashlib.scrypt(sound_key(phrase).encode(), salt=salt, **_SCRYPT)


def private_words(phrases: list[str]) -> list[str]:
    """The words of the passphrases said on a call, for the scrub of a public text. Small words are left out."""
    return sorted({word for phrase in phrases for word in re.findall(r"[a-z]+", phrase.lower()) if word not in _SMALL_WORDS})


def masked(text: str, phrases: list[str]) -> str:
    """`text` with each passphrase changed to "[passphrase]". For the call summary, which becomes a note."""
    for phrase in sorted({" ".join(phrase.split()) for phrase in phrases if plain(phrase)}, key=len, reverse=True):
        text = re.sub(r"\s+".join(re.escape(word) for word in phrase.split()), MASK, text, flags=re.IGNORECASE)
    return text


@dataclass(frozen=True)
class Phrase:
    """The passphrase row of one person on one number. It does not hold the passphrase."""

    row_id: int
    name: str
    failed: int  # wrong tries in a row
    calls: frozenset[str] = frozenset()  # the ids of the calls that this passphrase opens

    @property
    def locked(self) -> bool:
        return self.failed >= LOCK_AT


class PhraseStore:
    """The passphrases, in the database of the leads. It takes the open connection of the lead store."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS caller_phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller_number TEXT NOT NULL,
                name TEXT NOT NULL,
                salt BLOB NOT NULL,
                hash BLOB NOT NULL,
                created_at TEXT NOT NULL,
                failed INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute("CREATE TABLE IF NOT EXISTS phrase_calls (phrase_id INTEGER NOT NULL, call_control_id TEXT NOT NULL, PRIMARY KEY (phrase_id, call_control_id))")
        self._conn.commit()

    def _rows(self, caller_number: str, name: str) -> list[tuple[int, str, bytes, bytes, int]]:
        rows = self._conn.execute("SELECT id, name, salt, hash, failed FROM caller_phrases WHERE caller_number = ? ORDER BY id DESC", (caller_number,)).fetchall()
        return [row for row in rows if same_person(name, row[1])]

    def _calls(self, row_id: int) -> frozenset[str]:
        return frozenset(row[0] for row in self._conn.execute("SELECT call_control_id FROM phrase_calls WHERE phrase_id = ?", (row_id,)))

    def find(self, caller_number: str, name: str) -> Phrase | None:
        """The newest passphrase row of the person called `name` on this number."""
        rows = self._rows(caller_number, name) if name.strip() else []
        return Phrase(row_id=rows[0][0], name=rows[0][1], failed=rows[0][4], calls=self._calls(rows[0][0])) if rows else None

    def check(self, caller_number: str, name: str, phrase: str) -> Phrase | None:
        """The row when `phrase` is this person's passphrase. A wrong try is counted; a right one sets the count to zero."""
        rows = self._rows(caller_number, name) if name.strip() and plain(phrase) else []
        if not rows:
            return None
        row_id, row_name, salt, stored, failed = rows[0]
        if failed >= LOCK_AT:
            return None
        right = hmac.compare_digest(_hash(phrase, salt), stored)
        self._conn.execute("UPDATE caller_phrases SET failed = ? WHERE id = ?", (0 if right else failed + 1, row_id))
        self._conn.commit()
        return Phrase(row_id=row_id, name=row_name, failed=0, calls=self._calls(row_id)) if right else None

    def add_call(self, row_id: int, call_control_id: str) -> None:
        """The caller said this passphrase on this call, so the passphrase opens the call from now on."""
        self._conn.execute("INSERT OR IGNORE INTO phrase_calls (phrase_id, call_control_id) VALUES (?, ?)", (row_id, call_control_id))
        self._conn.commit()

    def set(self, caller_number: str, name: str, phrase: str, *, calls: Iterable[str]) -> None:
        """Keep `phrase` for this person. It takes the place of an earlier one, and it opens `calls` only."""
        salt = secrets.token_bytes(16)
        self.forget(caller_number, name)
        row_id = self._conn.execute(
            "INSERT INTO caller_phrases (caller_number, name, salt, hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (caller_number, name.strip(), salt, _hash(phrase, salt), datetime.now(UTC).isoformat()),
        ).lastrowid
        self._conn.executemany("INSERT OR IGNORE INTO phrase_calls (phrase_id, call_control_id) VALUES (?, ?)", [(row_id, call) for call in set(calls)])
        self._conn.commit()

    def forget(self, caller_number: str, name: str) -> None:
        for row in self._rows(caller_number, name):
            self._conn.execute("DELETE FROM caller_phrases WHERE id = ?", (row[0],))
            self._conn.execute("DELETE FROM phrase_calls WHERE phrase_id = ?", (row[0],))
        self._conn.commit()
