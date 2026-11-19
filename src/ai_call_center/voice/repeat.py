"""One answer that says the same words two times.

An OpenAI Realtime answer can hold more than one audio item. Sometimes the second item is a copy of
the first. Measured 2026-11-17 on a real call: the greeting agent got the exact words of the
greeting, and its one answer had two audio items with those words. The caller heard the full
greeting two times, 18 seconds in all. No caller sound was involved.

The audio of an answer arrives much faster than it plays. So the server can hold a later item for
a moment, read the first letters of its text, and drop the item when it is a copy.
"""

from __future__ import annotations

import re

PASS, HOLD, DROP = "pass", "hold", "drop"
REPEAT_CHECK_LETTERS = 24  # a later item that starts with this many letters of an earlier item is a copy


def _letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


class RepeatGuard:
    """Says what to do with each audio item of an answer: let it pass, hold it, or drop it."""

    def __init__(self) -> None:
        self._response = ""
        self._text: dict[str, str] = {}
        self._verdict: dict[str, str] = {}

    def _see(self, response_id: str, item_id: str) -> None:
        if response_id != self._response:
            self._response, self._text, self._verdict = response_id, {}, {}
        if item_id not in self._verdict:
            self._verdict[item_id] = HOLD if self._verdict else PASS  # the first item of an answer always plays
            self._text[item_id] = ""

    def audio(self, response_id: str, item_id: str) -> str:
        """A piece of audio came for this item."""
        self._see(response_id, item_id)
        return self._verdict[item_id]

    def text(self, response_id: str, item_id: str, delta: str) -> str:
        """A piece of the item's text came. The text decides a held item."""
        self._see(response_id, item_id)
        self._text[item_id] += delta
        if self._verdict[item_id] == HOLD:
            self._verdict[item_id] = self._judge(item_id, ended=False)
        return self._verdict[item_id]

    def settle(self, item_id: str) -> str:
        """The item has ended. A held item gets its last verdict."""
        if self._verdict.get(item_id) == HOLD:
            self._verdict[item_id] = self._judge(item_id, ended=True)
        return self._verdict.get(item_id, PASS)

    def give_up(self, item_id: str) -> str:
        """The item was held too long without a verdict. It plays."""
        if self._verdict.get(item_id) == HOLD:
            self._verdict[item_id] = PASS
        return self._verdict.get(item_id, PASS)

    def _judge(self, item_id: str, *, ended: bool) -> str:
        mine = _letters(self._text[item_id])
        earlier = [_letters(text) for other, text in self._text.items() if other != item_id and self._verdict[other] == PASS]
        if not mine:
            return PASS if ended else HOLD
        if not any(text.startswith(mine) for text in earlier):
            return PASS  # the item says something new
        if len(mine) >= REPEAT_CHECK_LETTERS or (ended and mine in earlier):
            return DROP
        return PASS if ended else HOLD
