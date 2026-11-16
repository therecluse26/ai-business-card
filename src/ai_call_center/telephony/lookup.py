"""Caller lookup: carrier, line type, area, and registered name for the calling number.

The lookup starts when Telnyx reports the incoming call, before we answer, so the result is
usually ready when the media stream opens a second later. Results are cached in SQLite, so a
number is looked up once.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from ai_call_center.callers import UNKNOWN_NUMBERS
from ai_call_center.leads import LeadStore, NumberInfo

log = logging.getLogger(__name__)


class NumberLookup(Protocol):
    async def lookup_number(self, number: str, *, with_name: bool) -> NumberInfo: ...


class CallerLookup:
    def __init__(self, client: NumberLookup, store: LeadStore, *, mode: str = "carrier+name") -> None:
        self._client = client
        self._store = store
        self._mode = mode
        self._tasks: dict[str, asyncio.Task[NumberInfo | None]] = {}

    def start(self, number: str) -> None:
        """Begin a lookup in the background. Safe to call for any number, any number of times."""
        if self._mode == "off" or number in UNKNOWN_NUMBERS or number in self._tasks:
            return
        if self._store.number_info_for(number) is not None:
            return
        self._tasks[number] = asyncio.create_task(self._run(number))

    async def _run(self, number: str) -> NumberInfo | None:
        try:
            info = await self._client.lookup_number(number, with_name=self._mode == "carrier+name")
            self._store.save_number_info(number, info)
            return info
        except Exception as exc:  # noqa: BLE001 - a lookup must never break a call
            log.warning("number lookup failed: %s", exc)
            return None
        finally:
            self._tasks.pop(number, None)

    async def get(self, number: str, *, timeout: float = 0.4) -> NumberInfo | None:
        """The cached result, or the in-flight one if it lands within the timeout. Never raises."""
        if self._mode == "off" or number in UNKNOWN_NUMBERS:
            return None
        cached = self._store.number_info_for(number)
        if cached is not None:
            return cached
        task = self._tasks.get(number)
        if task is None:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout)
        except TimeoutError:
            log.info("number lookup not ready after %.0f ms; the call starts without it", timeout * 1000)
            return None
