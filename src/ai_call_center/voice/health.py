"""Which voice providers are answering.

When a provider fails during a call, the server remembers it for a few minutes. New calls, and a
call that restarts its media stream to fail over, then build their agents on another provider
from the start. After the cooldown the provider is tried again.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

COOLDOWN_SECS = 300.0


class ProviderHealth:
    """One per server. Keys are voice leg ids, for example "gemini_live"."""

    def __init__(self, *, cooldown_secs: float = COOLDOWN_SECS, clock: Callable[[], float] = time.monotonic) -> None:
        self._cooldown = cooldown_secs
        self._clock = clock
        self._down_until: dict[str, float] = {}
        self.reasons: dict[str, str] = {}

    def mark_down(self, leg: str, reason: str = "") -> None:
        if not self.is_down(leg):
            log.error("voice provider %s is DOWN for %.0f s: %s", leg, self._cooldown, reason[:200])
        self._down_until[leg] = self._clock() + self._cooldown
        self.reasons[leg] = reason

    def is_down(self, leg: str) -> bool:
        until = self._down_until.get(leg)
        if until is None:
            return False
        if self._clock() >= until:
            del self._down_until[leg]
            log.info("voice provider %s: cooldown over; it will be tried again", leg)
            return False
        return True

    def down(self) -> frozenset[str]:
        return frozenset(leg for leg in list(self._down_until) if self.is_down(leg))
