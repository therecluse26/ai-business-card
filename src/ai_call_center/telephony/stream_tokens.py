"""One-time tokens for the media stream.

The media WebSocket is a public address. A webhook carries Telnyx's signature, and the guardrails run
there. A media stream carries no signature. Without a check, anyone who sends Telnyx's start message
gets an AI session: a paid one, outside every guardrail, with no phone call.

So the server makes the link itself. When it tells Telnyx to start a stream for a call, it puts a
random token in the stream's URL: `/media/<token>`. The media endpoint takes a stream only when its
token is known, was not used before, is not too old, and belongs to the call that the stream names.

The token is in the path, not in a query string, so it reaches the server unchanged. It shows up in
the access log. That does no harm: a token works one time.

Tokens are kept in memory. After a restart of the server, a stream that Telnyx was about to open is
refused; the caller hears nothing and calls again.
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from collections.abc import Callable


class StreamTokens:
    def __init__(self, base_url: str, *, ttl_secs: float = 60.0, keep: int = 1000, clock: Callable[[], float] = time.monotonic) -> None:
        self.base_url = base_url.rstrip("/")
        self._ttl = ttl_secs
        self._keep = keep
        self._clock = clock
        self._open: OrderedDict[str, tuple[str, float]] = OrderedDict()  # token -> (call, time it ends); oldest first

    def url_for(self, call_control_id: str) -> str:
        """The stream URL to give Telnyx for this call. Each use makes a new token."""
        now = self._clock()
        while self._open and (len(self._open) >= self._keep or next(iter(self._open.values()))[1] <= now):
            self._open.popitem(last=False)  # too old, or too many: the oldest goes
        token = secrets.token_urlsafe(24)
        self._open[token] = (call_control_id, now + self._ttl)
        return f"{self.base_url}/{token}"

    def redeem(self, token: str) -> str | None:
        """The call this token was made for, or None. A token works one time."""
        entry = self._open.pop(token, None)
        if entry is None:
            return None
        call_control_id, ends = entry
        return call_control_id if self._clock() < ends else None
