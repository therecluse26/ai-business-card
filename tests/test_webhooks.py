"""Telnyx webhook handling: an incoming call is answered with media streaming to this server."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_call_center.config import Settings
from ai_call_center.telephony.webhooks import handle_event


@dataclass
class FakeTelephony:
    answered: list[tuple[str, str]] = field(default_factory=list)

    async def answer_with_stream(self, call_control_id: str, *, stream_url: str) -> None:
        self.answered.append((call_control_id, stream_url))


def settings() -> Settings:
    return Settings(_env_file=None, public_url="https://example.ts.net", telnyx_app_id="app-1")


async def test_incoming_call_is_answered_with_the_media_stream_url():
    telephony = FakeTelephony()
    event = {"event_type": "call.initiated", "payload": {"call_control_id": "v2:abc", "direction": "incoming", "from": "+15550001", "to": "+15550002", "state": "parked"}}

    await handle_event(event, settings(), telephony)

    assert telephony.answered == [("v2:abc", "wss://example.ts.net/media")]


async def test_outgoing_call_initiated_is_ignored():
    telephony = FakeTelephony()
    event = {"event_type": "call.initiated", "payload": {"call_control_id": "v2:abc", "direction": "outgoing"}}

    await handle_event(event, settings(), telephony)

    assert telephony.answered == []


async def test_other_events_do_not_touch_telephony():
    telephony = FakeTelephony()
    for event_type in ("call.answered", "streaming.started", "streaming.failed", "call.hangup"):
        await handle_event({"event_type": event_type, "payload": {"call_control_id": "v2:abc"}}, settings(), telephony)

    assert telephony.answered == []


def test_urls_derive_from_public_url():
    s = settings()
    assert s.webhook_url == "https://example.ts.net/webhooks/telnyx"
    assert s.media_stream_url == "wss://example.ts.net/media"
