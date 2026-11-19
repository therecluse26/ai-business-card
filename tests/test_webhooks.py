"""Telnyx webhook handling: an incoming call is answered with media streaming to this server."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ai_call_center.config import Settings
from ai_call_center.telephony.webhooks import HandledCalls, handle_event


@dataclass
class FakeTelephony:
    answered: list[tuple[str, str]] = field(default_factory=list)
    fail_next_answer: bool = False

    async def answer_with_stream(self, call_control_id: str, *, stream_url: str) -> None:
        if self.fail_next_answer:
            self.fail_next_answer = False
            raise RuntimeError("Telnyx did not take the command")
        self.answered.append((call_control_id, stream_url))


def incoming(call_control_id: str = "v2:abc") -> dict:
    return {"event_type": "call.initiated", "payload": {"call_control_id": call_control_id, "direction": "incoming", "from": "+15550001", "to": "+15550002", "state": "parked"}}


def settings() -> Settings:
    return Settings(_env_file=None, public_url="https://example.ts.net", telnyx_app_id="app-1")


async def test_incoming_call_is_answered_with_the_media_stream_url():
    telephony = FakeTelephony()
    event = {"event_type": "call.initiated", "payload": {"call_control_id": "v2:abc", "direction": "incoming", "from": "+15550001", "to": "+15550002", "state": "parked"}}

    await handle_event(event, settings(), telephony)

    assert telephony.answered == [("v2:abc", "wss://example.ts.net/media")]


async def test_repeated_call_initiated_answers_the_call_one_time():
    telephony, handled = FakeTelephony(), HandledCalls()

    await handle_event(incoming(), settings(), telephony, handled=handled)
    await handle_event(incoming(), settings(), telephony, handled=handled)

    assert telephony.answered == [("v2:abc", "wss://example.ts.net/media")]


async def test_a_different_call_is_answered():
    telephony, handled = FakeTelephony(), HandledCalls()

    await handle_event(incoming("v2:abc"), settings(), telephony, handled=handled)
    await handle_event(incoming("v2:def"), settings(), telephony, handled=handled)

    assert [ccid for ccid, _ in telephony.answered] == ["v2:abc", "v2:def"]


async def test_a_repeat_may_answer_when_the_first_answer_failed():
    telephony, handled = FakeTelephony(fail_next_answer=True), HandledCalls()

    with pytest.raises(RuntimeError):
        await handle_event(incoming(), settings(), telephony, handled=handled)
    await handle_event(incoming(), settings(), telephony, handled=handled)

    assert telephony.answered == [("v2:abc", "wss://example.ts.net/media")]


def test_handled_calls_keeps_only_the_newest():
    handled = HandledCalls(keep=2)
    for ccid in ("a", "b", "c"):
        assert handled.first_time(ccid)

    assert handled.first_time("a")  # the oldest call was dropped
    assert not handled.first_time("c")


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
