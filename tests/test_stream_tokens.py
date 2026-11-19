"""The media endpoint takes a stream only with a one-time token that this server gave Telnyx for a call."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ai_call_center.config import Settings
from ai_call_center.telephony import media
from ai_call_center.telephony.stream_tokens import StreamTokens
from ai_call_center.telephony.transfers import Fallback, Snapshot, TransferManager
from ai_call_center.telephony.webhooks import handle_event

BASE = "wss://example.test/media"


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def token_of(url: str) -> str:
    assert url.startswith(BASE + "/")
    return url.removeprefix(BASE + "/")


def test_a_token_works_one_time_for_its_call():
    tokens = StreamTokens(BASE)
    token = token_of(tokens.url_for("v3:call-1"))
    assert len(token) >= 32
    assert tokens.redeem(token) == "v3:call-1"
    assert tokens.redeem(token) is None


def test_each_stream_gets_its_own_token():
    tokens = StreamTokens(BASE)
    assert tokens.url_for("v3:call-1") != tokens.url_for("v3:call-1")


def test_an_unknown_token_is_refused():
    tokens = StreamTokens(BASE)
    tokens.url_for("v3:call-1")
    assert tokens.redeem("a-guess") is None
    assert tokens.redeem("") is None


def test_a_token_that_is_too_old_is_refused():
    clock = Clock()
    tokens = StreamTokens(BASE, ttl_secs=60, clock=clock)
    token = token_of(tokens.url_for("v3:call-1"))
    clock.now += 61
    assert tokens.redeem(token) is None


def test_old_tokens_do_not_pile_up():
    clock = Clock()
    tokens = StreamTokens(BASE, ttl_secs=60, keep=3, clock=clock)
    first = token_of(tokens.url_for("v3:call-1"))
    for n in range(2, 5):
        tokens.url_for(f"v3:call-{n}")
    assert len(tokens._open) == 3
    assert tokens.redeem(first) is None  # the oldest went
    clock.now += 61
    tokens.url_for("v3:call-9")
    assert len(tokens._open) == 1  # the ones that were too old went


# --- the media endpoint ---


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Stand in for Telnyx's start message and for the call session, which would open AI sessions."""
    started: list[dict] = []

    async def fake_parse(websocket):
        return "telnyx", await websocket.receive_json()

    class FakeSession:
        def __init__(self, *, call_data, **_):
            self.call_data = call_data

        async def run(self) -> None:
            started.append(self.call_data)

    monkeypatch.setattr(media, "parse_telephony_websocket", fake_parse)
    monkeypatch.setattr(media, "CallSession", FakeSession)
    monkeypatch.setattr(media, "get_settings", lambda: Settings(_env_file=None))
    return started


def media_app(tokens: StreamTokens | None) -> TestClient:
    app = FastAPI()
    app.include_router(media.router)
    app.state.telephony = object()
    if tokens is not None:
        app.state.stream_tokens = tokens
    return TestClient(app)


def refused(client: TestClient, path: str) -> bool:
    try:
        with client.websocket_connect(path) as ws:
            ws.send_json({"call_id": "v3:call-1", "stream_id": "s1"})
            ws.receive_json()
    except WebSocketDisconnect:
        return True
    return False


def test_a_stream_with_its_token_starts_a_call_session(sessions):
    tokens = StreamTokens(BASE)
    token = token_of(tokens.url_for("v3:call-1"))
    with media_app(tokens).websocket_connect(f"/media/{token}") as ws:
        ws.send_json({"call_id": "v3:call-1", "stream_id": "s1"})
    assert sessions == [{"call_id": "v3:call-1", "stream_id": "s1"}]


def test_a_stream_without_a_token_starts_nothing(sessions):
    client = media_app(StreamTokens(BASE))
    assert refused(client, "/media")
    assert refused(client, "/media/a-guess")
    assert sessions == []


def test_a_token_cannot_be_used_for_a_second_stream(sessions):
    tokens = StreamTokens(BASE)
    token = token_of(tokens.url_for("v3:call-1"))
    client = media_app(tokens)
    with client.websocket_connect(f"/media/{token}") as ws:
        ws.send_json({"call_id": "v3:call-1", "stream_id": "s1"})
    assert refused(client, f"/media/{token}")
    assert len(sessions) == 1


def test_a_token_for_one_call_does_not_open_a_stream_for_another_call(sessions):
    tokens = StreamTokens(BASE)
    token = token_of(tokens.url_for("v3:call-1"))
    with pytest.raises(WebSocketDisconnect), media_app(tokens).websocket_connect(f"/media/{token}") as ws:
        ws.send_json({"call_id": "v3:another-call", "stream_id": "s1"})
        ws.receive_json()
    assert sessions == []


def test_a_server_with_no_tokens_refuses_every_stream(sessions):
    assert refused(media_app(None), "/media/anything")
    assert sessions == []


# --- the two places that ask Telnyx for a stream ---


class FakeTelephony:
    def __init__(self) -> None:
        self.urls: list[tuple[str, str]] = []

    async def answer_with_stream(self, call_control_id: str, *, stream_url: str) -> None:
        self.urls.append((call_control_id, stream_url))

    async def start_streaming(self, call_control_id: str, *, stream_url: str) -> None:
        self.urls.append((call_control_id, stream_url))

    async def stop_streaming(self, call_control_id: str) -> None:
        pass


async def test_an_incoming_call_is_answered_with_a_token_for_that_call():
    telephony, tokens = FakeTelephony(), StreamTokens(BASE)
    event = {"event_type": "call.initiated", "payload": {"call_control_id": "v3:call-1", "direction": "incoming", "from": "+15550001", "to": "+15550002"}}

    await handle_event(event, Settings(_env_file=None, public_url="https://example.test"), telephony, tokens=tokens)  # type: ignore[arg-type]

    [(call, url)] = telephony.urls
    assert call == "v3:call-1"
    assert tokens.redeem(token_of(url)) == "v3:call-1"


async def test_a_stream_that_restarts_gets_a_new_token():
    telephony, tokens = FakeTelephony(), StreamTokens(BASE)
    transfers = TransferManager(telephony, stream_url=BASE, tokens=tokens)  # type: ignore[arg-type]

    await transfers._restart("v3:call-1", Snapshot(entry_agent="contact", reason="a test"), Fallback())

    [(call, url)] = telephony.urls
    assert call == "v3:call-1"
    assert tokens.redeem(token_of(url)) == "v3:call-1"
