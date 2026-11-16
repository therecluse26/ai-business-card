"""Provider health, and when the watchdog gives up on a voice service."""

import asyncio
from types import SimpleNamespace

from pipecat.frames.frames import ErrorFrame

from ai_call_center.voice import call_session
from ai_call_center.voice.call_session import ServiceWatchdog, VoiceLeg
from ai_call_center.voice.health import ProviderHealth


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_provider_is_down_until_the_cooldown_ends():
    clock = Clock()
    health = ProviderHealth(cooldown_secs=300, clock=clock)
    assert not health.is_down("gemini_live") and health.down() == frozenset()

    health.mark_down("gemini_live", "no audio")
    clock.now += 299
    assert health.is_down("gemini_live") and health.down() == frozenset({"gemini_live"})
    assert not health.is_down("openai_realtime")

    clock.now += 2
    assert not health.is_down("gemini_live") and health.down() == frozenset()


def test_a_second_failure_restarts_the_cooldown():
    clock = Clock()
    health = ProviderHealth(cooldown_secs=300, clock=clock)
    health.mark_down("openai_realtime", "first")
    clock.now += 200
    health.mark_down("openai_realtime", "second")
    clock.now += 200
    assert health.is_down("openai_realtime")
    assert health.reasons["openai_realtime"] == "second"


# --- the watchdog ---


class FakeService:
    """Stands in for an OpenAI Realtime service: it does not reconnect by itself."""

    is_usable = True


def watchdog(monkeypatch, *, recover_fails: bool = False):
    service = FakeService()
    leg = VoiceLeg("classifier", "openai_realtime", service)
    dead: list[tuple[str, str]] = []
    recovered: list[str] = []

    async def reset(_service):
        if recover_fails:
            raise RuntimeError("cannot connect")
        recovered.append("reset")

    async def replay(_service, _context):
        recovered.append("replay")

    monkeypatch.setattr(call_session, "reset_session", reset)
    monkeypatch.setattr(call_session, "replay_and_respond", replay)
    monkeypatch.setattr(call_session, "self_heals", lambda _service: False)
    switcher = SimpleNamespace(strategy=SimpleNamespace(active_service=service))
    dog = ServiceWatchdog({"classifier": leg}, switcher, context=None, on_dead=lambda vl, reason: dead.append((vl.agent, reason)))
    tasks: list[asyncio.Task] = []
    dog.create_task = lambda coro, *_a, **_k: tasks.append(asyncio.ensure_future(coro)) or tasks[-1]

    async def error(text: str) -> None:
        frame = ErrorFrame(error=text)
        frame.processor = service
        dog._check(frame)  # noqa: SLF001
        for task in tasks:
            await task
        tasks.clear()

    return SimpleNamespace(error=error, dead=dead, recovered=recovered, service=service, dog=dog)


async def test_one_error_is_recovered_by_a_reconnect_and_a_replay(monkeypatch):
    w = watchdog(monkeypatch)
    await w.error("Error: a client event was rejected")
    assert w.recovered == ["reset", "replay"] and w.dead == []


async def test_an_error_right_after_a_recovery_means_the_service_is_dead(monkeypatch):
    w = watchdog(monkeypatch)
    await w.error("Error: connection closed")
    await w.error("Error: connection closed")
    assert w.recovered == ["reset", "replay"]  # no second try
    assert [agent for agent, _ in w.dead] == ["classifier"]
    await w.error("Error: connection closed")
    assert len(w.dead) == 1  # reported once


async def test_a_refused_account_is_dead_at_once(monkeypatch):
    w = watchdog(monkeypatch)
    await w.error("Error: error=RealtimeError(type='invalid_request_error', code='invalid_api_key', message='Incorrect API key provided')")
    assert w.recovered == [] and len(w.dead) == 1


async def test_a_recovery_that_fails_means_the_service_is_dead(monkeypatch):
    w = watchdog(monkeypatch, recover_fails=True)
    await w.error("Error: connection closed")
    assert len(w.dead) == 1 and "recovery failed" in w.dead[0][1]


async def test_spaced_errors_are_recovered_twice_and_then_the_service_is_dead(monkeypatch):
    w = watchdog(monkeypatch)
    monkeypatch.setattr(call_session, "RELAPSE_SECS", 0.0)  # every error counts as a new one
    await w.error("Error: one")
    await w.error("Error: two")
    assert w.recovered == ["reset", "replay"] * 2 and w.dead == []
    await w.error("Error: three")
    assert len(w.dead) == 1 and "too many recoveries" in w.dead[0][1]


async def test_a_service_pipecat_gave_up_on_is_dead(monkeypatch):
    w = watchdog(monkeypatch)
    w.service.is_usable = False
    frame = ErrorFrame(error="LLMSwitcher: HandoffGeminiLive#1 can no longer do its job")
    frame.processor = object()  # reported by the switcher, not by the service
    w.dog._check(frame)  # noqa: SLF001
    assert len(w.dead) == 1 and "gave up" in w.dead[0][1]
