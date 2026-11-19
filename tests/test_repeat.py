"""One answer that says the same words two times: the second copy is dropped."""

import asyncio
from types import SimpleNamespace

from ai_call_center.voice.legs import HELD_AUDIO_MAX_CHARS, OnceOpenAIRealtime
from ai_call_center.voice.repeat import DROP, HOLD, PASS, RepeatGuard

# The greeting of the real call of 2026-11-17, which the caller heard two times.
GREETING = "Hi, you've reached Digital Reformer, Brad Magyar's AI business card. I'm an AI assistant that Brad built. Am I speaking with Steve?"


def test_the_first_item_of_an_answer_always_plays():
    guard = RepeatGuard()
    assert guard.audio("r1", "a") == PASS
    assert guard.text("r1", "a", GREETING) == PASS


def test_a_second_item_with_the_same_words_is_dropped():
    guard = RepeatGuard()
    guard.text("r1", "a", GREETING)
    assert guard.audio("r1", "b") == HOLD  # no text yet: not known
    assert guard.text("r1", "b", "Hi, you've") == HOLD  # too few letters to be sure
    assert guard.text("r1", "b", " reached Digital Reformer,") == DROP
    assert guard.audio("r1", "b") == DROP


def test_a_second_item_with_new_words_plays():
    guard = RepeatGuard()
    guard.text("r1", "a", "Got it.")
    assert guard.audio("r1", "b") == HOLD
    assert guard.text("r1", "b", "What part of") == PASS
    assert guard.audio("r1", "b") == PASS


def test_a_short_copy_is_dropped_when_the_item_ends():
    guard = RepeatGuard()
    guard.text("r1", "a", "Go ahead.")
    assert guard.text("r1", "b", "Go ahead.") == HOLD
    assert guard.settle("b") == DROP


def test_a_short_item_that_only_starts_the_same_plays_when_it_ends():
    guard = RepeatGuard()
    guard.text("r1", "a", "Go ahead with your question.")
    assert guard.text("r1", "b", "Go") == HOLD
    assert guard.settle("b") == PASS


def test_an_item_with_no_text_plays_and_a_new_answer_starts_fresh():
    guard = RepeatGuard()
    guard.text("r1", "a", GREETING)
    assert guard.audio("r1", "b") == HOLD
    assert guard.give_up("b") == PASS
    assert guard.audio("r2", "c") == PASS  # the same words in a later answer are not this guard's business
    assert guard.text("r2", "c", GREETING) == PASS


def _service():
    service = OnceOpenAIRealtime(api_key="test")
    played = []

    async def play(kind, evt):
        played.append((kind, evt.item_id, evt.delta))

    service._play = play  # noqa: SLF001
    return service, played


def _evt(item_id, delta, response_id="r1"):
    return SimpleNamespace(response_id=response_id, item_id=item_id, delta=delta)


def _done(status="completed"):
    return SimpleNamespace(response=SimpleNamespace(status=status, usage=None))


def test_the_service_plays_the_greeting_one_time():
    service, played = _service()

    async def go():
        await service._handle_evt_audio_transcript_delta(_evt("a", GREETING))  # noqa: SLF001
        await service._handle_evt_audio_delta(_evt("a", "AAAA"))  # noqa: SLF001
        await service._handle_evt_audio_delta(_evt("b", "BBBB"))  # noqa: SLF001
        await service._handle_evt_audio_transcript_delta(_evt("b", GREETING))  # noqa: SLF001
        await service._handle_evt_audio_delta(_evt("b", "BBBB"))  # noqa: SLF001

    asyncio.run(go())
    assert played == [("text", "a", GREETING), ("audio", "a", "AAAA")]
    assert service._held == []  # noqa: SLF001


def test_the_service_plays_a_held_item_with_new_words_in_order():
    service, played = _service()

    async def go():
        await service._handle_evt_audio_transcript_delta(_evt("a", "Got it."))  # noqa: SLF001
        await service._handle_evt_audio_delta(_evt("b", "B1"))  # noqa: SLF001
        assert played == [("text", "a", "Got it.")]  # the second item waits for its text
        await service._handle_evt_audio_transcript_delta(_evt("b", "What part of the work"))  # noqa: SLF001
        await service._handle_evt_audio_delta(_evt("b", "B2"))  # noqa: SLF001

    asyncio.run(go())
    assert played == [("text", "a", "Got it."), ("audio", "b", "B1"), ("text", "b", "What part of the work"), ("audio", "b", "B2")]


def test_the_service_never_holds_audio_for_long():
    service, played = _service()

    async def go():
        await service._handle_evt_audio_transcript_delta(_evt("a", GREETING))  # noqa: SLF001
        await service._handle_evt_audio_delta(_evt("b", "B" * HELD_AUDIO_MAX_CHARS))  # noqa: SLF001
        await service._handle_evt_audio_delta(_evt("b", "BB"))  # noqa: SLF001

    asyncio.run(go())
    assert [kind for kind, item, _ in played if item == "b"] == ["audio", "audio"]


def test_held_audio_does_not_play_after_the_caller_speaks_or_the_answer_is_cancelled(monkeypatch):
    service, played = _service()

    async def nothing(*_args, **_kwargs):
        return None

    monkeypatch.setattr("pipecat.services.openai.realtime.llm.OpenAIRealtimeLLMService._handle_evt_response_done", nothing)

    async def go():
        await service._handle_evt_audio_transcript_delta(_evt("a", GREETING))  # noqa: SLF001
        await service._handle_evt_audio_delta(_evt("b", "B1"))  # noqa: SLF001
        await service._handle_evt_response_done(_done("cancelled"))  # noqa: SLF001
        await service._handle_evt_audio_transcript_delta(_evt("c", "Go ahead.", "r2"))  # noqa: SLF001
        await service._handle_evt_audio_delta(_evt("d", "D1", "r2"))  # noqa: SLF001
        await service._handle_evt_response_done(_done())  # noqa: SLF001

    asyncio.run(go())
    assert ("audio", "b", "B1") not in played
    assert played[-1] == ("audio", "d", "D1")  # an item with no text plays when the answer ends
