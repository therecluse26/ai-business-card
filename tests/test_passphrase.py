"""The passphrase check: a match of phone number and name opens no notes; the caller's passphrase does."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ai_call_center.callers import UNKNOWN_NUMBERS
from ai_call_center.graph.state import CallContext
from ai_call_center.graph.tools import forget, identify, save_lead_record, set_phrase, verify
from ai_call_center.passphrase import LOCK_AT, PhraseStore, masked, private_words, sound_key, usable

OLD, FIRST = "2026-11-01T15:00:00+00:00", "2026-11-10T15:00:00+00:00"


def _past_call(leads, ctx, call_id, started, summary, topic):
    """A finished call of Dana on the number of `ctx`, at a fixed time."""
    leads.start_call(call_id, ctx.caller_number)
    leads.finish_call(call_id, agents=["classifier"], summary=summary, caller_name="Dana Reyes", topic=topic)
    leads._conn.execute("UPDATE calls SET started_at = ? WHERE call_control_id = ?", (started, call_id))


def _next_call(ctx: CallContext, call_id: str) -> CallContext:
    """A new call from the same number: the same stores, and nothing else from the call before."""
    return CallContext(call_control_id=call_id, caller_number=ctx.caller_number, owner_phone=ctx.owner_phone, leads=ctx.leads, telephony=ctx.telephony, notifier=ctx.notifier, legs=ctx.legs, mode="text")


def _set(ctx: CallContext, phrase: str) -> str:
    set_phrase(ctx, phrase)
    return set_phrase(ctx, phrase)


@pytest.fixture
def dana(ctx, leads):
    """Dana called two times. On the second call (FIRST) she set the passphrase "purple lantern"."""
    _past_call(leads, ctx, "call-old", OLD, "Asked about a secret merger tool.", "a merger tool")
    _past_call(leads, ctx, "call-first", FIRST, "Asked about a booking app for a clinic.", "a booking app")
    PhraseStore(leads.connection).set(ctx.caller_number, "Dana Reyes", "purple lantern", calls=["call-first"])
    return ctx


def test_the_comparison_ignores_capitals_spaces_and_spelling_of_the_same_sound():
    assert sound_key("Purple Lantern.") == sound_key("purple lanturn") == sound_key("purplelantern")
    assert sound_key("purple lantern") != sound_key("green lantern")
    assert sound_key("route 66") != sound_key("route 67")
    assert usable("purple lantern", name="Dana Reyes") and not usable("ok") and not usable("Dana", name="Dana Reyes") and not usable("dana reyes", name="Dana Reyes")


def test_a_recognised_caller_is_asked_for_the_passphrase_and_gets_no_notes_before_it(dana):
    result = identify(dana, "Dana")
    assert '"Welcome back, Dana. I believe we have talked before. Just to verify, may I have the short passphrase that you gave me?"' in result
    assert "verify_passphrase" in result and "booking" not in result and "merger" not in result
    assert dana.identity_confirmed and not dana.identity_verified and dana.caller_history == ""


def test_the_right_passphrase_opens_the_notes_from_the_call_where_it_was_set(dana):
    identify(dana, "Dana")
    result = verify(dana, "Purple lanturn")  # as a speech model can hear it
    assert '"Thank you, that matches. Last time you asked about a booking app. Want to pick up there, or start something new?"' in result
    assert "booking app for a clinic" in result and dana.identity_verified and dana.open_calls == ["call-first"]
    assert "merger" not in result  # the call before the passphrase stays closed
    assert "verified already" in verify(dana, "purple lantern")
    # This call was made with the passphrase, so the passphrase opens it on the next call.
    assert PhraseStore(dana.leads.connection).find(dana.caller_number, "Dana").calls == {"call-first", dana.call_control_id}


def test_two_wrong_tries_close_the_check_for_the_call(dana):
    identify(dana, "Dana")
    first = verify(dana, "green lantern")
    assert '"That does not match what I have. Want to try one more time?"' in first and "booking" not in first
    second = verify(dana, "red lantern")
    assert "That does not match either." in second and "booking" not in second and dana.phrase_closed and not dana.identity_verified
    assert "no passphrase to check" in verify(dana, "purple lantern")  # the right one comes too late on this call
    assert dana.caller_history == "" and dana.spoken_phrases == ["green lantern", "red lantern"]


def test_a_caller_who_gives_no_passphrase_goes_on_with_no_notes(dana):
    identify(dana, "Dana")
    result = verify(dana, "")
    assert '"No problem. I can\'t use the notes from your earlier calls without it, but I can still help. What can I do for you today?"' in result
    assert dana.phrase_closed and not dana.identity_verified and "booking" not in result


def test_many_wrong_tries_over_many_calls_lock_the_passphrase(dana):
    call = dana
    for number in range(LOCK_AT):
        if number % 2 == 0:
            call = _next_call(dana, f"call-guess-{number}")
            identify(call, "Dana")
        verify(call, f"wrong guess {number}")
    late = _next_call(dana, "call-late")
    result = identify(late, "Dana")
    assert "Welcome Dana back" in result and "passphrase that you gave me" not in result and "booking" not in result
    assert "no passphrase to check" in verify(late, "purple lantern")  # locked for good: the right one opens nothing now


def test_verify_does_nothing_for_a_caller_who_was_not_asked(ctx, leads):
    identify(ctx, "Maria")
    assert "no passphrase to check" in verify(ctx, "purple lantern")
    assert not ctx.identity_verified


def test_a_new_passphrase_is_said_back_before_it_is_kept(ctx, leads):
    leads.start_call(ctx.call_control_id, ctx.caller_number)
    identify(ctx, "Dana Reyes")
    first = set_phrase(ctx, "purple lantern")
    assert '"I heard: purple lantern. Is that right?"' in first and PhraseStore(leads.connection).find(ctx.caller_number, "Dana") is None
    changed = set_phrase(ctx, "purple lamp")  # the caller corrects it: said back again, not kept yet
    assert "I heard: purple lamp." in changed and PhraseStore(leads.connection).find(ctx.caller_number, "Dana") is None
    done = set_phrase(ctx, "Purple Lamp")
    assert "On your next call, I will ask for that passphrase" in done and ctx.phrase_set
    row = PhraseStore(leads.connection).find(ctx.caller_number, "Dana")
    assert row is not None and row.calls == {ctx.call_control_id}
    stored = " ".join(str(value) for value in leads._conn.execute("SELECT * FROM caller_phrases").fetchone())
    assert "purple" not in stored.lower() and "lamp" not in stored.lower()  # the table holds a hash, not the passphrase


def test_a_passphrase_is_refused_when_it_is_too_short_the_name_or_from_a_hidden_number(ctx, leads):
    identify(ctx, "Dana Reyes")
    assert "Not kept" in set_phrase(ctx, "ok") and "Not kept" in set_phrase(ctx, "Dana")
    hidden = replace(ctx, caller_number=next(iter(UNKNOWN_NUMBERS - {""})), caller_name="Dana Reyes")
    assert "number is hidden" in _set(hidden, "purple lantern")
    assert PhraseStore(leads.connection).find(ctx.caller_number, "Dana Reyes") is None


def test_a_caller_who_forgot_the_passphrase_can_set_a_new_one_and_it_never_opens_the_old_notes(dana, leads):
    # A person with a faked number says "I forgot it", and sets a new passphrase on this call.
    leads.start_call(dana.call_control_id, dana.caller_number)
    identify(dana, "Dana")
    verify(dana, "")
    assert "On your next call" in _set(dana, "silver anchor")
    leads.finish_call(dana.call_control_id, agents=["classifier"], summary="Asked about a website.", caller_name="Dana Reyes", topic="a website")
    later = _next_call(dana, "call-later")
    identify(later, "Dana")
    assert "no passphrase to check" not in verify(later, "purple lantern") and not later.identity_verified  # the old one is gone
    result = verify(later, "silver anchor")
    assert later.identity_verified and "a website" in result
    assert "booking" not in result and "merger" not in result  # what came before the new passphrase stays closed


def test_a_passphrase_from_a_faked_number_never_opens_a_call_of_the_real_caller(ctx, leads):
    # The real caller has no passphrase. A person with a faked number leaves a message and sets one.
    _past_call(leads, ctx, "call-real-1", OLD, "Asked about a secret merger tool.", "a merger tool")
    leads.start_call(ctx.call_control_id, ctx.caller_number)
    identify(ctx, "Dana")
    _set(ctx, "silver anchor")
    leads.finish_call(ctx.call_control_id, agents=["classifier"], summary="Asked about prices.", caller_name="Dana Reyes", topic="prices")
    # The real caller calls again. They do not know that passphrase, so the call goes on with no notes.
    real = _next_call(ctx, "call-real-2")
    leads.start_call("call-real-2", ctx.caller_number)
    identify(real, "Dana")
    verify(real, "")
    leads.finish_call("call-real-2", agents=["classifier"], summary="Asked about a hospital contract.", caller_name="Dana Reyes", topic="a hospital contract")
    # The person with the faked number comes back and says their passphrase. They get their own call, and nothing more.
    fake = _next_call(ctx, "call-fake-2")
    identify(fake, "Dana")
    result = verify(fake, "silver anchor")
    assert fake.identity_verified and "prices" in result
    assert "merger" not in result and "hospital" not in result


def test_a_verified_caller_who_changes_the_passphrase_keeps_what_the_old_one_opened(dana, leads):
    identify(dana, "Dana")
    verify(dana, "purple lantern")
    _set(dana, "silver anchor")
    assert PhraseStore(leads.connection).find(dana.caller_number, "Dana").calls == {"call-first", dana.call_control_id}


def test_forget_me_deletes_the_passphrase(dana, leads):
    identify(dana, "Dana")
    forget(dana)
    assert leads._conn.execute("SELECT COUNT(*) FROM caller_phrases").fetchone()[0] == 0 and leads._conn.execute("SELECT COUNT(*) FROM phrase_calls").fetchone()[0] == 0
    assert "Not kept" in set_phrase(dana, "silver anchor")  # a forgotten caller sets nothing on this call


async def test_a_saved_message_brings_the_offer_one_time(ctx, leads):
    result, _ = await save_lead_record(ctx, name="Dana Reyes", need="A booking app.")
    assert "I can remember your details for next time" in result and "not a password you use anywhere else" in result and "set_passphrase" in result
    again, _ = await save_lead_record(ctx, name="Dana Reyes", need="A booking app.", email="dana@clinic.test")
    assert "remember your details" not in again


async def test_no_offer_for_a_verified_caller_a_hidden_number_or_with_the_setting_off(dana, ctx, leads):
    identify(dana, "Dana")
    verify(dana, "purple lantern")
    result, _ = await save_lead_record(dana, name="Dana Reyes", need="More work on the booking app.")
    assert "remember your details" not in result
    off = _next_call(ctx, "call-off")
    off.phrase_offer = False
    result, _ = await save_lead_record(off, name="Sam Lee", need="A question.")
    assert "remember your details" not in result
    hidden = replace(_next_call(ctx, "call-hidden"), caller_number="anonymous")
    result, _ = await save_lead_record(hidden, name="Sam Lee", need="A question.", email="sam@lee.test")
    assert "Lead saved" in result and "remember your details" not in result


def test_a_passphrase_is_masked_in_the_summary_and_its_words_are_private_in_a_public_text():
    assert masked("The caller set the passphrase Purple  lantern and left.", ["purple lantern"]) == "The caller set the passphrase [passphrase] and left."
    assert private_words(["The Purple Lantern", "of my garden"]) == ["garden", "lantern", "purple"]
