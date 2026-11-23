"""The note that tells an agent what to say when a caller's turn may hold no words (voice/cutoff.py)."""

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, LLMFullResponseStartFrame, TTSTextFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from ai_call_center.voice.cutoff import NOT_CAUGHT, CallerSoundWatch, cut_off_note, quiet_sound_note

DOWN = FrameDirection.DOWNSTREAM
QUESTION = "Want to pick up there, or start something new?"


def silent_after(text: str) -> CallerSoundWatch:
    """A watch where the agent said `text` and then stopped speaking."""
    watch = CallerSoundWatch(on_sound=None)
    watch.observe(LLMFullResponseStartFrame(), DOWN, now=0.0)
    watch.observe(BotStartedSpeakingFrame(), DOWN, now=0.1)
    watch.observe(TTSTextFrame(text, text_type="sentence"), DOWN, now=0.2)
    watch.observe(BotStoppedSpeakingFrame(), DOWN, now=5.0)
    return watch


def test_a_sound_in_silence_asks_the_caller_to_say_it_again():
    note, cut = silent_after(f"Welcome back, Mirela. {QUESTION}").observe(UserStartedSpeakingFrame(), DOWN, now=9.0)
    assert not cut
    assert f'Say only this: "{NOT_CAUGHT}"' in note
    assert QUESTION not in note  # the question it asked before is not asked again after an "um"


def test_each_sound_in_silence_gets_the_same_note():
    watch = silent_after(QUESTION)
    first = watch.observe(UserStartedSpeakingFrame(), DOWN, now=9.0)
    second = watch.observe(UserStartedSpeakingFrame(), DOWN, now=12.0)
    assert first == second == (quiet_sound_note(), False)


def test_a_short_no_is_an_answer_not_a_noise():
    for note in (quiet_sound_note(), cut_off_note("Is there anything else?")):
        case_1 = note.split("Case 2")[0]
        assert '"yes" or "no"' in case_1


def test_an_mm_hm_that_cut_the_agent_off_lets_it_finish_the_sentence():
    watch = CallerSoundWatch(on_sound=None)
    watch.observe(LLMFullResponseStartFrame(), DOWN, now=0.0)
    watch.observe(BotStartedSpeakingFrame(), DOWN, now=0.0)
    watch.observe(TTSTextFrame("Thanks, that's filed. When this call ends, it goes to the issue list.", text_type="sentence"), DOWN, now=0.1)
    note, cut = watch.observe(UserStartedSpeakingFrame(), DOWN, now=2.0)
    assert cut
    case_1, case_2 = note.split("Case 2")
    assert '"mm-hm"' in case_2 and '"okay"' in case_2 and '"mm-hm"' not in case_1  # it says the caller listens: not an answer
    assert 'Say this again, in these words, and then go on as you planned: "When this call ends, it goes to the issue list."' in case_2


def test_a_sound_in_silence_has_no_rule_for_a_listening_word():
    assert '"mm-hm"' not in quiet_sound_note()  # with the agent silent, a short "yes" answers it


def test_a_cut_sentence_is_still_said_again():
    watch = CallerSoundWatch(on_sound=None)
    watch.observe(LLMFullResponseStartFrame(), DOWN, now=0.0)
    watch.observe(BotStartedSpeakingFrame(), DOWN, now=0.0)
    watch.observe(TTSTextFrame("Thanks, that's filed. Is there anything else?", text_type="sentence"), DOWN, now=0.1)
    note, cut = watch.observe(UserStartedSpeakingFrame(), DOWN, now=0.5)
    assert cut
    assert "Thanks, that's filed." in note and NOT_CAUGHT not in note
