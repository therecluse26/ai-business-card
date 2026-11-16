"""The voice leveler: every agent voice at one loudness."""

import math

import numpy as np
import pytest

from ai_call_center.agents.specs import LEG_VOICES, VOICE_LEVELS_DBFS
from ai_call_center.voice.leveler import MAX_GAIN, MIN_GAIN, VoiceLeveler, dbfs_to_amplitude, scale_pcm16

RATE = 24000
TARGET = -22.0


def speech(dbfs: float, secs: float = 0.04) -> bytes:
    """A 300 Hz tone at the given RMS level, as 16-bit PCM."""
    t = np.arange(int(RATE * secs)) / RATE
    wave = np.sin(2 * math.pi * 300 * t) * dbfs_to_amplitude(dbfs) * math.sqrt(2)
    return (wave * 32768).clip(-32768, 32767).astype(np.int16).tobytes()


def level(audio: bytes) -> float:
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64) / 32768
    return 20 * math.log10(math.sqrt(float(np.mean(samples * samples))))


def leveler(voice: str, seeds: dict[str, float] | None = None) -> VoiceLeveler:
    return VoiceLeveler(lambda: voice, target_dbfs=TARGET, seeds=seeds or {})


def play(lv: VoiceLeveler, voice: str, dbfs: float, secs: float) -> bytes:
    """Run audio through the leveler in 40 ms chunks; return the last chunk as the caller hears it."""
    out = b""
    for _ in range(int(secs / 0.04)):
        chunk = speech(dbfs)
        out = scale_pcm16(chunk, lv.gain(voice, chunk, RATE))
    return out


def test_a_measured_voice_is_at_the_target_from_its_first_chunk():
    lv = leveler("Charon", {"Charon": -15.0})
    first = speech(-15.0)
    assert level(scale_pcm16(first, lv.gain("Charon", first, RATE))) == pytest.approx(TARGET, abs=0.3)


def test_a_loud_voice_and_a_quiet_voice_end_at_the_same_level():
    lv = leveler("x")
    loud, quiet = play(lv, "Charon", -15.0, 3.0), play(lv, "coral", -26.0, 3.0)
    assert level(loud) == pytest.approx(TARGET, abs=0.5) and level(quiet) == pytest.approx(TARGET, abs=0.5)


def test_an_unmeasured_voice_plays_unchanged_until_enough_speech_is_measured():
    lv = leveler("new")
    chunk = speech(-15.0)
    assert lv.gain("new", chunk, RATE) == 1.0  # 40 ms is not a measurement
    assert level(play(lv, "new", -15.0, 2.0)) == pytest.approx(TARGET, abs=0.5)


def test_pauses_do_not_change_the_measured_level():
    lv = leveler("Charon", {"Charon": -15.0})
    before = lv.gain("Charon", speech(-15.0), RATE)
    for _ in range(100):  # four seconds of near silence
        lv.gain("Charon", speech(-60.0), RATE)
    assert lv.gain("Charon", speech(-15.0), RATE) == pytest.approx(before, rel=0.02)


def test_gain_has_limits_and_audio_never_wraps():
    lv = leveler("x")
    play(lv, "whisper", -44.0, 3.0)
    play(lv, "shout", -3.0, 3.0)
    assert lv.gain("whisper", speech(-44.0), RATE) == pytest.approx(MAX_GAIN)
    assert lv.gain("shout", speech(-3.0), RATE) == pytest.approx(MIN_GAIN)
    loud = np.frombuffer(scale_pcm16(speech(-3.0), 4.0), dtype=np.int16)
    assert loud.max() == 32767 and loud.min() == -32768  # clipped, not wrapped around


def test_the_default_voice_of_every_agent_has_a_measured_level():
    from ai_call_center.agents.specs import DEFAULT_LEGS

    for agent, leg in DEFAULT_LEGS.items():
        assert LEG_VOICES[leg][agent] in VOICE_LEVELS_DBFS, f"measure the level of {LEG_VOICES[leg][agent]} and add it to VOICE_LEVELS_DBFS"
