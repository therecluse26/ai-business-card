"""How the caller sounds: one voice reading for each caller turn.

A voice reading is what Pipecat Voice Reader, a small model on this server, hears in the audio of a
caller's turn: the tone of voice, how activated the voice is, laughter, the speech rate. It reads the
sound only, not the words (see `voice/voice_reader.py`).

A reading is a measurement of one turn, not a fact about a person. So the texts here give counts, and
the lead email says that a model measured them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median

TONES = ("positive", "neutral", "sad", "frustrated")  # the tones the model tells apart
FRUSTRATED_AT = 0.6  # the tone "frustrated" at this confidence or more: the agent gets a note (voice/voice_reader.py)
LAUGHTER_AT = 0.5  # the chance of laughter in a turn, from which the turn counts as one with laughter


@dataclass(frozen=True)
class VoiceReading:
    """What the voice model heard in one caller turn."""

    turn: int  # the caller turn, counted from 1
    tone: str  # one of TONES
    confidence: float  # how sure the model is of the tone, 0 to 1
    arousal: float  # how activated the voice is, 0 (flat) to 1 (highly animated)
    laughter: float  # the chance the turn has laughter in it, 0 to 1
    speech_rate: float  # syllables per second; relaxed conversation is about 4
    sarcasm: float  # the chance the delivery alone sounds sarcastic, 0 to 1; weak without the words
    upset: float  # how clearly the caller is upset, 0 to 1
    seconds: float  # how much speech the model read; it reads at most the last 8 seconds of a turn
    millis: int  # how long the read took

    @property
    def frustrated(self) -> bool:
        return self.tone == "frustrated" and self.confidence >= FRUSTRATED_AT

    @property
    def laughed(self) -> bool:
        return self.laughter >= LAUGHTER_AT


def keep_reading(readings: list[VoiceReading], reading: VoiceReading) -> None:
    """One reading for each turn. A turn is read again each time the caller pauses, and the newest read stands."""
    if readings and readings[-1].turn == reading.turn:
        readings[-1] = reading
    else:
        readings.append(reading)


def _turns(count: int) -> str:
    return f"{count} turn" if count == 1 else f"{count} turns"


def _tones(readings: list[VoiceReading]) -> str:
    """ "neutral in 5, positive in 1", most frequent first."""
    counts = Counter(reading.tone for reading in readings)
    return ", ".join(f"{tone} in {count}" for tone, count in sorted(counts.items(), key=lambda pair: (-pair[1], TONES.index(pair[0]) if pair[0] in TONES else 9)))


def voice_summary(readings: list[VoiceReading]) -> str:
    """One line for the owner's lead email. "" when the model read no turn."""
    if not readings:
        return ""
    laughs = sum(reading.laughed for reading in readings)
    parts = [f"Tone of voice in {_turns(len(readings))}: {_tones(readings)}."]
    if laughs:
        parts.append(f"Laughter in {_turns(laughs)}.")
    parts.append("A model on the server measured this from the sound alone. It is a guide, not a fact.")
    return " ".join(parts)


def voice_sentence(readings: list[VoiceReading]) -> str:
    """What the tour agent can say about the caller's voice on this call. "" when the model read no turn."""
    if not readings:
        return ""
    laughs = sum(reading.laughed for reading in readings)
    rate = median(reading.speech_rate for reading in readings)
    millis = round(median(reading.millis for reading in readings))
    return (
        f"A small model on this server, Pipecat Voice Reader, heard the sound of the caller's voice in {_turns(len(readings))}: {_tones(readings)}. "
        f"Speech rate: about {rate:.1f} syllables per second. "
        f"Laughter: {f'in {_turns(laughs)}' if laughs else 'none'}. "
        f"One reading took about {millis} milliseconds. It reads the sound only, not the words."
    )
