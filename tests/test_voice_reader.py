"""The voice reader hears each caller turn. No test loads the real model or uses the network:
a fake model gives the values, and a scripted detector says where the speech is."""

from __future__ import annotations

from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import BotStartedSpeakingFrame, InputAudioRawFrame

from ai_call_center.caller_voice import VoiceReading, keep_reading, voice_sentence, voice_summary
from ai_call_center.graph.tools import call_details, describe_call, forget
from ai_call_center.leads import Lead
from ai_call_center.notify import lead_email, plain_text
from ai_call_center.voice.voice_reader import FRUSTRATED_NOTE, NOTE_TURNS, CallerVoiceReader, FrustrationNote, TurnAudio, reading_from

RATE = 8000
FRAME = b"\x01\x00" * 160  # 20 ms at 8 kHz
QUIET, STARTING, SPEAKING, STOPPING = VADState.QUIET, VADState.STARTING, VADState.SPEAKING, VADState.STOPPING


def reading(turn: int = 1, tone: str = "neutral", confidence: float = 0.9, **values) -> VoiceReading:
    fields = dict(arousal=0.3, laughter=0.02, speech_rate=3.5, sarcasm=0.01, upset=0.05, seconds=2.0, millis=58)
    return VoiceReading(turn=turn, tone=tone, confidence=confidence, **{**fields, **values})


def say(audio: TurnAudio, frames: int) -> bool:
    """A stretch of speech of `frames` frames, then the pause that ends it. True when the stretch ended."""
    audio.feed(FRAME, STARTING)
    for _ in range(frames - 1):
        audio.feed(FRAME, SPEAKING)
    audio.feed(FRAME, STOPPING)
    return audio.feed(FRAME, QUIET)


# --- the speech of a turn ---


def test_a_turn_is_every_stretch_of_speech_until_the_agent_answers():
    audio = TurnAudio(RATE)
    assert not audio.feed(FRAME, QUIET) and audio.turn == 0 and audio.seconds == 0
    assert say(audio, 50) and audio.turn == 1
    first = audio.seconds
    assert 1.0 < first < 1.1  # 51 frames of speech, and one frame from before it
    assert say(audio, 50) and audio.turn == 1 and audio.seconds > first * 1.9  # a pause in a turn: the same turn, more speech

    audio.agent_spoke()
    assert say(audio, 25) and audio.turn == 2 and audio.seconds < first  # a new turn starts empty


def test_a_sound_that_the_detector_does_not_confirm_is_not_speech():
    audio = TurnAudio(RATE)
    audio.feed(FRAME, STARTING)
    assert not audio.feed(FRAME, QUIET) and audio.turn == 0 and audio.clip() == b""


def test_the_audio_before_the_confirmed_start_is_kept_and_a_turn_holds_its_last_8_seconds():
    audio = TurnAudio(RATE)
    for _ in range(100):
        audio.feed(FRAME, QUIET)
    say(audio, 10)
    assert audio.seconds == 0.3 + 11 * 0.02  # 0.3 s of lead, then the stretch
    for _ in range(3):
        say(audio, 200)
    assert audio.seconds == 8.0


# --- the model's values ---


def test_a_reading_names_the_likeliest_tone_and_keeps_values_in_range():
    values = {"tone": {"positive": 0.1, "neutral": 0.2, "sad": 0.05, "frustrated": 0.65}, "arousal": 1.4, "laughter": -0.1, "speech_rate": 4.2, "sarcasm": 0.03, "upset": 0.5, "sad": 0.1}
    made = reading_from(values, turn=3, seconds=2.5, millis=61)
    assert (made.turn, made.tone, made.confidence, made.arousal, made.laughter, made.speech_rate) == (3, "frustrated", 0.65, 1.0, 0.0, 4.2)
    assert made.frustrated and not made.laughed
    assert not reading(tone="frustrated", confidence=0.59).frustrated  # under the limit
    assert not reading(tone="sad", confidence=0.95).frustrated
    assert reading(laughter=0.5).laughed


def test_the_newest_read_of_a_turn_stands():
    readings: list[VoiceReading] = []
    keep_reading(readings, reading(1))
    keep_reading(readings, reading(1, tone="positive"))
    keep_reading(readings, reading(2))
    assert [(r.turn, r.tone) for r in readings] == [(1, "positive"), (2, "neutral")]


# --- where a reading goes ---


def test_the_texts_give_counts_and_say_that_a_model_measured_them():
    assert voice_summary([]) == "" and voice_sentence([]) == ""
    readings = [reading(1), reading(2, tone="positive", laughter=0.8), reading(3), reading(4, tone="frustrated", confidence=0.7, speech_rate=5.0)]
    summary = voice_summary(readings)
    assert summary.startswith("Tone of voice in 4 turns: neutral in 2, positive in 1, frustrated in 1. Laughter in 1 turn.")
    assert "a guide, not a fact" in summary
    sentence = voice_sentence(readings)
    assert "Pipecat Voice Reader" in sentence and "in 4 turns: neutral in 2, positive in 1, frustrated in 1" in sentence
    assert "about 3.5 syllables per second" in sentence and "Laughter: in 1 turn." in sentence and "about 58 milliseconds" in sentence
    assert "Laughter: none." in voice_sentence([reading(1)])


def test_the_call_stats_and_the_lead_email_carry_the_readings(ctx):
    facts = dict(visited=["classifier"], active_agent="tour", lead_saved=False, transfer_requested=False)
    assert "Voice Reader" not in describe_call(ctx, **facts)
    assert call_details(ctx, kind="message", notes="", visited=["classifier"]).voice == ""

    ctx.voice_readings += [reading(1), reading(2, tone="positive")]
    assert "heard the sound of the caller's voice in 2 turns: positive in 1, neutral in 1" in describe_call(ctx, **facts)
    details = call_details(ctx, kind="message", notes="", visited=["classifier"])
    lead = Lead(name="Ada", email="", company="", need="A project.", caller_number="+15555550123", call_control_id="v2:test-call", created_at="2026-11-19T20:00:00+00:00")
    text = plain_text(lead_email(lead, row_id=1, sender="card@example.test", to="owner@example.test", details=details))
    assert "Caller's voice" in text and "Tone of voice in 2 turns" in text


def test_a_forgotten_caller_has_no_readings(ctx):
    ctx.voice_readings.append(reading(1))
    forget(ctx)
    assert ctx.voice_readings == []


# --- the processor ---


class FakeModel:
    ready = True

    def __init__(self, tone: str = "neutral") -> None:
        self.tone = tone
        self.clips: list[tuple[int, int]] = []

    def read(self, pcm: bytes, sample_rate: int) -> dict:
        self.clips.append((len(pcm), sample_rate))
        return {"tone": {self.tone: 0.9}, "arousal": 0.3, "laughter": 0.0, "speech_rate": 3.5, "sarcasm": 0.0, "upset": 0.1}


class ScriptedDetector:
    """Says "speaking" for the frames whose numbers are in `speech`."""

    def __init__(self, speech: range) -> None:
        self.speech, self.frame = speech, -1

    async def analyze_audio(self, audio: bytes) -> VADState:
        self.frame += 1
        return SPEAKING if self.frame in self.speech else QUIET


async def hear(reader: CallerVoiceReader, frames: int) -> None:
    for _ in range(frames):
        await reader.observe(InputAudioRawFrame(audio=FRAME, sample_rate=RATE, num_channels=1))
    if reader._reads is not None:
        await reader._reads


async def test_the_reader_reads_a_turn_when_the_caller_pauses():
    got: list[VoiceReading] = []

    async def on_reading(made: VoiceReading) -> None:
        got.append(made)

    model = FakeModel()
    reader = CallerVoiceReader(model, on_reading, detector=lambda rate: ScriptedDetector(range(10, 80)))
    await hear(reader, 9)
    assert got == []  # no speech yet
    await hear(reader, 80)
    assert [(r.turn, r.tone) for r in got] == [(1, "neutral")] and model.clips[0][1] == RATE
    assert 1.4 <= got[0].seconds <= 1.7

    await reader.observe(BotStartedSpeakingFrame())  # the agent answers: the next speech is turn 2
    reader._detector.speech = range(100, 160)
    await hear(reader, 80)
    assert [r.turn for r in got] == [1, 2]


async def test_the_reader_skips_short_speech_a_model_that_is_not_loaded_and_a_forgotten_caller():
    got: list[VoiceReading] = []

    async def on_reading(made: VoiceReading) -> None:
        got.append(made)

    short = CallerVoiceReader(FakeModel(), on_reading, detector=lambda rate: ScriptedDetector(range(10, 30)))  # 0.4 s
    await hear(short, 60)

    not_loaded = FakeModel()
    not_loaded.ready = False
    await hear(CallerVoiceReader(not_loaded, on_reading, detector=lambda rate: ScriptedDetector(range(10, 80))), 100)

    forgotten = FakeModel()
    await hear(CallerVoiceReader(forgotten, on_reading, active=lambda: False, detector=lambda rate: ScriptedDetector(range(10, 80))), 100)
    assert got == [] and not_loaded.clips == [] and forgotten.clips == []


async def test_a_model_error_does_not_stop_the_call():
    class Broken(FakeModel):
        def read(self, pcm: bytes, sample_rate: int) -> dict:
            raise RuntimeError("no model")

    async def on_reading(made: VoiceReading) -> None:
        raise AssertionError("no reading is expected")

    await hear(CallerVoiceReader(Broken(), on_reading, detector=lambda rate: ScriptedDetector(range(10, 80))), 100)


# --- the note about frustration ---


class Session:
    """Stands for an agent's session: it holds the notes that were sent and not taken out."""

    def __init__(self, takes_notes: bool = True) -> None:
        self.takes_notes, self.notes, self.sent = takes_notes, [], 0
        self.holder = "contact"

    async def send(self, text: str):
        if not self.takes_notes:
            return None
        self.sent += 1
        self.notes.append((self.holder, text))
        return self.notes[-1]

    async def forget(self, note) -> None:
        self.notes.remove(note)

    def current(self, note) -> bool:
        return note[0] == self.holder


async def test_a_frustrated_turn_sends_one_note_and_the_note_leaves_after_two_turns():
    session = Session()
    note = FrustrationNote(session.send, session.forget, session.current)
    assert not await note.on_reading(reading(1)) and session.notes == []
    assert await note.on_reading(reading(2, tone="frustrated", confidence=0.8))
    assert session.notes == [("contact", FRUSTRATED_NOTE)]
    assert not await note.on_reading(reading(2, tone="frustrated", confidence=0.9))  # the same turn, read again: no second note
    assert not await note.on_reading(reading(3)) and note.held  # the note stays for the next turn
    assert not await note.on_reading(reading(2 + NOTE_TURNS)) and session.notes == [] and not note.held
    assert session.sent == 1


async def test_a_new_frustrated_turn_keeps_the_note_longer():
    session = Session()
    note = FrustrationNote(session.send, session.forget, session.current)
    await note.on_reading(reading(1, tone="frustrated", confidence=0.8))
    await note.on_reading(reading(2, tone="frustrated", confidence=0.8))
    await note.on_reading(reading(3))
    assert note.held and session.sent == 1
    await note.on_reading(reading(4))
    assert not note.held


async def test_after_a_handoff_the_old_note_is_taken_out_and_the_next_agent_gets_its_own():
    session = Session()
    note = FrustrationNote(session.send, session.forget, session.current)
    await note.on_reading(reading(1, tone="frustrated", confidence=0.8))
    session.holder = "consultant"
    await note.on_reading(reading(2))
    assert session.notes == [] and not note.held  # the contact agent's session no longer holds a note
    assert await note.on_reading(reading(3, tone="frustrated", confidence=0.8))
    assert session.notes == [("consultant", FRUSTRATED_NOTE)]


async def test_an_agent_that_takes_no_note_gets_none():
    session = Session(takes_notes=False)
    note = FrustrationNote(session.send, session.forget, session.current)
    assert not await note.on_reading(reading(1, tone="frustrated", confidence=0.8)) and not note.held
