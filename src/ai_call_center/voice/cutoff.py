"""When a sound comes from the caller's line, the agent learns what to say if it has no words.

A voice provider takes a cough or a noise for the start of a caller's turn. The agent's audio
stops at once, and the provider then answers the sound. The model hears that the sound has no
words, but it does not know that it was cut off, or in which sentence. On 2026-11-17 an agent
answered four coughs with "Sorry, I didn't catch that", and a prompt rule alone gave "Go ahead?"
in place of the greeting. So this code tells the agent the one fact it cannot see: the sentence
that was cut. A cough while the agent is silent got a new greeting ("Hi, I'm glad you called"),
so in that case the agent is told its last question. The agent still decides if the caller's
turn has words: it hears the audio.

No voice detection mode tells a cough from a word (measured 2026-11-17: OpenAI server VAD at
threshold 0.8, OpenAI semantic VAD, and the local Silero detector all took coughs for speech).
"""

import re
import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

SPOKEN_CHARS_PER_SEC = 15.0  # about 2.5 words a second; places the cut in the text
CUT_TEXT_MAX_CHARS = 420  # how much of the answer the agent is asked to say again
JUST_INTERRUPTED_SECS = 0.5  # a caller's turn this soon after an interruption of the agent is the cause of it

ANSWERED = "answered"

_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def cut_sentence(text: str, heard_secs: float) -> str:
    """The agent's answer from the start of the sentence that was playing after `heard_secs`.

    `text` is the whole answer so far as the voice service wrote it. The service writes faster
    than the audio plays, so the text usually runs past the cut: that is what the caller missed.
    """
    text = " ".join(text.split())
    if not text:
        return ""
    heard_chars = max(heard_secs, 0.0) * SPOKEN_CHARS_PER_SEC
    start = 0
    for part in _SENTENCE_END.split(text):
        end = start + len(part)
        if heard_chars <= end:
            break
        start = min(end + 1, len(text))
    else:
        start = text.rfind(part)  # the estimate runs past the text: the last sentence was playing
    rest = text[start:].strip() or text
    if len(rest) > CUT_TEXT_MAX_CHARS:
        rest = rest[:CUT_TEXT_MAX_CHARS].rsplit(" ", 1)[0]
    return rest


def last_question(text: str) -> str:
    """The question that ends the agent's answer `text`. "" when it does not end with one."""
    parts = _SENTENCE_END.split(" ".join(text.split()))
    return parts[-1] if parts and parts[-1].endswith("?") else ""


_NOTE_START = "This is a note from the phone system, not from the caller. It replaces any earlier note from the phone system about a sound. "
_DECIDE = (
    "Listen to the caller's turn that comes now, and decide which of two cases it is. "
    "Case 1: the caller said clear words. A request, a question, an answer to what you asked, or words that ask you to stop "
    '("wait", "stop", "hold on", "excuse me") are clear words. Then forget this note: {words} '
    'Case 2: a cough or a noise. The turn has no clear words, or only an "um", or a word or two that do not fit what you just said '
    '(a cough often sounds like "bye", "thank you", "hi" or "you"). Then it is not a request: '
    "do not answer it, do not apologise, do not comment on it, and do not greet again. {noise}"
)


def cut_off_note(said: str) -> str:
    """What the agent is told, in its own session, at the moment a sound cuts it off."""
    again = f'Say this again, in these words, and then go on as you planned: "{said}"' if said else "Say again the sentence that you had started, and then go on as you planned."
    return (
        _NOTE_START
        + "A sound on the caller's line cut you off in the middle of a sentence. "
        + _DECIDE.format(words="answer the caller's words, and do not say again what you were saying.", noise=again)
    )


def quiet_sound_note(question: str) -> str:
    """What the agent is told when a caller's turn starts while the agent is silent."""
    say = f'Say only this: "{question}"' if question else 'Say only this: "Go ahead."'
    return _NOTE_START + _DECIDE.format(words="answer the caller's words as usual.", noise=say)


class CallerSoundWatch(FrameProcessor):
    """Sits after the voice services. Calls `on_sound(note, cut)` when a caller's turn starts.

    `note` is `cut_off_note` when the turn started while the agent spoke (`cut` is True), and
    `quiet_sound_note` when the agent was silent. `on_sound` runs at the start of the caller's
    sound, so the note reaches the agent before the provider answers the turn. `on_answer` runs
    when the agent next starts to speak: the note has done its work and must go.
    """

    def __init__(self, on_sound, on_answer=None) -> None:
        super().__init__()
        self._on_sound = on_sound
        self._on_answer = on_answer
        self._note_out = False
        self._text = ""
        self._speaking = False
        self._started_at = 0.0
        self._interrupted_at = 0.0
        self._noted = False  # one note for one stretch of agent speech

    def observe(self, frame: Frame, direction: FrameDirection, now: float | None = None) -> tuple[str, bool] | str | None:
        """Track the frame. Returns (note, cut) at the start of a caller's turn, ANSWERED when the
        agent starts to speak after a note, else None."""
        now = time.monotonic() if now is None else now
        if isinstance(frame, LLMFullResponseStartFrame):
            self._text = ""
        elif isinstance(frame, TTSTextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._text += frame.text
        elif isinstance(frame, BotStartedSpeakingFrame):
            if not self._speaking:
                self._speaking, self._started_at, self._noted = True, now, False
                if self._note_out:
                    self._note_out = False
                    return ANSWERED
        elif isinstance(frame, InterruptionFrame):
            if self._speaking:
                self._interrupted_at = now
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking = False
        elif isinstance(frame, UserStartedSpeakingFrame):
            cut = self._speaking or now - self._interrupted_at <= JUST_INTERRUPTED_SECS
            if cut and not self._noted:
                self._noted = True
                self._interrupted_at = 0.0
                self._note_out = True
                return cut_off_note(cut_sentence(self._text, now - self._started_at)), True
            if not cut:
                self._note_out = True
                return quiet_sound_note(last_question(self._text)), False
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        sound = self.observe(frame, direction)
        if sound == ANSWERED:
            if self._on_answer:
                await self._on_answer()
        elif sound is not None:
            await self._on_sound(*sound)
        await self.push_frame(frame, direction)
