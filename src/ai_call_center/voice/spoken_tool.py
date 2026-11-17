"""A handoff that the agent said aloud and did not make.

A speech-to-speech model sometimes speaks a tool call as text. Measured 2026-11-17 on Amazon Nova:
the consultant agent said "Let me bring in the right agent to get your details. <call transfer_ to_
contact", made no tool call, and went silent. The caller heard "call transfer to underscore
underscore contact" and had to ask "what?".

The server reads the agent's text. When the agent stops speaking and its text has the name of a
handoff tool, the server is told, and it makes the handoff itself.
"""

from __future__ import annotations

import re

from pipecat.frames.frames import BotStoppedSpeakingFrame, Frame, LLMFullResponseStartFrame, LLMTextFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

TEXT_TAIL_CHARS = 400  # a tool name comes at the end of what the agent said


def spoken_handoff(text: str, agents) -> str:
    """The agent whose handoff tool the text names, or "".

    The tool name is transfer_to_<agent>. A model writes it with spaces in it ("transfer_ to_ contact"),
    so spaces do not count. An underscore must be there: "I can transfer to contact you" is plain speech.
    """
    names = "|".join(sorted((re.escape(a) for a in agents), key=len, reverse=True))
    if not names:
        return ""
    found = ""
    for match in re.finditer(rf"transfer(_*)to(_*)({names})", re.sub(r"\s+", "", text.lower())):
        if match.group(1) or match.group(2):
            found = match.group(3)  # the last one named is where the agent wants the caller to go
    return found


class SpokenToolWatch(FrameProcessor):
    """Tells the server when the agent said a handoff tool's name. Sits after the voice services."""

    def __init__(self, agents, on_spoken) -> None:
        super().__init__()
        self._agents = tuple(agents)
        self._on_spoken = on_spoken  # async (target agent)
        self._text = ""

    def observe(self, frame: Frame, direction: FrameDirection) -> str:
        """The handoff target to act on now, or ""."""
        if isinstance(frame, LLMFullResponseStartFrame):
            self._text = ""
        elif isinstance(frame, (LLMTextFrame, TTSTextFrame)) and direction == FrameDirection.DOWNSTREAM:
            self._text = (self._text + " " + frame.text)[-TEXT_TAIL_CHARS:]
        elif isinstance(frame, BotStoppedSpeakingFrame):
            target, self._text = spoken_handoff(self._text, self._agents), ""
            return target
        return ""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if target := self.observe(frame, direction):
            await self._on_spoken(target)
        await self.push_frame(frame, direction)
