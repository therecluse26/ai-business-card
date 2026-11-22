"""Respellings: a word written so that a voice model says it right ("Magyar" -> "Mag-yar").

The voice models here are speech-to-speech models. They take no phonetic markup (no SSML, no IPA), and a
hint in the prompt ("say it as ...") loses against the sound the model learned for the spelling. What
does work: the model says a word the way the word is written in the text it reads. So the server
changes the written word to its respelling in everything a voice model reads, and changes it back in
everything a person reads (the lead email, the call summary, the log).

`RESPELLINGS` in `.env` holds the pairs: "Magyar=Mag-yar". The text chat does not use them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Not part of a web or email address, and not a part of a longer word: "brad.magyar@example.com" and
# "example.com/magyar" stay as they are.
_BEFORE = r"(?<![\w@./-])"
_AFTER = r"(?![\w@/-]|\.\w)"


def _pattern(word: str) -> re.Pattern[str]:
    return re.compile(f"{_BEFORE}{re.escape(word)}{_AFTER}", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Respelling:
    pairs: tuple[tuple[str, str], ...] = ()  # (written, spoken)

    @classmethod
    def parse(cls, text: str) -> Respelling:
        """ "Magyar=Mag-yar, Nguyen=Win" as pairs. A pair with an empty side is an error, so a typing
        mistake in `.env` stops the server at start and not in the middle of a call."""
        pairs: list[tuple[str, str]] = []
        for part in text.replace(";", ",").split(","):
            if not part.strip():
                continue
            written, _, spoken = (piece.strip() for piece in part.partition("="))
            if not written or not spoken:
                raise ValueError(f"RESPELLINGS: {part.strip()!r} must be <written word>=<how to write it for the voice>")
            if _pattern(written).search(spoken):
                raise ValueError(f"RESPELLINGS: the respelling {spoken!r} must not contain the word {written!r}")
            pairs.append((written, spoken))
        return cls(tuple(pairs))

    def __bool__(self) -> bool:
        return bool(self.pairs)

    def spoken(self, text: str) -> str:
        """`text` for a voice model: each written word is its respelling."""
        for written, spoken in self.pairs:
            text = _pattern(written).sub(lambda _: spoken, text)
        return text

    def written(self, text: str) -> str:
        """`text` for a person to read: each respelling is the written word again."""
        for written, spoken in self.pairs:
            text = _pattern(spoken).sub(lambda _: written, text)
        return text

    def written_messages(self, messages: list[Any]) -> list[Any]:
        """Copies of Pipecat context messages with the written words, for the summary model and the log."""
        if not self:
            return messages
        out: list[Any] = []
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                msg = {**msg, "content": self.written(msg["content"])}
            elif isinstance(msg, dict) and isinstance(msg.get("content"), list):
                parts = [{**p, "text": self.written(p["text"])} if isinstance(p, dict) and isinstance(p.get("text"), str) else p for p in msg["content"]]
                msg = {**msg, "content": parts}
            out.append(msg)
        return out

    def instructions(self, text: str) -> str:
        """An agent's instructions for a voice model: the respellings, and one note that explains them.

        The note gives the true spelling as letters only. The written word itself is not in the text,
        so the model has no second spelling to say."""
        if not self:
            return text
        lines = [f'- "{spoken}" is spelled {", ".join(written.upper())}.' for written, spoken in self.pairs]
        return (
            f"{self.spoken(text)}\n\n"
            "Respelled words:\n"
            "- Some words in these instructions are respelled, so that you say them right. Say each one as one word, the way it is written here.\n"
            "- Say the letters of the true spelling only when the caller asks how the word is spelled.\n" + "\n".join(lines)
        )

    def tool(self, handler):
        """`handler` with the respellings on both sides: the tool's result goes to the voice model with
        the respellings, and the model's arguments come in with the written words."""
        if not self:
            return handler

        async def respelled(params) -> None:
            callback = params.result_callback

            async def result_callback(result, *, properties=None) -> None:
                await callback(self.spoken(result) if isinstance(result, str) else result, properties=properties)

            params.result_callback = result_callback
            params.arguments = {k: self.written(v) if isinstance(v, str) else v for k, v in dict(params.arguments or {}).items()}
            await handler(params)

        return respelled
