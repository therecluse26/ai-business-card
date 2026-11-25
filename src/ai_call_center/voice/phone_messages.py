"""Fixed texts that the phone system puts in the shared context, and how to read them back.

`call_session.py` writes these messages. A voice leg that takes the call as text, not as a
replayed context (Phonic, see `phonic_leg.py`), reads them back: it must know the greeting
order and the repeated "call so far" block when it sees them.
"""

from __future__ import annotations


def connect_message(words: str) -> str:
    """The first message of a call, with the exact greeting. Left free, a voice model said only
    "I'm the AI assistant on this line", and once spoke its own thoughts first. Measured 2026-11-16."""
    return f'The call has just connected. Say exactly these words to the caller, and nothing before or after them: "{words}"'


CONNECT_MARK = "The call has just connected. Say exactly these words"
GREETING_DONE = "The call connected, and the greeting agent said the greeting."
_CONNECT_HEAD = connect_message("")[:-1]  # everything before the words and the closing quote


def greeting_of(text: str) -> str | None:
    """The words of a `connect_message`, exactly as they went in. None for any other text."""
    if text.startswith(_CONNECT_HEAD) and text.endswith('"') and len(text) > len(_CONNECT_HEAD):
        return text[len(_CONNECT_HEAD) : -1]
    return None


# The first words of the block that `call_so_far` adds to a handoff instruction.
CALL_SO_FAR_MARK = "The call so far, word for word"


def without_call_so_far(text: str) -> str:
    """The text with the repeated "call so far" block cut off. The block is always last."""
    at = text.find("\n\n" + CALL_SO_FAR_MARK)
    return text if at < 0 else text[:at]
