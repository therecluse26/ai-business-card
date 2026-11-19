"""Quick yes/no decisions from Jev, the decision model of TypeSafe AI.

A word list reads words, not meaning: "can I talk to him?" and "I can't talk right now" hold the same
word. So the server asks Jev one typed question and reads the probability of a yes. Jev writes no text.

A slow request, a failed request, or no API key gives no decision, and the word list decides in its
place. So Jev can never stop a call or delay it by more than the time limit.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

CALLER_TURNS_SENT = 4  # the caller's last turns, with the agent's lines between them
LINE_MAX_CHARS = 400  # a longer line is cut; the request stays small and fast

# One question, and a high value means yes. The criteria say where yes ends: a caller who only
# agrees to the agent's offer has not asked in their own words.
LIVE_CALL_QUESTION = "Did the caller ask, in their own words, to speak with the owner on this call?"
LIVE_CALL_YES = (
    "The caller asks to speak or talk with the owner, to be connected, transferred or put through to the owner, "
    "to get the owner on the line, or to reach a real person. A request that names no later time is a request for this call."
)
LIVE_CALL_NO = (
    "The caller wants to leave a message, send an email, get a call back, or book a later time. "
    "Or the caller only asks about the owner. Or the caller only said yes or okay to an offer that the agent made."
)


@dataclass(frozen=True)
class Decision:
    """One answer from the decision model."""

    probability: float  # that the answer is yes, 0 to 1
    millis: int  # how long the request took
    model: str  # the model version that answered, for example "jev-1.14.0"


@runtime_checkable  # the tool schema of a LangChain tool checks the call context's fields with isinstance
class Decider(Protocol):
    async def asked_for_live_call(self, lines: list[str], *, owner_name: str = "") -> Decision | None:
        """The probability that the caller asked for a live call with the owner. None = no decision."""
        ...

    async def warm(self) -> None:
        """Open the connection, so the next request does not pay for it."""
        ...


def conversation_state(lines: list[str], *, owner_name: str = "") -> dict[str, Any]:
    """What the model reads: the end of the call, from the caller's fourth-last turn on."""
    caller_turns = [i for i, line in enumerate(lines) if line.startswith("caller:")]
    recent = lines[caller_turns[-CALLER_TURNS_SENT] :] if len(caller_turns) >= CALLER_TURNS_SENT else lines
    conversation = []
    for line in recent:
        speaker, _, text = line.partition(":")
        conversation.append({"speaker": speaker.strip(), "text": text.strip()[:LINE_MAX_CHARS]})
    owner = owner_name or "the owner"
    # Only the end of the call is sent, so the greeting that names the owner can be missing. Without
    # this line Jev cannot tell who "him" is: "Can I speak to him?" got 0.74, and 0.98 with the greeting.
    call = f"A phone call to the AI business card of {owner}. An AI agent answers. A person who is not named, such as 'him' or 'her', is {owner}."
    return {"owner": owner, "call": call, "conversation": conversation}


class JevDecider:
    """Asks Jev over the TypeSafe API. One client for the process: a new client pays for a new connection."""

    def __init__(self, *, api_key: str, model: str = "jev-latest", time_limit: float = 0.5, client: Any = None) -> None:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        self.model = model
        self.time_limit = time_limit
        # No retries: a second try cannot fit in the time limit, and the code's own answer is the fallback.
        self._client = client or AsyncTypeSafeClient(api_key=api_key, model=model, retry=RetryPolicy(max_retries=0), timeout=time_limit)

    async def asked_for_live_call(self, lines: list[str], *, owner_name: str = "") -> Decision | None:
        from typesafe_sdk import Noul, NoulCriteria

        started = time.monotonic()
        try:
            # The SDK's timeout is per HTTP step (connect, write, read). wait_for is the limit for the whole request.
            response = await asyncio.wait_for(
                self._client.system_one(
                    state=conversation_state(lines, owner_name=owner_name),
                    questions={"live_call": Noul(instructions=LIVE_CALL_QUESTION, criteria=NoulCriteria(true=LIVE_CALL_YES, false=LIVE_CALL_NO))},
                ),
                self.time_limit,
            )
            probability = float(response.nouls["live_call"].noul)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"a probability of {probability}")
        except Exception as error:  # a timeout, the network, a refused key, a changed response
            log.warning("Jev gave no decision after %d ms (%s: %s)", _millis(started), type(error).__name__, error)
            return None
        return Decision(probability=probability, millis=_millis(started), model=str(response.model))

    async def warm(self) -> None:
        try:
            await asyncio.wait_for(self._client.models.list(), self.time_limit)
        except Exception as error:
            log.debug("Jev warm-up failed (%s)", type(error).__name__)

    async def aclose(self) -> None:
        await self._client.aclose()


def _millis(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


_default: JevDecider | None = None


def default_decider(settings: Any) -> Decider | None:
    """The process-wide Jev client, or None when TYPESAFE_API_KEY is not set (the word lists decide alone)."""
    global _default
    key = settings.typesafe_api_key.get_secret_value()
    if not key:
        return None
    if _default is None:
        try:
            _default = JevDecider(api_key=key, model=settings.jev_model, time_limit=settings.jev_timeout_ms / 1000)
        except Exception as error:  # a key with spaces in it, for example
            log.error("Jev decisions are off: the client could not be made (%s: %s)", type(error).__name__, error)
            return None
        log.info("Jev decisions are on: model %s, time limit %d ms", settings.jev_model, settings.jev_timeout_ms)
    return _default
