"""`uv run check-jev`: ask Jev about sample caller lines, to check the key, the question, and the time limit.

For each sample it prints the word list's answer, Jev's probability, what the transfer check does with
it, and the time. The first request opens the connection, so it is the slowest. Use the times to set
JEV_TIMEOUT_MS. The lines are made up; no caller's words are sent.
"""

from __future__ import annotations

import asyncio
import sys

from ai_call_center.config import get_settings
from ai_call_center.decisions import JevDecider
from ai_call_center.graph.tools import LIVE_CALL_NO, LIVE_CALL_YES, asked_for_the_owner

CHECK_TIME_LIMIT = 5.0  # generous on purpose: this run measures the time, it does not enforce the limit

# (the end of a call, does the caller want a live call). A line with no "agent:" at its start is the caller's.
SAMPLES = [
    ("Can I speak to him?", True),
    ("Is Brad around? I'd love a quick word with him.", True),
    ("Any chance you could get him on the line for me right now?", True),
    ("I'd rather just chat with him directly, if he's free.", True),
    ("Put me through to him, please.", True),
    ("Is there a real person I can talk to?", True),
    ("I don't want to leave a message. I want to talk to him.", True),
    ("agent: I can take a message, or connect you to him now.\nConnect me, please.", True),
    ("I would like to leave a message with him.", False),
    ("Just have him call me back. I can't talk right now.", False),
    ("Could he give me a call next week?", False),
    ("No need to bother him. Just pass on my number.", False),
    ("Can you email him my details?", False),
    ("What kind of projects does he take on?", False),
    ("Yes.", False),
    ("agent: Would you like me to connect you to him?\nSure.", False),
]


def sample_lines(sample: str, owner_name: str) -> list[str]:
    """A real call starts with the agent's greeting, which names the owner. Without it, "him" has no meaning."""
    greeting = f"agent: This is {owner_name}'s AI business card. How can I help?"
    return [greeting] + [line if line.startswith("agent:") else f"caller: {line}" for line in sample.split("\n")]


async def run() -> int:
    settings = get_settings()
    key = settings.typesafe_api_key.get_secret_value()
    if not key:
        print("TYPESAFE_API_KEY is not set in .env. Make a key at console.typesafe.ai, add it, and run this again.")
        return 1
    decider = JevDecider(api_key=key, model=settings.jev_model, time_limit=CHECK_TIME_LIMIT)
    limit = settings.jev_timeout_ms
    print(f"model {settings.jev_model}; JEV_TIMEOUT_MS={limit}; yes at {LIVE_CALL_YES}, no at {LIVE_CALL_NO}\n")
    print(f"{'word list':<10}{'Jev':>6}{'ms':>7}  {'transfer check':<24}{'':<7}caller line")
    failed = over = wrong = unsure = 0
    for sample, wants_live_call in SAMPLES:
        lines = sample_lines(sample, settings.owner_name)
        said = sample.replace("\n", " / ")
        listed = asked_for_the_owner(lines)
        decision = await decider.asked_for_live_call(lines, owner_name=settings.owner_name)
        if decision is None:
            failed += 1
            print(f"{'match' if listed else 'miss':<10}{'-':>6}{'-':>7}  {'no decision':<31}{said}")
            continue
        p = decision.probability
        outcome = "transfer goes ahead" if p >= LIVE_CALL_YES else "agent asks the caller" if p > LIVE_CALL_NO else "no transfer"
        slow = decision.millis > limit
        over += slow
        middle = LIVE_CALL_NO < p < LIVE_CALL_YES
        unsure += middle and wants_live_call
        wrong += (p >= LIVE_CALL_YES) != wants_live_call and not middle
        mark = "ok" if (p >= LIVE_CALL_YES) == wants_live_call and not middle else "?" if middle else "WRONG"
        print(f"{'match' if listed else 'miss':<10}{p:>6.2f}{decision.millis:>6}{'*' if slow else ' '}  {outcome:<24}{mark:<7}{said}")
    await decider.aclose()
    print()
    if failed:
        print(f"{failed} request(s) gave no decision. The log line above each one names the error.")
    if over:
        print(f"* {over} request(s) took longer than JEV_TIMEOUT_MS. On a call, the word list decides for those.")
    if wrong:
        print(f"{wrong} line(s) got a clear answer that is not the expected one. Read them before you trust the limits.")
    if unsure:
        print(f"{unsure} request(s) for a live call got a question from the agent, not a transfer. The caller must ask twice.")
    if not (failed or over or wrong or unsure):
        print("All requests gave the expected answer inside the time limit.")
    return 1 if failed else 0


def main() -> None:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    sys.exit(asyncio.run(run()))
