"""`uv run check-jev`: ask Jev about sample caller lines, to check the key, the question, and the time limit.

For each sample it prints the word list's answer, Jev's probability, what the transfer check does with
it, and the time. The first request opens the connection, so it is the slowest. Use the times to set
JEV_TIMEOUT_MS. The sample lines are in `evals/live_call.json`, the same file that `uv run score-harness`
reads. The lines are made up; no caller's words are sent.
"""

from __future__ import annotations

import asyncio
import sys

from ai_call_center.config import get_settings
from ai_call_center.decisions import JevDecider
from ai_call_center.graph.tools import LIVE_CALL_NO, LIVE_CALL_YES, asked_for_the_owner
from ai_call_center.score import load_live_call, sample_lines

CHECK_TIME_LIMIT = 5.0  # generous on purpose: this run measures the time, it does not enforce the limit


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
    for sample in load_live_call():
        wants_live_call = sample.wants_live_call
        lines = sample_lines(sample.lines, settings.owner_name)
        said = " / ".join(sample.lines)
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
