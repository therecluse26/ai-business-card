"""Analytics for each call: the figures, the answer times, and the grader's scores.

Two readers use this data. A public page shows how the calls went. The hill climbing loop compares
the scores of one version of the prompts and the code with the scores of the next.

The tables hold no phone number, no name, no summary, and no transcript. Two fields stay private all
the same: the call id, which joins a record to its LangSmith trace, and the grader's notes, which can
say what happened on a call. `public_report` is the only way out for a reader who is not the owner,
and it copies only the fields that it names. A new field is private until someone adds it there.

A caller who asked to be forgotten gets no record, and the notes of their earlier calls are deleted.

Run `uv run call-report` to print the public report. `--notes` prints the private notes too.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel

from ai_call_center.agents.specs import LEG_PLATFORMS
from ai_call_center.grading import FAIL, NA, PASS, RULES, Score, describe_scores, grade

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
VOICE, TEXT = "voice", "text"  # the channel of a call: the phone, or the text mode
MESSAGE, TRANSFER, NO_OUTCOME = "message", "transfer", "none"  # how a call ended for the owner
MEETING = "meeting"  # the caller booked a meeting on Cal.com (meetings.py); it counts before a message
OUTCOMES = (MESSAGE, TRANSFER, NO_OUTCOME, MEETING)
# The outcome of one call can say something about one caller, so the report gives totals only, and
# only from this many calls on. Limit: a reader who keeps two reports can still see the change between them.
MIN_CALLS_FOR_OUTCOMES = 5
DURATION_STEP_SECS = 10  # the report rounds a call's length to this


@dataclass(frozen=True)
class Answer:
    """One timed answer: from the end of the caller's speech to the start of the agent's audio."""

    leg: str  # the voice leg that answered, for example "openai_realtime"
    seconds: float
    agent: str = ""


@dataclass(frozen=True)
class CallAnalytics:
    call_id: str  # private
    day: str  # the US Eastern date the call started on; the time of day is not kept
    channel: str  # VOICE or TEXT
    version: str  # the git commit of the prompts and the code that took the call
    test: bool  # the owner's own phone, or the terminal simulator: not a caller
    duration_secs: int
    agents: list[str]  # in the order they had the call
    outcome: str  # one of OUTCOMES
    caller_turns: int = 0
    failovers: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0  # the voice models only, at list prices (`voice/stats.py`)
    answers: list[Answer] = field(default_factory=list)
    handoff_secs: list[float] = field(default_factory=list)
    scores: list[Score] = field(default_factory=list)  # each note is private
    readings: dict[str, str] = field(default_factory=dict)  # private: how the call sounded (`grading.READINGS`)
    seq: int = 0  # the record's number, set by the store; it stays the same when a record is saved again


class AnalyticsStore:
    """Three tables: one row for each call, one for each timed answer, one for each score.

    It takes a path, or the open connection of another store on the same database."""

    def __init__(self, db: Path | str | sqlite3.Connection) -> None:
        self._own = not isinstance(db, sqlite3.Connection)
        if isinstance(db, sqlite3.Connection):
            self._conn = db
        else:
            if str(db) != ":memory:":
                Path(db).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_analytics (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL UNIQUE,
                day TEXT NOT NULL,
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                test INTEGER NOT NULL,
                duration_secs INTEGER NOT NULL,
                agents TEXT NOT NULL,
                outcome TEXT NOT NULL,
                caller_turns INTEGER NOT NULL,
                failovers INTEGER NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                handoff_secs TEXT NOT NULL,
                readings TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_answers (
                call_id TEXT NOT NULL,
                n INTEGER NOT NULL,
                agent TEXT NOT NULL,
                leg TEXT NOT NULL,
                seconds REAL NOT NULL,
                PRIMARY KEY (call_id, n)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_scores (
                call_id TEXT NOT NULL,
                rule TEXT NOT NULL,
                verdict TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                agent TEXT NOT NULL DEFAULT '',
                cause TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (call_id, rule)
            )
            """
        )
        # A database from before the cause codes has no `agent` and no `cause` column.
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(call_scores)")}
        for column in ("agent", "cause"):
            if column not in columns:
                self._conn.execute(f"ALTER TABLE call_scores ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
        # A database from before the grader heard a call has no `readings` column.
        if "readings" not in {row[1] for row in self._conn.execute("PRAGMA table_info(call_analytics)")}:
            self._conn.execute("ALTER TABLE call_analytics ADD COLUMN readings TEXT NOT NULL DEFAULT '{}'")
        self._conn.commit()

    def save(self, call: CallAnalytics) -> None:
        """Write the record of one call. A call that is saved again keeps its number, and the new figures replace the old.

        A call is saved again when it comes back from a transfer that nobody answered."""
        self._conn.execute(
            """
            INSERT INTO call_analytics (call_id, day, channel, version, test, duration_secs, agents, outcome, caller_turns,
                                        failovers, tokens_in, tokens_out, cost_usd, handoff_secs, readings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(call_id) DO UPDATE SET
                day = excluded.day, channel = excluded.channel, version = excluded.version, test = excluded.test,
                duration_secs = excluded.duration_secs, agents = excluded.agents, outcome = excluded.outcome,
                caller_turns = excluded.caller_turns, failovers = excluded.failovers, tokens_in = excluded.tokens_in,
                tokens_out = excluded.tokens_out, cost_usd = excluded.cost_usd, handoff_secs = excluded.handoff_secs,
                readings = excluded.readings
            """,
            (
                call.call_id,
                call.day,
                call.channel,
                call.version,
                int(call.test),
                int(call.duration_secs),
                " -> ".join(call.agents),
                call.outcome,
                int(call.caller_turns),
                int(call.failovers),
                int(call.tokens_in),
                int(call.tokens_out),
                float(call.cost_usd),
                json.dumps([round(secs, 3) for secs in call.handoff_secs]),
                json.dumps(call.readings),
            ),
        )
        self._conn.execute("DELETE FROM call_answers WHERE call_id = ?", (call.call_id,))
        self._conn.executemany(
            "INSERT INTO call_answers (call_id, n, agent, leg, seconds) VALUES (?, ?, ?, ?, ?)",
            [(call.call_id, n, answer.agent, answer.leg, round(answer.seconds, 3)) for n, answer in enumerate(call.answers, start=1)],
        )
        self._conn.execute("DELETE FROM call_scores WHERE call_id = ?", (call.call_id,))
        self._conn.executemany(
            "INSERT OR REPLACE INTO call_scores (call_id, rule, verdict, note, agent, cause) VALUES (?, ?, ?, ?, ?, ?)",
            [(call.call_id, score.rule, score.verdict, score.note, score.agent, score.cause) for score in call.scores],
        )
        self._conn.commit()

    def forget(self, call_id: str) -> None:
        """Delete the whole record of one call: the caller on it asked to be forgotten."""
        for table in ("call_analytics", "call_answers", "call_scores"):
            self._conn.execute(f"DELETE FROM {table} WHERE call_id = ?", (call_id,))
        self._conn.commit()

    def drop_notes(self, call_ids: list[str]) -> None:
        """Delete the grader's notes of these calls. The figures and the verdicts stay: they say nothing about a person."""
        self._conn.executemany("UPDATE call_scores SET note = '' WHERE call_id = ?", [(call_id,) for call_id in call_ids])
        self._conn.commit()

    def calls(self) -> list[CallAnalytics]:
        """Every record, oldest first."""
        answers: dict[str, list[Answer]] = {}
        for call_id, agent, leg, seconds in self._conn.execute("SELECT call_id, agent, leg, seconds FROM call_answers ORDER BY call_id, n"):
            answers.setdefault(call_id, []).append(Answer(leg=leg, seconds=seconds, agent=agent))
        order = {rule.key: n for n, rule in enumerate(RULES)}
        scores: dict[str, list[Score]] = {}
        for call_id, rule, verdict, note, agent, cause in self._conn.execute("SELECT call_id, rule, verdict, note, agent, cause FROM call_scores"):
            scores.setdefault(call_id, []).append(Score(rule, verdict, note, agent=agent, cause=cause))
        rows = self._conn.execute(
            "SELECT seq, call_id, day, channel, version, test, duration_secs, agents, outcome, caller_turns,"
            " failovers, tokens_in, tokens_out, cost_usd, handoff_secs, readings FROM call_analytics ORDER BY seq"
        ).fetchall()
        return [
            CallAnalytics(
                seq=row[0],
                call_id=row[1],
                day=row[2],
                channel=row[3],
                version=row[4],
                test=bool(row[5]),
                duration_secs=row[6],
                agents=[agent for agent in row[7].split(" -> ") if agent],
                outcome=row[8],
                caller_turns=row[9],
                failovers=row[10],
                tokens_in=row[11],
                tokens_out=row[12],
                cost_usd=row[13],
                handoff_secs=json.loads(row[14]),
                readings=json.loads(row[15] or "{}"),
                answers=answers.get(row[1], []),
                scores=sorted(scores.get(row[1], []), key=lambda score: order.get(score.rule, len(order))),
            )
            for row in rows
        ]

    def close(self) -> None:
        if self._own:
            self._conn.close()


# --- the version of the harness ---


@lru_cache
def harness_version(configured: str = "") -> str:
    """The git commit of the prompts and the code that take the calls. Each record names it, so the
    scores of one version can be compared with the next.

    The container image has it in HARNESS_VERSION (`scripts/fly-deploy.sh`). On a laptop git gives it,
    with "-dirty" when a tracked file has a change that is not committed."""
    if configured.strip():
        return configured.strip()
    here = Path(__file__).parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{commit}-dirty" if commit and changed else commit or "unknown"


# --- one call's record, made when the call ends ---


def call_day(started_at: datetime) -> str:
    return started_at.astimezone(EASTERN).date().isoformat()


async def record_call(
    db: Path | str | sqlite3.Connection,
    call: CallAnalytics,
    lines: list[str],
    *,
    summary_tries: int = 0,
    profile: str = "",
    owner: str = "",
    grader: BaseChatModel | None = None,
    config: dict[str, Any] | None = None,
    decider: Any = None,
    audio: Any = None,
) -> CallAnalytics:
    """Grade the call and save its record. `call` holds the figures; `lines` is the transcript, which is read here and not stored.

    `decider` is the decision grader and `audio` is the call audio for it. The audio is not stored either."""
    graded = await grade(lines, visited=call.agents, summary_tries=summary_tries, profile=profile, owner=owner, model=grader, config=config, decider=decider, audio=audio)
    if graded.by:
        log.info("call %s: the model rules were checked by %s", call.call_id, graded.by)
    record = replace(call, caller_turns=sum(1 for line in lines if line.startswith("caller:")), scores=graded.scores, readings=graded.readings)
    store = AnalyticsStore(db)
    try:
        store.save(record)
    finally:
        store.close()
    return record


def forget_call(db: Path | str | sqlite3.Connection, call_id: str) -> None:
    store = AnalyticsStore(db)
    try:
        store.forget(call_id)
    finally:
        store.close()


# --- the public report ---


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _score(calls: list[CallAnalytics]) -> dict[str, Any]:
    """Passed and failed verdicts over these calls. The score is the passed share of the two; "na" does not count."""
    verdicts = [score.verdict for call in calls for score in call.scores]
    passed, failed = verdicts.count(PASS), verdicts.count(FAIL)
    return {"passed": passed, "failed": failed, "score": round(passed / (passed + failed), 2) if passed + failed else None}


def _public_call(call: CallAnalytics) -> dict[str, Any]:
    known = {rule.key for rule in RULES}
    return {
        "n": call.seq,
        "day": call.day,
        "channel": call.channel,
        "version": call.version,
        "duration_secs": int(round(call.duration_secs / DURATION_STEP_SECS) * DURATION_STEP_SECS),
        "caller_turns": call.caller_turns,
        "agents": list(call.agents),
        "failovers": call.failovers,
        "tokens_in": call.tokens_in,
        "tokens_out": call.tokens_out,
        "cost_cents": round(call.cost_usd * 100, 1),
        "answers": [{"agent": answer.agent, "provider": LEG_PLATFORMS.get(answer.leg, answer.leg), "secs": round(answer.seconds, 2)} for answer in call.answers],
        "handoff_secs": [round(secs, 2) for secs in call.handoff_secs],
        # The verdict of each rule. The note of a score never leaves the store through this report.
        "scores": {score.rule: score.verdict for score in call.scores if score.rule in known},
    }


def _public_check(check: Any) -> dict[str, Any]:
    """One check that a merged fix held (`fix_check.FixCheck`): fixed words, versions, and counts. The issue number stays private.

    `new_version` is the first version that held the fix. The `new` counts are over all `new_versions` versions that hold it."""
    return {
        "rule": check.rule,
        "verdict": check.verdict,
        "pull": check.pull,
        "fixed_version": check.fixed_version,
        "new_version": check.new_version,
        "new_versions": check.new_versions,
        "fixed": {"failed": check.fixed_failed, "graded": check.fixed_graded},
        "new": {"failed": check.new_failed, "graded": check.new_graded},
    }


def public_report(calls: list[CallAnalytics], *, tests: bool = False, fix_checks: Iterable[Any] = ()) -> dict[str, Any]:
    """What a public page may show. Test calls are left out unless `tests` is set.

    The report holds no call id, no time of day, no grader's note, and no outcome of a single call.
    `fix_checks` are the stored checks that a fix held (`fix_check.py`)."""
    shown = [call for call in calls if tests or not call.test]
    versions: dict[str, list[CallAnalytics]] = {}
    for call in shown:
        versions.setdefault(call.version, []).append(call)
    outcomes = {outcome: sum(1 for call in shown if call.outcome == outcome) for outcome in OUTCOMES} if len(shown) >= MIN_CALLS_FOR_OUTCOMES else None
    return {
        "rules": [
            {
                "key": rule.key,
                "title": rule.title,
                "checked_by": "a model that hears the call" if rule.hears else "a model" if rule.check else "code",
                **{verdict: sum(1 for call in shown for score in call.scores if score.rule == rule.key and score.verdict == verdict) for verdict in (PASS, FAIL, NA)},
            }
            for rule in RULES
        ],
        "totals": {
            "calls": len(shown),
            **_score(shown),
            "median_answer_secs": _median([answer.seconds for call in shown for answer in call.answers]),
            "outcomes": outcomes,
        },
        # One row for each version, in the order the versions first took a call: the trend that the hill climbing loop must move.
        "versions": [
            {"version": version, "calls": len(group), **_score(group), "median_answer_secs": _median([answer.seconds for call in group for answer in call.answers])}
            for version, group in versions.items()
        ],
        "calls": [_public_call(call) for call in shown],
        # One row for each merged fix of a rule: did the rule fail less often after the fix was deployed?
        "fix_checks": [_public_check(check) for check in fix_checks if check.rule in {rule.key for rule in RULES} and check.verdict in ("held", "not_held")],
    }


def dated_report(calls: list[CallAnalytics], *, tests: bool = False, fix_checks: Iterable[Any] = ()) -> dict[str, Any]:
    """The public report with the US Eastern date it was made on. This is the file that a public page reads."""
    return {"generated_on": call_day(datetime.now(UTC)), **public_report(calls, tests=tests, fix_checks=fix_checks)}


def main() -> None:
    """`uv run call-report [--tests] [--notes] [--publish]`: the public report as JSON.

    `--notes` prints each call's private scores as text. `--publish` uploads the report (`publish.py`) and prints nothing else."""
    from ai_call_center.config import get_settings

    args = sys.argv[1:]
    if "--publish" in args:
        import asyncio
        import logging

        from ai_call_center.publish import publish_report, target

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        if target(get_settings()) is None:
            sys.exit("The upload is not set up. Set the four REPORT_R2 settings in .env: see the README.")
        sys.exit(0 if asyncio.run(publish_report(get_settings())) else 1)
    store = AnalyticsStore(get_settings().leads_db)
    try:
        calls = store.calls()
    finally:
        store.close()
    if "--notes" in args:
        print("PRIVATE: the notes can say what happened on a call. Do not publish this output.")
        for call in calls:
            if call.test and "--tests" not in args:
                continue
            print(f"{call.seq:>4}  {call.day}  {call.channel:<5}  {call.version:<20}  {' -> '.join(call.agents)}: {describe_scores(call.scores)}")
        return
    from ai_call_center.fix_check import stored_checks  # fix_check reads this module, so the import waits until here

    print(json.dumps(dated_report(calls, tests="--tests" in args, fix_checks=stored_checks(get_settings().leads_db)), indent=2))


if __name__ == "__main__":
    main()
