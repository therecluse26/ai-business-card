"""After the deploy: a check that the fix of a failed rule held.

A failed rule sends a fix request (`improve.py`). The job opens a pull request, the owner merges it, and
the merge deploys. This module reads the calls after the merge, and says if the rule now fails less
often. It runs after each graded call, right after the fix requests.

The unit is a merged fix, not a harness version: the server deploys many times in a day, so one
version rarely takes enough calls. For each rule with a fix request that no check answered yet:

1. It finds the newest merged pull request of the rule (branch `improve/<rule>-<run id>`). An older
   fix whose check did not fire yet is dropped: the newer fix replaces it. One check for each fix.
2. It asks GitHub if the version of this call holds the merge commit. A table keeps each answer, so
   each (pull request, version) pair costs one compare at most.
3. The "after" calls are the graded calls at each version that holds the fix ("na" does not count).
   It waits until they reach `FIX_CHECK_CALLS`.
4. The baseline is the newest fix request of the rule at a version that does not hold the fix.
   It looks at `BASELINE_TRIES` versions at most.
5. The fix held when the fail share after it is at most half the fail share at the baseline.
6. A fix that did not hold opens one GitHub issue with the label `needs-owner`. That label starts no job.

The stored `new_version` is the first version that held the fix, by the order of calls. The "new"
counts are over all versions that hold it. The issue holds only fixed words, commits, counts, and the
link of the pull request. No word of a call. The counts include test calls, as a fix request's do.

The GitHub requests use the token of the fix request. Listing pull requests needs "Pull requests" (read),
the compare needs "Contents" (read), and the issue needs "Issues" (write).
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_call_center.analytics import AnalyticsStore, CallAnalytics, call_day
from ai_call_center.config import Settings
from ai_call_center.feedback import _call, _request
from ai_call_center.grading import FAIL, NA, rule_of
from ai_call_center.improve import COMMIT, RequestStore, Target, target
from ai_call_center.issue_queue import NEEDS_OWNER

log = logging.getLogger(__name__)

HELD, NOT_HELD = "held", "not_held"
VERDICTS = (HELD, NOT_HELD)
BASELINE_TRIES = 3  # versions of fix requests that the baseline looks at; each can cost one compare
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
IN_VERSION = ("ahead", "identical")  # the compare status when the version holds the merge commit
NOTE = (
    "The server wrote this issue from counts of the grader's verdicts. It holds no word of a call. "
    "The label `needs-owner` starts no job. The label `build` starts the build job for this issue."
)

Call = Callable[[Any], Any]


@dataclass(frozen=True)
class FixCheck:
    """One check of one merged fix of one rule. Each field is a fixed word, a commit, a count, or a date."""

    rule: str
    fixed_version: str  # the version of the fix request: the calls before the fix
    new_version: str  # the first version that held the fix, by the order of calls
    fixed_failed: int
    fixed_graded: int
    new_failed: int  # over all versions that hold the fix
    new_graded: int
    verdict: str  # HELD or NOT_HELD
    day: str  # the US Eastern date of the check
    pull: int = 0  # the number of the merged pull request
    issue: int = 0  # the issue that a NOT_HELD verdict opened; 0 = none
    new_versions: int = 1  # the versions that hold the fix and took a call


@dataclass(frozen=True)
class Pull:
    """A merged pull request of a fix job, as far as this check needs it."""

    number: int
    merge_commit: str


class CheckStore:
    """The checks, one row for each (rule, pull request), and the answers of the compare. It takes a path, or the open connection of another store."""

    def __init__(self, db: Path | str | sqlite3.Connection) -> None:
        self._own = not isinstance(db, sqlite3.Connection)
        self._conn = db if isinstance(db, sqlite3.Connection) else sqlite3.connect(str(db), check_same_thread=False)
        RequestStore(self._conn)  # makes the table `fix_requests` on a new database, so the reads below work
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(fix_checks)")}
        if columns and "request_seq" not in columns:
            self._conn.execute("DROP TABLE fix_checks")  # the first form of the table, one row for each version; no build with it shipped
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fix_checks (
                rule TEXT NOT NULL,
                pull INTEGER NOT NULL,
                fixed_version TEXT NOT NULL,
                new_version TEXT NOT NULL,
                new_versions INTEGER NOT NULL,
                fixed_failed INTEGER NOT NULL,
                fixed_graded INTEGER NOT NULL,
                new_failed INTEGER NOT NULL,
                new_graded INTEGER NOT NULL,
                verdict TEXT NOT NULL,
                day TEXT NOT NULL,
                issue INTEGER NOT NULL DEFAULT 0,
                request_seq INTEGER NOT NULL,
                PRIMARY KEY (rule, pull)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fix_holds (
                pull INTEGER NOT NULL,
                version TEXT NOT NULL,
                holds INTEGER NOT NULL,
                PRIMARY KEY (pull, version)
            )
            """
        )
        self._conn.commit()

    def open_rules(self) -> list[str]:
        """The rules with a fix request newer than the baseline of their last check. Such a request waits for a fix."""
        rows = self._conn.execute(
            "SELECT rule FROM fix_requests GROUP BY rule"
            " HAVING MAX(rowid) > COALESCE((SELECT MAX(request_seq) FROM fix_checks WHERE fix_checks.rule = fix_requests.rule), 0)"
        )
        return [row[0] for row in rows]

    def request_versions(self, rule: str) -> list[tuple[str, int]]:
        """(version, row number of its newest fix request) for the rule's fix requests, newest first, each version one time."""
        rows = self._conn.execute("SELECT version, MAX(rowid) AS seq FROM fix_requests WHERE rule = ? GROUP BY version ORDER BY seq DESC", (rule,))
        return [(version, seq) for version, seq in rows]

    def checked(self, rule: str, pull: int) -> bool:
        return self._conn.execute("SELECT 1 FROM fix_checks WHERE rule = ? AND pull = ?", (rule, pull)).fetchone() is not None

    def known_hold(self, pull: int, version: str) -> bool | None:
        row = self._conn.execute("SELECT holds FROM fix_holds WHERE pull = ? AND version = ?", (pull, version)).fetchone()
        return None if row is None else bool(row[0])

    def keep_hold(self, pull: int, version: str, holds: bool) -> None:
        self._conn.execute("INSERT OR REPLACE INTO fix_holds (pull, version, holds) VALUES (?, ?, ?)", (pull, version, int(holds)))
        self._conn.commit()

    def held_versions(self, pull: int) -> set[str]:
        return {row[0] for row in self._conn.execute("SELECT version FROM fix_holds WHERE pull = ? AND holds = 1", (pull,))}

    def add(self, check: FixCheck, request_seq: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO fix_checks (rule, pull, fixed_version, new_version, new_versions, fixed_failed, fixed_graded, new_failed, new_graded, verdict, day, issue, request_seq)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *(check.rule, check.pull, check.fixed_version, check.new_version, check.new_versions, check.fixed_failed, check.fixed_graded),
                *(check.new_failed, check.new_graded, check.verdict, check.day, check.issue, request_seq),
            ),
        )
        self._conn.commit()

    def checks(self) -> list[FixCheck]:
        """Every stored check, oldest first."""
        rows = self._conn.execute(
            "SELECT rule, fixed_version, new_version, fixed_failed, fixed_graded, new_failed, new_graded, verdict, day, pull, issue, new_versions FROM fix_checks ORDER BY rowid"
        ).fetchall()
        return [FixCheck(*row) for row in rows]

    def close(self) -> None:
        if self._own:
            self._conn.close()


def stored_checks(db: Path | str | sqlite3.Connection) -> list[FixCheck]:
    """The stored checks, for the report."""
    store = CheckStore(db)
    try:
        return store.checks()
    finally:
        store.close()


def counts(calls: list[CallAnalytics], rule: str, versions: set[str]) -> tuple[int, int]:
    """(failed, graded) calls at these versions for this rule. "na" is not a graded call."""
    verdicts = [score.verdict for call in calls if call.version in versions for score in call.scores if score.rule == rule and score.verdict != NA]
    return verdicts.count(FAIL), len(verdicts)


def held(fixed_failed: int, fixed_graded: int, new_failed: int, new_graded: int) -> bool:
    """True when the fail share after the fix is at most half the share before. Integers only, so no rounding decides."""
    return 2 * new_failed * fixed_graded <= fixed_failed * new_graded


# --- GitHub ---


def merged_pulls(to: Target, call: Call = _call) -> list[dict[str, Any]]:
    """The closed pull requests that were merged, newest change first. One request serves each rule of a call."""
    rows = call(_request(to, "GET", "/pulls?state=closed&sort=updated&direction=desc&per_page=100"))
    return [row for row in rows if isinstance(row, dict) and row.get("merged_at")] if isinstance(rows, list) else []


def merged_fix(to: Target, rule: str, pulls: list[dict[str, Any]]) -> Pull | None:
    """The newest merged pull request of a fix job for this rule, or None."""
    branch = re.compile(rf"^improve/{re.escape(rule)}-\d+$")  # improve.yml: improve/<rule>-<run id>
    found: list[tuple[str, Pull]] = []
    for row in pulls:
        head = row.get("head") if isinstance(row.get("head"), dict) else {}
        repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
        number, commit, merged = row.get("number"), row.get("merge_commit_sha"), row.get("merged_at")
        if not (branch.match(str(head.get("ref") or "")) and str(repo.get("full_name") or "").casefold() == to.repository.casefold()):
            continue  # a branch of another repository can have the same name
        if isinstance(number, int) and not isinstance(number, bool) and isinstance(commit, str) and FULL_COMMIT.match(commit) and isinstance(merged, str):
            found.append((merged, Pull(number=number, merge_commit=commit)))
    return max(found, key=lambda pair: pair[0])[1] if found else None  # GitHub's ISO times in UTC sort as text


def holds(to: Target, merge_commit: str, version: str, call: Call = _call) -> bool:
    """True when the harness version holds the merge commit.

    The version is a short commit (`git rev-parse --short=12`). A merge commit that starts with it is the
    same commit, and needs no request. Else GitHub's compare takes the short commit as it is."""
    if merge_commit.startswith(version):
        return True
    found = call(_request(to, "GET", f"/compare/{merge_commit}...{urllib.parse.quote(version)}?per_page=1"))
    return isinstance(found, dict) and found.get("status") in IN_VERSION


async def _holds(store: CheckStore, to: Target, pull: Pull, version: str, call: Call) -> bool:
    """`holds`, asked one time for each (pull request, version). A commit does not change, so the answer stays true."""
    known = store.known_hold(pull.number, version)
    if known is None:
        known = await asyncio.to_thread(holds, to, pull.merge_commit, version, call)
        store.keep_hold(pull.number, version, known)
    return known


def issue_text(check: FixCheck, *, repository: str) -> tuple[str, str]:
    """(title, body) of the issue for a fix that did not hold. Fixed words, commits, counts, and a link only."""
    rule = rule_of(check.rule)
    after = f"harness version `{check.new_version}`" + (f" and {check.new_versions - 1} later versions" if check.new_versions > 1 else "")
    title = f"The fix for the rule `{check.rule}` did not hold"
    lines = [
        f"The grader checked the rule `{check.rule}` after the fix was deployed. The rule fails about as often as before.",
        "",
        f"- Rule: `{check.rule}`" + (f": {rule.title}." if rule else ""),
        f"- The fix: https://github.com/{repository}/pull/{check.pull}",
        f"- Before the fix: harness version `{check.fixed_version}`, {check.fixed_failed} of {check.fixed_graded} graded calls failed the rule.",
        f"- After the fix: {after}, {check.new_failed} of {check.new_graded} graded calls failed the rule.",
        "- A fix holds when the fail share after it is at most half the fail share before it.",
        "",
        "---",
        NOTE,
    ]
    return title, "\n".join(lines) + "\n"


def open_issue(to: Target, check: FixCheck, call: Call = _call) -> int:
    """Open the issue for a fix that did not hold. It gives the issue number, or 0 when GitHub gave none."""
    title, body = issue_text(check, repository=to.repository)
    made = call(_request(to, "POST", "/issues", {"title": title, "body": body, "labels": [NEEDS_OWNER]}))
    number = made.get("number") if isinstance(made, dict) else None
    labels = (made.get("labels") or []) if isinstance(made, dict) else []
    if NEEDS_OWNER not in {label.get("name") for label in labels if isinstance(label, dict)}:
        # Without the label the queue would not start a job either: an issue of the owner with no label waits. The owner sorts it.
        log.warning("issue #%s has no label %s: make the label in the repository", number, NEEDS_OWNER)
    return number if isinstance(number, int) and not isinstance(number, bool) else 0


# --- after a call ---


async def _check_rule(store: CheckStore, to: Target, rule: str, pull: Pull, *, version: str, calls: list[CallAnalytics], limit: int, day: str, call: Call) -> tuple[FixCheck, int] | None:
    """The check of one merged fix and the row of its baseline fix request, or None when it cannot fire yet. A GitHub error raises."""
    if not await _holds(store, to, pull, version, call):
        return None  # a version before the merge: the count after the fix did not change
    after = store.held_versions(pull.number)
    new_failed, new_graded = counts(calls, rule, after)
    if new_graded < limit:
        return None
    # The baseline must not hold the fix. A fix request at a version that holds it is a failure after the fix.
    fixed_version, request_seq = "", 0
    for asked, seq in store.request_versions(rule)[:BASELINE_TRIES]:
        if not await _holds(store, to, pull, asked, call):
            fixed_version, request_seq = asked, seq
            break
    fixed_failed, fixed_graded = counts(calls, rule, {fixed_version}) if fixed_version else (0, 0)
    if not fixed_failed:
        return None  # no baseline, or the failed calls of the baseline were forgotten
    first = next(made.version for made in calls if made.version in after)
    check = FixCheck(
        rule=rule,
        fixed_version=fixed_version,
        new_version=first,
        fixed_failed=fixed_failed,
        fixed_graded=fixed_graded,
        new_failed=new_failed,
        new_graded=new_graded,
        verdict=HELD if held(fixed_failed, fixed_graded, new_failed, new_graded) else NOT_HELD,
        day=day,
        pull=pull.number,
        new_versions=len({made.version for made in calls if made.version in after}),
    )
    if check.verdict == NOT_HELD:
        check = replace(check, issue=await asyncio.to_thread(open_issue, to, check, call))
    # A fix request after the baseline stays open: it is a failure after this fix, and waits for a fix of its own.
    return check, request_seq


async def check_fixes(
    settings: Settings,
    db: Path | str | sqlite3.Connection,
    record: CallAnalytics,
    *,
    call: Call = _call,
    now: datetime | None = None,
) -> list[FixCheck]:
    """Check each rule whose newest merged fix has enough calls after it. It gives the checks it stored.

    It never raises: the end of a call must not fail because of a check. A GitHub error is one log line,
    and no check is stored, so a later call checks again."""
    to = target(settings)
    if to is None or not settings.fix_check or not COMMIT.match(record.version):
        return []
    version, limit = record.version, max(1, settings.fix_check_calls)
    stored: list[FixCheck] = []
    try:
        analytics = AnalyticsStore(db)
        store = CheckStore(db)
        try:
            rules = [rule for rule in store.open_rules() if rule_of(rule) is not None]
            if not rules:
                return []  # no fix request waits for a check: GitHub is not asked
            calls, day = analytics.calls(), call_day(now or datetime.now(UTC))
            pulls: list[dict[str, Any]] | None = None
            for rule in rules:
                try:
                    if pulls is None:
                        pulls = await asyncio.to_thread(merged_pulls, to, call)
                    pull = merged_fix(to, rule, pulls)
                    if pull is None or store.checked(rule, pull.number):
                        continue  # no merged fix yet, or this fix has its check; a newer fix gets a check of its own
                    found = await _check_rule(store, to, rule, pull, version=version, calls=calls, limit=limit, day=day, call=call)
                except Exception as error:
                    # The error of urllib holds the status and its reason. It does not hold the token. HTTP 403 = a permission is missing.
                    log.warning("fix check for %s: GitHub did not answer, a later call checks again: %s: %s", rule, type(error).__name__, error)
                    continue
                if found is None:
                    continue
                check, request_seq = found
                store.add(check, request_seq)
                stored.append(check)
                log.info(
                    "fix check for %s: %s (pull request #%s); failed %d of %d at %s, %d of %d at %d versions from %s%s",
                    *(rule, check.verdict, check.pull, check.fixed_failed, check.fixed_graded, check.fixed_version),
                    *(check.new_failed, check.new_graded, check.new_versions, check.new_version),
                    f"; issue #{check.issue}" if check.issue else "",
                )
        finally:
            store.close()
            analytics.close()
    except Exception:
        log.exception("could not check the fixes at version %s", version)
    return stored
