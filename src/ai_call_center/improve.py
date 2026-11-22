"""A fix request: when a call fails a rule, the server starts the job on GitHub that proposes a fix.

This is the last part of the improvement loop. The grader scores each call after hangup
(`grading.py`). For each failed rule, this module starts the workflow `.github/workflows/improve.yml`
in the repository. There a coding agent changes one prompt or one word list, a fixed scorer measures
the change (`score.py`), and the job opens a pull request with the evidence. Nobody starts the job by hand.

What leaves the server is a fix request: the rule, the agent at fault, the cause code, the harness
version, and two counts. Each of the three names is a word from a fixed list in `grading.py`, and the
code here checks that. So a fix request cannot hold what a caller said, and it cannot hold an
instruction that a caller planted. The repository is public; a fix request is safe to show there.

Limits on the cost:

- One fix request for each (rule, agent, cause) at one harness version. The same failure on the next
  call starts no second job. After a fix is deployed the version is new, so the failure can ask again.
- At most IMPROVE_DAILY_LIMIT fix requests in one day (US Eastern).
- Only a version that is a clean git commit asks: a laptop with changed files does not.

The start of the job is one POST to GitHub's REST API ("create a workflow dispatch event"), written
with the standard library. The token needs one permission, on this repository only: "Actions" (write).
With that token nobody can change code. The three IMPROVE settings turn this on; with the token or
the repository empty, nothing is sent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ai_call_center.analytics import AnalyticsStore, CallAnalytics, call_day
from ai_call_center.config import Settings
from ai_call_center.grading import AGENTS, FAIL, NA, OTHER, rule_of

log = logging.getLogger(__name__)

API = "https://api.github.com"
SEND_TIME_LIMIT = 10.0  # seconds; the call is over, so nobody waits for this
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
WORKFLOW = re.compile(r"^[A-Za-z0-9_.-]+\.ya?ml$")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")  # a clean version; "-dirty" and "unknown" do not match


@dataclass(frozen=True)
class FixRequest:
    """One failed rule, as it leaves the server. Every field is a word from a fixed list, a commit, or a count."""

    rule: str
    agent: str  # "" = the grader did not name one
    cause: str
    version: str
    failed_calls: int = 1  # graded calls at this version that failed the rule
    graded_calls: int = 1  # graded calls at this version where the rule applied

    def inputs(self) -> dict[str, str]:
        """The inputs of the workflow. GitHub wants each one as a string."""
        return {
            "rule": self.rule,
            "agent": self.agent,
            "cause": self.cause,
            "version": self.version,
            "failed_calls": str(self.failed_calls),
            "graded_calls": str(self.graded_calls),
        }


@dataclass(frozen=True)
class Target:
    repository: str  # "owner/name"
    workflow: str  # the file name of the workflow
    ref: str  # the branch that holds the workflow
    token: str

    @property
    def url(self) -> str:
        return f"{API}/repos/{self.repository}/actions/workflows/{self.workflow}/dispatches"


def target(settings: Settings) -> Target | None:
    """Where a fix request goes, or None when it is not set up or a value has a wrong form."""
    made = Target(
        repository=settings.improve_repo.strip(),
        workflow=settings.improve_workflow.strip(),
        ref=settings.improve_ref.strip(),
        token=settings.improve_github_token.get_secret_value().strip(),
    )
    if not (made.token and made.repository):
        return None
    if not (REPOSITORY.match(made.repository) and WORKFLOW.match(made.workflow) and made.ref):
        log.warning("IMPROVE_REPO must be owner/name and IMPROVE_WORKFLOW a .yml file name; no fix request is sent")
        return None
    return made


class RequestStore:
    """One row for each fix request that GitHub accepted. It takes a path, or the open connection of another store."""

    def __init__(self, db: Path | str | sqlite3.Connection) -> None:
        self._own = not isinstance(db, sqlite3.Connection)
        self._conn = db if isinstance(db, sqlite3.Connection) else sqlite3.connect(str(db), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fix_requests (
                rule TEXT NOT NULL,
                agent TEXT NOT NULL,
                cause TEXT NOT NULL,
                version TEXT NOT NULL,
                day TEXT NOT NULL,
                PRIMARY KEY (rule, agent, cause, version)
            )
            """
        )
        self._conn.commit()

    def asked(self, request: FixRequest) -> bool:
        row = self._conn.execute("SELECT 1 FROM fix_requests WHERE rule = ? AND agent = ? AND cause = ? AND version = ?", (request.rule, request.agent, request.cause, request.version)).fetchone()
        return row is not None

    def asked_on(self, day: str) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM fix_requests WHERE day = ?", (day,)).fetchone()[0]

    def add(self, request: FixRequest, day: str) -> None:
        self._conn.execute("INSERT OR IGNORE INTO fix_requests (rule, agent, cause, version, day) VALUES (?, ?, ?, ?, ?)", (request.rule, request.agent, request.cause, request.version, day))
        self._conn.commit()

    def close(self) -> None:
        if self._own:
            self._conn.close()


def fix_requests(record: CallAnalytics, calls: list[CallAnalytics]) -> list[FixRequest]:
    """One fix request for each rule that this call failed. `calls` are the stored records, for the two counts.

    A name that is not in its fixed list is not sent: the agent becomes "", the cause becomes "other",
    and a rule that the code does not know gives no fix request. A rule with `fix_request=False` gives none
    either: no prompt and no word list can repair it."""
    if not COMMIT.match(record.version):
        return []
    made = []
    for score in record.scores:
        rule = rule_of(score.rule)
        if score.verdict != FAIL or rule is None or not rule.fix_request:
            continue
        same_version = [found.verdict for call in calls if call.version == record.version for found in call.scores if found.rule == rule.key and found.verdict != NA]
        made.append(
            FixRequest(
                rule=rule.key,
                agent=score.agent if score.agent in AGENTS else "",
                cause=score.cause if score.cause in {code for code, _ in rule.causes} else OTHER,
                version=record.version,
                failed_calls=max(1, same_version.count(FAIL)),
                graded_calls=max(1, len(same_version)),
            )
        )
    return made


def dispatch_request(to: Target, request: FixRequest) -> urllib.request.Request:
    """The request that starts the workflow with the inputs of one fix request."""
    body = json.dumps({"ref": to.ref, "inputs": request.inputs()}).encode()
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {to.token}",
        "Content-Type": "application/json",
        "User-Agent": "voice-card-fix-request",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    return urllib.request.Request(to.url, data=body, method="POST", headers=headers)


def _send(request: urllib.request.Request) -> int:
    with urllib.request.urlopen(request, timeout=SEND_TIME_LIMIT) as response:  # noqa: S310 - the URL is always https://api.github.com, made in Target.url
        return response.status


async def request_fixes(
    settings: Settings,
    db: Path | str | sqlite3.Connection,
    record: CallAnalytics,
    *,
    send: Callable[[urllib.request.Request], int] = _send,
    now: datetime | None = None,
) -> list[FixRequest]:
    """Start one job for each new failure of this call. It gives the fix requests that GitHub accepted.

    It never raises: the end of a call must not fail because a job did not start. A failure is one log
    line, and the fix request is not stored, so the same failure on a later call asks again."""
    to = target(settings)
    if to is None:
        return []
    sent: list[FixRequest] = []
    try:
        analytics = AnalyticsStore(db)
        store = RequestStore(db)
        try:
            day = call_day(now or datetime.now(UTC))
            for request in fix_requests(record, analytics.calls()):
                if store.asked(request):
                    continue
                if store.asked_on(day) >= settings.improve_daily_limit:
                    log.info("fix request for %s not sent: the limit of %d for one day is reached", request.rule, settings.improve_daily_limit)
                    break
                try:
                    status = await asyncio.to_thread(send, dispatch_request(to, request))
                except Exception as error:
                    # The error of urllib holds the status and its reason. It does not hold the token.
                    log.warning("fix request for %s did not start a job in %s: %s", request.rule, to.repository, error)
                    continue
                store.add(request, day)
                sent.append(request)
                log.info("fix request sent to %s (HTTP %s): rule %s, agent %s, cause %s, version %s", to.repository, status, request.rule, request.agent or "-", request.cause, request.version)
        finally:
            store.close()
            analytics.close()
    except Exception:
        log.exception("could not make the fix requests of call %s", record.call_id)
    return sent
