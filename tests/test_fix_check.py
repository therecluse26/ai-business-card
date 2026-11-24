"""After the deploy: the check that a merged fix of a failed rule held. GitHub is a fake; no request leaves the test."""

from __future__ import annotations

import json
import urllib.error
from datetime import UTC, datetime

from pydantic import SecretStr

from ai_call_center.analytics import AnalyticsStore, CallAnalytics, public_report
from ai_call_center.config import Settings
from ai_call_center.fix_check import HELD, NOT_HELD, CheckStore, FixCheck, check_fixes, held, issue_text, stored_checks
from ai_call_center.grading import FAIL, NA, PASS, Score
from ai_call_center.improve import FixRequest, RequestStore
from ai_call_center.issue_queue import QueueIssue, plan

RULE = "name_asked_once"
OLD = "6e77d417184e"  # the version of the fix request
NEW = "a1b2c3d4e5f6"  # a version after the merge
LATER = ("b1b2c3d4e5f6", "c1b2c3d4e5f6")  # two more deploys after the merge
MERGE = "f" * 40  # the merge commit of the fix
REPO = "someone/voice-card"
NOW = datetime(2026, 11, 23, 15, 0, tzinfo=UTC)
SECRET_NOTE = "The caller Zebulon asked about the Quixote project."


def settings(**over) -> Settings:
    base = {"improve_github_token": SecretStr("test-token"), "improve_repo": REPO}
    return Settings(_env_file=None, **{**base, **over})


def call(call_id: str, version: str, verdict: str, rule: str = RULE) -> CallAnalytics:
    return CallAnalytics(
        call_id=call_id, day="2026-11-22", channel="voice", version=version, test=False, duration_secs=60, agents=["classifier", "contact"], outcome="none",
        scores=[Score(rule, verdict, SECRET_NOTE, agent="contact", cause="asked_again_after_handoff" if verdict == FAIL else "")],
    )  # fmt: skip


def pull(number: int = 7, *, rule: str = RULE, merged: str | None = "2026-11-22T10:00:00Z", repo: str = REPO, commit: str = MERGE) -> dict:
    return {"number": number, "head": {"ref": f"improve/{rule}-123456", "repo": {"full_name": repo}}, "merged_at": merged, "created_at": "2026-11-21T18:00:00Z", "merge_commit_sha": commit}


class GitHub:
    """A GitHub in memory: the closed pull requests, which version holds which merge commit, and the issues it opened."""

    def __init__(self, pulls: list[dict] | None = None, *, holds: dict[str, set[str]] | None = None, fail_on: str = "") -> None:
        self.pulls = [pull()] if pulls is None else pulls
        self.holds = {MERGE: {NEW, *LATER}} if holds is None else holds
        self.fail_on = fail_on
        self.requests: list = []
        self.issues: list[dict] = []

    def compares(self, version: str) -> int:
        return sum(1 for request in self.requests if "/compare/" in request.full_url and f"...{version}?" in request.full_url)

    def __call__(self, request):
        self.requests.append(request)
        path = request.full_url.removeprefix(f"https://api.github.com/repos/{REPO}")
        if self.fail_on and path.startswith(self.fail_on):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", None, None)  # type: ignore[arg-type]
        if request.get_method() == "GET" and path.startswith("/pulls?"):
            return self.pulls
        if request.get_method() == "GET" and path.startswith("/compare/"):
            base, head = path.removeprefix("/compare/").split("?")[0].split("...")
            return {"status": "ahead" if head in self.holds.get(base, set()) else "behind"}
        if request.get_method() == "POST" and path == "/issues":
            self.issues.append(json.loads(request.data))
            return {"number": 40 + len(self.issues), "labels": [{"name": name} for name in self.issues[-1]["labels"]]}
        raise AssertionError(f"no fake for {request.get_method()} {path}")


def save(db, *records: CallAnalytics) -> CallAnalytics:
    store = AnalyticsStore(db)
    for record in records:
        store.save(record)
    store.close()
    return records[-1] if records else None  # type: ignore[return-value]


def ask(db, version: str, rule: str = RULE) -> None:
    requests = RequestStore(db)
    requests.add(FixRequest(rule=rule, agent="contact", cause="asked_again_after_handoff", version=version), "2026-11-21")
    requests.close()


def seed(db, *, before: list[str], after: list[str], version: str = NEW) -> CallAnalytics:
    """The calls before the fix and after it, and one fix request at the old version. It gives the last call."""
    save(db, *(call(f"old-{n}", OLD, verdict) for n, verdict in enumerate(before)))
    ask(db, OLD)
    return save(db, *(call(f"new-{version}-{n}", version, verdict) for n, verdict in enumerate(after)))


async def test_with_too_few_calls_after_the_fix_nothing_is_stored(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = seed(db, before=[FAIL, FAIL], after=[PASS, PASS, PASS, NA, PASS, NA])  # 4 graded calls; "na" does not count
    github = GitHub()
    assert await check_fixes(settings(), db, last, call=github, now=NOW) == []
    assert stored_checks(db) == [] and github.compares(OLD) == 0  # the baseline is asked only when the count is reached


async def test_with_no_fix_request_waiting_github_is_not_asked(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = save(db, *(call(f"c{n}", NEW, PASS) for n in range(5)))
    github = GitHub()
    assert await check_fixes(settings(), db, last, call=github, now=NOW) == [] and github.requests == []


async def test_with_no_merged_fix_for_the_rule_nothing_is_stored(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = seed(db, before=[FAIL, FAIL], after=[PASS] * 5)
    pulls = [pull(1, merged=None), pull(2, rule="handoff_intro"), pull(3, repo="stranger/voice-card"), {**pull(4), "head": {"ref": f"improve/issue-{RULE}-9", "repo": {"full_name": REPO}}}]
    github = GitHub(pulls)
    assert await check_fixes(settings(), db, last, call=github, now=NOW) == []
    assert len(github.requests) == 1 and stored_checks(db) == []  # one list of pull requests, no compare


async def test_a_version_before_the_merge_does_not_count(tmp_path):
    db = tmp_path / "leads.sqlite3"
    seed(db, before=[FAIL, FAIL], after=[PASS] * 3, version="0123456789ab")  # deployed before the merge
    github = GitHub()
    assert await check_fixes(settings(), db, call("x", "0123456789ab", PASS), call=github, now=NOW) == []
    for n in range(4):
        last = save(db, call(f"after-{n}", NEW, PASS))
        assert await check_fixes(settings(), db, last, call=github, now=NOW) == []  # 4 calls after the merge, not 7
    last = save(db, call("after-4", NEW, PASS))
    [check] = await check_fixes(settings(), db, last, call=github, now=NOW)
    assert (check.new_version, check.new_graded, check.new_versions) == (NEW, 5, 1)


async def test_calls_at_three_versions_that_hold_the_fix_reach_the_count_together(tmp_path):
    db = tmp_path / "leads.sqlite3"
    save(db, *(call(f"old-{n}", OLD, FAIL) for n in range(2)))
    ask(db, OLD)
    github = GitHub()
    stored = []
    for n, version in enumerate([NEW, NEW, LATER[0], LATER[0], LATER[1]]):
        last = save(db, call(f"after-{n}", version, PASS))
        stored += await check_fixes(settings(), db, last, call=github, now=NOW)
    [check] = stored
    assert check == FixCheck(rule=RULE, fixed_version=OLD, new_version=NEW, fixed_failed=2, fixed_graded=2, new_failed=0, new_graded=5, verdict=HELD, day="2026-11-23", pull=7, new_versions=3)
    assert stored_checks(db) == [check]


async def test_each_pull_request_and_version_asks_compare_one_time(tmp_path):
    db = tmp_path / "leads.sqlite3"
    seed(db, before=[FAIL, FAIL], after=[])
    github = GitHub()
    for n in range(4):
        last = save(db, call(f"after-{n}", NEW, PASS))
        await check_fixes(settings(), db, last, call=github, now=NOW)
    assert github.compares(NEW) == 1 and github.compares(OLD) == 0
    last = save(db, call("after-4", NEW, PASS))
    assert await check_fixes(settings(), db, last, call=github, now=NOW)
    assert github.compares(NEW) == 1 and github.compares(OLD) == 1


async def test_a_fix_held_when_the_rule_fails_at_most_half_as_often(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = seed(db, before=[FAIL, FAIL, PASS], after=[PASS, PASS, FAIL, PASS, PASS])  # 2 of 3 before, 1 of 5 after
    github = GitHub()
    [check] = await check_fixes(settings(), db, last, call=github, now=NOW)
    assert (check.verdict, check.fixed_failed, check.fixed_graded, check.new_failed, check.new_graded) == (HELD, 2, 3, 1, 5)
    assert github.issues == []
    assert held(2, 4, 1, 4) and not held(2, 4, 2, 5)  # exactly half holds; a little more does not


async def test_a_merge_commit_that_starts_with_the_short_version_needs_no_compare(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = seed(db, before=[FAIL], after=[PASS] * 5)
    github = GitHub([pull(commit=NEW + "0" * 28)], holds={})
    [check] = await check_fixes(settings(), db, last, call=github, now=NOW)
    assert check.verdict == HELD and github.compares(NEW) == 0 and github.compares(OLD) == 1


async def test_a_newer_merged_fix_replaces_an_older_check_that_did_not_fire(tmp_path):
    db = tmp_path / "leads.sqlite3"
    newer = "e" * 40
    last = seed(db, before=[FAIL, FAIL], after=[PASS] * 3)
    github = GitHub([pull(7), pull(8, merged="2026-11-22T20:00:00Z", commit=newer)], holds={MERGE: {NEW, LATER[0]}, newer: {LATER[0]}})
    assert await check_fixes(settings(), db, last, call=github, now=NOW) == []
    last = save(db, *(call(f"later-{n}", LATER[0], PASS) for n in range(5)))
    [check] = await check_fixes(settings(), db, last, call=github, now=NOW)
    assert (check.pull, check.new_version, check.new_graded) == (8, LATER[0], 5)  # the 3 calls at NEW hold only the older fix
    assert [found.pull for found in stored_checks(db)] == [8]


async def test_the_baseline_skips_a_fix_request_at_a_version_that_holds_the_fix(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = seed(db, before=[FAIL, FAIL], after=[PASS, PASS, PASS, PASS, FAIL])
    ask(db, NEW)  # the rule failed after the fix too, and sent a fix request at the new version
    github = GitHub()
    [check] = await check_fixes(settings(), db, last, call=github, now=NOW)
    assert (check.fixed_version, check.fixed_failed, check.fixed_graded) == (OLD, 2, 2)


async def test_a_fix_request_after_the_fix_waits_for_a_fix_of_its_own(tmp_path):
    db = tmp_path / "leads.sqlite3"
    newer = "e" * 40
    last = seed(db, before=[FAIL, FAIL], after=[PASS, PASS, FAIL, FAIL, FAIL])
    ask(db, NEW)
    github = GitHub(holds={MERGE: {NEW, LATER[0]}, newer: {LATER[0]}})
    [first] = await check_fixes(settings(), db, last, call=github, now=NOW)
    github.pulls = [pull(7), pull(8, merged="2026-11-22T20:00:00Z", commit=newer)]
    last = save(db, *(call(f"later-{n}", LATER[0], PASS) for n in range(5)))
    [second] = await check_fixes(settings(), db, last, call=github, now=NOW)
    assert (first.pull, first.verdict, second.pull, second.fixed_version, second.verdict) == (7, NOT_HELD, 8, NEW, HELD)


async def test_a_fix_that_did_not_hold_opens_exactly_one_issue_with_no_word_of_a_call(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = seed(db, before=[FAIL, FAIL, PASS], after=[FAIL, PASS, FAIL, PASS, FAIL])  # 2 of 3 before, 3 of 5 after
    github = GitHub()
    [check] = await check_fixes(settings(), db, last, call=github, now=NOW)
    assert (check.verdict, check.issue, check.new_failed, check.new_graded) == (NOT_HELD, 41, 3, 5)
    [issue] = github.issues
    assert issue["labels"] == ["needs-owner"]
    assert issue["title"] == f"The fix for the rule `{RULE}` did not hold"
    assert f"https://github.com/{REPO}/pull/7" in issue["body"] and OLD in issue["body"] and NEW in issue["body"]
    assert "3 of 5 graded calls" in issue["body"] and "2 of 3 graded calls" in issue["body"]
    assert "Zebulon" not in json.dumps(issue) and "Quixote" not in json.dumps(issue)

    # A second call makes no second check, and asks GitHub nothing: no fix request waits.
    asked = len(github.requests)
    last = save(db, call("new-later", NEW, FAIL))
    assert await check_fixes(settings(), db, last, call=github, now=NOW) == []
    assert len(github.requests) == asked and len(github.issues) == 1 and len(stored_checks(db)) == 1


async def test_a_github_error_stores_nothing_and_raises_nothing(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = seed(db, before=[FAIL, FAIL], after=[FAIL] * 5)
    for path in ("/pulls", "/compare/", "/issues"):
        github = GitHub(fail_on=path)
        assert await check_fixes(settings(), db, last, call=github, now=NOW) == []
        assert stored_checks(db) == []
    github = GitHub()
    assert [check.verdict for check in await check_fixes(settings(), db, last, call=github, now=NOW)] == [NOT_HELD]


async def test_a_version_that_is_not_a_clean_commit_does_nothing(tmp_path):
    db = tmp_path / "leads.sqlite3"
    seed(db, before=[FAIL, FAIL], after=[PASS] * 5)
    for version in (f"{NEW}-dirty", "unknown", ""):
        github = GitHub()
        assert await check_fixes(settings(), db, call("x", version, PASS), call=github, now=NOW) == []
        assert github.requests == []
    assert stored_checks(db) == []


async def test_it_is_off_with_no_token_no_repository_or_fix_check_false(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = seed(db, before=[FAIL, FAIL], after=[PASS] * 5)
    for made in (settings(improve_github_token=SecretStr("")), settings(improve_repo=""), settings(fix_check=False)):
        github = GitHub()
        assert await check_fixes(made, db, last, call=github, now=NOW) == [] and github.requests == []


async def test_the_limit_of_calls_is_a_setting(tmp_path):
    db = tmp_path / "leads.sqlite3"
    last = seed(db, before=[FAIL], after=[PASS] * 2)
    assert await check_fixes(settings(), db, last, call=GitHub(), now=NOW) == []
    assert [check.verdict for check in await check_fixes(settings(fix_check_calls=2), db, last, call=GitHub(), now=NOW)] == [HELD]


def test_the_public_report_shows_each_check_with_fixed_words_and_counts_only(tmp_path):
    db = tmp_path / "leads.sqlite3"
    store = CheckStore(db)
    store.add(FixCheck(rule=RULE, fixed_version=OLD, new_version=NEW, fixed_failed=2, fixed_graded=3, new_failed=3, new_graded=5, verdict=NOT_HELD, day="2026-11-23", pull=7, issue=41, new_versions=2), 1)
    store.add(FixCheck(rule="an_old_rule", fixed_version=OLD, new_version=NEW, fixed_failed=1, fixed_graded=1, new_failed=0, new_graded=5, verdict=HELD, day="2026-11-23", pull=8), 2)
    store.close()
    report = public_report([], fix_checks=stored_checks(db))
    assert report["fix_checks"] == [
        {"rule": RULE, "verdict": NOT_HELD, "pull": 7, "fixed_version": OLD, "new_version": NEW, "new_versions": 2, "fixed": {"failed": 2, "graded": 3}, "new": {"failed": 3, "graded": 5}}
    ]
    assert public_report([])["fix_checks"] == []


def test_the_issue_of_the_owner_with_only_needs_owner_starts_no_job_and_no_triage():
    title, body = issue_text(FixCheck(RULE, OLD, NEW, 2, 3, 3, 5, NOT_HELD, "2026-11-23", pull=7, new_versions=3), repository=REPO)
    assert "2 later versions" in body
    issue = QueueIssue(number=41, title=title, body=body, labels=frozenset({"needs-owner"}), author="someone")
    assert plan([issue], owner="someone") == []
