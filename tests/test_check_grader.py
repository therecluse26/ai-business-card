"""The floor of `check-grader --set`: the weekly check (grader-check.yml) fails only below it.

No test here sends a request. A fake decision grader gives made-up verdicts for the made-up calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import pytest

from ai_call_center import check_grader
from ai_call_center.check_grader import GRADER_CALLS, floor_faults, run_set
from ai_call_center.config import get_settings
from ai_call_center.grading import FAIL, PASS, Grade, Score, model_rules


def test_the_floor_holds_at_its_numbers() -> None:
    assert floor_faults(33, 2, min_found=33, max_alarms=2) == []
    assert floor_faults(36, 0, min_found=33, max_alarms=2) == []


def test_too_few_found_is_below_the_floor() -> None:
    assert floor_faults(32, 0, min_found=33, max_alarms=2) == ["32 found, and the floor is 33"]


def test_too_many_false_alarms_is_below_the_floor() -> None:
    assert floor_faults(36, 3, min_found=33, max_alarms=2) == ["3 false alarm(s), and the floor allows 2"]


def test_both_faults_are_named() -> None:
    assert len(floor_faults(0, 9, min_found=33, max_alarms=2)) == 2


def test_no_floor_for_a_count_that_is_not_given() -> None:
    assert floor_faults(0, 99, min_found=None, max_alarms=None) == []
    assert floor_faults(0, 99, min_found=1, max_alarms=None) == ["0 found, and the floor is 1"]


class FakeDecider:
    """Fails each planted rule, except the misses; fails the alarms too. It asks nothing."""

    def __init__(self, by_lines: dict[tuple[str, ...], dict], misses: set[tuple[str, str]], alarms: set[tuple[str, str]]) -> None:
        self.by_lines, self.misses, self.alarms = by_lines, misses, alarms
        self.last_answers: dict[str, Any] = {}
        self.last_lines: list = []

    async def grade(self, lines: list[str], **_: Any) -> Grade:
        call = self.by_lines[tuple(lines)]
        scores = []
        for rule in model_rules():
            must = call["fail"].get(rule.key)
            key = (call["id"], rule.key)
            if must is not None and key not in self.misses:
                scores.append(Score(rule.key, FAIL, agent=must["agent"], cause=must["causes"][0]))
            elif must is None and key in self.alarms:
                scores.append(Score(rule.key, FAIL, agent="contact", cause="other"))
            else:
                scores.append(Score(rule.key, PASS))
        return Grade(scores, by="a fake")


def _run(monkeypatch: pytest.MonkeyPatch, *, misses: int = 0, alarms: int = 0, min_found: int | None = None, max_alarms: int | None = None) -> int:
    calls = json.loads(GRADER_CALLS.read_text())["calls"]
    owner = get_settings().owner_name
    by_lines = {tuple(line.format(owner=owner) for line in call["lines"]): call for call in calls}
    planted = [(call["id"], key) for call in calls for key in call["fail"]]
    clean = [(call["id"], rule.key) for call in calls for rule in model_rules() if rule.key not in call["fail"]]
    fake = FakeDecider(by_lines, set(planted[:misses]), set(clean[:alarms]))
    monkeypatch.setattr(check_grader, "default_decision_grader", lambda settings: fake)
    args = argparse.Namespace(calls=None, only="decision", min_found=min_found, max_false_alarms=max_alarms)
    return asyncio.run(run_set(args))


def test_a_run_with_no_floor_fails_on_one_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch) == 0
    assert _run(monkeypatch, misses=1) == 1


def test_a_run_at_the_floor_passes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    planted = sum(len(call["fail"]) for call in json.loads(GRADER_CALLS.read_text())["calls"])
    assert _run(monkeypatch, misses=2, alarms=1, min_found=planted - 2, max_alarms=1) == 0
    assert "it holds" in capsys.readouterr().out


def test_a_run_below_the_floor_fails(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    planted = sum(len(call["fail"]) for call in json.loads(GRADER_CALLS.read_text())["calls"])
    assert _run(monkeypatch, misses=3, min_found=planted - 2, max_alarms=1) == 1
    assert f"{planted - 3} found, and the floor is {planted - 2}" in capsys.readouterr().out
    assert _run(monkeypatch, alarms=2, min_found=planted - 2, max_alarms=1) == 1
