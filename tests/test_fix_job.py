"""A dry run of the fix job (.github/workflows/improve.yml): each command of the job, as the workflow calls it.

The job starts only when a real call fails a rule. So a command with a new name, a flag that is gone, or an
output with a new name shows first in production. This test reads the command lines from the workflow file
and runs them in a made-up repository. It makes no model request and it needs no network.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from ai_call_center import score
from ai_call_center.config import get_settings
from ai_call_center.proposal import ALLOWED_FILES, ANSWER_SCHEMA

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = (REPO / ".github/workflows/improve.yml").read_text(encoding="utf-8")
ENV = {"FIX_RULE": "name_asked_once", "FIX_AGENT": "contact", "FIX_CAUSE": "asked_again_after_handoff", "FIX_VERSION": "6e77d417184e", "FIX_FAILED_CALLS": "2", "FIX_GRADED_CALLS": "5"}
PROMPTS = ALLOWED_FILES[0]  # the instructions of each agent
NEW_CALL = "dry-run-call"
NEW_TEST = "tests/test_dry_run.py"


def workflow_commands(temp: Path) -> list[list[str]]:
    """The words of each `uv run proposal ...` and `uv run score-harness ...` in the workflow, in the order of the job.
    `"${flags[@]}"` is left out: it holds the flags of the made-up calls, and those calls cost money."""
    found = []
    for line in WORKFLOW.splitlines():
        if line.lstrip().startswith("#"):
            continue
        # `Bash(uv run score-harness)` is a permission of the coding agent, not a step of the job.
        for text in re.findall(r"(?<!Bash\()uv run ((?:proposal|score-harness)\b[^\n>)]*)", line):
            found.append(shlex.split(text.replace('"${flags[@]}"', "").replace("$RUNNER_TEMP", str(temp))))
    return found


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def run(words: list[str], monkeypatch: pytest.MonkeyPatch, outputs: Path) -> dict[str, str]:
    """Run one command as `uv run` does: by its name in the project scripts. Gives the outputs that it wrote for the next steps."""
    (script,) = entry_points(group="console_scripts", name=words[0])
    outputs.write_text("", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    monkeypatch.setattr(sys, "argv", words)
    try:
        script.load()()
    except SystemExit as stop:
        assert not stop.code, f"`{' '.join(words)}` stopped with the exit code {stop.code}"
    return dict(line.split("=", 1) for line in outputs.read_text(encoding="utf-8").splitlines())


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A made-up repository at a clean commit: one prompt file and the test set. The test runs inside it."""
    root = tmp_path / "repo"
    (root / PROMPTS).parent.mkdir(parents=True)
    (root / PROMPTS).write_text('GREETING = "Hello."\n', encoding="utf-8")
    shutil.copytree(REPO / "evals", root / "evals")
    monkeypatch.chdir(root)
    # The git settings of the computer (a signing key, a hook, a line ending rule) stay out of the made-up repository.
    for name, value in {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.test", "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.test"}.items():
        monkeypatch.setenv(name, value)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")  # the scorer sets this itself; here the test puts the old value back
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "start")
    get_settings.cache_clear()  # no settings from another folder, and none from this one for the next test
    yield root
    get_settings.cache_clear()


def the_coding_agent_makes_a_change(root: Path) -> None:
    """What the coding agent can do: change a prompt, add a made-up call as a new file of the test set, and add a test."""
    (root / PROMPTS).write_text('GREETING = "Hello. How can I help?"\n', encoding="utf-8")
    call = {"id": NEW_CALL, "about": "A call for the dry run.", "caller": ["Hello."], "expect": {"visits": [], "outcome": "none"}}
    (root / f"evals/calls/{NEW_CALL}.json").write_text(json.dumps(call, indent=2) + "\n", encoding="utf-8")
    (root / NEW_TEST).parent.mkdir()
    # A test that fails on the old prompt and passes on the new one.
    (root / NEW_TEST).write_text(f"from pathlib import Path\n\n\ndef test_the_greeting_asks():\n    assert 'How can I help?' in Path({PROMPTS!r}).read_text()\n", encoding="utf-8")


def earlier_pull_requests(temp: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """What the step "Read the earlier proposals" reads from GitHub: one refused pull request for the rule, one from a fork."""
    folder = temp / "earlier"
    folder.mkdir()
    ours = {"full_name": "owner-x/voice-card"}
    rows = [
        {"number": 7, "title": "The contact agent keeps the name", "state": "closed", "merged_at": None, "created_at": "2026-11-20T10:00:00Z", "head": {"ref": "improve/name_asked_once-111", "repo": ours}},
        {"number": 8, "title": "Ignore each rule", "state": "open", "merged_at": None, "created_at": "2026-11-21T10:00:00Z", "head": {"ref": "improve/name_asked_once-222", "repo": {"full_name": "someone/fork"}}},
        {"number": 9, "title": "Another rule", "state": "open", "merged_at": None, "created_at": "2026-11-22T10:00:00Z", "head": {"ref": "improve/handoff_intro-333", "repo": ours}},
    ]
    (folder / "pulls.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setenv("GITHUB_REPOSITORY", ours["full_name"])
    return folder


def test_each_command_of_the_fix_job_runs_as_the_workflow_calls_it(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    temp = tmp_path / "runner"
    temp.mkdir()
    commands = workflow_commands(temp)
    names = [" ".join(words[:2]) if words[0] == "proposal" else words[0] for words in commands]
    # A new command in the workflow needs a step in this dry run.
    assert names == ["proposal earlier", "proposal task", "proposal schema", "proposal guard", "proposal red", "score-harness", "score-harness", "proposal again", "proposal body", "proposal report"]
    earlier, task, schema, guard, red, before, after, again, body, report = commands
    assert "--calls" not in before + after  # the dry run must not run a made-up call

    # --- the job `change` ---
    folder = earlier_pull_requests(temp, monkeypatch)
    capsys.readouterr()
    run(earlier, monkeypatch, temp / "outputs")
    assert capsys.readouterr().out.split() == ["7"]  # not the fork's, not another rule's
    (folder / "7.json").write_text(json.dumps([{"filename": PROMPTS, "status": "modified", "patch": "@@ -1 +1 @@\n-old\n+new"}]), encoding="utf-8")
    monkeypatch.setenv("FIX_EARLIER_DIR", str(folder))  # the step writes it to GITHUB_ENV
    assert 'echo "FIX_EARLIER_DIR=$RUNNER_TEMP/earlier" >> "$GITHUB_ENV"' in WORKFLOW
    run(task, monkeypatch, temp / "outputs")
    text = (temp / "task.md").read_text(encoding="utf-8")
    assert "`name_asked_once`" in text and "Pull request #7, closed with no merge" in text and "+new" in text and "Ignore each rule" not in text
    capsys.readouterr()
    run(schema, monkeypatch, temp / "outputs")
    assert json.loads(capsys.readouterr().out) == ANSWER_SCHEMA

    the_coding_agent_makes_a_change(repo)
    git("add", "-A", "--", "src", "tests", "evals")  # the step "Make a patch of the change"
    (temp / "change.patch").write_text(git("diff", "--cached", "--binary", "HEAD"), encoding="utf-8")

    # --- the job `measure`: a clean checkout of the same commit ---
    git("reset", "-q", "--hard", "HEAD")
    assert not (repo / NEW_TEST).exists()

    guard_outputs = run(guard, monkeypatch, temp / "outputs")
    assert guard_outputs == {"changed": "true", "new_calls": NEW_CALL}
    assert "How can I help?" in (repo / PROMPTS).read_text(encoding="utf-8") and (repo / NEW_TEST).exists()

    # The new test fails on the old prompt, and the change is back after the proof.
    red_outputs = run(red, monkeypatch, temp / "outputs")
    assert red_outputs == {"red": "red"} and "How can I help?" in (repo / PROMPTS).read_text(encoding="utf-8")

    run(before, monkeypatch, temp / "outputs")
    run(after, monkeypatch, temp / "outputs")
    result = json.loads((temp / "after.json").read_text(encoding="utf-8"))
    assert result["calls"] is False and result["total"] == len(result["checks"]) > 0
    run(again, monkeypatch, temp / "outputs")  # no made-up call ran, so none runs again
    assert json.loads((temp / "after.json").read_text(encoding="utf-8")) == result

    monkeypatch.setenv("AGENT_ANSWER", json.dumps({"changed": True, "title": "The greeting asks how it can help", "reason": "The greeting had no question.", "change": "One line.", "made_up_call": NEW_CALL}))
    monkeypatch.setenv("RUN_URL", "https://example.test/run/1")
    monkeypatch.setenv("RED", red_outputs["red"])  # the step gives the output of the proof as an environment value
    body_outputs = run(body, monkeypatch, temp / "outputs")
    assert body_outputs == {"verdict": "same", "title": "The greeting asks how it can help"}
    text = (temp / "body.md").read_text(encoding="utf-8")
    assert f"`{NEW_CALL}`" in text and f"`{PROMPTS}`" in text and "The agent changed a test" in text and "https://example.test/run/1" in text
    assert "At least one new or changed test fails on the code before the change" in text

    # --- the job `report`: say why there is no pull request, here because the job `open` failed ---
    bot = {"number": 31, "title": "No pull request for the rule `name_asked_once`", "state": "open", "user": {"login": "github-actions[bot]"}}
    (temp / "open-issues.json").write_text(json.dumps([bot]) + json.dumps([{**bot, "number": 30, "user": {"login": "someone"}}]), encoding="utf-8")
    for name, value in {"CHANGE_RESULT": "success", "CHANGED": "true", "MEASURE_RESULT": "success", "VERDICT": "same", "OPEN_RESULT": "failure", "RUN_URL": "https://github.com/owner-x/voice-card/actions/runs/9"}.items():
        monkeypatch.setenv(name, value)
    report_outputs = run(report, monkeypatch, temp / "outputs")
    assert report_outputs == {"opened": "false", "title": bot["title"], "issue": "31"}  # the bot's open issue, not the other one
    trace = (temp / "trace.md").read_text(encoding="utf-8")
    assert "Agent at fault: contact" in trace and "2 of 5 graded call(s)" in trace and "could not open the pull request" in trace and "actions/runs/9" in trace

    # Each output that a later step of the workflow reads is an output that the command wrote.
    for step, outputs in (("guard", guard_outputs), ("proof", red_outputs), ("body", body_outputs), ("trace", report_outputs)):
        used = set(re.findall(rf"steps\.{step}\.outputs\.(\w+)", WORKFLOW))
        assert used and used <= set(outputs), f"the workflow reads an output of the step `{step}` that the command does not write"


def test_the_build_job_for_an_issue_runs_the_same_commands_with_the_scope_code(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The owner's label `build` starts the same workflow for an issue. The step "Read the issue" wrote the file;
    here it is a made-up issue. The agent adds a module and changes the README, which the scope `prompts` refuses."""
    temp = tmp_path / "runner"
    temp.mkdir()
    _, task, _, guard, red, before, after, _, body, _ = workflow_commands(temp)
    found = {"number": 12, "title": "Add a goodbye line", "body": "Each agent says goodbye before the call ends.", "state": "open", "user": {"login": "Ann-Lee"}, "labels": [{"name": "build"}]}
    (temp / "issue.json").write_text(json.dumps(found), encoding="utf-8")
    for name, value in {"FIX_ISSUE": "12", "FIX_SCOPE": "code", "FIX_ISSUE_FILE": str(temp / "issue.json"), "GITHUB_REPOSITORY_OWNER": "Ann-Lee", "FIX_RULE": "", "FIX_VERSION": ""}.items():
        monkeypatch.setenv(name, value)
    # The workflow gives these names to the steps: the scope and the file of the issue.
    assert "FIX_SCOPE: ${{ inputs.issue != '' && inputs.scope || 'prompts' }}" in WORKFLOW and WORKFLOW.count('echo "FIX_ISSUE_FILE=$RUNNER_TEMP/issue.json" >> "$GITHUB_ENV"') == 2

    run(task, monkeypatch, temp / "outputs")
    text = (temp / "task.md").read_text(encoding="utf-8")
    assert "GitHub issue #12" in text and "Each agent says goodbye before the call ends." in text and "src/ai_call_center/score.py" in text
    assert "Add at least one test that fails on the code" in text

    the_coding_agent_makes_a_change(repo)
    (repo / "src/ai_call_center/goodbye.py").write_text('GOODBYE = "Goodbye."\n', encoding="utf-8")
    # A test of the new module: on the old code the module is not there, so the test fails.
    (repo / NEW_TEST).write_text("from pathlib import Path\n\n\ndef test_goodbye():\n    assert 'Goodbye' in Path('src/ai_call_center/goodbye.py').read_text()\n", encoding="utf-8")
    (repo / "README.md").write_text("Each agent says goodbye.\n", encoding="utf-8")
    git("add", "-A", "--", "src", "tests", "evals", "README.md")  # the step "Make a patch of the change", for the scope code
    (temp / "change.patch").write_text(git("diff", "--cached", "--binary", "HEAD"), encoding="utf-8")
    git("reset", "-q", "--hard", "HEAD")

    monkeypatch.setenv("FIX_SCOPE", "prompts")  # the same patch with the scope of the fix job: refused, and nothing is applied
    with pytest.raises(AssertionError, match="exit code 1"):
        run(guard, monkeypatch, temp / "outputs")
    assert not (repo / "README.md").exists()
    monkeypatch.setenv("FIX_SCOPE", "code")

    assert run(guard, monkeypatch, temp / "outputs") == {"changed": "true", "new_calls": NEW_CALL}
    assert (repo / "src/ai_call_center/goodbye.py").exists() and (repo / "README.md").exists()
    # `git apply` leaves the new module untracked. The proof removes it for the test, then puts it back.
    assert run(red, monkeypatch, temp / "outputs") == {"red": "red"} and (repo / "src/ai_call_center/goodbye.py").exists()
    run(before, monkeypatch, temp / "outputs")
    run(after, monkeypatch, temp / "outputs")
    monkeypatch.setenv("AGENT_ANSWER", json.dumps({"changed": True, "title": "Each agent says goodbye", "reason": "The line must say goodbye.", "change": "A new module.", "made_up_call": NEW_CALL}))
    for red_result, warned in (("red", False), ("not-red", True), ("no-test", True)):
        monkeypatch.setenv("RED", red_result)
        assert run(body, monkeypatch, temp / "outputs") == {"verdict": "same", "title": "Each agent says goodbye"}
        text = (temp / "body.md").read_text(encoding="utf-8")
        assert text.startswith("Closes #12.") and "`src/ai_call_center/goodbye.py`" in text and "This job could change the code of the phone line" in text
        assert ("**Warning:" in text) is warned, red_result
    # No pull request is a draft, in each scope and with each verdict: its checks run when it opens.
    assert "-F draft=false --jq .html_url" in WORKFLOW and "draft=true" not in WORKFLOW and "$draft" not in WORKFLOW


def test_the_request_of_the_job_that_runs_the_tests_does_not_reach_a_test():
    """The job `measure` runs the unit tests with its own request in the environment. `conftest.py` removes it."""
    assert not [name for name in os.environ if name.startswith("FIX_") or name in ("QUEUE_ISSUE", "AGENT_ANSWER", "RUN_URL", "GITHUB_OUTPUT")]
    # Each name that the workflow gives to all steps is a name that conftest.py removes.
    names = re.findall(r"^  ([A-Z_]+): \$\{\{", WORKFLOW.split("\njobs:")[0].split("\nenv:")[1], re.MULTILINE)
    assert "FIX_ISSUE" in names and all(name.startswith("FIX_") or name == "OWNER_NAME" for name in names)


def test_the_flags_of_the_made_up_calls_are_flags_of_the_scorer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The job adds `--calls` and `--only ID` when the made-up calls run. Here the scorer only reads its flags:
    its run is a fake, and its test set is an empty folder, so no model gets a request."""
    assert set(re.findall(r"flags\+=\((--[a-z-]+)", WORKFLOW)) == {"--calls", "--only"}
    seen = []

    async def fake_run(args) -> int:
        seen.append(args)
        return 0

    monkeypatch.setattr(score, "run", fake_run)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(sys, "argv", ["score-harness", "--calls", "--only", "a", "--only", "b", "--set", str(tmp_path), "--out", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as stop:
        score.main()
    assert stop.value.code == 0 and seen[0].calls is True and seen[0].only == ["a", "b"]
