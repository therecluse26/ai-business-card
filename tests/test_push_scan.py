"""The scan before each push (scripts/git-hooks): the filter of .env values, and a real push to a throwaway remote.

Each value here is made up. The repositories are in a temp folder, with no global git config.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parent.parent / "scripts" / "git-hooks"
_spec = importlib.util.spec_from_file_location("push_scan", HOOKS / "push_scan.py")
assert _spec and _spec.loader
push_scan = importlib.util.module_from_spec(_spec)
sys.modules["push_scan"] = push_scan  # a dataclass reads its module from sys.modules
_spec.loader.exec_module(push_scan)

FAKE_KEY = "sk-made-up-4f1c9a7e2b"  # made up for this test
FAKE_PHONE = "+15555550142"  # a 555 number: never a real line


def test_read_env_strips_quotes_export_and_comments() -> None:
    text = "# a comment\nexport A_KEY='quoted value'\nB_KEY=plain # note\n\nNO_EQUALS\nC_KEY=\"x # kept\"\n"
    assert push_scan.read_env(text) == [("A_KEY", "quoted value"), ("B_KEY", "plain"), ("C_KEY", "x # kept")]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("SMTP_PORT", "587"),
        ("FEEDBACK_ISSUES", "true"),
        ("LIMIT", "1234567"),
        ("LLM_MODEL", "anthropic:claude-opus-5-5"),
        ("GEMINI_LIVE_MODEL", "models/some-live-model"),
        ("LEADS_DB", "data/leads.sqlite3"),
        ("OWNER_NAME", "Pat Example"),
        ("WEBSITE_URL", "https://example.com/about"),
        ("SHORT", "abc1234"),
        ("REGION", "us-east-1"),  # in .env.example
    ],
)
def test_plainly_public_values_are_skipped(key: str, value: str) -> None:
    assert push_scan.plainly_public(key, value, {"us-east-1"})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ANTHROPIC_API_KEY", FAKE_KEY),
        ("OWNER_PHONE", FAKE_PHONE),
        ("ACCOUNT", "12345678"),  # 8 digits: can be a number of a person
        ("HOOK_URL", "https://hooks.example.com/services/T000/B000/abcdefghijklmnopqrstuvwx"),
        ("DB_URL_WITH_LOGIN", "https://user:pass@example.com/"),
        ("SIGNED_URL", "https://example.com/file?sig=abc"),
        ("OWNER_EMAIL", "pat@example.com"),
    ],
)
def test_secret_values_are_kept(key: str, value: str) -> None:
    assert not push_scan.plainly_public(key, value, set())


def test_pushed_revs_handle_a_new_branch_an_update_and_a_deletion() -> None:
    zero, a, b = "0" * 40, "a" * 40, "b" * 40
    stdin = f"refs/heads/new {a} refs/heads/new {zero}\nrefs/heads/gone {zero} refs/heads/gone {b}\n"
    assert push_scan.pushed_revs(stdin, "origin") == [[a, "--not", "--remotes=origin"]]


# --- a real push to a throwaway remote ---


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name, value in {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }.items():
        monkeypatch.setenv(name, value)
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    _git(tmp_path, "init", "--quiet", "--bare", str(remote))
    _git(tmp_path, "init", "--quiet", "--initial-branch=main", str(work))
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "config", "core.hooksPath", str(HOOKS))
    (work / ".gitignore").write_text(".env\n")
    (work / ".env.example").write_text("SMTP_PORT=587\nREGION=us-east-1\n")
    (work / "README.md").write_text("A throwaway repository.\n")
    _git(work, "add", ".")
    _git(work, "commit", "--quiet", "-m", "first")
    return work


def _commit(work: Path, name: str, text: str) -> None:
    (work / name).write_text(text)
    _git(work, "add", name)
    _git(work, "commit", "--quiet", "-m", f"add {name}")


def _push(work: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _git(work, "push", "--quiet", "origin", *args, check=False)


def test_no_env_says_so_and_passes(repo: Path) -> None:
    done = _push(repo, "main")
    assert done.returncode == 0, done.stderr
    assert "no .env here" in done.stdout + done.stderr


def test_a_value_of_env_stops_the_push_and_is_not_printed(repo: Path) -> None:
    (repo / ".env").write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\nREGION=us-east-1\nSMTP_PORT=587\n")
    _push(repo, "main")
    _commit(repo, "notes.txt", "one line\nthe key is " + FAKE_KEY + "\n")
    done = _push(repo, "main")
    said = done.stdout + done.stderr
    assert done.returncode != 0
    assert "the value of ANTHROPIC_API_KEY from .env is in notes.txt, line 2" in said
    assert "--no-verify" in said
    assert FAKE_KEY not in said


def test_a_value_added_and_removed_again_is_still_found(repo: Path) -> None:
    (repo / ".env").write_text(f"OWNER_PHONE={FAKE_PHONE}\n")
    _push(repo, "main")
    _commit(repo, "a.txt", f"call {FAKE_PHONE}\n")
    _commit(repo, "a.txt", "call the office\n")
    assert _push(repo, "main").returncode != 0


def test_a_clean_push_passes_with_public_values(repo: Path) -> None:
    (repo / ".env").write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\nREGION=us-east-1\nLLM_MODEL=anthropic:claude-opus-5-5\n")
    _commit(repo, "docs.txt", "Set REGION=us-east-1 and LLM_MODEL=anthropic:claude-opus-5-5.\n")
    done = _push(repo, "main")
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_new_branch_scans_only_its_own_commits(repo: Path) -> None:
    (repo / ".env").write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\n")
    _commit(repo, "old.txt", FAKE_KEY + "\n")
    assert _push(repo, "--no-verify", "main").returncode == 0  # an old leak, pushed past the hook on purpose
    _git(repo, "switch", "--quiet", "-c", "feature")
    _commit(repo, "new.txt", "clean\n")
    assert _push(repo, "feature").returncode == 0
    _commit(repo, "leak.txt", FAKE_KEY + "\n")
    done = _push(repo, "feature")
    assert done.returncode != 0
    said = done.stdout + done.stderr
    assert "leak.txt" in said and "old.txt" not in said


def test_a_deleted_branch_passes(repo: Path) -> None:
    (repo / ".env").write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\n")
    _git(repo, "switch", "--quiet", "-c", "old")
    _commit(repo, "x.txt", FAKE_KEY + "\n")
    assert _push(repo, "--no-verify", "old").returncode == 0
    assert _push(repo, "--delete", "old").returncode == 0
