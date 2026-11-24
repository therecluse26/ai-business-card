"""The scan before each push: no value of the local .env and no key that gitleaks knows leaves this machine.

The repository is public. Git runs `pre-push` next to this file, and it runs this file. Turn it on one time:

    git config core.hooksPath scripts/git-hooks

Two scans, of the added lines of the commits that the push sends, and nothing else:
1. Each value of `.env` that can be a secret. The message names the key of `.env`, the file and the line.
   It never prints the value.
2. gitleaks, when it is on PATH.

A finding stops the push. `git push --no-verify` goes past a false alarm. Only the Python standard library is used.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

NO_COMMIT = "0" * 40
SHORTEST = 8  # a shorter value matches too much text, and a key that short is not safe anyway
SETTING_WORDS = {"true", "false", "yes", "no", "on", "off", "enabled", "disabled", "none", "null"}
# A setting with the word MODEL in its name, or one of the last words below, holds a name or a place, not a secret.
# A model name is in code and docs. Whole words only: `_LOG` must not match `_LOGIN`.
PUBLIC_LAST_WORDS = {"PATH", "DIR", "DB", "LOG"}
# The owner's name is on the business card (profile.md), so it is in many tracked files.
PUBLIC_KEYS = {"OWNER_NAME", "OWNER_NAME_PRONUNCIATION", "BRAND_NAME", "RESPELLINGS"}
TOKEN_SEGMENT = 20  # a path segment this long in a URL can be a token, as in a webhook URL
SHOWN = 20  # findings printed; the rest are counted


@dataclass(frozen=True)
class Finding:
    key: str
    commit: str
    path: str
    line: int


def read_env(text: str) -> list[tuple[str, str]]:
    """The (key, value) pairs of a .env text. A comment and a line with no `=` are skipped."""
    pairs = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()  # a comment after a value with no quotes
        pairs.append((key, value))
    return pairs


def is_public_url(value: str) -> bool:
    """A web address with no user, no password, no query and no long path segment. Such an address holds no key."""
    if not re.match(r"(?i)^(https?|wss?)://", value):
        return False
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    if not parts.hostname or "@" in parts.netloc or parts.query or parts.fragment:
        return False
    return all(len(segment) < TOKEN_SEGMENT for segment in parts.path.split("/"))


def plainly_public(key: str, value: str, examples: set[str]) -> bool:
    """True for a value that is not a secret: see the filter in the README."""
    if len(value) < SHORTEST or value.lower() in SETTING_WORDS:
        return True
    if re.fullmatch(r"[+-]?\d{1,7}(\.\d+)?", value):
        return True  # a plain number shorter than 8 digits; a longer one can be a phone number
    if value in examples:
        return True  # the value is in the tracked .env.example, so it is public already
    name = key.upper()
    words = name.split("_")
    if name in PUBLIC_KEYS or "MODEL" in words or words[-1] in PUBLIC_LAST_WORDS:
        return True
    return is_public_url(value)


def secret_values(pairs: Iterable[tuple[str, str]], examples: set[str]) -> list[tuple[str, str]]:
    return [(key, value) for key, value in pairs if not plainly_public(key, value, examples)]


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True, errors="replace").stdout


def known(commit: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode == 0


def pushed_revs(stdin: str, remote: str) -> list[list[str]]:
    """For each ref that the push updates: the arguments of `git rev-list` for the commits that it sends.

    Git gives one line for each ref: `<local ref> <local sha> <remote ref> <remote sha>`.
    A deleted ref sends no commit. For a new ref, the commits on a branch of the remote are not sent again."""
    revs = []
    for line in stdin.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        _, local, _, there = fields
        if local == NO_COMMIT:
            continue
        args = [local, "--not", f"--remotes={remote}"]
        if there != NO_COMMIT and known(there):
            args.append(there)
        revs.append(args)
    return revs


def added_lines(revs: list[str]) -> Iterator[tuple[str, str, int, str]]:
    """(commit, file, line number, text) for each added line of each commit in `revs`."""
    out = git("log", "--patch", "--unified=0", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames", "--diff-merges=first-parent", "--format=commit %H", *revs)
    commit = path = ""
    number = 0
    for line in out.splitlines():
        if line.startswith("commit ") and len(line) == 47:
            commit, path = line[7:], ""
        elif line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("@@"):
            found = re.search(r"\+(\d+)", line)
            number = int(found.group(1)) if found else 0
        elif line.startswith("+") and path:
            yield commit, path, number, line[1:]
            number += 1


def scan(revs: list[str], secrets: list[tuple[str, str]]) -> list[Finding]:
    findings = []
    for commit, path, number, text in added_lines(revs):
        findings += [Finding(key, commit, path, number) for key, value in secrets if value in text]
    return findings


def env_files(top: Path) -> list[Path]:
    """The .env of this checkout, and the .env of the main checkout when this is a worktree."""
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    places = [top / ".env", common.parent / ".env"]
    return [path for i, path in enumerate(places) if path.is_file() and path.resolve() not in {p.resolve() for p in places[:i]}]


def run_gitleaks(revs: list[list[str]]) -> int:
    """0 = gitleaks found nothing or is not installed, 1 = a finding, 2 = it could not scan."""
    if shutil.which("gitleaks") is None:
        print("push scan: gitleaks is not on PATH, so only the .env scan ran.")
        return 0
    worst = 0
    for args in revs:
        # Exit code 3 marks a leak, so a fault of gitleaks itself (exit 1) is not read as a leak.
        # --verbose prints each finding with its file and line; --redact keeps the secret out of it.
        command = ["gitleaks", "git", "--no-banner", "--redact", "--verbose", "--log-level", "warn", "--exit-code", "3", "--log-opts", " ".join(args), "."]
        done = subprocess.run(command)
        if done.returncode == 3:
            worst = max(worst, 1)
        elif done.returncode != 0:
            print(f"push scan: gitleaks failed (exit {done.returncode}), so it scanned nothing.")
            worst = 2
    return worst


def main(argv: list[str], stdin: str) -> int:
    remote = argv[1] if len(argv) > 1 else "origin"
    top = Path(git("rev-parse", "--show-toplevel").strip())
    revs = pushed_revs(stdin, remote)
    if not revs:
        return 0  # only deleted refs: nothing leaves this machine
    findings: list[Finding] = []
    files = env_files(top)
    if not files:
        print("push scan: no .env here, so no value of it is checked.")
    else:
        example = top / ".env.example"
        examples = {value for _, value in read_env(example.read_text())} if example.is_file() else set()
        secrets = secret_values((pair for path in files for pair in read_env(path.read_text(errors="replace"))), examples)
        for args in revs:
            findings += scan(args, secrets)
    findings = sorted(set(findings), key=lambda f: (f.path, f.line, f.key, f.commit))
    for finding in findings[:SHOWN]:
        print(f"push scan: the value of {finding.key} from .env is in {finding.path}, line {finding.line} (commit {finding.commit[:10]}).")
    if len(findings) > SHOWN:
        print(f"push scan: {len(findings) - SHOWN} more finding(s) of the .env values.")
    leaks = run_gitleaks(revs)
    if not findings and not leaks:
        return 0
    print(
        "push scan: the push is stopped. Remove the value from each commit (for example `git commit --amend` or `git rebase -i`)."
        "\npush scan: when it is a false alarm, push with `git push --no-verify`."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv, sys.stdin.read()))
