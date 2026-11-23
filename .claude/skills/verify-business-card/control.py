#!/usr/bin/env python3
"""The harness of the verify-business-card skill: start, check, drive and clean up a verification run.

Run each command from the repository root:

    uv run python .claude/skills/verify-business-card/control.py <command> ...

    new                              make a run folder under data/verify/ and print its path
    doctor  --run DIR                read-only: is this run safe and ready to drive?
    chat    --run DIR --name N --say "..." [--say "..."] [--from NUMBER] [--model M]
                                     one text call with the agents (PAID: model requests)
    server  --run DIR                start the phone server on 127.0.0.1:8011, with no link to Telnyx
    http    METHOD URL [--json BODY] send one HTTP request and print the status, the type and the body's start
    ws      URL                      open a WebSocket and print how the server answers
    rows    --run DIR                print what the run stored: leads, calls, passphrase rows
    cleanup --run DIR                stop the server of this run, keep the evidence, delete the database

Every command uses the same safe settings (`SAFE`): no email to the owner, no request to Telnyx, no
trace, no upload, no job on GitHub, and a database in the run folder.
"""

from __future__ import annotations

import argparse
import os
import pty
import re
import select
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PORT = 8011
PROMPT = re.compile(rb"caller \(\w+\)> $")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TURN_SECS = 180
# A text call runs on this model, not on LLM_MODEL of `.env`. The live line runs small realtime voice models
# (OPENAI_REALTIME_MODEL, NOVA_SONIC_MODEL, GEMINI_LIVE_MODEL in config.py). A large text model follows a prompt
# that a small voice model does not, so a pass on a large model proves less about the line. No vendor publishes
# a text model that is equal to its realtime model; this is the nearest small model that the repo has a key for.
PROOF_MODEL = "anthropic:claude-haiku-5-5"

# Each of these is empty or off so that a verification run reaches nothing outside this computer,
# except the model providers that a text call needs. An empty value wins over the value in `.env`.
SAFE = {
    "LLM_MODEL": PROOF_MODEL,  # see PROOF_MODEL; `chat --model` gives another one
    "SMTP_HOST": "",  # no email to the owner: a saved lead is recorded, not sent
    "PUBLIC_URL": "",  # with TELNYX_APP_ID: the server does not change the webhook URL of the live phone line
    "TELNYX_APP_ID": "",
    "CALLER_LOOKUP": "off",  # no paid number lookup
    "SMS_ENABLED": "false",
    "LANGSMITH_TRACING": "false",
    "TYPESAFE_API_KEY": "",  # no request to Jev: the word list decides a transfer
    "REPORT_R2_BUCKET": "",  # no upload of the public report
    "IMPROVE_GITHUB_TOKEN": "",  # no fix job and no issue on GitHub
    "GRADER_MODEL": "",  # no paid grade after a text call; the code rules still give their scores
    "GRADER_DECISION_MODEL": "",
    "TRANSFER_HOURS": "",  # a live transfer is allowed at any time, so a run gives the same result at night
    "VOICE_READER": "false",  # the server starts without the local voice model
    "SAVE_CALLER_AUDIO": "false",
    "HOST": "127.0.0.1",
    "PORT": str(PORT),
}


def safe_env(run: Path) -> dict[str, str]:
    return {**os.environ, **SAFE, "LEADS_DB": str(run / "leads.sqlite3"), "SERVER_LOG": str(run / "server.log")}


def run_folder(value: str) -> Path:
    run = Path(value).resolve()
    if not run.is_dir() or run.parent != ROOT / "data" / "verify":
        sys.exit(f"{value} is not a run folder. Make one with the `new` command.")
    return run


def health() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as answer:
            return answer.status == 200
    except (urllib.error.URLError, OSError):
        return False


def server_pid(run: Path) -> int:
    """The process group of the server that this run started, or 0 when it is not running."""
    try:
        pid = int((run / "server.pid").read_text())
        os.killpg(pid, 0)
    except (OSError, ValueError):
        return 0
    return pid


def new() -> int:
    run = ROOT / "data" / "verify" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    print(run)
    return 0


def doctor(run: Path) -> int:
    code = (
        "from ai_call_center.config import get_settings as g; s = g(); "
        "print('|'.join(str(v) for v in (s.leads_db, s.smtp_host, s.public_url, s.telnyx_app_id, s.caller_lookup, "
        "s.improve_github_token.get_secret_value() != '', s.report_r2_bucket, s.grader_model, s.grader_decision_model, s.llm_model, s.summary_model, s.port)))"
    )
    out = subprocess.run(["uv", "run", "python", "-c", code], cwd=ROOT, env=safe_env(run), capture_output=True, text=True, check=False)
    if out.returncode:
        print(f"FAIL  the settings do not load:\n{out.stderr[-600:]}")
        return 1
    db, smtp, public, app, lookup, token, bucket, grader, decision, model, summary, port = out.stdout.strip().split("\n")[-1].split("|")
    checks = [
        ("the database is in the run folder", Path(db).resolve() == run / "leads.sqlite3", db),
        ("no email reaches the owner (SMTP_HOST is empty)", smtp == "", smtp),
        ("the server has no link to Telnyx (PUBLIC_URL and TELNYX_APP_ID are empty)", public == "" and app == "", f"{public} {app}"),
        ("no number lookup, no GitHub token, no report upload", lookup == "off" and token == "False" and bucket == "", f"{lookup} {token} {bucket}"),
        ("no paid grader after a text call", grader == "" and decision == "", f"{grader} {decision}"),
        (f"the server port is {PORT}", port == str(PORT), port),
    ]
    ok = True
    for name, passed, value in checks:
        ok = ok and passed
        print(f"{'ok  ' if passed else 'FAIL'}  {name}" + ("" if passed else f": {value!r}"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip())
    print(f"info  the code is commit {commit}{', with changes in src/ that are not committed' if dirty else ''}")
    print(f"info  a text call uses {model} for the agents and {summary} for the summary. Each text call costs money.")
    pid = server_pid(run)
    if pid:
        up = health()
        ok = ok and up
        print(f"{'ok  ' if up else 'FAIL'}  the server of this run (process group {pid}) answers on port {PORT}")
    elif health():
        ok = False
        print(f"FAIL  something answers on port {PORT}, and this run did not start it. Do not drive it.")
    else:
        print("info  no server runs for this run (start one with the `server` command when a feature needs it)")
    return 0 if ok else 1


def server(run: Path) -> int:
    if server_pid(run):
        print(f"The server of this run is already up on http://127.0.0.1:{PORT}")
        return 0
    if health():
        sys.exit(f"Something else answers on port {PORT}. This harness does not drive a server that it did not start.")
    with (run / "server.out").open("ab") as out:
        process = subprocess.Popen(["uv", "run", "serve"], cwd=ROOT, env=safe_env(run), stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    (run / "server.pid").write_text(str(process.pid))
    for _ in range(120):
        if process.poll() is not None:
            sys.exit(f"The server stopped at start. Read {run / 'server.out'}")
        if health():
            print(f"The server is up on http://127.0.0.1:{PORT} (process group {process.pid}). Its output is in {run / 'server.out'}")
            return 0
        time.sleep(0.5)
    os.killpg(process.pid, signal.SIGTERM)
    sys.exit(f"The server did not answer /health in 60 s. Read {run / 'server.out'}")


def read_turn(fd: int, pid: int) -> tuple[bytes, bool]:
    """Read until the chat asks for the caller's next line. Returns the output, and True when the chat ended."""
    data, deadline = b"", time.monotonic() + TURN_SECS
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 1.0)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:  # the chat closed its terminal
            return data, True
        if not chunk:
            return data, True
        data += chunk
        if PROMPT.search(data.replace(b"\r", b"")):
            return data, False
    os.kill(pid, signal.SIGTERM)
    return data + f"\n[harness] no answer in {TURN_SECS} s; the chat was stopped\n".encode(), True


def chat(run: Path, name: str, number: str, lines: list[str], model: str) -> int:
    env = safe_env(run)
    if model:
        env["LLM_MODEL"] = model
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(ROOT)
        os.execvpe("uv", ["uv", "run", "chat", "--from", number], env)
    output, ended = read_turn(fd, pid)
    for line in [*lines, "/quit"]:
        if ended:
            break
        os.write(fd, line.encode() + b"\n")
        more, ended = read_turn(fd, pid)
        output += more
    _, status = os.waitpid(pid, 0)
    text = ANSI.sub("", output.decode("utf-8", "replace")).replace("\r", "")
    header = f"# text call {name!r} from {number}, {datetime.now().isoformat(timespec='seconds')}, model: {model or PROOF_MODEL}\n"
    file = run / f"chat-{name}.txt"
    file.write_text(header + text, encoding="utf-8")
    print(text)
    print(f"\n[harness] transcript: {file}")
    done = "[hung up]" in text and os.waitstatus_to_exitcode(status) == 0
    if not done:
        print("[harness] FAIL: the chat did not reach its end (no '[hung up]' line)")
    return 0 if done else 1


def _tls():
    """The uv-managed Python on this computer finds no system certificates, so an HTTPS request uses the list of `certifi`."""
    import ssl

    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def http(method: str, url: str, body: str) -> int:
    request = urllib.request.Request(url, method=method.upper(), data=body.encode() if body else None, headers={"content-type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(request, timeout=10, context=_tls()) as answer:
            status, kind, data = answer.status, answer.headers.get("content-type", ""), answer.read()
    except urllib.error.HTTPError as refused:
        status, kind, data = refused.code, refused.headers.get("content-type", ""), refused.read()
    except (urllib.error.URLError, OSError) as error:
        print(f"{method.upper()} {url} -> NO SERVER: {error}")
        return 0
    shown = data[:200].decode("utf-8", "replace") if kind.startswith(("application/json", "text/")) else ""
    print(f"{method.upper()} {url} -> HTTP {status}, {kind or 'no content type'}, {len(data)} bytes" + (f": {shown}" if shown else ""))
    return 0


def ws(url: str) -> int:
    import asyncio

    import websockets

    async def attempt() -> str:
        try:
            async with websockets.connect(url, open_timeout=5) as socket:
                try:
                    await asyncio.wait_for(socket.recv(), timeout=3)
                except TimeoutError:
                    return "ACCEPTED: the stream stayed open"
                except websockets.ConnectionClosed as closed:
                    return f"ACCEPTED, then closed with code {closed.rcvd.code if closed.rcvd else 'none'}"
                return "ACCEPTED: the server sent data"
        except websockets.InvalidStatus as refused:
            return f"REFUSED before the stream opened: HTTP {refused.response.status_code}"
        except websockets.ConnectionClosed as closed:
            return f"REFUSED: closed with code {closed.rcvd.code if closed.rcvd else 'none'}"
        except OSError as error:
            return f"NO SERVER: {error}"

    print(f"{url} -> {asyncio.run(attempt())}")
    return 0


QUERIES = {
    "leads": "SELECT id, name, email, company, need, caller_number, call_control_id FROM leads ORDER BY id",
    "calls": "SELECT call_control_id, caller_number, ended_at IS NOT NULL AS ended, agents, caller_name, topic, summary FROM calls ORDER BY started_at",
    "caller_phrases (no salt, no hash)": "SELECT id, caller_number, name, failed FROM caller_phrases ORDER BY id",
    "phrase_calls": "SELECT phrase_id, call_control_id FROM phrase_calls ORDER BY phrase_id, call_control_id",
}


def rows(run: Path) -> str:
    db = run / "leads.sqlite3"
    if not db.exists():
        return "(this run has no database: no text call and no server ran)\n"
    out = []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for title, query in QUERIES.items():
        out.append(f"## {title}")
        try:
            found = conn.execute(query).fetchall()
        except sqlite3.OperationalError:
            out.append("(no such table)")
            continue
        out += [" | ".join(str(value) for value in row) for row in found] or ["(no rows)"]
    conn.close()
    return "\n".join(out) + "\n"


def cleanup(run: Path) -> int:
    pid = server_pid(run)
    if pid:
        os.killpg(pid, signal.SIGTERM)
        for _ in range(20):
            if not server_pid(run):
                break
            time.sleep(0.25)
        else:
            os.killpg(pid, signal.SIGKILL)
        print(f"Stopped the server of this run (process group {pid}).")
    (run / "server.pid").unlink(missing_ok=True)
    if (run / "leads.sqlite3").exists():
        (run / "rows-at-cleanup.txt").write_text(rows(run), encoding="utf-8")
        for file in run.glob("leads.sqlite3*"):
            file.unlink()
        print("Kept the stored rows as rows-at-cleanup.txt, and deleted the database.")
    print(f"The evidence stays in {run}:")
    for file in sorted(run.iterdir()):
        print(f"  {file.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("new")
    for name in ("doctor", "server", "rows", "cleanup"):
        commands.add_parser(name).add_argument("--run", required=True)
    talk = commands.add_parser("chat")
    talk.add_argument("--run", required=True)
    talk.add_argument("--name", required=True, help="a short name for the transcript file, for example call-1")
    talk.add_argument("--say", action="append", required=True, help="one line of the caller; give it one time for each turn")
    talk.add_argument("--from", dest="number", default="+15555550142", help="the caller's number (made up)")
    talk.add_argument("--model", default="", help=f"the model for the agents (default {PROOF_MODEL}, not LLM_MODEL of .env)")
    commands.add_parser("ws").add_argument("url")
    request = commands.add_parser("http")
    request.add_argument("method")
    request.add_argument("url")
    request.add_argument("--json", default="", help="a JSON body to send")
    args = parser.parse_args()
    if args.command == "new":
        return new()
    if args.command == "ws":
        return ws(args.url)
    if args.command == "http":
        return http(args.method, args.url, args.json)
    run = run_folder(args.run)
    if args.command == "doctor":
        return doctor(run)
    if args.command == "server":
        return server(run)
    if args.command == "rows":
        print(rows(run), end="")
        return 0
    if args.command == "cleanup":
        return cleanup(run)
    return chat(run, args.name, args.number, args.say, args.model)


if __name__ == "__main__":
    sys.exit(main())
