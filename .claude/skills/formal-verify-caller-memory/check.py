"""Every order of caller actions, run on the real caller-memory code: does data of one person reach another?

Run: uv run python .claude/skills/formal-verify-caller-memory/check.py

The property: on a call, the agents and a new lead get no data that another person left on an earlier call.

The world: two people use one phone number and have one first name. Each can call, say their name, say a
passphrase, set a passphrase, leave a message, ask to be forgotten, and hang up. Neither says the other
person's passphrase. The search runs the real functions (`graph/tools.py`: `identify`, `verify`,
`set_phrase`, `save_lead_record`, `forget`; `crm.py`: `request_status`, `sync_call`) on a real SQLite
store, for every order of those actions, up to `--calls` calls. It is a breadth-first search, so each
fault comes with its shortest list of steps.

Each person has secret words of three kinds: a surname, a note word (in their message, email address,
company, call summary and topic), and a passphrase word. A fault is a secret word of one person in what
the code gives the agents or stores on a call of the other person, before that person said the word.

`--mutants` proves that the search can fail: it runs the search on broken copies of the package.
SKILL.md, next to this file, says how to read a result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ai_call_center
from ai_call_center import crm as crm_module
from ai_call_center import leads as leads_module
from ai_call_center import passphrase
from ai_call_center.agents.specs import Welcome
from ai_call_center.analytics import AnalyticsStore
from ai_call_center.callers import CallSummary
from ai_call_center.crm import CrmStore
from ai_call_center.feedback import FeedbackStore
from ai_call_center.graph import tools
from ai_call_center.graph.state import CallContext
from ai_call_center.leads import LeadStore
from ai_call_center.passphrase import PhraseStore
from ai_call_center.telephony import RecordingTelephony

NUMBER = "+15555550123"
DEFAULT_CALLS = 3
SAVES_IN_A_CALL = 1  # a bound of the search: how many times a caller leaves a message on one call
# The search sets `passphrase.LOCK_AT` to this value, so that a lock is inside the bound of three calls.
# The code compares the count of wrong tries with LOCK_AT; the value 5 of the product needs four calls.
SEARCH_LOCK_AT = 2

# The tables that the caller-memory code writes. A state of the search is the rows of these tables.
TABLES = {
    "leads": "id, name, email, company, need, caller_number, call_control_id, created_at, timeline, size",
    "calls": "call_control_id, caller_number, started_at, ended_at, agents, summary, caller_name, topic",
    "caller_phrases": "id, caller_number, name, salt, hash, created_at, failed",
    "phrase_calls": "phrase_id, call_control_id",
    "crm_calls": "call_control_id, caller_number, name, person_id, deal_id, deal_made, task_id, noted, entry, state, tries, error, created_at",
    "sqlite_sequence": "name, seq",  # the next row number of each table
}

SURNAME, NOTE, PHRASE = "surname", "note word", "passphrase word"


@dataclass(frozen=True)
class Person:
    label: str
    name: str  # the name this person gives on each call
    surname: str
    note: str  # the word in everything this person leaves on a call
    phrase: str
    phrase_word: str

    @property
    def secrets(self) -> dict[str, str]:
        return {self.surname: SURNAME, self.note: NOTE, self.phrase_word: PHRASE}


REAL = Person("the real caller", "Sam Carter", "carter", "quartz", "silver lantern", "lantern")
# The search runs one time for each second person. The two cases take different paths in the code:
# the first matches the real caller by the first name only, the second by the full name.
WORLDS = {
    "shared phone": Person("a second person with the same first name", "Sam Pike", "pike", "pelican", "copper kettle", "kettle"),
    "faked number": Person("a person with a faked number who knows the real caller's full name", "Sam Carter", "pike", "pelican", "copper kettle", "kettle"),
}
WRONG_PHRASE = "wooden ladle"
ALL_PHRASES = (REAL.phrase, "copper kettle", WRONG_PHRASE)

# What the code gives the agents, or stores, on a call. Each is a place where a secret word is a fault.
TOOL_RESULT = "tool result"  # the text that a tool gives back to the voice model
HISTORY = "caller_history"  # the notes of earlier calls in the call context
CALLER_NAME = "caller_name"  # the name that every agent gets in its handoff message (`known_name_note`)
NEW_LEAD = "new lead"  # the lead that a save writes on this call
CRM_RECORD = "CRM record"  # a record of the owner's CRM that crm_lookup read on this call; the agent gets its status and stage
PLACES = (TOOL_RESULT, HISTORY, CALLER_NAME, NEW_LEAD, CRM_RECORD)

# Faults that are known and reported, and not fixed yet: (kind of secret word, place). A run passes
# only when it finds exactly these. When the product code fixes one, remove it here (SKILL.md).
# No finding is open. An entry is (kind, place): the text that says when it was found and why it is still open.
KNOWN_FINDINGS: dict[tuple[str, str], str] = {}

# The search must reach each of these, or a pass means nothing: no data was ever released.
WITNESSES = (
    "a verified caller got the notes of their own earlier call",
    "a verified caller reused their own email address on file",
    "a caller set a passphrase",
    "a passphrase replaced another person's passphrase",
    "a passphrase was locked after wrong tries",
    "a caller was forgotten",
    "a verified caller got the stage of their own request from the CRM",
)

# The fields of the call context that the caller-memory code reads or writes.
CTX_FIELDS = (
    "caller_history",
    "identity_confirmed",
    "identity_verified",
    "open_calls",
    "phrase_tries",
    "phrase_closed",
    "phrase_offered",
    "phrase_set",
    "held_phrase",
    "returning_name",
    "identified",
    "forgotten",
    "caller_name",
    "transfer_note_id",
)

SAVES = {
    # name given to save_lead, email, company
    "new": (True, "{note}@example.test", "{Note} Works"),
    "on file": (True, "on file", ""),
    "on file, no name": (False, "on file", ""),
}

WORDS = {
    "identify": "The agent calls identify_caller with that name.",
    "verify own": "The caller says their own passphrase (verify_passphrase).",
    "verify wrong": "The caller says a wrong passphrase (verify_passphrase).",
    "verify none": "The caller gives no passphrase (verify_passphrase with an empty one).",
    "set": "The caller gives their own passphrase to set (set_passphrase).",
    "save new": "The caller leaves a message with a new email address and company (save_lead).",
    "save on file": 'The caller leaves a message; the agent passes the email "on file" (save_lead).',
    "save on file, no name": 'The caller leaves a message; the agent passes no name and the email "on file" (save_lead).',
    "forget": "The caller asks to be forgotten (forget_caller).",
    "crm lookup": "The caller asks where their request stands (crm_lookup).",
    "end": "The call ends. Its summary is stored, and the owner's CRM gets the call.",
}


class Clock:
    """Stands in for `leads.now_iso`: one second later on each use, so the order of the records is fixed."""

    def __init__(self) -> None:
        self.tick = 0

    def __call__(self) -> str:
        self.tick += 1
        return (datetime(2026, 1, 1, 15, tzinfo=UTC) + timedelta(seconds=self.tick)).isoformat(timespec="seconds")


@dataclass(frozen=True)
class NoEmail:
    """Stands in for the email to the owner. The email goes to the owner, not to an agent, so the property does not cover it."""

    delivers = True

    async def send_lead(self, lead: object, *, row_id: int, details: object = None) -> None:
        return None


class SearchCrm:
    """Stands in for the owner's CRM. It holds nothing: the ID of a record is its content, so each state of the
    search is in the tables. It has no person that the owner added by hand, and each deal is open."""

    def __init__(self) -> None:
        self.read: list[str] = []  # each record that a get_record gave

    async def tool(self, name: str, arguments: dict) -> dict:
        if name == "search":
            return {"hits": []}
        if name == "create_person":
            return {"id": f"person: {arguments['firstName']} {arguments['lastName']}".strip()}
        if name in ("create_company", "create_deal"):
            return {"id": f"{name.removeprefix('create_')}: {arguments['name']}"}
        if name in ("add_note", "create_task", "update_person"):
            return {"id": name}
        if name == "list_pipelines":
            return {"pipelines": [{"id": "pipeline", "name": "Client work", "stages": [{"id": "stage", "name": "New lead"}]}]}
        if name == "get_record":
            record = {"id": arguments["id"], "name": arguments["id"].partition(": ")[2], "status": "open", "stageId": "stage"}
            self.read.append(json.dumps(record))
            return {"record": record, "timeline": []}
        raise ValueError(name)  # pragma: no cover


@dataclass(frozen=True)
class Leak:
    kind: str  # SURNAME, NOTE or PHRASE
    place: str
    word: str
    text: str


class World:
    """One store, and at most one call in progress. `snapshot` and `restore` move it between states."""

    def __init__(self, second: Person) -> None:
        self.people = (REAL, second)
        # What each person knows from the start: their own secret words, and the name they give.
        self.known = tuple(frozenset(person.secrets) | {word.lower() for word in person.name.split()} for person in self.people)
        self.clock = Clock()
        leads_module.now_iso = self.clock  # the time of a call
        tools.now_iso = self.clock  # the time of a lead
        crm_module.now_iso = self.clock  # the time of a CRM entry
        # The cost of the hash does not change which passphrase matches. The real setting takes 34 ms.
        passphrase._SCRYPT = {"n": 2, "r": 1, "p": 1, "dklen": 32}
        passphrase.LOCK_AT = SEARCH_LOCK_AT
        self.loop = asyncio.new_event_loop()
        self.store = LeadStore(":memory:")
        for table in (PhraseStore, AnalyticsStore, FeedbackStore, CrmStore):  # every table exists before the first snapshot
            table(self.store.connection)
        self.crm = SearchCrm()
        self.welcome = Welcome(owner="Test Owner", companies="from two companies", tip="how this works")
        self.ctx: CallContext | None = None
        self.who = -1
        self.saves = 0  # messages left on the call in progress
        self.owners: list[int] = []  # who made each call, oldest first
        self.seen: set[str] = set()  # the witnesses that the last action reached

    # --- states ---

    def snapshot(self) -> tuple:
        conn = self.store.connection
        rows = tuple(tuple(conn.execute(f"SELECT {columns} FROM {table} ORDER BY rowid")) for table, columns in TABLES.items())
        ctx = None
        if self.ctx is not None:
            ctx = tuple(tuple(value) if isinstance(value := getattr(self.ctx, name), list) else value for name in CTX_FIELDS)
        return (rows, ctx, self.who, self.saves, tuple(self.owners), self.clock.tick)

    def restore(self, state: tuple) -> None:
        rows, ctx, self.who, self.saves, owners, self.clock.tick = state
        conn = self.store.connection
        for (table, columns), table_rows in zip(TABLES.items(), rows, strict=True):
            conn.execute(f"DELETE FROM {table}")
            conn.executemany(f"INSERT INTO {table} ({columns}) VALUES ({', '.join('?' * (columns.count(',') + 1))})", table_rows)
        conn.commit()
        self.owners = list(owners)
        self.ctx = None
        if ctx is not None:
            self.ctx = self._new_ctx(f"call-{len(self.owners)}")
            for name, value in zip(CTX_FIELDS, ctx, strict=True):
                setattr(self.ctx, name, list(value) if isinstance(value, tuple) else value)

    @staticmethod
    def key(state: tuple) -> tuple:
        """The state without what cannot change a later result: the times, the salts, the row numbers."""
        (leads, calls, phrases, phrase_calls, crm_calls, _), ctx, who, saves, owners, _ = state
        named = []
        for row_id, _, name, salt, stored, _, failed in phrases:
            which = next((phrase for phrase in ALL_PHRASES if passphrase._hash(phrase, salt) == stored), "?")
            named.append((name, which, failed, tuple(sorted(call for phrase_id, call in phrase_calls if phrase_id == row_id))))
        return (
            tuple((lead[1], lead[2], lead[3], lead[4], lead[6]) for lead in leads),
            tuple((call[0], call[3] is not None, call[5], call[6], call[7]) for call in sorted(calls, key=lambda call: (call[2], call[0]))),
            tuple(named),
            # The call, the name, the person, the deal, "this call made the deal", the state. The rows are in the order of the calls.
            tuple((row[0], row[2], row[3], row[4], row[5], row[9]) for row in crm_calls),
            ctx,
            who,
            saves,
            owners,
        )

    def other_tables_are_empty(self) -> list[str]:
        """The tables with rows that a snapshot does not hold. Each one is a fault of this search, not of the product."""
        conn = self.store.connection
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return [table for table in tables if table not in TABLES and conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]]

    def _new_ctx(self, call_id: str) -> CallContext:
        return CallContext(
            call_control_id=call_id,
            caller_number=NUMBER,
            owner_phone="+15555550199",
            leads=self.store,
            telephony=RecordingTelephony(),
            notifier=NoEmail(),
            mode="voice",
            welcome=self.welcome,
            owner_name="Test Owner",
            crm=self.crm,
        )

    # --- actions ---

    @staticmethod
    def actions(state: tuple, max_calls: int) -> list[tuple]:
        _, ctx, _, saves, owners, _ = state
        if ctx is None:
            return [("call", who) for who in (0, 1)] if len(owners) < max_calls else []
        return [(name,) for name in WORDS if not name.startswith("save") or saves < SAVES_IN_A_CALL]

    def apply(self, action: tuple) -> list[Leak]:
        """Do one action with the real code. Returns each secret word that reached a place it must not reach."""
        self.seen = set()
        if action[0] == "call":
            self.who, self.saves = action[1], 0
            self.owners.append(self.who)
            call_id = f"call-{len(self.owners)}"
            self.store.start_call(call_id, NUMBER)
            self.ctx = self._new_ctx(call_id)
            return []
        ctx, me = self.ctx, self.people[self.who]
        assert ctx is not None
        step, lead = action[0], {}
        self.crm.read = []
        if step == "identify":
            text = tools.identify(ctx, me.name)
        elif step.startswith("verify"):
            was_verified = ctx.identity_verified
            text = tools.verify(ctx, {"verify own": me.phrase, "verify wrong": WRONG_PHRASE, "verify none": ""}[step])
            if ctx.identity_verified and not was_verified and me.note in ctx.caller_history.lower():
                self.seen.add(WITNESSES[0])
            if any(row[0] >= passphrase.LOCK_AT for row in self.store.connection.execute("SELECT failed FROM caller_phrases")):
                self.seen.add(WITNESSES[4])
        elif step == "set":
            before = self._phrases_on_file()
            text = tools.set_phrase(ctx, me.phrase)
            if before != (after := self._phrases_on_file()):
                self.seen.add(WITNESSES[2])
                if before and before[0] != self.who and after[0] == self.who:
                    self.seen.add(WITNESSES[3])
        elif step.startswith("save"):
            self.saves += 1
            with_name, email, company = SAVES[step.removeprefix("save ")]
            text, lead = self.loop.run_until_complete(
                tools.save_lead_record(
                    ctx,
                    name=me.name if with_name else "",
                    need=f"A message about {me.note}",
                    email=email.format(note=me.note),
                    company=company.format(Note=me.note.title()),
                    visited=["classifier", "contact"],
                )
            )
            if email == "on file" and lead.get("email") == f"{me.note}@example.test":
                self.seen.add(WITNESSES[1])
        elif step == "forget":
            text, _ = tools.forget(ctx)
            self.seen.add(WITNESSES[5])
        elif step == "crm lookup":
            text = self.loop.run_until_complete(crm_module.request_status(ctx))
            if ctx.identity_verified and any(me.note in record.lower() for record in self.crm.read):
                self.seen.add(WITNESSES[6])
        elif step == "end":
            # What `call_session` and `chat.run` do at the end. The summary model writes what was said on the call.
            # Each call is about a project, so each call that is not forgotten leaves a lead: a person and a deal in the CRM.
            if ctx.forgotten:
                self.store.finish_call(ctx.call_control_id, agents=["classifier"], summary="")
            else:
                summary = CallSummary(summary=f"{me.name} asked about {me.note}.", caller_name=ctx.caller_name or me.name, topic=f"the {me.note} project", interest="project")
                self.store.finish_call(ctx.call_control_id, agents=["classifier"], summary=summary.summary, caller_name=summary.caller_name, topic=summary.topic)
                self.loop.run_until_complete(crm_module.sync_call(ctx, summary, visited=["classifier"]))
            self.ctx, self.who, self.saves = None, -1, 0
            return []
        else:  # pragma: no cover
            raise ValueError(step)
        places = {TOOL_RESULT: text, HISTORY: ctx.caller_history, CALLER_NAME: ctx.caller_name, NEW_LEAD: " ".join(lead.values()), CRM_RECORD: " ".join(self.crm.read)}
        leaks = []
        for place, shown in places.items():
            for word, kind in self.people[1 - self.who].secrets.items():
                if word in shown.lower() and word not in self.known[self.who]:
                    leaks.append(Leak(kind, place, word, shown))
        return leaks

    def _phrases_on_file(self) -> list[int]:
        """Whose passphrase each row holds: 0 for the real caller, 1 for the second person."""
        rows = self.store.connection.execute("SELECT salt, hash FROM caller_phrases ORDER BY id").fetchall()
        return [next((who for who, person in enumerate(self.people) if passphrase._hash(person.phrase, salt) == stored), -1) for salt, stored in rows]


def describe(action: tuple, call: int, people: tuple[Person, Person]) -> str:
    if action[0] == "call":
        person = people[action[1]]
        return f"Call {call} starts. It is from {person.label}, who gives the name {person.name!r}."
    return WORDS[action[0]]


def excerpt(leak: Leak) -> str:
    at = leak.text.lower().index(leak.word)
    start, end = max(0, at - 70), min(len(leak.text), at + len(leak.word) + 50)
    return ("..." if start else "") + " ".join(leak.text[start:end].split()) + ("..." if end < len(leak.text) else "")


def search_world(world_name: str, max_calls: int) -> dict:
    """Breadth first over every state that the two people can reach in `max_calls` calls."""
    logging.disable(logging.CRITICAL)
    world = World(WORLDS[world_name])
    state = world.snapshot()
    first = world.key(state)
    parents: dict[tuple, tuple | None] = {first: None}
    queue = deque([(first, state)])
    findings: dict[tuple[str, str], dict] = {}
    witnesses: set[str] = set()
    steps_done = 0
    while queue:
        key, state = queue.popleft()
        for action in world.actions(state, max_calls):
            world.restore(state)
            leaks = world.apply(action)
            steps_done += 1
            witnesses |= world.seen
            for leak in leaks:
                if (leak.kind, leak.place) not in findings:
                    path, at = [action], key
                    while parents[at] is not None:
                        at, earlier = parents[at]
                        path.append(earlier)
                    steps, calls = [], 0
                    for one in reversed(path):
                        calls += one[0] == "call"
                        steps.append(describe(one, calls, world.people))
                    findings[(leak.kind, leak.place)] = {"world": world_name, "kind": leak.kind, "place": leak.place, "word": leak.word, "steps": steps, "text": excerpt(leak)}
            after = world.snapshot()
            new = world.key(after)
            if new not in parents:
                parents[new] = (key, action)
                queue.append((new, after))
    if stray := world.other_tables_are_empty():
        raise SystemExit(f"The code wrote to {stray}, which the search does not snapshot. Add each table to TABLES in {Path(__file__).name}.")
    return {"states": len(parents), "steps": steps_done, "findings": findings, "witnesses": witnesses}


def search(max_calls: int) -> dict:
    states = steps = 0
    findings: dict[tuple[str, str], dict] = {}
    witnesses: set[str] = set()
    with ProcessPoolExecutor(max_workers=len(WORLDS)) as pool:  # the two worlds share nothing
        results = list(pool.map(search_world, WORLDS, [max_calls] * len(WORLDS)))
    for result in results:
        states, steps, witnesses = states + result["states"], steps + result["steps"], witnesses | result["witnesses"]
        for where, finding in result["findings"].items():  # the shortest list of steps of the two worlds
            if where not in findings or len(finding["steps"]) < len(findings[where]["steps"]):
                findings[where] = finding
    return {"package": str(Path(ai_call_center.__file__).parent), "calls": max_calls, "states": states, "steps": steps, "findings": list(findings.values()), "witnesses": sorted(witnesses)}


def verdict(result: dict) -> tuple[bool, list[str]]:
    """True when the run found exactly the known findings and reached every witness."""
    lines = [f"Searched every order of actions over {result['calls']} calls: {result['states']} states, {result['steps']} steps."]
    ok = True
    found = {(finding["kind"], finding["place"]) for finding in result["findings"]}
    for finding in result["findings"]:
        known = (finding["kind"], finding["place"]) in KNOWN_FINDINGS
        ok = ok and known
        lines.append("")
        lines.append(f"{'KNOWN FINDING, still open' if known else 'NEW FINDING'}: a {finding['kind']} of the other person is in the {finding['place']}.")
        lines.append(f"  The second person is {WORLDS[finding['world']].label}.")
        lines += [f"  {n}. {step}" for n, step in enumerate(finding["steps"], 1)]
        lines.append(f"  The {finding['place']} then holds {finding['word']!r}: {finding['text']}")
    for gone in sorted(set(KNOWN_FINDINGS) - found) if result["calls"] >= DEFAULT_CALLS else []:  # a shorter search does not reach each one
        ok = False
        lines.append("")
        lines.append(f"A KNOWN FINDING NO LONGER HAPPENS: {gone[0]} in the {gone[1]}. If the code fixed it, remove it from KNOWN_FINDINGS.")
    missing = [witness for witness in WITNESSES if witness not in result["witnesses"]]
    if missing and result["calls"] >= DEFAULT_CALLS:
        ok = False
        lines.append("")
        lines += [f"NOT REACHED, so the search is too weak: {witness}." for witness in missing]
    lines.append("")
    if ok and not result["findings"]:
        lines.append("PASS: no data of one person reached a call of the other person.")
    elif ok:
        lines.append("PASS, with the known findings above. No new finding.")
    else:
        lines.append("FAIL.")
    return ok, lines


# Broken copies of the package. Each one removes a check that the property needs, so the search must
# find a note word of the other person. `old` must be in the file exactly one time.
MUTANTS = (
    ("verify gives the notes of every call, not only the calls that the passphrase opens", "graph/tools.py", ", only_calls=right.calls)", ")"),
    ("identify gives the notes after a name match, with no passphrase check", "graph/tools.py", "    ctx.returning_name = memory.name\n", "    ctx.returning_name = memory.name\n    ctx.caller_history = history_text(memory)\n"),
    (
        "a new passphrase opens every earlier call of the name, not only its own call",
        "graph/tools.py",
        "calls=[ctx.call_control_id, *(ctx.open_calls if ctx.identity_verified else [])]",
        "calls=[ctx.call_control_id, *(call.call_control_id for call in ctx.leads.calls_for(ctx.caller_number, limit=30))]",
    ),
    # Not in this list on purpose: `save_lead_record` without its `ctx.identity_verified` test. Only `verify`
    # fills `ctx.open_calls`, and only after a right passphrase, so the `open_calls` filter on the next line of
    # the product code holds the property alone. The code without that test behaves the same (run of 2026-11-22).
    ("save_lead reuses the contact of a call that the passphrase does not open", "graph/tools.py", "if lead.call_control_id in ctx.open_calls and same_person(confirmed, lead.name)]", "if same_person(confirmed, lead.name)]"),
    ("identify gives the agents the name on record after a name match, with no passphrase check", "graph/tools.py", "    ctx.caller_name = name\n    on_file =", "    ctx.caller_name = memory.name\n    on_file ="),
    ("every passphrase is right", "passphrase.py", "right = hmac.compare_digest(_hash(phrase, salt), stored)", "right = True"),
    (
        "the CRM lookup reads a deal that the call of another person made",
        "crm.py",
        "made = {row.deal_id: row for row in records if row.deal_id and row.deal_made}",
        "made = {row.deal_id: row for row in records if row.deal_id}",
    ),
    (
        "the CRM lookup reads the records of every call of the number, not only the calls that the passphrase opens",
        "crm.py",
        "records = CrmStore(ctx.leads.connection).records_for(ctx.open_calls)",
        "records = CrmStore(ctx.leads.connection).records_for([row[0] for row in ctx.leads.connection.execute('SELECT call_control_id FROM crm_calls WHERE caller_number = ?', (ctx.caller_number,))])",
    ),
    # Not in this list on purpose: `request_status` without its `ctx.identity_verified` test. Only `verify` fills
    # `ctx.open_calls`, so `records_for(ctx.open_calls)` reads nothing for a caller who is not verified.
)


def run_mutants(max_calls: int) -> bool:
    source = Path(ai_call_center.__file__).parent
    ok = True
    for name, file, old, new in MUTANTS:
        text = (source / file).read_text(encoding="utf-8")
        if text.count(old) != 1:
            ok = False
            print(f"STALE     {name}\n          {file} no longer has this text one time: {old!r}. Update MUTANTS to the new code.")
            continue
        with tempfile.TemporaryDirectory(prefix="caller-memory-mutant-") as folder:
            shutil.copytree(source, Path(folder) / "ai_call_center", ignore=shutil.ignore_patterns("__pycache__"))
            (Path(folder) / "ai_call_center" / file).write_text(text.replace(old, new), encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": folder, "PYTHONDONTWRITEBYTECODE": "1"}
            run = subprocess.run([sys.executable, __file__, "--json", "--calls", str(max_calls)], env=env, capture_output=True, text=True, check=False)
            if run.returncode not in (0, 1) or not run.stdout.strip():
                ok = False
                print(f"ERROR     {name}\n{run.stderr[-800:]}")
                continue
            result = json.loads(run.stdout)
            if not result["package"].startswith(folder):
                ok = False
                print(f"ERROR     {name}\n          the search ran on {result['package']}, not on the broken copy")
                continue
            new_findings = [finding for finding in result["findings"] if (finding["kind"], finding["place"]) not in KNOWN_FINDINGS]
            if new_findings:
                first = min(new_findings, key=lambda finding: len(finding["steps"]))
                print(f"CAUGHT    {name}\n          a {first['kind']} in the {first['place']}, after {len(first['steps'])} steps")
            else:
                ok = False
                print(f"NOT CAUGHT {name}\n          the search passed on broken code, so it is too weak")
    print()
    print("PASS: the search fails on every broken copy." if ok else "FAIL.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calls", type=int, default=DEFAULT_CALLS, help=f"how many calls the search covers (default {DEFAULT_CALLS})")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--mutants", action="store_true", help="run the search on broken copies of the package; each one must fail")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.mutants:
        return 0 if run_mutants(args.calls) else 1
    started = time.monotonic()
    result = search(args.calls)
    ok, lines = verdict(result)
    if args.json:
        print(json.dumps(result))
    else:
        print("\n".join(lines))
        print(f"({time.monotonic() - started:.0f} s, on {result['package']})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
