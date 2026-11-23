"""The owner's CRM (zepto-crm), reached over MCP.

After a call that left a lead for the owner, code writes it to the CRM. No model chooses these writes, and
none of them runs while the caller is on the line.

The rules of the writes. They are the same for every call, and this module is the only place that has them:

  Which call    A call that gave the owner a lead: a saved message, a note of a request for a live call, or
                a talk about a project or a role with no message (the warm lead email). A lead email and a
                CRM lead go together. A caller who asked to be forgotten leaves nothing.
  Person        One for each caller. The line uses a person that exists: an earlier call of the same number
                and name, a CRM person with that number and name, or the CRM person with that email
                address. The line fills a field of that person only when the field is empty (surname, email
                address, phone number, company). It never changes a field that has a value.
  Entry         One entry of the kind "call" on the person's timeline for each call. Never changed.
  Deal          Each lead is in one deal, in the first stage of the first pipeline. A person with an open
                deal from this line gets no second deal: the call joins the open one. The line never
                moves, renames or closes a deal. The stage belongs to the owner.
  Task          One "Follow up with <name>" for each call, due on the day of the call, on the deal. There
                is none when the call went to the owner's phone. The line never closes a task.
  Test call     A call from the owner's own phone gets the same records. Each one has "Test" in its name,
                so the owner can find them and delete them.

An entry that the CRM did not take waits in the table `crm_calls`, and the end of the next call sends it
again. So a CRM that is down loses nothing: the lead is in the store and in the owner's email already.

One read reaches an agent (`request_status`): where the caller's own request stands. A phone number can
be faked, so it needs the passphrase check of the call, and it reads only a deal that a call with that
passphrase made. It gives a status, a stage and two dates: no name, no note, no contact detail.

The MCP server has no session: each tool call is one POST with a Bearer token, and the answer is one
event of an event stream. So this module needs no MCP package.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ai_call_center.callers import EASTERN, UNKNOWN_NUMBERS, CallSummary
from ai_call_center.leads import now_iso
from ai_call_center.names import same_person

if TYPE_CHECKING:
    from ai_call_center.graph.state import CallContext

log = logging.getLogger(__name__)

WARM = ("project", "role")  # the interests that make a lead of a call with no message (the warm lead email)
UNNAMED = "Unnamed caller"
SYNC_LIMIT_SECS = 60.0  # all the writes after one call; what is not written by then waits for the next call
STATUS_LIMIT_SECS = 4.0  # the read during a call; after this the agent says that it cannot check now
STATUS_DEALS = 3  # the deals that one read gets, newest first
# The CRM refuses these for what the entry holds, so the same entry gets the same answer the next time.
# The longest text that the CRM takes in each field. A longer one is cut: a topic comes from a summary model.
NAME_MAX, TITLE_MAX, TASK_MAX, NOTES_MAX, BODY_MAX, QUERY_MAX = 120, 200, 300, 5_000, 50_000, 100
REFUSED = ("validation_failed", "bad_request")
# The CRM cannot be reached at all. The entries that still wait are not tried on this run.
DOWN = ("unreachable", "unauthenticated", "forbidden")


class CrmError(Exception):
    """A tool call that did not work. `code` is the CRM's error code, or one of this module: "unreachable", "protocol"."""

    def __init__(self, code: str, message: str = "", details: Any = None) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.details = details


@runtime_checkable  # the tool schema of a LangChain tool checks the call context's fields with isinstance
class Crm(Protocol):
    async def tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call one tool of the CRM's MCP server. Returns its JSON result. Raises CrmError."""
        ...


class McpCrm:
    """The CRM's MCP server over Streamable HTTP. One client for the process: a new client pays for a new connection."""

    def __init__(self, *, url: str, token: str, time_limit: float = 10.0, client: Any = None) -> None:
        import httpx

        self.url = url
        self._ids = 0
        self._client = client or httpx.AsyncClient(
            timeout=time_limit,
            follow_redirects=False,  # a redirect is the sign-in page of Cloudflare Access, not the CRM
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """One JSON-RPC request. Returns its `result`."""
        import httpx

        self._ids += 1
        try:
            response = await self._client.post(self.url, json={"jsonrpc": "2.0", "id": self._ids, "method": method, "params": params or {}})
        except httpx.HTTPError as error:
            raise CrmError("unreachable", type(error).__name__) from error
        if 300 <= response.status_code < 400:
            raise CrmError("unreachable", "the address gave a redirect: Cloudflare Access has no Bypass policy for /ext/")
        if response.status_code in (401, 403):
            raise CrmError("unauthenticated", f"status {response.status_code}: the API token is not valid")
        if response.status_code >= 400:
            raise CrmError("unreachable", f"status {response.status_code}")
        message = _rpc_message(response.text)
        if "error" in message:
            raise CrmError("protocol", str(message["error"].get("message", "")))
        return message.get("result") or {}

    async def tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self.request("tools/call", {"name": name, "arguments": arguments})
        try:
            data = json.loads(result["content"][0]["text"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise CrmError("protocol", f"{name} gave no JSON result") from error
        if result.get("isError"):
            fault = data.get("error", {}) if isinstance(data, dict) else {}
            raise CrmError(str(fault.get("code") or "internal"), str(fault.get("message") or ""), fault.get("details"))
        return data

    async def aclose(self) -> None:
        await self._client.aclose()


def _rpc_message(text: str) -> dict[str, Any]:
    """The JSON-RPC message of an answer. The server sends it as the `data:` line of one event, or as plain JSON."""
    for line in text.splitlines():
        if line.startswith("data:"):
            text = line[5:]
            break
    try:
        message = json.loads(text)
    except ValueError as error:
        raise CrmError("protocol", "the answer is not JSON") from error
    if not isinstance(message, dict):
        raise CrmError("protocol", "the answer is not a JSON-RPC message")
    return message


def crm_configured(settings: Any) -> bool:
    return bool(settings.crm_mcp_url and settings.crm_api_token.get_secret_value())


_default: McpCrm | None = None


def default_crm(settings: Any) -> Crm | None:
    """The process-wide CRM client, or None when CRM_MCP_URL or CRM_API_TOKEN is not set (nothing is written or read)."""
    global _default
    if not crm_configured(settings):
        return None
    if _default is None:
        _default = McpCrm(url=settings.crm_mcp_url, token=settings.crm_api_token.get_secret_value(), time_limit=settings.crm_timeout_secs)
        log.info("The CRM is on: MCP, time limit %.0f s for each request", settings.crm_timeout_secs)
    return _default


# --- what one call writes ---


@dataclass(frozen=True)
class CrmEntry:
    """What one call leaves in the CRM."""

    call_control_id: str
    caller_number: str  # "" when the number is hidden
    name: str
    email: str = ""
    company: str = ""
    title: str = ""  # of the timeline entry
    body: str = ""  # of the timeline entry
    deal: str = ""  # the name of the deal that is made when the person has no open one
    task: str = ""  # the title of the task; "" = no task
    task_notes: str = ""
    due: str = ""  # of the task, YYYY-MM-DD


def entry_for_call(ctx: CallContext, summary: CallSummary, *, visited: list[str], transferred: bool = False, test: bool = False, now: datetime | None = None) -> CrmEntry | None:
    """The CRM entry for a call that is over, or None when the call left nothing for the owner.

    The rules are at the top of this module. A call counts when the owner got a lead from it: a saved message,
    a note of a request for a live call, or a talk about a project or a role (the warm lead email).
    `transferred`: the owner's phone rings or the owner is on the line, so there is no task to follow up yet.
    `test`: the call is from the owner's own phone. It gets the same records, each with "Test" in its name.
    """
    if ctx.forgotten:
        return None
    saved = ctx.leads.lead_for_call(ctx.call_control_id)
    row_id, lead = saved if saved else (0, None)
    warm = summary.interest in WARM
    hidden = ctx.caller_number in UNKNOWN_NUMBERS
    if lead is None and (not warm or hidden):  # the same rule as the warm lead email (`send_warm_lead_at_call_end`)
        return None
    name = " ".join(((lead.name if lead else "") or ctx.caller_name or summary.caller_name or UNNAMED).split())
    company = lead.company if lead and lead.company.lower() != "none" else ""
    now = now or datetime.now(UTC)
    topic = " ".join(summary.topic.split())
    lines = [f"Message: {lead.need}" if lead else f"The caller left no message. What they talked about: {summary.owner_note or summary.summary or 'the call summary is empty'}"]
    when, size = (lead.timeline if lead else "") or summary.timeline, (lead.size if lead else "") or summary.size
    lines += [f"{label}: {value}" for label, value in (("Start", when), ("Size", size), ("Company", company)) if value]
    if lead and summary.summary:
        lines.append(f"Summary of the call: {summary.summary}")
    if ctx.identity_verified:
        lines.append("Caller: verified with their passphrase.")
    elif ctx.identity_confirmed:
        lines.append("Caller: the name matched an earlier caller on this number. No passphrase check, so it is not proof.")
    else:
        lines.append("Caller: new on this number.")
    reach = [f"email {lead.email}"] if lead and lead.email else []
    reach.append("a hidden number" if hidden else f"the number {ctx.caller_number}")
    lines.append(f"Reach them by: {', '.join(reach)}.")
    minutes = max(1, round((now - ctx.started_at).total_seconds() / 60))
    lines.append(f"Agents: {' -> '.join(visited)}. About {minutes} min, on {now.astimezone(EASTERN).strftime('%b %-d at %-I:%M %p Eastern')}.")
    if row_id:
        lines.append(f"Lead record {row_id} of the phone line.")
    follow_up = not transferred
    about = topic or (f"a {summary.interest}" if warm else "a message")
    mark = "Test: " if test else ""
    return CrmEntry(
        call_control_id=ctx.call_control_id,
        caller_number="" if hidden else ctx.caller_number,
        name=name,
        email=lead.email if lead else "",
        company=company,
        title=("Test call to the AI line" if test else "Call to the AI line") + (f": {topic}" if topic else ""),
        body="\n".join(lines),
        deal=f"{mark}{name}: {about}",
        task=f"{mark}Follow up with {name}" + (f": {topic}" if topic else "") if follow_up else "",
        task_notes=(lead.need if lead else summary.owner_note or summary.summary) if follow_up else "",
        due=now.astimezone(EASTERN).date().isoformat() if follow_up else "",
    )


# --- the link between a call and its CRM records ---

WAITING, WRITTEN, REFUSED_STATE = "waiting", "written", "refused"


@dataclass(frozen=True)
class CrmCall:
    call_control_id: str
    caller_number: str
    name: str
    person_id: str
    deal_id: str
    deal_made: bool  # this call made the deal; false = it uses the deal of an earlier call
    task_id: str
    noted: str  # a digest of the timeline entry that is in the CRM
    entry: str  # the entry that waits, as JSON; "" when it is written
    state: str
    tries: int
    created_at: str


_COLUMNS = "call_control_id, caller_number, name, person_id, deal_id, deal_made, task_id, noted, entry, state, tries, created_at"


class CrmStore:
    """One row for each call that has a CRM entry: the IDs of its CRM records, and the entry while it waits.

    It keeps its table in the lead store's database, as `PhraseStore` does.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crm_calls (
                call_control_id TEXT PRIMARY KEY,
                caller_number TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                person_id TEXT NOT NULL DEFAULT '',
                deal_id TEXT NOT NULL DEFAULT '',
                deal_made INTEGER NOT NULL DEFAULT 0,
                task_id TEXT NOT NULL DEFAULT '',
                noted TEXT NOT NULL DEFAULT '',
                entry TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL DEFAULT 'waiting',
                tries INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def queue(self, entry: CrmEntry) -> None:
        """Make the entry wait. A second entry for one call (the call came back after a transfer) replaces the first; the IDs stay."""
        self._conn.execute(
            "INSERT INTO crm_calls (call_control_id, caller_number, name, entry, state, created_at) VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(call_control_id) DO UPDATE SET name = excluded.name, entry = excluded.entry, state = excluded.state, tries = 0, error = ''",
            (entry.call_control_id, entry.caller_number, entry.name, json.dumps(asdict(entry)), WAITING, now_iso()),
        )
        self._conn.commit()

    def get(self, call_control_id: str) -> CrmCall | None:
        row = self._conn.execute(f"SELECT {_COLUMNS} FROM crm_calls WHERE call_control_id = ?", (call_control_id,)).fetchone()
        return _crm_call(row) if row else None

    def waiting(self) -> list[str]:
        """The calls with an entry that the CRM does not have yet, oldest first."""
        return [row[0] for row in self._conn.execute("SELECT call_control_id FROM crm_calls WHERE state = ? ORDER BY created_at, rowid", (WAITING,))]

    def update(self, call_control_id: str, **values: Any) -> None:
        names = ", ".join(f"{name} = ?" for name in values)
        self._conn.execute(f"UPDATE crm_calls SET {names} WHERE call_control_id = ?", (*values.values(), call_control_id))
        self._conn.commit()

    def failed(self, call_control_id: str, error: str, *, refused: bool) -> None:
        self._conn.execute(
            "UPDATE crm_calls SET tries = tries + 1, error = ?, state = ? WHERE call_control_id = ?",
            (error[:300], REFUSED_STATE if refused else WAITING, call_control_id),
        )
        self._conn.commit()

    def person_for(self, caller_number: str, name: str) -> str:
        """The CRM person of an earlier call from this number by a person with this name, or ""."""
        if not caller_number or not name:
            return ""
        rows = self._conn.execute(
            "SELECT person_id, name FROM crm_calls WHERE caller_number = ? AND person_id != '' ORDER BY created_at DESC, rowid DESC", (caller_number,)
        )
        return next((person_id for person_id, stored in rows if same_crm_person(name, stored)), "")

    def deals_of(self, person_id: str, *, exclude: str) -> list[str]:
        """The deals that earlier calls of this CRM person have, newest first."""
        rows = self._conn.execute(
            "SELECT deal_id FROM crm_calls WHERE person_id = ? AND deal_id != '' AND call_control_id != ? ORDER BY created_at DESC, rowid DESC",
            (person_id, exclude),
        )
        return list(dict.fromkeys(row[0] for row in rows))

    def forget_person(self, person_id: str) -> None:
        """The CRM no longer has this person (the owner deleted or merged it). The next write finds or makes the person again."""
        self._conn.execute("UPDATE crm_calls SET person_id = '', deal_id = '', deal_made = 0, task_id = '', noted = '' WHERE person_id = ?", (person_id,))
        self._conn.commit()

    def records_for(self, call_ids: list[str]) -> list[CrmCall]:
        """The rows of these calls that are in the CRM, newest first."""
        if not call_ids:
            return []
        marks = ", ".join("?" * len(call_ids))
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM crm_calls WHERE call_control_id IN ({marks}) AND person_id != '' ORDER BY created_at DESC, rowid DESC", call_ids
        )
        return [_crm_call(row) for row in rows]

    def forget(self, call_ids: list[str]) -> None:
        """The caller asked to be forgotten: the links of their calls go, and so does an entry that still waits.
        What the CRM has already stays with the owner, as the email does."""
        self._conn.executemany("DELETE FROM crm_calls WHERE call_control_id = ?", [(call_id,) for call_id in call_ids])
        self._conn.commit()


def _crm_call(row: tuple) -> CrmCall:
    return CrmCall(*row[:5], bool(row[5]), *row[6:])


# --- the writes ---


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def same_crm_person(spoken: str, stored: str) -> bool:
    """True when two names are one person for the CRM: `same_person` (the first names match), and no two different surnames.

    The line itself matches by the first name only, because a caller says "Sam" on one call and "Sam Carter" on the next.
    That is too wide for the owner's records: "Sam Pike" and "Sam Carter" on one shared phone are two persons.
    """
    if not same_person(spoken, stored):
        return False
    a, b = spoken.split(), stored.split()
    return len(a) < 2 or len(b) < 2 or a[-1].casefold() == b[-1].casefold()


def _record_gone(error: CrmError) -> bool:
    """True when the CRM says that the linked record does not exist. It answers that as a fault of the field `recordId`."""
    if error.code == "not_found":
        return True
    details = error.details if isinstance(error.details, list) else []
    return error.code == "validation_failed" and any(isinstance(item, dict) and item.get("path") == "recordId" for item in details)


async def _company(crm: Crm, name: str) -> str:
    """The ID of the company with this name; it is made when the CRM has none. "" when there is no name, or the CRM refused."""
    if not name.strip():
        return ""
    try:
        name = name.strip()[:TITLE_MAX]
        found = await crm.tool("search", {"query": name[:QUERY_MAX], "types": ["company"], "limit": 5})
        for hit in found.get("hits", []):
            if str(hit.get("title", "")).casefold() == name.casefold():
                return str(hit["id"])
        return str((await crm.tool("create_company", {"name": name}))["id"])
    except CrmError as error:
        if error.code in DOWN:
            raise
        log.warning("CRM: no company for %r (%s)", name, error)  # the person is written without it; the timeline entry names the company
        return ""


async def _find_or_make_person(crm: Crm, entry: CrmEntry) -> tuple[str, bool]:
    """The CRM person for the caller of this entry, and True when this call made the person.

    The CRM has one unique field for a person: the email address. So the search goes by the phone number,
    and a hit counts only with the same name (`same_crm_person`) and the same number on the record.
    """
    digits = _digits(entry.caller_number)
    if digits:
        # The CRM search matches whole words. A number is one word when it is stored as +15551234567, which is how this line stores it.
        found = await crm.tool("search", {"query": digits, "types": ["person"], "limit": 10})
        named = [hit for hit in found.get("hits", []) if same_crm_person(entry.name, str(hit.get("title", "")))]
        for hit in named[:3]:
            person = (await crm.tool("get_record", {"type": "person", "id": hit["id"], "timeline": 0})).get("record") or {}
            if _digits(str(person.get("phone") or "")) == digits:
                return str(hit["id"]), False
    first, _, last = entry.name.partition(" ")
    fields: dict[str, Any] = {"firstName": first[:NAME_MAX], "lastName": last.strip()[:NAME_MAX]}
    if entry.caller_number:
        fields["phone"] = entry.caller_number
    if company_id := await _company(crm, entry.company):
        fields["companyId"] = company_id
    try:
        return str((await crm.tool("create_person", {**fields, **({"email": entry.email} if entry.email else {})}))["id"]), True
    except CrmError as error:
        existing = error.details.get("existingId") if isinstance(error.details, dict) else None
        if error.code == "conflict" and existing:
            return str(existing), False  # a person with this email address is in the CRM
        if error.code == "validation_failed" and entry.email:
            # An address as a voice model heard it can be no address. The timeline entry still has it.
            return str((await crm.tool("create_person", fields))["id"]), True
        raise


async def _fill_person(crm: Crm, person_id: str, entry: CrmEntry) -> None:
    """A person that the CRM has already gets what this call adds: a surname, an email address, a phone number, a company.

    Only a field that is empty is filled. A field with a value stays: the owner can have corrected it.
    Raises CrmError "not_found" when the CRM no longer has the person.
    """
    person = (await crm.tool("get_record", {"type": "person", "id": person_id, "timeline": 0})).get("record") or {}
    surname = entry.name.partition(" ")[2].strip() if entry.name != UNNAMED else ""
    new: dict[str, Any] = {}
    if surname and not person.get("lastName"):
        new["lastName"] = surname[:NAME_MAX]
    if entry.caller_number and not person.get("phone"):
        new["phone"] = entry.caller_number
    if entry.company and not person.get("companyId") and (company_id := await _company(crm, entry.company)):
        new["companyId"] = company_id
    email = entry.email if not person.get("email") else ""
    for fields in ({**new, "email": email}, new) if email else (new,):
        if not fields:
            return
        try:
            await crm.tool("update_person", {"id": person_id, **fields})
            return
        except CrmError as error:
            if error.code in DOWN or error.code == "not_found":
                raise
            # The address is no address, or another person has it: the other fields are sent without it. The entry has it.
            log.warning("CRM: the person %s did not take %s (%s)", person_id, ", ".join(fields), error)


async def _open_deal(crm: Crm, store: CrmStore, person_id: str, *, exclude: str) -> str:
    """An open deal that an earlier call of this person has, or "". A second call about one project makes no second deal."""
    for deal_id in store.deals_of(person_id, exclude=exclude)[:STATUS_DEALS]:
        try:
            deal = (await crm.tool("get_record", {"type": "deal", "id": deal_id, "timeline": 0})).get("record") or {}
        except CrmError as error:
            if error.code != "not_found":
                raise
            continue
        if deal.get("status") == "open":
            return deal_id
    return ""


async def _write(crm: Crm, store: CrmStore, call_control_id: str) -> None:
    """Write the entry of one call. Each step that worked is kept, so a second try starts after it."""
    row = store.get(call_control_id)
    if row is None or not row.entry:
        return
    entry = CrmEntry(**json.loads(row.entry))
    person_id = row.person_id
    if not person_id:
        person_id, made = store.person_for(entry.caller_number, entry.name), False
        if not person_id:
            person_id, made = await _find_or_make_person(crm, entry)
        if not made:
            try:
                await _fill_person(crm, person_id, entry)
            except CrmError as error:
                if error.code != "not_found":
                    raise
                store.forget_person(person_id)
                raise CrmError("gone", "the CRM no longer has this person") from error
        store.update(call_control_id, person_id=person_id)
    digest = _digest(entry.body)
    if digest != row.noted:
        try:
            await crm.tool("add_note", {"type": "person", "recordId": person_id, "kind": "call", "title": entry.title[:TITLE_MAX], "body": entry.body[:BODY_MAX]})
        except CrmError as error:
            if _record_gone(error):
                store.forget_person(person_id)
                raise CrmError("gone", "the CRM no longer has this person") from error
            raise
        store.update(call_control_id, noted=digest)
    deal_id = row.deal_id
    if not deal_id:
        deal_id, made = await _open_deal(crm, store, person_id, exclude=call_control_id), False
        if not deal_id:
            company_id = await _company(crm, entry.company)
            deal = await crm.tool("create_deal", {"name": (entry.deal or f"{entry.name}: a message")[:TITLE_MAX], "personIds": [person_id], **({"companyId": company_id} if company_id else {})})
            deal_id, made = str(deal["id"]), True
        store.update(call_control_id, deal_id=deal_id, deal_made=int(made))
    if entry.task and not row.task_id:
        task = await crm.tool("create_task", {"title": entry.task[:TASK_MAX], "notes": entry.task_notes[:NOTES_MAX] or None, "dueDate": entry.due or None, "type": "deal", "recordId": deal_id})
        store.update(call_control_id, task_id=str(task["id"]))
    store.update(call_control_id, entry="", state=WRITTEN, error="")


_lock: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = None


def _one_at_a_time() -> asyncio.Lock:
    """Two calls that end together must not write the same waiting entry two times."""
    global _lock
    loop = asyncio.get_running_loop()
    if _lock is None or _lock[0] is not loop:
        _lock = (loop, asyncio.Lock())
    return _lock[1]


async def write_waiting(crm: Crm, connection: sqlite3.Connection) -> int:
    """Write each entry that waits, oldest first. Returns how many are in the CRM now. Never raises."""
    store, written = CrmStore(connection), 0
    async with _one_at_a_time():
        for call_control_id in store.waiting():
            for attempt in (1, 2):
                try:
                    await _write(crm, store, call_control_id)
                    written += 1
                except CrmError as error:
                    if error.code == "gone" and attempt == 1:
                        continue  # the link to a deleted person is cleared; this try makes the person again
                    refused = error.code in REFUSED
                    store.failed(call_control_id, str(error), refused=refused)
                    if refused:
                        # The entry stays in the table with the CRM's reason. The lead is in the owner's email.
                        log.error("CRM: the entry of call %s was refused and is not sent again: %s", call_control_id, error)
                    else:
                        log.warning("CRM: the entry of call %s was not written; it waits for the next call end (%s)", call_control_id, error)
                    if error.code in DOWN:
                        return written
                except Exception:  # a changed result of a tool, for example
                    log.exception("CRM: the entry of call %s was not written; it waits for the next call end", call_control_id)
                    store.failed(call_control_id, "an error in this server; see the log", refused=False)
                break
    return written


async def sync_call(ctx: CallContext, summary: CallSummary, *, visited: list[str], transferred: bool = False, test: bool = False) -> None:
    """At the end of a call: queue what the call leaves for the CRM, then write each entry that waits. Never raises."""
    if ctx.crm is None:
        return
    try:
        entry = entry_for_call(ctx, summary, visited=visited, transferred=transferred, test=test)
        if entry is not None:
            CrmStore(ctx.leads.connection).queue(entry)
        written = await asyncio.wait_for(write_waiting(ctx.crm, ctx.leads.connection), SYNC_LIMIT_SECS)
        if written or entry is not None:
            log.info("call %s: CRM: %d entr%s written%s", ctx.call_control_id, written, "y" if written == 1 else "ies", "" if entry is not None else "; this call left none")
    except Exception:
        log.exception("call %s: the CRM write failed; an entry that was queued waits for the next call end", ctx.call_control_id)


# --- the read ---

NOT_CONNECTED = "No CRM is connected to this line, so there is nothing to look up. Say that you cannot check it, and offer to take a message."
NOT_VERIFIED = (
    "Nothing was read. A CRM record is given only after the passphrase check, and this caller has not passed it on this call. "
    "Say in one sentence that you can look up a request only for a caller who gave their passphrase at the start of the call. "
    "Do not ask for a passphrase now. Offer to take a message."
)


def _day(moment: datetime) -> str:
    return moment.astimezone(EASTERN).strftime("%b %-d")


def _iso_day(iso_utc: str) -> str:
    try:
        return _day(datetime.fromisoformat(iso_utc))
    except ValueError:
        return "an earlier day"


def _deal_line(deal: dict[str, Any], stages: dict[str, tuple[str, int, int]], *, asked: str, owner: str) -> str:
    """One deal, as the agent gets it: the status, the stage and the day the stage began. Not its name: the owner can write anything there."""
    status = deal.get("status")
    if status == "won":
        return f"- {asked}: {owner} marked it as won."
    if status == "lost":
        return f"- {asked}: {owner} closed it."
    if deal.get("stageId") not in stages:
        return f"- {asked}: open."
    name, place, count = stages[deal["stageId"]]
    entered = deal.get("stageEnteredAt")
    since = f" since {_day(datetime.fromtimestamp(entered / 1000, UTC))}" if isinstance(entered, int | float) else ""
    return f'- {asked}: open, in the stage "{name}" (stage {place} of {count}){since}.'


async def request_status(ctx: CallContext) -> str:
    """Where the caller's own earlier request stands in the CRM. The text for the model.

    Only after the passphrase check, and only from the calls that the passphrase opens (`ctx.open_calls`).
    A deal counts only when one of those calls made it: a call that only joined the deal of an earlier
    call can be the call of another person with the same name on the same number. Such a caller hears
    that the owner has their message, and no stage.
    """
    owner = ctx.owner_name.split()[0] if ctx.owner_name.strip() else "the owner"
    if ctx.crm is None:
        return NOT_CONNECTED
    if ctx.forgotten or not ctx.identity_verified:
        return NOT_VERIFIED
    records = CrmStore(ctx.leads.connection).records_for(ctx.open_calls)
    if not records:
        return f"The CRM has no record from this caller's earlier calls. Say that in one sentence, and offer to take a message for {owner}."
    made = {row.deal_id: row for row in records if row.deal_id and row.deal_made}
    deal_ids = list(made)[:STATUS_DEALS]
    lines: list[str] = []
    if deal_ids:
        try:
            pipelines, *deals = await asyncio.wait_for(
                asyncio.gather(ctx.crm.tool("list_pipelines", {}), *(ctx.crm.tool("get_record", {"type": "deal", "id": deal_id, "timeline": 0}) for deal_id in deal_ids)),
                STATUS_LIMIT_SECS,
            )
            stages = {
                stage["id"]: (stage["name"], place, len(pipeline["stages"]))
                for pipeline in pipelines.get("pipelines", [])
                for place, stage in enumerate(pipeline.get("stages", []), 1)
            }
            for deal_id, got in zip(deal_ids, deals, strict=True):
                lines.append(_deal_line(got.get("record") or {}, stages, asked=f"The request from {_iso_day(made[deal_id].created_at)}", owner=owner))
        except Exception as error:  # the CRM is down or slow, or a result has another shape: the call goes on
            log.warning("call %s: the CRM read failed (%s: %s)", ctx.call_control_id, type(error).__name__, error)
            return f"The CRM did not answer. Say in one sentence that you cannot check it right now, and that {owner} has their message."
    else:
        lines.append(f"- {owner}'s CRM has the caller's message from {_iso_day(records[0].created_at)}. This call can give no stage for it. Say only that {owner} has the message.")
    return (
        f"Read from {owner}'s CRM over MCP, just now:\n"
        + "\n".join(lines)
        + "\nTell the caller in one or two sentences. You may say that you read it from the CRM over MCP. "
        "Say nothing more about the record than this. Never promise a date, an answer or a result."
    )
