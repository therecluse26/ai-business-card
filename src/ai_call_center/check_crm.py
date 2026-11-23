"""`uv run check-crm`: ask the owner's CRM over MCP, to check the address, the token, and the time of a request.

It reads only: the tool list, the pipelines, and five searches for a word that no record has. It prints no
record and no token. Then it shows the CRM entries of this server's calls that still wait (see crm.py).

`uv run check-crm --send` also writes the entries that wait. The end of each call does that too.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time

from ai_call_center.config import get_settings
from ai_call_center.crm import REFUSED_STATE, WAITING, CrmError, CrmStore, McpCrm, crm_configured, write_waiting

TOOLS_USED = ("search", "get_record", "list_pipelines", "create_person", "create_company", "create_deal", "add_note", "create_task")


async def run(send: bool) -> int:
    settings = get_settings()
    if not crm_configured(settings):
        print("CRM_MCP_URL or CRM_API_TOKEN is not set in .env. Make an API token in the CRM (Settings, API tokens), add both, and run this again.")
        return 1
    crm = McpCrm(url=settings.crm_mcp_url, token=settings.crm_api_token.get_secret_value(), time_limit=settings.crm_timeout_secs)
    try:
        started = time.monotonic()
        listed = await crm.request("tools/list")
        first = (time.monotonic() - started) * 1000
        names = {tool["name"] for tool in listed.get("tools", [])}
        missing = [name for name in TOOLS_USED if name not in names]
        print(f"The CRM answers: {len(names)} tools. First request: {first:.0f} ms (it opens the connection).")
        if missing:
            print(f"These tools are missing, and crm.py needs them: {', '.join(missing)}.")
        config = await crm.tool("list_pipelines", {})
        pipelines = config.get("pipelines", [])
        if pipelines:
            stages = [stage["name"] for stage in pipelines[0].get("stages", [])]
            print(f"A new deal goes to the pipeline {pipelines[0]['name']!r}, stage {stages[0]!r}. Stages: {', '.join(stages)}.")
        else:
            print("The CRM has no pipeline, so a deal cannot be made. Make one in the CRM settings.")
        times = []
        for _ in range(5):
            started = time.monotonic()
            await crm.tool("search", {"query": "zzzzqqqq", "limit": 1})
            times.append(round((time.monotonic() - started) * 1000))
        print(f"Five searches, in ms: {', '.join(map(str, times))}. CRM_TIMEOUT_SECS={settings.crm_timeout_secs:g}.")
        connection = sqlite3.connect(str(settings.leads_db))
        store = CrmStore(connection)
        counts = dict(connection.execute("SELECT state, COUNT(*) FROM crm_calls GROUP BY state").fetchall())
        print(f"Entries of this server: {sum(counts.values())} in total, {counts.get(WAITING, 0)} waiting, {counts.get(REFUSED_STATE, 0)} refused by the CRM.")
        for call_id, error in connection.execute("SELECT call_control_id, error FROM crm_calls WHERE state = ? ORDER BY created_at", (REFUSED_STATE,)):
            print(f"  refused: call {call_id}: {error}")
        if send and (waiting := len(store.waiting())):
            print(f"Sent now: {await write_waiting(crm, connection)} of {waiting} waiting.")
        connection.close()
        return 1 if missing or not pipelines else 0
    except CrmError as error:
        print(f"The CRM gave no answer: {error}")
        return 1
    finally:
        await crm.aclose()


def main() -> None:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    sys.exit(asyncio.run(run("--send" in sys.argv[1:])))
