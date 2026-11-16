"""FastAPI server: Telnyx webhooks, the media WebSocket, and a health check.

Run: uv run serve
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from ai_call_center.config import get_settings
from ai_call_center.guardrails import CallGuard
from ai_call_center.leads import LeadStore
from ai_call_center.telephony import media, webhooks
from ai_call_center.telephony.lookup import CallerLookup
from ai_call_center.telephony.telnyx_client import TelnyxTelephony
from ai_call_center.telephony.transfers import TransferManager

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.telephony = TelnyxTelephony(settings)
    app.state.lookups = CallerLookup(app.state.telephony, LeadStore(settings.leads_db), mode=settings.caller_lookup)
    app.state.guard = CallGuard(
        LeadStore(settings.leads_db),
        max_calls_per_number_per_day=settings.max_calls_per_number_per_day,
        max_concurrent_calls=settings.max_concurrent_calls,
        daily_minutes_cap=settings.daily_minutes_cap,
        max_call_minutes=settings.max_call_minutes,
        exempt_numbers=frozenset({settings.owner_phone}),
    )
    app.state.transfers = TransferManager(app.state.telephony, stream_url=settings.media_stream_url, ring_secs=settings.transfer_ring_secs)
    if settings.telnyx_app_id:
        try:
            if not await app.state.telephony.outbound_profile_id(settings.telnyx_app_id):
                app.state.transfers.unavailable = "the Telnyx application has no outbound voice profile"
                log.warning("LIVE TRANSFER IS OFF: the Telnyx application has no outbound voice profile. Run `uv run setup-transfer`, then restart.")
        except Exception as exc:  # noqa: BLE001 - do not block startup on this check
            log.error("could not check the Telnyx outbound voice profile: %s", exc)
    if settings.public_url and settings.telnyx_app_id:
        try:
            changed = await app.state.telephony.ensure_webhook_url(settings.telnyx_app_id, settings.webhook_url)
            log.info("Telnyx webhook URL %s: %s", "updated" if changed else "already correct", settings.webhook_url)
        except Exception as exc:  # noqa: BLE001 - the server is still useful; the user can fix the portal by hand
            log.error("could not set the Telnyx webhook URL: %s", exc)
    else:
        log.warning("PUBLIC_URL or TELNYX_APP_ID not set; the Telnyx webhook URL was not checked")
    log.info("ready: webhooks at %s, media at %s, number %s", settings.webhook_url, settings.media_stream_url, settings.telnyx_phone_number or "(none)")
    yield


app = FastAPI(title="AI business card", lifespan=lifespan)
app.include_router(webhooks.router)
app.include_router(media.router)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("ai_call_center.app:app", host="127.0.0.1", port=get_settings().port, log_level="info")


if __name__ == "__main__":
    main()
