"""FastAPI server: Telnyx webhooks, the media WebSocket, and a health check.

Run: uv run serve
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from ai_call_center.config import get_settings
from ai_call_center.telephony import media, webhooks
from ai_call_center.telephony.telnyx_client import TelnyxTelephony

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.telephony = TelnyxTelephony(settings)
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
