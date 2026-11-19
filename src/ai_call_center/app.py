"""FastAPI server: Telnyx webhooks, the media WebSocket, and a health check.

Run: uv run serve
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Response

from ai_call_center import tracing
from ai_call_center.config import Settings, get_settings
from ai_call_center.guardrails import CallGuard
from ai_call_center.leads import LeadStore
from ai_call_center.telephony import media, webhooks
from ai_call_center.telephony.hold_audio import ringback_wav
from ai_call_center.telephony.lookup import CallerLookup
from ai_call_center.telephony.stream_tokens import StreamTokens
from ai_call_center.telephony.telnyx_client import TelnyxTelephony
from ai_call_center.telephony.transfers import TransferManager
from ai_call_center.voice.health import ProviderHealth

log = logging.getLogger(__name__)


def agent_legs_line(settings: Settings) -> str:
    """One line that says which provider and voice each agent uses on this server, for the start-up log."""
    from ai_call_center.graph.build import default_specs
    from ai_call_center.voice.legs import resolve_spec

    parts = []
    for name, spec in default_specs().items():
        runs = resolve_spec(spec, settings)
        moved = "" if runs.leg == spec.leg else f" (set to {spec.leg}, which has no key)"
        parts.append(f"{name}={runs.leg}/{runs.voice}{moved}")
    return ", ".join(parts)


def log_to_file(path: str) -> None:
    """Send the server's log lines and Pipecat's log lines to one file as well as to the terminal."""
    from pathlib import Path

    from loguru import logger

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        file.replace(file.with_name(file.name + ".previous"))
    handler = logging.FileHandler(file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logger.add(str(file), level="DEBUG", encoding="utf-8")  # Pipecat logs with loguru


async def check_telnyx_application(telephony: TelnyxTelephony, transfers: TransferManager, settings: Settings) -> None:
    """Can the Telnyx application make a transfer, and does it send its webhooks to this server?"""
    if settings.telnyx_app_id:
        try:
            if not await telephony.outbound_profile_id(settings.telnyx_app_id):
                transfers.unavailable = "the Telnyx application has no outbound voice profile"
                log.warning("LIVE TRANSFER IS OFF: the Telnyx application has no outbound voice profile. Run `uv run setup-transfer`, then restart.")
        except Exception as exc:  # noqa: BLE001 - the server still takes calls without this check
            log.error("could not check the Telnyx outbound voice profile: %s", exc)
    if settings.public_url and settings.telnyx_app_id:
        try:
            changed = await telephony.ensure_webhook_url(settings.telnyx_app_id, settings.webhook_url)
            log.info("Telnyx webhook URL %s: %s", "updated" if changed else "already correct", settings.webhook_url)
        except Exception as exc:  # noqa: BLE001 - the server is still useful; the user can fix the portal by hand
            log.error("could not set the Telnyx webhook URL: %s", exc)
    else:
        log.warning("PUBLIC_URL or TELNYX_APP_ID not set; the Telnyx webhook URL was not checked")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    tracing.start(settings)  # before any call builds a pipeline
    log.info("voice providers at start: %s", agent_legs_line(settings))
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
    app.state.handled = webhooks.HandledCalls()  # so a repeated call.initiated webhook does not answer a call twice
    app.state.health = ProviderHealth()  # which voice providers are answering; shared by all calls
    # The media endpoint takes a stream only with a one-time token that this server gave Telnyx for a call.
    app.state.stream_tokens = StreamTokens(settings.media_stream_url)
    app.state.transfers = TransferManager(
        app.state.telephony,
        stream_url=settings.media_stream_url,
        ring_secs=settings.transfer_ring_secs,
        detect_voicemail=settings.transfer_voicemail_detection,
        screen=settings.transfer_screening and bool(settings.telnyx_app_id),  # the call to the owner needs the app id
        hold_audio_url=settings.hold_audio_url,
        accept_secs=settings.transfer_accept_secs,
        tokens=app.state.stream_tokens,
    )
    # The two checks of the Telnyx application take about 0.7 s. They run while the server already takes
    # calls, because a call can be the reason the server started. A call waits for them only where it
    # needs the result (`TransferManager.settled`).
    checks = asyncio.create_task(check_telnyx_application(app.state.telephony, app.state.transfers, settings), name="telnyx-start-checks")
    app.state.transfers.checked = checks
    log.info("ready: webhooks at %s, media at %s/<token>, number %s", settings.webhook_url, settings.media_stream_url, settings.telnyx_phone_number or "(none)")
    yield
    checks.cancel()


app = FastAPI(title="AI business card", lifespan=lifespan)
app.include_router(webhooks.router)
app.include_router(media.router)


@app.get("/audio/ringback.wav")
async def ringback() -> Response:
    """The ring tone a caller hears while the owner's phone rings. Telnyx fetches it."""
    return Response(content=ringback_wav(), media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if get_settings().server_log:
        log_to_file(get_settings().server_log)
    uvicorn.run("ai_call_center.app:app", host=get_settings().host, port=get_settings().port, log_level="info")


if __name__ == "__main__":
    main()
