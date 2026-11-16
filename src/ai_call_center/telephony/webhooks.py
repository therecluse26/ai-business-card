"""POST /webhooks/telnyx: verify the signature, acknowledge fast, act in the background."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Request, Response

from ai_call_center.config import Settings, get_settings
from ai_call_center.telephony.telnyx_client import TelnyxTelephony

log = logging.getLogger(__name__)
router = APIRouter()


def verify_and_parse(telephony: TelnyxTelephony, raw: str, headers) -> dict:
    """Raise on a bad signature. Return the event's `data` dict."""
    telephony.client.webhooks.unwrap(raw, headers=headers)  # raises if the Ed25519 signature is invalid
    return json.loads(raw)["data"]


async def handle_event(data: dict, settings: Settings, telephony: TelnyxTelephony) -> None:
    event_type = data.get("event_type")
    payload = data.get("payload") or {}
    ccid = payload.get("call_control_id")
    log.info("telnyx event %s call=%s state=%s", event_type, ccid, payload.get("state"))

    if event_type == "call.initiated" and payload.get("direction") == "incoming":
        log.info("incoming call from %s to %s", payload.get("from"), payload.get("to"))
        await telephony.answer_with_stream(ccid, stream_url=settings.media_stream_url)
    elif event_type == "call.answered":
        pass  # streaming was requested on answer; Telnyx opens the WebSocket next
    elif event_type == "streaming.failed":
        log.error("streaming failed: %s", payload.get("failure_reason"))
    elif event_type == "call.hangup":
        log.info("call ended: cause=%s source=%s", payload.get("hangup_cause"), payload.get("hangup_source"))


@router.post("/webhooks/telnyx")
async def telnyx_webhook(request: Request, background: BackgroundTasks) -> Response:
    settings = get_settings()
    telephony: TelnyxTelephony = request.app.state.telephony
    raw = (await request.body()).decode()
    try:
        data = verify_and_parse(telephony, raw, request.headers)
    except Exception as exc:  # noqa: BLE001 - any failure here means "not from Telnyx"
        log.warning("rejected webhook: %s", exc)
        return Response(status_code=400)
    background.add_task(handle_event, data, settings, telephony)
    return Response(status_code=200)
