"""POST /webhooks/telnyx: verify the signature, acknowledge fast, act in the background."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict

from fastapi import APIRouter, BackgroundTasks, Request, Response

from ai_call_center.config import Settings, get_settings
from ai_call_center.guardrails import CallGuard
from ai_call_center.telephony.lookup import CallerLookup
from ai_call_center.telephony.stream_tokens import StreamTokens
from ai_call_center.telephony.telnyx_client import TelnyxTelephony, decode_state
from ai_call_center.telephony.transfers import TransferManager

log = logging.getLogger(__name__)
router = APIRouter()


class HandledCalls:
    """The incoming calls this server already acted on. Telnyx can deliver one webhook more than once."""

    def __init__(self, keep: int = 1000) -> None:
        self._keep = keep
        self._ids: OrderedDict[str, None] = OrderedDict()

    def first_time(self, call_control_id: str) -> bool:
        """True the first time for a call, False for a repeat. Records the call."""
        if call_control_id in self._ids:
            return False
        self._ids[call_control_id] = None
        while len(self._ids) > self._keep:
            self._ids.popitem(last=False)  # the oldest call
        return True

    def forget(self, call_control_id: str) -> None:
        self._ids.pop(call_control_id, None)


def verify_and_parse(telephony: TelnyxTelephony, raw: str, headers) -> dict:
    """Raise on a bad signature. Return the event's `data` dict."""
    telephony.client.webhooks.unwrap(raw, headers=headers)  # raises if the Ed25519 signature is invalid
    return json.loads(raw)["data"]


async def handle_event(
    data: dict,
    settings: Settings,
    telephony: TelnyxTelephony,
    lookups: CallerLookup | None = None,
    guard: CallGuard | None = None,
    transfers: TransferManager | None = None,
    handled: HandledCalls | None = None,
    tokens: StreamTokens | None = None,
) -> None:
    event_type = data.get("event_type")
    payload = data.get("payload") or {}
    ccid = payload.get("call_control_id")
    state = decode_state(payload.get("client_state"))
    log.info("telnyx event %s call=%s state=%s", event_type, ccid, payload.get("state"))
    if transfers is not None:
        await transfers.on_event(event_type, ccid, state, str(payload.get("result") or ""), str(payload.get("digits") or ""))  # the owner's leg of a live transfer: answered, or not

    if event_type == "call.initiated" and payload.get("direction") == "incoming":
        caller = payload.get("from") or ""
        if handled is not None and not handled.first_time(ccid):
            # A second answer command would fail, and the guard would judge the same call twice.
            log.info("repeated call.initiated for call %s ignored", ccid)
            return
        log.info("incoming call from %s to %s", caller, payload.get("to"))
        try:
            admission = guard.admit(caller) if guard is not None else None
            if admission is not None and not admission.allowed:
                # Turned away before any AI session opens: a short spoken reason, then a hang-up.
                log.warning("call %s from %s turned away: %s", ccid, caller, admission.reason)
                await telephony.answer_and_say(ccid, admission.message)
                return
            if lookups is not None:
                lookups.start(caller)  # runs while we answer; the session reads the result
            # The token in the URL lets the media endpoint know this stream: it is for a call that passed the guard.
            stream_url = tokens.url_for(ccid) if tokens is not None else settings.media_stream_url
            await telephony.answer_with_stream(ccid, stream_url=stream_url)
        except Exception:
            if handled is not None:
                handled.forget(ccid)  # the call got no answer, so a repeat of the webhook may try again
            raise
    elif event_type == "call.answered":
        # A normal call needs nothing here: streaming was requested on answer. A call that was
        # turned away carries the sentence to say.
        if text := state.get("say"):
            await telephony.speak(ccid, text, then_hangup=True)
    elif event_type == "call.speak.ended":
        if state.get("then") == "hangup":
            await telephony.hangup(ccid)
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
    background.add_task(handle_event, data, settings, telephony, getattr(request.app.state, "lookups", None), getattr(request.app.state, "guard", None), getattr(request.app.state, "transfers", None), getattr(request.app.state, "handled", None), getattr(request.app.state, "stream_tokens", None))
    return Response(status_code=200)
