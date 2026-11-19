"""WS /media/<token>: Telnyx bidirectional media stream in, a Pipecat call session out.

The token is the proof that this server asked Telnyx for the stream (see stream_tokens.py).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket
from pipecat.runner.utils import parse_telephony_websocket

from ai_call_center.config import get_settings
from ai_call_center.voice.call_session import CallSession

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/media/{token}")
async def media_stream(websocket: WebSocket, token: str) -> None:
    tokens = getattr(websocket.app.state, "stream_tokens", None)
    expected_call = tokens.redeem(token) if tokens is not None else None
    if expected_call is None:
        # Not a stream this server asked for: refused before the WebSocket opens, so no AI session starts.
        log.warning("media stream refused: its token is unknown, used, or too old")
        await websocket.close(code=1008)
        return
    await websocket.accept()
    transport_type, call_data = await parse_telephony_websocket(websocket)
    if transport_type != "telnyx":
        log.warning("unexpected media stream from %s; closing", transport_type)
        await websocket.close()
        return
    if call_data.get("call_id") != expected_call:
        log.warning("media stream refused: its token is for call %s, and the stream names call %s", expected_call, call_data.get("call_id"))
        await websocket.close(code=1008)
        return
    log.info("media stream for call %s (stream %s, encoding %s)", call_data.get("call_id"), call_data.get("stream_id"), call_data.get("outbound_encoding"))
    transfers = getattr(websocket.app.state, "transfers", None)
    # A stream that starts again after a transfer that did not connect: the contact agent returns.
    resume = transfers.take_resume(call_data.get("call_id")) if transfers else None
    if resume is not None:
        log.info("call %s: media stream restarted; the %s agent takes the call (%s)", call_data.get("call_id"), resume.entry_agent, resume.reason)
    session = CallSession(
        websocket=websocket,
        call_data=call_data,
        settings=get_settings(),
        telephony=websocket.app.state.telephony,
        lookups=getattr(websocket.app.state, "lookups", None),
        transfers=transfers,
        resume=resume,
        health=getattr(websocket.app.state, "health", None),
    )
    await session.run()
