"""WS /media: Telnyx bidirectional media stream in, a Pipecat call session out."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket
from pipecat.runner.utils import parse_telephony_websocket

from ai_call_center.config import get_settings
from ai_call_center.voice.call_session import CallSession

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/media")
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    transport_type, call_data = await parse_telephony_websocket(websocket)
    if transport_type != "telnyx":
        log.warning("unexpected media stream from %s; closing", transport_type)
        await websocket.close()
        return
    log.info("media stream for call %s (stream %s, encoding %s)", call_data.get("call_id"), call_data.get("stream_id"), call_data.get("outbound_encoding"))
    session = CallSession(websocket=websocket, call_data=call_data, settings=get_settings(), telephony=websocket.app.state.telephony)
    await session.run()
