"""Telnyx Call Control commands used by the server. Implements the Telephony seam."""

from __future__ import annotations

import logging
from uuid import uuid4

from telnyx import AsyncTelnyx

from ai_call_center.config import Settings

log = logging.getLogger(__name__)


class TelnyxTelephony:
    """Thin wrapper over the Telnyx SDK. Every command gets a fresh command_id, so a retry is safe."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncTelnyx(
            api_key=settings.telnyx_api_key.get_secret_value(),
            public_key=settings.telnyx_public_key or None,
        )

    @staticmethod
    def _cid() -> str:
        return str(uuid4())

    async def answer_with_stream(self, call_control_id: str, *, stream_url: str) -> None:
        """Answer the call and start bidirectional PCMU media streaming in one command."""
        await self.client.calls.actions.answer(
            call_control_id,
            command_id=self._cid(),
            stream_url=stream_url,
            stream_track="inbound_track",
            stream_bidirectional_mode="rtp",
            stream_bidirectional_codec="PCMU",
        )

    async def start_streaming(self, call_control_id: str, *, stream_url: str) -> None:
        await self.client.calls.actions.start_streaming(
            call_control_id,
            command_id=self._cid(),
            stream_url=stream_url,
            stream_track="inbound_track",
            stream_bidirectional_mode="rtp",
            stream_bidirectional_codec="PCMU",
        )

    async def stop_streaming(self, call_control_id: str) -> None:
        await self.client.calls.actions.stop_streaming(call_control_id, command_id=self._cid())

    async def transfer(self, call_control_id: str, *, to: str) -> None:
        await self.client.calls.actions.transfer(call_control_id, to=to, command_id=self._cid())

    async def transfer_from_stream(self, call_control_id: str, *, to: str) -> None:
        """Transfer a call that has an active media stream.

        Telnyx refuses a transfer while streaming is on (error 90045, "Media Streaming is used"),
        so the stream stops first. Stopping it closes our media WebSocket, which ends the Pipecat
        session. If the transfer then fails, the caller hears a spoken apology and the call ends.
        """
        await self.stop_streaming(call_control_id)
        try:
            await self.transfer(call_control_id, to=to)
        except Exception:
            log.exception("transfer of %s to %s failed after streaming stopped", call_control_id, to)
            try:
                await self.speak(call_control_id, "Sorry, I could not connect the call. The owner will follow up with you. Goodbye.")
                await self.hangup(call_control_id)
            except Exception:
                log.exception("fallback speak/hangup for %s failed", call_control_id)
            raise

    async def speak(self, call_control_id: str, text: str) -> None:
        await self.client.calls.actions.speak(
            call_control_id, payload=text, voice="Telnyx.KokoroTTS.af", command_id=self._cid()
        )

    async def hangup(self, call_control_id: str) -> None:
        await self.client.calls.actions.hangup(call_control_id, command_id=self._cid())

    async def ensure_webhook_url(self, app_id: str, url: str) -> bool:
        """Point the Voice API application at this server. Returns True when it changed the URL."""
        current = await self.client.call_control_applications.retrieve(app_id)
        data = current.model_dump().get("data") or {}
        if data.get("webhook_event_url") == url:
            return False
        await self.client.call_control_applications.update(app_id, application_name=data.get("application_name") or "ai-business-card", webhook_event_url=url)
        log.info("Telnyx application %s webhook_event_url set to %s", app_id, url)
        return True
