"""Telnyx Call Control commands used by the server. Implements the Telephony seam."""

from __future__ import annotations

import base64
import json
import logging
from uuid import uuid4

from telnyx import AsyncTelnyx

from ai_call_center.config import Settings
from ai_call_center.leads import NumberInfo

log = logging.getLogger(__name__)


def encode_state(state: dict) -> str:
    """Telnyx `client_state`: base64 text that Telnyx sends back on the webhooks that follow a command."""
    return base64.b64encode(json.dumps(state).encode()).decode()


def decode_state(value: str | None) -> dict:
    if not value:
        return {}
    try:
        state = json.loads(base64.b64decode(value))
    except ValueError:
        return {}
    return state if isinstance(state, dict) else {}


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

    async def transfer(self, call_control_id: str, *, to: str, timeout_secs: int | None = None, state: dict | None = None) -> None:
        """Call `to` and bridge it to this call when it answers. The call must have no media stream
        (Telnyx error 90045); `TransferManager` stops the stream first.

        `state` is set as the new leg's `client_state`, so the webhooks for the owner's leg can be
        told apart. The owner's phone shows this line's number as the caller ID.
        """
        extra: dict = {}
        if timeout_secs:
            extra["timeout_secs"] = timeout_secs
        if state:
            extra["target_leg_client_state"] = encode_state(state)
        if self.settings.telnyx_phone_number:
            extra["from_"] = self.settings.telnyx_phone_number
        await self.client.calls.actions.transfer(call_control_id, to=to, command_id=self._cid(), **extra)

    async def send_text(self, to: str, body: str) -> None:
        """One SMS from the line's number. US carriers deliver it only once the number has a
        messaging profile and a registered 10DLC campaign."""
        await self.client.messages.send(to=to, from_=self.settings.telnyx_phone_number, text=body, type="SMS")

    async def lookup_number(self, number: str, *, with_name: bool) -> NumberInfo:
        """Telnyx Number Lookup: carrier and line type, plus the registered caller name when asked."""
        types = ["carrier", "caller-name"] if with_name else ["carrier"]
        response = await self.client.number_lookup.retrieve(number, extra_query={"type": types}, timeout=5.0)
        data = response.data
        if data is None:
            return NumberInfo()
        carrier, port, cnam = data.carrier, data.portability, data.caller_name
        return NumberInfo(
            caller_name=(cnam.caller_name if cnam else "") or "",
            carrier=(carrier.name if carrier else "") or (port.spid_carrier_name if port else "") or "",
            line_type=(carrier.type if carrier else "") or (port.line_type if port else "") or "",
            city=(port.city if port else "") or "",
            state=(port.state if port else "") or "",
        )

    async def speak(self, call_control_id: str, text: str, *, then_hangup: bool = False, state: dict | None = None) -> None:
        """Telnyx text-to-speech on the call. With `then_hangup`, the webhook handler hangs up when
        Telnyx reports the speech has ended; a hang-up sent right away would cut the speech off.
        `state` is set as the call's `client_state`; it replaces any state set earlier."""
        state = {"then": "hangup"} if then_hangup else state
        extra = {"client_state": encode_state(state)} if state else {}
        await self.client.calls.actions.speak(
            call_control_id, payload=text, voice="Telnyx.KokoroTTS.af", command_id=self._cid(), **extra
        )

    async def answer_and_say(self, call_control_id: str, text: str) -> None:
        """Answer with no media stream (no AI session, no model cost). The webhook handler speaks
        `text` when Telnyx reports the call answered, then hangs up."""
        await self.client.calls.actions.answer(call_control_id, command_id=self._cid(), client_state=encode_state({"say": text}))

    async def hangup(self, call_control_id: str) -> None:
        await self.client.calls.actions.hangup(call_control_id, command_id=self._cid())

    async def outbound_profile_id(self, app_id: str) -> str:
        """The outbound voice profile of the Voice API application, or "" when it has none."""
        current = await self.client.call_control_applications.retrieve(app_id)
        data = current.model_dump().get("data") or {}
        return str((data.get("outbound") or {}).get("outbound_voice_profile_id") or "")

    async def ensure_outbound_profile(self, app_id: str, *, name: str = "ai-business-card") -> tuple[str, bool, bool]:
        """Give the application an outbound voice profile, which a live transfer needs.

        Returns (profile id, created now, assigned now). A new profile is limited on purpose:
        US destinations only, two calls at a time, and a spend limit of 5 USD a day.
        """
        current = await self.client.call_control_applications.retrieve(app_id)
        app = current.model_dump().get("data") or {}
        assigned = str((app.get("outbound") or {}).get("outbound_voice_profile_id") or "")
        if assigned:
            return assigned, False, False
        profiles = (await self.client.outbound_voice_profiles.list()).model_dump().get("data") or []
        profile_id = next((str(p["id"]) for p in profiles if p.get("name") == name), "")
        created = False
        if not profile_id:
            made = await self.client.outbound_voice_profiles.create(
                name=name,
                whitelisted_destinations=["US"],
                concurrent_call_limit=2,
                daily_spend_limit="5.00",
                daily_spend_limit_enabled=True,
            )
            profile_id = str((made.model_dump().get("data") or {})["id"])
            created = True
        await self.client.call_control_applications.update(
            app_id,
            application_name=app.get("application_name") or name,
            webhook_event_url=app.get("webhook_event_url"),
            outbound={"outbound_voice_profile_id": profile_id},
        )
        log.info("Telnyx application %s now uses outbound voice profile %s", app_id, profile_id)
        return profile_id, created, True

    async def ensure_webhook_url(self, app_id: str, url: str) -> bool:
        """Point the Voice API application at this server. Returns True when it changed the URL."""
        current = await self.client.call_control_applications.retrieve(app_id)
        data = current.model_dump().get("data") or {}
        if data.get("webhook_event_url") == url:
            return False
        await self.client.call_control_applications.update(app_id, application_name=data.get("application_name") or "ai-business-card", webhook_event_url=url)
        log.info("Telnyx application %s webhook_event_url set to %s", app_id, url)
        return True
