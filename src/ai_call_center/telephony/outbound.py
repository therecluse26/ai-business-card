"""One-time Telnyx setup for the live transfer: an outbound voice profile on the Voice API application.

A transfer is an outbound call from the application to the owner's phone. Telnyx refuses it
(error 10010, "Connection has no Outbound Profile assigned") until the application has an
outbound voice profile.

Run: uv run setup-transfer
"""

from __future__ import annotations

import asyncio
import sys

from ai_call_center.config import get_settings
from ai_call_center.telephony.telnyx_client import TelnyxTelephony


async def run() -> int:
    settings = get_settings()
    if not settings.telnyx_app_id or not settings.telnyx_api_key.get_secret_value():
        print("TELNYX_API_KEY and TELNYX_APP_ID must be set in .env. Run: bash scripts/telnyx-setup.sh")
        return 1
    telephony = TelnyxTelephony(settings)
    profile_id, created, assigned = await telephony.ensure_outbound_profile(settings.telnyx_app_id)
    print(f"Outbound voice profile {profile_id}: {'created' if created else 'already there'}.")
    print(f"Application {settings.telnyx_app_id}: {'profile assigned now' if assigned else 'profile was already assigned'}.")
    print("A live transfer can now call the owner's phone. Restart the server: uv run serve")
    return 0


def main() -> None:
    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
