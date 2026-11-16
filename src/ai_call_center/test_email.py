"""Send one sample lead email to the owner, to check the SMTP settings in `.env`.

Run: uv run test-email

It uses the same code path as a real lead: `notifier_from_settings` and `send_lead`. It writes
nothing to the lead table.
"""

from __future__ import annotations

import asyncio
import smtplib
import sys
from datetime import UTC, datetime

from ai_call_center.config import Settings, get_settings
from ai_call_center.leads import Lead
from ai_call_center.notify import SmtpNotifier, notifier_from_settings


def missing_settings(settings: Settings) -> list[str]:
    """Names of the `.env` settings that a lead email needs and that are empty."""
    missing = [name for name, value in (("OWNER_EMAIL", settings.owner_email), ("SMTP_HOST", settings.smtp_host)) if not value]
    if settings.smtp_host and settings.smtp_username and not settings.smtp_password.get_secret_value():
        missing.append("SMTP_PASSWORD")
    return missing


def sample_lead() -> Lead:
    return Lead(
        name="Test Caller (uv run test-email)",
        email="test.caller@example.com",
        company="Example Company",
        need="This is a test of the lead email. No caller left this message.",
        caller_number="+15555550100",
        call_control_id="test-email",
        created_at=datetime.now(UTC).isoformat(),
    )


def explain(exc: Exception) -> str:
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return "The server refused the login. Check SMTP_USERNAME and SMTP_PASSWORD. Many providers need an app password or an API key here, not the account password."
    if isinstance(exc, smtplib.SMTPSenderRefused | smtplib.SMTPRecipientsRefused):
        return "The server refused the sender or the recipient. SMTP_FROM must be an address on a domain the provider has verified for you."
    if isinstance(exc, TimeoutError | ConnectionError | OSError):
        return "The server did not answer. Check SMTP_HOST and SMTP_PORT. Port 465 uses TLS from the start; port 587 uses STARTTLS."
    return "The send failed."


async def run(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    if missing := missing_settings(settings):
        print(f"Not sent. These settings are empty in .env: {', '.join(missing)}.")
        return 1
    notifier = notifier_from_settings(settings)
    if not isinstance(notifier, SmtpNotifier):
        print("Not sent. The SMTP settings are not complete.")
        return 1
    print(f"Sending a sample lead email through {notifier.host}:{notifier.port} ({'TLS' if notifier.port == 465 else 'STARTTLS'}), from {notifier.sender} ...")
    try:
        await notifier.send_lead(sample_lead(), row_id=0)
    except Exception as exc:  # noqa: BLE001 - every failure is reported the same way
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print(explain(exc))
        return 1
    print("Sent. Look for it at the address in OWNER_EMAIL; check the spam folder too.")
    print("A real lead uses the same path. Restart the server so it reads the new settings: uv run serve")
    return 0


def main() -> None:
    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
