"""Owner notifications. When the contact agent saves a lead, the owner gets an email with the caller's details.

The owner's address lives only in settings (.env). It never enters an agent prompt or a tool result,
so no agent can say it to a caller.
"""

from __future__ import annotations

import asyncio
import logging
import re
import smtplib
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from ai_call_center.leads import Lead

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
_EMAIL_RE = re.compile(r"^[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+$")


@runtime_checkable
class Notifier(Protocol):
    async def send_lead(self, lead: Lead, *, row_id: int) -> None:
        """Tell the owner about a new lead. Raise on failure."""


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _local_time(iso_utc: str) -> str:
    try:
        when = datetime.fromisoformat(iso_utc).astimezone(EASTERN)
    except ValueError:
        return iso_utc
    return when.strftime("%a %b %-d, %Y at %-I:%M %p %Z")


def lead_email(lead: Lead, *, row_id: int, sender: str, to: str) -> EmailMessage:
    """Build the email the owner receives. Reply-To is the caller, so the owner can reply directly."""
    msg = EmailMessage()
    msg["Subject"] = f"Voice business card lead: {_one_line(lead.name)}"[:200]
    msg["From"] = sender
    msg["To"] = to
    if _EMAIL_RE.match(lead.email):
        msg["Reply-To"] = lead.email
    msg.set_content(
        "A caller left their details on your voice business card.\n\n"
        f"Name:          {lead.name}\n"
        f"Email:         {lead.email}\n"
        f"Company:       {lead.company}\n"
        f"Interested in: {lead.need}\n"
        f"Caller number: {lead.caller_number}\n"
        f"Time:          {_local_time(lead.created_at)}\n"
        f"Call ID:       {lead.call_control_id}\n"
        f"Lead record:   {row_id}\n\n"
        "The agent read these details back to the caller, and the caller confirmed them.\n"
        "Speech recognition can still mishear an email address. If a reply bounces, call the number above.\n"
    )
    return msg


@dataclass
class SmtpNotifier:
    """Sends lead emails over SMTP. Port 465 uses implicit TLS; any other port uses STARTTLS."""

    host: str
    port: int
    username: str
    password: str
    sender: str
    to: str
    timeout: float = 15.0

    def _send(self, msg: EmailMessage) -> None:
        if self.port == 465:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout) as smtp:
                if self.username:
                    smtp.login(self.username, self.password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as smtp:
                smtp.starttls()
                if self.username:
                    smtp.login(self.username, self.password)
                smtp.send_message(msg)

    async def send_lead(self, lead: Lead, *, row_id: int) -> None:
        msg = lead_email(lead, row_id=row_id, sender=self.sender, to=self.to)
        await asyncio.to_thread(self._send, msg)


@dataclass
class RecordingNotifier:
    """Records lead emails instead of sending them. Used by tests, and by chat.py when SMTP is not set up."""

    sent: list[EmailMessage] = field(default_factory=list)
    to: str = "owner@example.invalid"

    async def send_lead(self, lead: Lead, *, row_id: int) -> None:
        self.sent.append(lead_email(lead, row_id=row_id, sender="voice-card@example.invalid", to=self.to))


def notifier_from_settings(settings) -> Notifier:
    """SMTP when OWNER_EMAIL and SMTP_HOST are set; otherwise a recording fake that sends nothing."""
    if settings.owner_email and settings.smtp_host:
        return SmtpNotifier(
            host=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password.get_secret_value(),
            sender=settings.smtp_from or settings.smtp_username or settings.owner_email,
            to=settings.owner_email,
        )
    log.warning("OWNER_EMAIL or SMTP_HOST is not set. Lead emails are recorded, not sent.")
    return RecordingNotifier()
