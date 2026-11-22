"""Owner notifications. When the contact agent saves a lead, the owner gets an email with the caller's details.

The owner's address lives only in settings (.env). It never enters an agent prompt or a tool result,
so no agent can say it to a caller.

The email leads with the caller's message, then how to reach them, then facts about the call.
It has an HTML part and a plain-text part with the same content. Every value comes from a
caller's speech or from a model, so the HTML part escapes all of them.
"""

from __future__ import annotations

import asyncio
import logging
import re
import smtplib
from html import escape
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from ai_call_center.leads import Lead

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
_EMAIL_RE = re.compile(r"^[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+$")


@dataclass(frozen=True)
class CallDetails:
    """Facts about the call that a lead came from. They make the email specific. Any field can be empty."""

    # "message": the caller left a message. "transfer": the caller asked to speak to the owner.
    # "warm-project" or "warm-role": the caller talked about a project or a role, and left no message.
    kind: str = "message"
    notes: str = ""  # what else the caller asked or said on the call, written by the agent
    agents: tuple[str, ...] = ()  # the agents the caller spoke with, in order
    seconds: int = 0  # how long the call had run when the lead was saved
    earlier_calls: int = 0  # finished calls from this number before this one
    confirmed_returning: bool = False  # the name the caller gave matched an earlier caller on this number
    line: str = ""  # from the number lookup, for example "mobile on Verizon"
    area: str = ""  # from the number lookup, for example "Akron, OH"
    registered_name: str = ""  # from the number lookup: the account holder, not always the caller
    voice: str = ""  # how the caller sounded, from the voice readings (caller_voice.voice_summary)


@runtime_checkable
class Notifier(Protocol):
    async def send_lead(self, lead: Lead, *, row_id: int, details: CallDetails | None = None) -> None:
        """Tell the owner about a new lead. Raise on failure."""


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _local_time(iso_utc: str) -> str:
    try:
        when = datetime.fromisoformat(iso_utc).astimezone(EASTERN)
    except ValueError:
        return iso_utc
    return when.strftime("%a %b %-d, %Y at %-I:%M %p %Z")


def _phone(number: str) -> str:
    """+13305550123 as +1 (330) 555-0123. Any other form is returned as it is."""
    if re.fullmatch(r"\+1\d{10}", number):
        return f"+1 ({number[2:5]}) {number[5:8]}-{number[8:]}"
    return number


def _duration(seconds: int) -> str:
    minutes, secs = divmod(max(seconds, 0), 60)
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


def _has(value: str) -> bool:
    return bool(value.strip()) and value.strip().lower() != "none"


def _rows(lead: Lead, row_id: int, details: CallDetails) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """(label, text, link) rows for the two tables: how to reach the caller, and facts about the call.
    A row with nothing to say is left out."""
    dialable = re.fullmatch(r"\+\d{8,15}", lead.caller_number) is not None
    reach: list[tuple[str, str, str]] = []
    if dialable:
        reach.append(("Call back", _phone(lead.caller_number), f"tel:{lead.caller_number}"))
    else:
        reach.append(("Call back", "The caller's number was hidden.", ""))
    if _has(lead.email):
        reach.append(("Email", lead.email, f"mailto:{lead.email}" if _EMAIL_RE.match(lead.email) else ""))
    if _has(lead.company):
        reach.append(("Company", _one_line(lead.company), ""))

    # The qualifying answers come first: they tell the owner how warm the lead is.
    call: list[tuple[str, str, str]] = []
    if _has(lead.timeline):
        call.append(("Wants to start", _one_line(lead.timeline), ""))
    if _has(lead.size):
        call.append(("Size of the work", _one_line(lead.size), ""))
    call.append(("When", _local_time(lead.created_at), ""))
    if details.seconds:
        call.append(("Call length so far", _duration(details.seconds), ""))
    if details.agents:
        call.append(("Spoke with", ", ".join(f"{a} agent" for a in details.agents), ""))
    if _has(details.notes):
        call.append(("Also on this call", _one_line(details.notes), ""))
    if _has(details.voice):
        call.append(("Caller's voice", _one_line(details.voice), ""))
    if details.confirmed_returning:
        call.append(("Returning caller", "Yes. The name they gave matched an earlier caller on this number.", ""))
    if details.earlier_calls:
        call.append(("Earlier calls from this number", str(details.earlier_calls), ""))
    line = ", ".join(part for part in (details.line, details.area) if _has(part))
    if line:
        call.append(("Phone line", line, ""))
    if _has(details.registered_name):
        call.append(("Name on the phone account", f"{details.registered_name.title()} (from a number lookup; it can be another person)", ""))
    if row_id:
        call.append(("Lead record", str(row_id), ""))
    call.append(("Call ID", lead.call_control_id, ""))
    return reach, call


FOOTNOTE = (
    "The agent read the details back to the caller, and the caller confirmed them. "
    "Speech recognition can still mishear a name or an email address."
)
WARM_FOOTNOTE = (
    "The caller left no message, so nobody read details back to them. "
    "The text above is the call summary, written by a model after the call."
)
WARM_TOPICS = {"warm-project": "a project", "warm-role": "a role"}
TRANSFER_FOOTNOTE = (
    "The agent saved this before it tried your phone, so the record exists even when the call does not reach you. "
    "Speech recognition can still mishear a name."
)


def _html(headline: str, message: str, reach: list[tuple[str, str, str]], call: list[tuple[str, str, str]], footnote: str) -> str:
    def table(title: str, rows: list[tuple[str, str, str]]) -> str:
        cells = []
        for label, text, link in rows:
            value = f'<a href="{escape(link, quote=True)}" style="color:#1a56db;text-decoration:none">{escape(text)}</a>' if link else escape(text)
            cells.append(
                '<tr><td style="padding:6px 16px 6px 0;color:#6b7280;vertical-align:top;white-space:nowrap">'
                f'{escape(label)}</td><td style="padding:6px 0;color:#111827;vertical-align:top">{value}</td></tr>'
            )
        return (
            f'<h2 style="margin:28px 0 8px;font-size:13px;letter-spacing:0.06em;text-transform:uppercase;color:#6b7280">{escape(title)}</h2>'
            f'<table role="presentation" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:15px;line-height:1.45">{"".join(cells)}</table>'
        )

    return (
        '<!doctype html><html><body style="margin:0;padding:24px 16px;background:#f3f4f6;'
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif\">"
        '<div style="max-width:560px;margin:0 auto;background:#ffffff;border-radius:10px;padding:28px 28px 24px">'
        '<div style="font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#6b7280">Voice business card</div>'
        f'<h1 style="margin:6px 0 18px;font-size:22px;line-height:1.3;color:#111827">{escape(headline)}</h1>'
        '<div style="border-left:4px solid #1a56db;background:#f5f8ff;padding:14px 16px;font-size:16px;line-height:1.5;color:#111827">'
        f"{escape(message)}</div>"
        f"{table('Reach them', reach)}{table('About the call', call)}"
        f'<p style="margin:26px 0 0;font-size:12px;line-height:1.5;color:#6b7280">{escape(footnote)}</p>'
        "</div></body></html>"
    )


def _text(headline: str, message: str, reach: list[tuple[str, str, str]], call: list[tuple[str, str, str]], footnote: str) -> str:
    def block(title: str, rows: list[tuple[str, str, str]]) -> str:
        return title + "\n" + "\n".join(f"  {label}: {text}" for label, text, _link in rows)

    return f"{headline}\n\n{message}\n\n{block('Reach them', reach)}\n\n{block('About the call', call)}\n\n{footnote}\n"


def plain_text(msg: EmailMessage) -> str:
    """The plain-text part of a lead email."""
    part = msg.get_body(preferencelist=("plain",))
    return part.get_content() if part is not None else ""


def lead_email(lead: Lead, *, row_id: int, sender: str, to: str, details: CallDetails | None = None) -> EmailMessage:
    """Build the email the owner receives. Reply-To is the caller, so the owner can reply directly."""
    details = details or CallDetails()
    name = _one_line(lead.name)
    company = f" ({_one_line(lead.company)})" if _has(lead.company) else ""
    if details.kind == "transfer":
        subject = f"{name}{company} asked to speak to you"
        headline = f"{name} asked to speak to you on your voice business card."
        footnote = TRANSFER_FOOTNOTE
    elif details.kind in WARM_TOPICS:
        topic = WARM_TOPICS[details.kind]
        subject = f"Warm lead: {name}{company} talked about {topic}, no message left"
        headline = f"{name} talked about {topic} on your voice business card, but left no message."
        footnote = WARM_FOOTNOTE
    else:
        subject = f"Message from {name}{company}"
        headline = f"{name} left you a message on your voice business card."
        footnote = FOOTNOTE
    message = " ".join(lead.need.split())
    reach, call = _rows(lead, row_id, details)

    msg = EmailMessage()
    msg["Subject"] = f"Voice card: {subject}"[:200]
    msg["From"] = sender
    msg["To"] = to
    if _EMAIL_RE.match(lead.email):
        msg["Reply-To"] = lead.email
    msg.set_content(_text(headline, message, reach, call, footnote))
    msg.add_alternative(_html(headline, message, reach, call, footnote), subtype="html")
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

    async def send_lead(self, lead: Lead, *, row_id: int, details: CallDetails | None = None) -> None:
        msg = lead_email(lead, row_id=row_id, sender=self.sender, to=self.to, details=details)
        await asyncio.to_thread(self._send, msg)


@dataclass
class RecordingNotifier:
    """Records lead emails instead of sending them. Used by tests, and by chat.py when SMTP is not set up."""

    sent: list[EmailMessage] = field(default_factory=list)
    to: str = "owner@example.invalid"
    delivers: bool = True  # False when it stands in for missing SMTP settings: nothing reaches the owner

    async def send_lead(self, lead: Lead, *, row_id: int, details: CallDetails | None = None) -> None:
        self.sent.append(lead_email(lead, row_id=row_id, sender="voice-card@example.invalid", to=self.to, details=details))


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
    return RecordingNotifier(delivers=False)
