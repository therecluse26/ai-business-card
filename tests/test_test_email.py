"""`uv run test-email`: one sample lead email through the real notifier path."""

import smtplib

from ai_call_center import test_email
from ai_call_center.config import Settings
from ai_call_center.notify import SmtpNotifier


def settings(**over) -> Settings:
    base = {"owner_email": "owner@example.com", "smtp_host": "smtp.example.com", "smtp_port": 587, "smtp_username": "user", "smtp_password": "secret"}
    return Settings(_env_file=None, **{**base, **over})


async def test_nothing_is_sent_when_settings_are_missing(monkeypatch, capsys):
    sent = []
    monkeypatch.setattr(SmtpNotifier, "_send", lambda self, msg: sent.append(msg))

    assert await test_email.run(settings(owner_email="", smtp_host="")) == 1

    assert sent == []
    out = capsys.readouterr().out
    assert "OWNER_EMAIL" in out and "SMTP_HOST" in out


async def test_a_missing_password_is_named(capsys):
    assert await test_email.run(settings(smtp_password="")) == 1
    assert "SMTP_PASSWORD" in capsys.readouterr().out


async def test_the_sample_lead_goes_to_the_owner_and_says_it_is_a_test(monkeypatch, capsys):
    sent = []
    monkeypatch.setattr(SmtpNotifier, "_send", lambda self, msg: sent.append(msg))

    assert await test_email.run(settings(smtp_from="card@example.com")) == 0

    assert len(sent) == 1
    msg = sent[0]
    assert msg["To"] == "owner@example.com" and msg["From"] == "card@example.com"
    assert "test of the lead email" in msg.get_content()
    out = capsys.readouterr().out
    assert "Sent." in out and "owner@example.com" not in out  # the owner's address is not printed


async def test_a_refused_login_is_explained(monkeypatch, capsys):
    def refuse(self, msg):
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(SmtpNotifier, "_send", refuse)

    assert await test_email.run(settings()) == 1

    out = capsys.readouterr().out
    assert "FAILED" in out and "app password or an API key" in out
