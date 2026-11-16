"""`uv run check-nova`: one short Nova Sonic session with the keys from `.env`."""

from ai_call_center import check_nova
from ai_call_center.config import Settings


def settings(**over) -> Settings:
    return Settings(_env_file=None, **{"nova_aws_access_key_id": "AKIATEST", "nova_aws_secret_access_key": "secret", **over})


async def test_nothing_is_opened_when_the_keys_are_missing(monkeypatch, capsys):
    async def never(_settings):
        raise AssertionError("no session may be opened without keys")

    monkeypatch.setattr(check_nova, "ask_once", never)
    assert await check_nova.run(settings(nova_aws_secret_access_key="")) == 1
    assert "NOVA_AWS_SECRET_ACCESS_KEY" in capsys.readouterr().out


async def test_an_answer_is_reported_with_its_delay(monkeypatch, capsys):
    async def answer(_settings):
        return 0.84, "The Nova leg works."

    monkeypatch.setattr(check_nova, "ask_once", answer)
    assert await check_nova.run(settings()) == 0
    out = capsys.readouterr().out
    assert "It works. First audio after 0.8 s." in out and "The Nova leg works." in out and "secret" not in out


async def test_a_refused_key_is_explained(monkeypatch, capsys):
    async def refuse(_settings):
        raise RuntimeError("AccessDeniedException: User is not authorized to perform bedrock:InvokeModelWithBidirectionalStream")

    monkeypatch.setattr(check_nova, "ask_once", refuse)
    assert await check_nova.run(settings()) == 1
    assert "bedrock:InvokeModelWithBidirectionalStream" in capsys.readouterr().out
