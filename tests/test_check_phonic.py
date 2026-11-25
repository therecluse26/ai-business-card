"""`uv run check-phonic`: one short Phonic conversation with the key from `.env`."""

from ai_call_center import check_phonic
from ai_call_center.config import Settings

KEY = "ph_test_key_never_shown"


def settings(**over) -> Settings:
    return Settings(_env_file=None, **{"phonic_api_key": KEY, **over})


async def test_nothing_is_opened_when_the_key_is_missing(monkeypatch, capsys):
    async def never(_settings):
        raise AssertionError("no conversation may be opened without a key")

    monkeypatch.setattr(check_phonic, "ask_once", never)
    assert await check_phonic.run(settings(phonic_api_key="")) == 1
    assert "PHONIC_API_KEY" in capsys.readouterr().out


async def test_an_answer_is_reported_with_its_times(monkeypatch, capsys):
    async def answer(_settings):
        return 0.91, 0.32, "The Phonic leg works.", "phonic-1"

    monkeypatch.setattr(check_phonic, "ask_once", answer)
    assert await check_phonic.run(settings()) == 0
    out = capsys.readouterr().out
    assert "It works. Ready after 0.9 s; first audio 0.32 s after the request. Model: phonic-1." in out
    assert "The Phonic leg works." in out and KEY not in out


async def test_a_refused_key_is_explained_and_never_shown(monkeypatch, capsys):
    async def refuse(_settings):
        raise check_phonic.CheckFailed(f"Phonic error: Invalid API key {KEY}")

    monkeypatch.setattr(check_phonic, "ask_once", refuse)
    assert await check_phonic.run(settings()) == 1
    out = capsys.readouterr().out
    assert "did not accept the key" in out and '"ph_"' in out and KEY not in out
