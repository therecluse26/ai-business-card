"""Respellings: the voice models read "Mag-yar", and people read "Magyar"."""

from types import SimpleNamespace

import pytest

from ai_call_center.agents.specs import Profile, build_specs, greeting_words
from ai_call_center.callers import transcript_lines
from ai_call_center.config import Settings
from ai_call_center.voice.call_session import voice_instructions
from ai_call_center.voice.respell import Respelling

NAME = Respelling.parse("Magyar=Mag-yar")


def test_the_voice_model_reads_the_respelling():
    hello = NAME.spoken(greeting_words("Brad Magyar", "Digital Reformer"))
    assert "Brad Mag-yar built" in hello
    assert "Magyar" not in hello


def test_every_form_of_the_word_is_respelled():
    assert NAME.spoken("Magyar's line. Ask MAGYAR, or magyar.") == "Mag-yar's line. Ask Mag-yar, or Mag-yar."


def test_people_read_the_written_word_again():
    assert NAME.written("Call Brad Mag-yar back. It is Mag-yar's call.") == "Call Brad Magyar back. It is Magyar's call."
    assert NAME.written(NAME.spoken("Brad Magyar")) == "Brad Magyar"


def test_an_address_and_a_longer_word_stay_as_they_are():
    text = "brad.magyar@example.com, example.com/magyar, magyar.dev, bradmagyar, Magyarorszag"
    assert NAME.spoken(text) == text


def test_a_respelled_text_does_not_change_a_second_time():
    assert NAME.spoken(NAME.spoken("Brad Magyar")) == "Brad Mag-yar"


def test_the_instructions_never_hold_the_written_word():
    spec = build_specs(Profile(summary="Brad Magyar builds software.", full="x"), owner_name="Brad Magyar", name_hint='Magyar is said "MAG-yar".')["classifier"]
    text = NAME.instructions(voice_instructions(spec, spec.leg))
    assert "Brad Mag-yar" in text
    assert "Magyar" not in text
    assert '"Mag-yar" is spelled M, A, G, Y, A, R.' in text


def test_no_respellings_change_nothing():
    off = Respelling.parse("")
    handler = object()
    assert not off
    assert off.spoken("Brad Magyar") == off.written("Brad Magyar") == off.instructions("Brad Magyar") == "Brad Magyar"
    assert off.tool(handler) is handler


def test_more_than_one_pair():
    two = Respelling.parse("Magyar=Mag-yar, Nguyen = Win")
    assert two.spoken("Magyar and Nguyen") == "Mag-yar and Win"
    assert two.written("Mag-yar and Win") == "Magyar and Nguyen"


@pytest.mark.parametrize("text", ["Magyar", "Magyar=", "=Mag-yar", "Magyar=the Magyar name"])
def test_a_bad_pair_stops_the_server_at_start(text):
    with pytest.raises(ValueError, match="RESPELLINGS"):
        Respelling.parse(text)
    with pytest.raises(ValueError, match="RESPELLINGS"):
        Settings(_env_file=None, respellings=text)


def test_the_summary_model_and_the_log_read_the_written_word():
    messages = [
        {"role": "developer", "content": "Say: Hi, Brad Mag-yar built this."},
        {"role": "user", "content": "Who is this?"},
        {"role": "assistant", "content": [{"type": "text", "text": "Brad Mag-yar's AI line."}]},
    ]
    assert transcript_lines(NAME.written_messages(messages)) == ["caller: Who is this?", "agent: Brad Magyar's AI line."]
    assert messages[2]["content"][0]["text"] == "Brad Mag-yar's AI line."  # the voice models keep the respelling


async def test_a_tool_result_is_respelled_and_its_arguments_are_written():
    said, saved = [], {}

    async def to_model(result, *, properties=None):
        said.append((result, properties))

    async def save_lead(params):
        saved.update(params.arguments)
        await params.result_callback("Saved. Tell the caller that Brad Magyar reads it today.", properties="p")

    params = SimpleNamespace(arguments={"need": "A call with Brad Mag-yar", "size": 3}, result_callback=to_model)
    await NAME.tool(save_lead)(params)

    assert saved == {"need": "A call with Brad Magyar", "size": 3}
    assert said == [("Saved. Tell the caller that Brad Mag-yar reads it today.", "p")]
