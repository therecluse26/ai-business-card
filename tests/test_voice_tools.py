"""Voice-mode tool schemas mirror the LangGraph tools, agent by agent."""

from ai_call_center.agents.specs import DEFAULT_VOICES
from ai_call_center.graph.build import all_tools
from ai_call_center.voice.tools import tools_for_agent, voice_tool_schemas


def test_every_graph_tool_has_a_voice_schema(specs):
    schemas = voice_tool_schemas(specs)
    assert set(schemas) == set(all_tools(specs))
    lead = schemas["save_lead"]
    assert lead.required == ["name", "need"]  # the owner can call back on the caller's number
    assert set(lead.properties) == {"name", "need", "email", "company", "callback_number", "call_notes", "timeline", "size"}
    assert lead.properties["email"] == {"type": "string"}
    assert schemas["get_call_stats"].properties == {}


def test_agent_tool_sets_match_text_mode(specs):
    schemas = voice_tool_schemas(specs)
    names = {name: [t.name for t in tools_for_agent(spec, schemas).standard_tools] for name, spec in specs.items()}
    assert names["classifier"] == [
        "identify_caller", "verify_passphrase", "set_passphrase", "forget_caller", "end_call",
        "transfer_to_profile", "transfer_to_tour", "transfer_to_consultant", "transfer_to_contact", "transfer_to_feedback",
    ]  # fmt: skip
    assert names["contact"][:3] == ["save_lead", "set_passphrase", "request_owner_transfer"]
    assert "transfer_to_contact" not in names["contact"]
    assert all("end_call" in tools for tools in names.values())  # any agent can hang up after a goodbye


def test_every_agent_can_reach_the_feedback_agent_and_it_can_hand_back(specs):
    schemas = voice_tool_schemas(specs)
    names = {name: [t.name for t in tools_for_agent(spec, schemas).standard_tools] for name, spec in specs.items()}
    assert all("transfer_to_feedback" in tools for agent, tools in names.items() if agent != "feedback")
    assert names["feedback"] == [
        "file_feedback", "forget_caller", "end_call",
        "transfer_to_classifier", "transfer_to_profile", "transfer_to_tour", "transfer_to_consultant", "transfer_to_contact",
    ]  # fmt: skip
    # A report about the line never rings the owner's phone.
    assert "request_owner_transfer" not in names["feedback"] and "save_lead" not in names["feedback"]
    feedback = schemas["file_feedback"]
    assert feedback.required == ["kind", "summary"] and set(feedback.properties) == {"kind", "summary"}


def test_each_agent_has_its_own_voice(specs):
    voices = [spec.voice for spec in specs.values()]
    assert len(set(voices)) == len(voices)
    assert specs["classifier"].voice == DEFAULT_VOICES["classifier"]


def test_the_feedback_agent_has_a_voice_that_no_other_agent_on_its_leg_uses():
    from ai_call_center.agents.specs import DEFAULT_LEGS, LEG_VOICES

    # Nova has three English voices, so agents share them there. The other legs have a voice for each agent.
    for leg in ("openai_realtime", "gemini_live"):
        others = {agent: voice for agent, voice in LEG_VOICES[leg].items() if agent != "feedback"}
        assert LEG_VOICES[leg]["feedback"] not in others.values()
    assert set(LEG_VOICES["nova_sonic"]) == set(DEFAULT_LEGS)  # a move to Nova still finds a voice


# --- a handoff: the agent that gets the call answers the caller ---


class _Result:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def __call__(self, text, **_kwargs) -> None:
        self.said.append(text)


def _handoff_parts(specs):
    from types import SimpleNamespace

    from ai_call_center.voice.call_session import CallerWords, VoiceCallState

    legs = {name: SimpleNamespace(agent=name, leg="nova_sonic" if name == "consultant" else "openai_realtime") for name in specs}
    session = SimpleNamespace(specs=specs, call_control_id="call-1")
    params = SimpleNamespace(result_callback=_Result())
    state = VoiceCallState(active_agent="contact", handoffs=["consultant", "contact"])
    return session, params, state, CallerWords(), {"legs": legs, "switcher": None, "worker": None, "context": None, "schemas": {}}


async def test_an_agent_cannot_hand_the_call_on_before_the_caller_speaks_to_it(specs, monkeypatch):
    """Measured 2026-11-16: consultant -> contact -> consultant -> contact in 10 s, and the caller
    heard "Let me bring in the right agent" three times."""
    import time

    from ai_call_center.voice import call_session
    from ai_call_center.voice.call_session import CallSession

    monkeypatch.setattr(call_session, "SPOKE_WAIT_SECS", 0.05)
    session, params, state, words, parts = _handoff_parts(specs)
    state.took_over_at = time.monotonic()  # the contact agent got the call a moment ago

    await CallSession._handoff(session, "consultant", params, state=state, caller_words=words, **parts)

    assert state.active_agent == "contact" and state.handoffs == ["consultant", "contact"]
    (said,) = params.result_callback.said
    assert said.startswith("No handoff was made.") and '"Hi, I\'m the contact agent, running on OpenAI."' in said


async def test_an_agent_can_hand_the_call_on_after_the_caller_speaks_to_it(specs):
    import time

    from ai_call_center.voice.call_session import CallSession

    session, params, state, words, parts = _handoff_parts(specs)
    state.took_over_at = time.monotonic() - 5
    words.last_spoke_at = time.monotonic()  # the caller answered the contact agent

    await CallSession._handoff(session, "consultant", params, state=state, caller_words=words, **parts)

    assert state.active_agent == "consultant"
    assert params.result_callback.said == ["Handed off to the consultant agent."]


async def test_words_that_arrive_just_after_the_tool_call_still_count(specs):
    """Nova reports the caller's words and calls the handoff tool in the same millisecond."""
    import asyncio
    import time

    from ai_call_center.voice.call_session import CallSession

    session, params, state, words, parts = _handoff_parts(specs)
    state.took_over_at = time.monotonic() - 5

    async def words_arrive() -> None:
        await asyncio.sleep(0.05)
        words.last_spoke_at = time.monotonic()

    late = asyncio.create_task(words_arrive())
    await CallSession._handoff(session, "consultant", params, state=state, caller_words=words, **parts)
    await late

    assert state.active_agent == "consultant"


async def test_an_agent_cannot_hand_the_call_to_itself(specs):
    from ai_call_center.voice.call_session import CallSession

    session, params, state, words, parts = _handoff_parts(specs)

    await CallSession._handoff(session, "contact", params, state=state, caller_words=words, **parts)

    assert state.handoffs == ["consultant", "contact"]
    assert params.result_callback.said[0].startswith("No handoff was made: you are the contact agent already.")


async def test_the_first_agent_can_hand_off_at_once(specs):
    from ai_call_center.voice.call_session import CallSession

    session, params, state, words, parts = _handoff_parts(specs)
    state.active_agent, state.handoffs = "classifier", []  # no handoff yet: took_over_at is 0.0

    await CallSession._handoff(session, "consultant", params, state=state, caller_words=words, **parts)

    assert state.active_agent == "consultant"


# --- the line noise before the greeting ---


def test_the_callers_audio_is_silence_until_the_greeting_starts():
    """Measured 2026-11-16: a 170 ms noise, 60 ms after the stream opened, was answered with a second greeting."""
    from pipecat.frames.frames import BotStartedSpeakingFrame, InputAudioRawFrame

    from ai_call_center.voice.call_session import OpeningQuiet

    def audio() -> InputAudioRawFrame:
        return InputAudioRawFrame(audio=b"\x01\x02" * 80, sample_rate=8000, num_channels=1)

    quiet = OpeningQuiet()
    click = audio()
    quiet.screen(click)
    assert click.audio == bytes(160) and not quiet.open

    quiet.screen(BotStartedSpeakingFrame())
    speech = audio()
    quiet.screen(speech)
    assert speech.audio == b"\x01\x02" * 80 and quiet.open


def test_the_callers_audio_opens_when_no_greeting_comes():
    from pipecat.frames.frames import InputAudioRawFrame

    from ai_call_center.voice.call_session import OpeningQuiet

    quiet = OpeningQuiet(limit_secs=0.0)
    frame = InputAudioRawFrame(audio=b"\x01\x02" * 80, sample_rate=8000, num_channels=1)
    quiet.screen(frame)
    assert frame.audio == b"\x01\x02" * 80 and quiet.open


# --- the caller's newest words, before Pipecat commits them to the shared context ---


async def _heard(words, *texts: str) -> None:
    """Record words as the processor does when a voice service reports them."""
    import time

    for text in texts:
        words._heard.append((time.monotonic(), text))  # noqa: SLF001


async def test_words_the_service_heard_count_before_the_context_holds_them():
    from ai_call_center.graph.tools import said_goodbye
    from ai_call_center.voice.call_session import CallerWords

    words = CallerWords()
    lines = ["agent: Anything else?", "caller: Tell me about the system.", "agent: It routes calls between agents."]
    assert words.with_newest(lines) == lines  # nothing heard: the transcript is unchanged
    await _heard(words, "That's all.", " Goodbye.")

    assert words.missing_from(lines) == "That's all. Goodbye."
    assert words.with_newest(lines)[-1] == "caller: That's all. Goodbye."
    assert not said_goodbye(lines) and said_goodbye(words.with_newest(lines))


async def test_words_already_in_the_transcript_are_not_added_twice():
    from ai_call_center.voice.call_session import CallerWords

    words = CallerWords()
    await _heard(words, "thanks can i hear about brad's background now")
    lines = ["agent: Would you like more?", "caller: Thanks. Can I hear about Brad's background now?"]

    assert words.missing_from(lines) == "" and words.with_newest(lines) == lines


async def test_old_words_do_not_count_as_new():
    import time

    from ai_call_center.voice.call_session import CALLER_WORDS_SECS, CallerWords

    words = CallerWords()
    words._heard.append((time.monotonic() - CALLER_WORDS_SECS - 1, "Connect me to Brad."))  # noqa: SLF001

    assert words.missing_from(["agent: Hello."]) == ""


async def test_the_next_agent_reads_the_callers_name_in_the_handoff_message(specs, monkeypatch):
    """Measured 2026-11-16: after "Am I speaking with Steve?" and a yes, the contact agent asked
    "who should I say is calling?" and then "what's your name?"."""
    import asyncio
    from types import SimpleNamespace

    from ai_call_center.voice import call_session
    from ai_call_center.voice.call_session import CallSession

    async def nothing(*_args, **_kwargs) -> None:
        return None

    for name in ("ensure_connected", "adopt_context", "_wait_for", "_answer", "_reset_in_background"):
        monkeypatch.setattr(call_session, name, nothing)
    monkeypatch.setattr(call_session, "tools_for_agent", lambda _spec, _schemas: None)

    class Context:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        def set_tools(self, _tools) -> None: ...

        def add_message(self, message: dict) -> None:
            self.messages.append(message)

    class Capture:
        async def __call__(self, text, properties=None) -> None:
            self.text, self.properties = text, properties

    session, _params, state, words, parts = _handoff_parts(specs)
    for leg in parts["legs"].values():
        leg.lock, leg.service = asyncio.Lock(), SimpleNamespace(create_task=lambda coro: coro.close())
    state.active_agent, state.handoffs = "classifier", []
    context, params = Context(), SimpleNamespace(result_callback=Capture())
    parts.update(context=context, worker=SimpleNamespace(queue_frames=nothing), switcher=SimpleNamespace(strategy=SimpleNamespace(active_service=None)))

    await CallSession._handoff(session, "contact", params, state=state, caller_words=words, known_name=lambda: "Steve Stevens", **parts)
    await params.result_callback.properties.on_context_updated()

    (message,) = context.messages
    assert message["role"] == "developer" and "The caller's name is Steve Stevens." in message["content"]
    assert "never ask who is calling" in message["content"] and state.took_over_at > 0

    # Words that Pipecat has not committed yet are added, so the next agent answers them.
    import time

    words._heard = [(time.monotonic(), "can you call brad for me")]
    words.last_spoke_at = time.monotonic()
    context.messages.clear()
    await CallSession._handoff(session, "consultant", params, state=state, caller_words=words, known_name=lambda: "", **parts)
    await params.result_callback.properties.on_context_updated()
    said, instruction = context.messages
    assert said == {"role": "user", "content": "can you call brad for me"} and instruction["role"] == "developer"
