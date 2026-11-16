"""Live transfer: stop the stream, call the owner, and bring the contact agent back if nobody picks up."""

from dataclasses import dataclass, field

from ai_call_center.config import Settings
from ai_call_center.telephony.telnyx_client import encode_state
from ai_call_center.telephony.transfers import APOLOGY, Fallback, Snapshot, TransferManager
from ai_call_center.telephony.webhooks import handle_event
from ai_call_center.voice.call_session import TRANSFER_DID_NOT_CONNECT, after_transfer_attempt, resume_message

CALL = "v3:caller-leg"
OWNER_LEG = "v3:owner-leg"
OWNER = "+15555550199"
STREAM = "wss://example.ts.net/media"


@dataclass
class FakeTelephony:
    log: list[tuple] = field(default_factory=list)
    fail: set[str] = field(default_factory=set)  # command names that raise

    def _do(self, name: str, *args) -> None:
        self.log.append((name, *args))
        if name in self.fail:
            raise RuntimeError(f"{name} refused")

    async def stop_streaming(self, call_control_id):
        self._do("stop_streaming", call_control_id)

    async def start_streaming(self, call_control_id, *, stream_url):
        self._do("start_streaming", call_control_id, stream_url)

    async def transfer(self, call_control_id, *, to, timeout_secs=None, state=None, detect_voicemail=False):
        self.detect_voicemail = detect_voicemail
        self._do("transfer", call_control_id, to, timeout_secs, state)

    async def speak(self, call_control_id, text, *, then_hangup=False, state=None):
        self._do("speak", call_control_id, text, then_hangup, *([state] if state else []))

    async def hangup(self, call_control_id):
        self._do("hangup", call_control_id)

    async def answer_with_stream(self, call_control_id, *, stream_url):
        self._do("answer_with_stream", call_control_id)


def manager(telephony: FakeTelephony) -> TransferManager:
    return TransferManager(telephony, stream_url=STREAM, ring_secs=20)


def snapshot() -> Snapshot:
    return Snapshot(messages=[{"role": "user", "content": "Connect me to the owner."}], handoffs=["contact"], lead_saved=True, caller_name="Ada")


def owner_event(event_type: str, *, tagged: bool = True, **payload) -> dict:
    tag = {"client_state": encode_state({"transfer_of": CALL})} if tagged else {}
    return {"event_type": event_type, "payload": {"call_control_id": OWNER_LEG, **tag, **payload}}


def settings() -> Settings:
    return Settings(_env_file=None, public_url="https://example.ts.net", telnyx_app_id="app-1")


async def test_transfer_stops_the_stream_first_and_tags_the_owner_leg():
    telephony = FakeTelephony()
    await manager(telephony).start(CALL, to=OWNER, snapshot=snapshot())
    assert telephony.log == [("stop_streaming", CALL), ("transfer", CALL, OWNER, 20, {"transfer_of": CALL})]


async def test_owner_does_not_pick_up_and_the_agent_comes_back_with_the_conversation():
    telephony = FakeTelephony()
    transfers = manager(telephony)
    await transfers.start(CALL, to=OWNER, snapshot=snapshot())

    # Telnyx reports the owner's leg ended without an answer.
    await handle_event(owner_event("call.hangup"), settings(), telephony, transfers=transfers)

    assert telephony.log[-1] == ("start_streaming", CALL, STREAM)
    resumed = transfers.take_resume(CALL)
    assert resumed.reason == "no_answer" and resumed.lead_saved and resumed.messages[0]["content"] == "Connect me to the owner."
    assert transfers.take_resume(CALL) is None  # taken once, by the media stream that starts again


async def test_owner_picks_up_and_a_later_hangup_resumes_nothing():
    telephony = FakeTelephony()
    transfers = manager(telephony)
    await transfers.start(CALL, to=OWNER, snapshot=snapshot())
    await handle_event(owner_event("call.answered"), settings(), telephony, transfers=transfers)
    await handle_event(owner_event("call.hangup"), settings(), telephony, transfers=transfers)
    assert [entry[0] for entry in telephony.log] == ["stop_streaming", "transfer"]
    assert transfers.take_resume(CALL) is None


# --- the owner's voicemail answers ---


async def test_voicemail_answers_and_the_agent_comes_back():
    telephony = FakeTelephony()
    transfers = TransferManager(telephony, stream_url=STREAM, ring_secs=20, detect_voicemail=True)
    await transfers.start(CALL, to=OWNER, snapshot=snapshot())
    assert telephony.detect_voicemail is True
    await handle_event(owner_event("call.answered"), settings(), telephony, transfers=transfers)
    await handle_event(owner_event("call.bridged"), settings(), telephony, transfers=transfers)
    # The detection event comes without the tag here: the manager knows the owner's leg by its id.
    await handle_event(owner_event("call.machine.detection.ended", tagged=False, result="machine"), settings(), telephony, transfers=transfers)
    assert telephony.log[-1] == ("hangup", OWNER_LEG)  # only the voicemail leg; the caller stays on the line
    await handle_event(owner_event("call.hangup"), settings(), telephony, transfers=transfers)

    assert telephony.log[-1] == ("start_streaming", CALL, STREAM)
    resumed = transfers.take_resume(CALL)
    assert resumed is not None and resumed.reason == "no_answer" and resumed.caller_name == "Ada"


async def test_a_person_answers_and_the_call_ends_when_the_owner_hangs_up():
    telephony = FakeTelephony()
    transfers = TransferManager(telephony, stream_url=STREAM, ring_secs=20, detect_voicemail=True)
    await transfers.start(CALL, to=OWNER, snapshot=snapshot())
    await handle_event(owner_event("call.answered"), settings(), telephony, transfers=transfers)
    for result in ("human", "not_sure"):  # only "machine" ends the owner's leg
        await handle_event(owner_event("call.machine.detection.ended", result=result), settings(), telephony, transfers=transfers)
    assert [entry[0] for entry in telephony.log] == ["stop_streaming", "transfer"]
    await handle_event(owner_event("call.hangup"), settings(), telephony, transfers=transfers)

    # The caller's leg is parked when the owner's leg ends, so the server ends it. No agent comes back.
    assert telephony.log[-1] == ("hangup", CALL)
    assert transfers.take_resume(CALL) is None


async def test_without_voicemail_detection_the_transfer_command_is_as_before():
    telephony = FakeTelephony()
    await manager(telephony).start(CALL, to=OWNER, snapshot=snapshot())
    assert telephony.detect_voicemail is False


async def test_transfer_command_fails_and_the_agent_comes_back():
    telephony = FakeTelephony(fail={"transfer"})
    transfers = manager(telephony)
    await transfers.start(CALL, to=OWNER, snapshot=snapshot())
    assert telephony.log[-1] == ("start_streaming", CALL, STREAM)
    assert transfers.take_resume(CALL).reason == "failed"


async def test_telnyx_voice_speaks_only_when_the_stream_cannot_start_again():
    telephony = FakeTelephony(fail={"transfer", "start_streaming"})
    transfers = manager(telephony)
    await transfers.start(CALL, to=OWNER, snapshot=snapshot())
    assert telephony.log[-1] == ("speak", CALL, APOLOGY, True)
    assert transfers.take_resume(CALL) is None


async def test_caller_hangs_up_while_the_owner_phone_rings():
    telephony = FakeTelephony()
    transfers = manager(telephony)
    await transfers.start(CALL, to=OWNER, snapshot=snapshot())
    await handle_event({"event_type": "call.hangup", "payload": {"call_control_id": CALL}}, settings(), telephony, transfers=transfers)
    await handle_event(owner_event("call.hangup"), settings(), telephony, transfers=transfers)
    assert [entry[0] for entry in telephony.log] == ["stop_streaming", "transfer"]  # no restart, no speech


async def test_no_transfer_when_the_stream_cannot_stop():
    telephony = FakeTelephony(fail={"stop_streaming"})
    await manager(telephony).start(CALL, to=OWNER, snapshot=snapshot())
    assert telephony.log == [("stop_streaming", CALL)]


def test_resume_message_tells_the_agent_what_happened():
    text = resume_message("no_answer", "Test Owner", lead_saved=True)
    assert "Test Owner did not pick up" in text and "Test Owner will call you back" in text and "Do not introduce yourself again" in text
    assert "take their name and a message" in resume_message("failed", "Test Owner", lead_saved=False)


def test_the_goodbye_instruction_is_gone_when_the_call_comes_back():
    messages = [
        {"role": "user", "content": "Connect me."},
        {"role": "assistant", "tool_calls": [{"id": "call_s", "type": "function", "function": {"name": "save_lead", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_s", "content": '"Lead saved as record 1."'},
        {"role": "assistant", "tool_calls": [{"id": "call_t", "type": "function", "function": {"name": "request_owner_transfer", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_t", "content": '"Say only this one line now: ..."'},
    ]
    resumed = after_transfer_attempt(messages)
    assert TRANSFER_DID_NOT_CONNECT in resumed[-1]["content"] and "Say only" not in str(resumed)
    assert resumed[:4] == messages[:4]  # everything else is unchanged
    assert "Say only" in messages[-1]["content"]  # the snapshot itself is not edited in place


# --- a voice provider stops answering: restart the stream, or use Telnyx's voice ---


def fallback(to: str = OWNER) -> Fallback:
    return Fallback(to=to, connecting="Connecting you now.", goodbye="Call again later.", no_answer="Nobody picked up.")


async def test_restart_stops_and_starts_the_stream_and_keeps_the_conversation_for_the_new_session():
    telephony = FakeTelephony()
    transfers = manager(telephony)
    snap = Snapshot(messages=[{"role": "user", "content": "Hello?"}], reason="provider_failed", entry_agent="profile", failovers=1)

    await transfers.restart(CALL, snapshot=snap, fallback=fallback())

    assert telephony.log == [("stop_streaming", CALL), ("start_streaming", CALL, STREAM)]
    resumed = transfers.take_resume(CALL)
    assert resumed is snap and resumed.entry_agent == "profile" and resumed.failovers == 1
    assert transfers.take_resume(CALL) is None  # handed out once


async def test_restart_goes_on_when_the_stream_is_already_stopped():
    telephony = FakeTelephony(fail={"stop_streaming"})
    transfers = manager(telephony)
    await transfers.restart(CALL, snapshot=Snapshot(reason="provider_failed"), fallback=fallback())
    assert telephony.log[-1] == ("start_streaming", CALL, STREAM)
    assert transfers.take_resume(CALL) is not None


async def test_restart_that_fails_speaks_and_then_transfers_to_the_owner():
    telephony = FakeTelephony(fail={"start_streaming"})
    transfers = manager(telephony)
    transfers.restart_retry_secs = 0.0

    await transfers.restart(CALL, snapshot=Snapshot(reason="provider_failed"), fallback=fallback())

    assert [entry[0] for entry in telephony.log].count("start_streaming") == 3  # tried three times
    assert transfers.take_resume(CALL) is None
    assert telephony.log[-1] == ("speak", CALL, "Connecting you now.", False, {"then": "transfer"})
    # Telnyx reports the sentence has ended; only then does the transfer start.
    await transfers.on_event("call.speak.ended", CALL, {"then": "transfer"})
    assert telephony.log[-1] == ("transfer", CALL, OWNER, 20, {"transfer_of": CALL})


async def test_last_resort_without_a_transfer_says_goodbye_and_hangs_up():
    telephony = FakeTelephony()
    transfers = manager(telephony)

    await transfers.last_resort(CALL, fallback=fallback(to=""))

    assert telephony.log == [("stop_streaming", CALL), ("speak", CALL, "Call again later.", True)]
    await transfers.on_event("call.speak.ended", CALL, {"then": "hangup"})
    assert not [entry for entry in telephony.log if entry[0] == "transfer"]


async def test_last_resort_transfer_that_nobody_answers_ends_the_call_and_starts_no_agent():
    telephony = FakeTelephony()
    transfers = manager(telephony)
    await transfers.last_resort(CALL, fallback=fallback())
    await transfers.on_event("call.speak.ended", CALL, {"then": "transfer"})

    await handle_event(owner_event("call.hangup"), settings(), telephony, transfers=transfers)

    assert telephony.log[-1] == ("speak", CALL, "Nobody picked up.", True)
    assert not [entry for entry in telephony.log if entry[0] == "start_streaming"]
    assert transfers.take_resume(CALL) is None


async def test_last_resort_transfer_that_the_owner_answers_needs_nothing_more():
    telephony = FakeTelephony()
    transfers = manager(telephony)
    await transfers.last_resort(CALL, fallback=fallback())
    await transfers.on_event("call.speak.ended", CALL, {"then": "transfer"})
    await handle_event(owner_event("call.answered"), settings(), telephony, transfers=transfers)
    count = len(telephony.log)

    await handle_event(owner_event("call.hangup"), settings(), telephony, transfers=transfers)

    assert len(telephony.log) == count


async def test_caller_hangs_up_during_the_spoken_fallback_and_no_transfer_starts():
    telephony = FakeTelephony()
    transfers = manager(telephony)
    await transfers.last_resort(CALL, fallback=fallback())

    await transfers.on_event("call.hangup", CALL, {"then": "transfer"})
    await transfers.on_event("call.speak.ended", CALL, {"then": "transfer"})

    assert not [entry for entry in telephony.log if entry[0] == "transfer"]


async def test_restart_tries_again_when_telnyx_refuses_the_first_start():
    class RefusesOnce(FakeTelephony):
        async def start_streaming(self, call_control_id, *, stream_url):
            self.log.append(("start_streaming", call_control_id, stream_url))
            if [entry[0] for entry in self.log].count("start_streaming") == 1:
                raise RuntimeError("the old stream is still closing")

    telephony = RefusesOnce()
    transfers = manager(telephony)
    transfers.restart_retry_secs = 0.0

    await transfers.restart(CALL, snapshot=Snapshot(reason="provider_failed"), fallback=fallback())

    assert [entry[0] for entry in telephony.log] == ["stop_streaming", "start_streaming", "start_streaming"]
    assert transfers.take_resume(CALL) is not None  # the new session still gets the conversation
