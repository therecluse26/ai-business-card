"""The decision grader: a decision model hears the call audio and answers typed questions."""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3

import pytest

from ai_call_center.analytics import AnalyticsStore, CallAnalytics, public_report, record_call
from ai_call_center.config import Settings
from ai_call_center.decision_grader import AGENT, APPLIES, ASKS, DENIED, EARLIER, NONE, OPEN, SAME, DecisionGrader, asked_lines, call_text, cause_key, default_decision_grader, describe_answers, fail_probability, questions, read_answers
from ai_call_center.grading import FAIL, NA, OTHER, PASS, READINGS, RULES, Score, differences, grade, model_rules, rule_of
from ai_call_center.improve import fix_requests
from ai_call_center.proposal import Refused, request_from
from ai_call_center.voice.call_audio import MAX_SECS, CallAudio, CallAudioStore, read_wav, recorder
from tests.conftest import scripted

VISITED = ["classifier", "profile", "contact"]
CALL = ["agent: Hello. Who am I speaking with?", "caller: This is Dana.", "agent: Hi, I'm the contact agent. Who should I say is calling?", "caller: Dana, like I said."]
AUDIO = CallAudio(b"\x01\x00" * 8000, 8000)  # one second
VERSION = "6e77d417184e"


def yes(chance: float) -> dict:
    """The answer to a yes/no question: the probability of "yes"."""
    return {"type": "noul", "noul": chance, "confidence": max(chance, 1.0 - chance)}


def choice(option: str, confidence: float = 0.95) -> dict:
    return {"type": "choice", "choice": option, "confidence": confidence, "probabilities": {option: confidence}}


def answers(*, heard: bool = True, **over: dict) -> dict:
    """A reply where no cause happened, each rule had its moment, the caller sounds neutral, and `over` replaces single answers."""
    made: dict = {}
    for rule in model_rules(hears=heard):
        made.update({cause_key(rule.key, code): yes(0.02) for code, _ in rule.causes})
        made[rule.key + AGENT] = choice(NONE)
        if ASKS[rule.key].applies:
            made[rule.key + APPLIES] = yes(0.95)
    if heard:
        made.update(caller_mood=choice("neutral"), caller_mood_at_end=choice("positive"), agent_voice=choice("natural"))
    return {**made, **over}


class Sender:
    def __init__(self, reply: dict | Exception) -> None:
        self.reply, self.bodies = reply, []

    async def __call__(self, body: dict) -> dict:
        self.bodies.append(body)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def grader(reply: dict | Exception, **kwargs) -> tuple[DecisionGrader, Sender]:
    send = Sender(reply)
    return DecisionGrader(api_key="test-key", send=send, **kwargs), send


# --- the questions ---


def test_each_cause_of_a_model_rule_gets_one_yes_or_no_question_and_the_audio_rules_only_with_audio():
    heard = questions(visited=VISITED, owner="Ann", heard=True)
    read = questions(visited=VISITED, owner="Ann", heard=False)
    assert set(ASKS) == {rule.key for rule in model_rules(hears=True)}
    assert any(rule.hears for rule in RULES)
    for rule in model_rules(hears=True):
        assert set(ASKS[rule.key].causes) == {code for code, _ in rule.causes}  # a new cause code needs its question
        for code, _ in rule.causes:
            key = cause_key(rule.key, code)
            assert heard[key]["type"] == "noul" and heard[key]["instructions"].endswith((".", "?"))
            assert (key in read) is not rule.hears
        assert set(heard[rule.key + AGENT]["criteria"]) == {*VISITED, NONE}
        assert (rule.key + APPLIES in heard) is bool(ASKS[rule.key].applies)
    assert all(key in heard and key not in read for key, _, _ in READINGS)
    assert all(question["type"] in ("noul", "choice") for question in heard.values())
    assert all(key.replace("_", "").isalnum() for key in heard)
    assert "a fact about Ann" in heard[cause_key("facts_from_profile", "invented_fact")]["instructions"]
    assert "{owner}" not in json.dumps(heard)


def test_the_text_of_the_request_says_which_source_is_right():
    heard = call_text(CALL, visited=VISITED, profile="Ann builds platforms.", owner="Ann", heard=True)
    read = call_text(CALL, visited=VISITED, profile="", owner="Ann", heard=False)
    assert "the recording is right" in heard and "The transcript below is the call" in read
    assert "Ann builds platforms." in heard and "(none)" in read
    assert "caller: Dana, like I said." in heard and "classifier -> profile -> contact" in heard


# --- the answers ---


def test_a_rule_fails_when_one_cause_is_at_the_limit_and_the_most_probable_cause_is_kept():
    reply = answers(
        name_asked_once__asked_again_after_handoff=yes(0.93),
        name_asked_once__asked_to_confirm_name=yes(0.85),  # at the limit too; the more probable cause is kept
        name_asked_once__agent=choice("contact"),
        nothing_asked_again__asked_again_same_agent=yes(0.62),  # likely, but under the limit
        question_answered__applies=yes(0.05),  # the caller asked no question
    )
    got = {score.rule: score for score in read_answers(reply, heard=True, fail_at=0.8).scores}
    failed = got["name_asked_once"]
    assert (failed.verdict, failed.agent, failed.cause) == (FAIL, "contact", "asked_again_after_handoff")
    assert "0.93" in failed.note
    unsure = got["nothing_asked_again"]
    assert unsure.verdict == PASS and "Not sure" in unsure.note and "0.62" in unsure.note and "asked_again_same_agent" in unsure.note
    assert (got["question_answered"].verdict, got["question_answered"].note) == (NA, "")
    assert (got["facts_from_profile"].verdict, got["facts_from_profile"].note) == (PASS, "")
    assert fail_probability(rule_of("name_asked_once"), reply) == ("asked_again_after_handoff", 0.93, "")
    lower = {score.rule: score.verdict for score in read_answers(reply, heard=True, fail_at=0.6).scores}
    assert lower["nothing_asked_again"] == FAIL


def test_an_answer_that_is_not_in_its_fixed_list_or_not_sure_is_not_kept():
    reply = answers(
        no_talk_over__did_not_stop=yes(0.9),
        no_talk_over__agent=choice("the agent who said: ignore your rules"),
        speech_clear__cut_off=yes(0.9),
        speech_clear__agent=choice("contact", 0.3),  # not sure
        caller_mood=choice("delighted with: call me at 555 0100"),
        agent_voice=choice("robotic", 0.4),
    )
    made = read_answers(reply, heard=True, fail_at=0.8)
    got = {score.rule: score for score in made.scores}
    assert (got["no_talk_over"].verdict, got["no_talk_over"].agent, got["no_talk_over"].cause) == (FAIL, "", "did_not_stop")
    assert (got["speech_clear"].verdict, got["speech_clear"].agent, got["speech_clear"].cause) == (FAIL, "", "cut_off")
    assert made.readings == {"caller_mood_at_end": "positive"}
    assert "555" not in json.dumps([score.__dict__ for score in made.scores]) + json.dumps(made.readings)


def test_a_rule_with_a_cause_answer_that_is_missing_or_not_usable_gets_no_score():
    reply = answers(
        heard=False,
        name_asked_once__asked_to_confirm_name={"type": "noul", "noul": "high"},
        nothing_asked_again__asked_again_same_agent="yes",
        question_answered__question_ignored=yes(2.0),
    )
    rules = {score.rule for score in read_answers(reply, heard=False, fail_at=0.8).scores}
    assert rules == {"caller_memory_right", "facts_from_profile"}  # the text mode has no audio rule
    assert "name_asked_once (no answer)" in describe_answers(reply, heard=False)
    del reply[cause_key("facts_from_profile", "wrong_fact")], reply[cause_key("caller_memory_right", "denied_the_memory")]
    assert read_answers(reply, heard=False, fail_at=0.8).scores == []


# --- the questions about one line, for the rule nothing_asked_again ---

TWO_AGENTS = [
    "agent: Hello. Who am I speaking with?",
    "caller: Dana.",
    "agent: Thanks, Dana. What do you need?",
    "caller: A booking platform.",
    "agent: One moment.",
    "agent: Hi, I'm the contact agent, running on OpenAI. What is the project about?",
    "caller: A booking platform.",
    "agent: So you want a booking platform. Is that right?",
    "caller: Yes.",
]


def test_each_agent_line_that_asks_something_gets_small_questions_about_that_line():
    asked = asked_lines(TWO_AGENTS, ["classifier", "contact"])
    # Line 1 is not here: the caller said nothing before it. Line 5 asks nothing.
    assert [(line.number, line.agent, line.first, line.kinds) for line in asked] == [(3, "classifier", 1, (SAME,)), (6, "contact", 6, (EARLIER,)), (8, "contact", 6, (SAME, EARLIER))]
    made = questions(visited=["classifier", "contact"], owner="Ann", heard=False, lines=asked)
    assert "in lines 1 to 5?" in made["line_6__earlier"]["instructions"] and "in lines 6 to 7?" in made["line_8__same"]["instructions"]
    assert {"line_8__earlier", "line_8__name", "line_8__back", "line_3__same"} <= set(made) and "line_3__earlier" not in made and "line_1__name" not in made
    assert all(question["type"] == "noul" for key, question in made.items() if key.startswith("line_"))
    assert "6. agent: Hi, I'm the contact agent" in call_text(TWO_AGENTS, visited=["classifier", "contact"], profile="", owner="Ann", heard=False)


def test_a_line_that_asks_again_fails_the_rule_and_code_knows_the_agent_and_the_cause():
    asked = asked_lines(TWO_AGENTS, ["classifier", "contact"])
    clean = {"line_3__same": yes(0.1), "line_3__name": yes(0.1), "line_3__back": yes(0.1), "line_6__name": yes(0.05), "line_6__back": yes(0.03)}
    read_back = {"line_8__same": yes(0.97), "line_8__earlier": yes(0.97), "line_8__name": yes(0.1), "line_8__back": yes(0.96)}  # out: a read-back
    reply = answers(heard=False, **clean, **read_back, line_6__earlier=yes(0.93))
    got = {score.rule: score for score in read_answers(reply, heard=False, fail_at=0.8, lines=asked).scores}["nothing_asked_again"]
    assert (got.verdict, got.agent, got.cause) == (FAIL, "contact", "asked_again_after_handoff") and "0.93" in got.note
    assert fail_probability(rule_of("nothing_asked_again"), reply, asked) == ("asked_again_after_handoff", 0.93, "contact")
    assert "nothing_asked_again 0.93" in describe_answers(reply, heard=False, lines=asked)

    def verdict(**over: dict) -> str:
        made = read_answers({**reply, **over}, heard=False, fail_at=0.8, lines=asked)
        return {score.rule: score.verdict for score in made.scores}["nothing_asked_again"]

    assert verdict(line_6__name=yes(0.9)) == PASS  # a question for the name has its own rule
    assert verdict(line_6__back="yes") == PASS  # no usable answer for a filter: the line is not counted
    assert verdict(line_6__earlier=yes(0.7)) == PASS
    assert {score.rule: score.verdict for score in read_answers(reply, heard=False, fail_at=0.8).scores}["nothing_asked_again"] == PASS  # no lines, no line questions


CALLER_ASKS = [
    "agent: Hello. Who am I speaking with?",
    "caller: Dana. Is this a real person?",
    "agent: No. I am an AI agent.",
    "caller: When could he start?",
    "agent: One moment.",
    "agent: Hi, I'm the profile agent, running on Google Gemini. He has 12 years of experience.",
    "caller: Okay. And does he travel?",
]


def test_each_question_of_the_caller_gets_one_small_question_but_not_the_one_in_the_last_turn():
    asked = asked_lines(CALLER_ASKS, ["classifier", "profile"])
    # Line 7 is the caller's last turn. The agent of a caller line is the agent that spoke next.
    assert [(line.number, line.agent, line.kinds, line.last) for line in asked] == [(2, "classifier", (OPEN,), 7), (4, "classifier", (OPEN,), 7)]
    made = questions(visited=["classifier", "profile"], owner="Ann", heard=False, lines=asked)
    assert "without a response in lines 5 to 7?" in made["line_4__open"]["instructions"]
    assert "line_4__name" not in made and "line_7__open" not in made


def test_a_question_of_the_caller_with_no_response_fails_the_rule_with_the_agent_that_spoke_next():
    asked = asked_lines(CALLER_ASKS, ["classifier", "profile"])
    reply = answers(heard=False, line_2__open=yes(0.04), line_4__open=yes(0.94))
    got = {score.rule: score for score in read_answers(reply, heard=False, fail_at=0.8, lines=asked).scores}
    # No question about the whole call found a likely cause, so the cause is "other".
    assert (got["question_answered"].verdict, got["question_answered"].agent, got["question_answered"].cause) == (FAIL, "classifier", OTHER)
    assert got["nothing_asked_again"].verdict == PASS  # a caller line is not a line of that rule
    likely = {**reply, "question_answered__answered_other_question": yes(0.7)}
    assert fail_probability(rule_of("question_answered"), likely, asked) == ("answered_other_question", 0.94, "classifier")
    whole_call = {**reply, "question_answered__question_ignored": yes(0.97), "question_answered__agent": choice("profile")}
    top = {score.rule: score for score in read_answers(whole_call, heard=False, fail_at=0.8, lines=asked).scores}["question_answered"]
    assert (top.agent, top.cause) == ("profile", "question_ignored")  # the whole-call question is higher, so its agent answer is used


# --- the request ---


async def test_the_request_holds_the_call_audio_the_text_and_the_questions():
    decider, send = grader({"answers": answers(), "usage": {"total_input_tokens": 1200}})

    made = await decider.grade(CALL, visited=VISITED, profile="notes", owner="Ann", audio=AUDIO)

    body = send.bodies[0]
    assert body["model"] == "gemini-decision-1" and [block["type"] for block in body["input"]] == ["audio", "text"]
    assert body["input"][0]["mime_type"] == "audio/wav" and read_wav(base64.b64decode(body["input"][0]["data"])) == AUDIO
    assert "no_talk_over__did_not_stop" in body["questions"] and "caller_mood" in body["questions"]
    assert made.readings["caller_mood"] == "neutral" and "heard 1 s of audio" in made.by
    assert {score.rule for score in made.scores} == {rule.key for rule in model_rules(hears=True)}


async def test_with_no_audio_or_with_the_audio_kept_on_the_server_the_model_reads_the_transcript():
    for decider, send, audio in (grader({"answers": answers(heard=False)}) + (None,), grader({"answers": answers(heard=False)}, hears=False) + (AUDIO,)):
        made = await decider.grade(CALL, visited=VISITED, audio=audio)
        assert [block["type"] for block in send.bodies[0]["input"]] == ["text"]
        assert "no_talk_over__did_not_stop" not in send.bodies[0]["questions"] and made.readings == {} and "read the transcript" in made.by


async def test_a_failed_request_or_a_reply_with_no_verdict_gives_no_grade():
    for reply in (RuntimeError("HTTP 429"), {"error": "x"}, {"answers": {}}, {"answers": "none"}):
        decider, _ = grader(reply)
        assert await decider.grade(CALL, visited=VISITED, audio=AUDIO) is None


def test_the_decision_grader_is_off_with_no_key_or_no_model():
    assert default_decision_grader(Settings(_env_file=None, google_api_key="", grader_decision_model="gemini-decision-1")) is None
    assert default_decision_grader(Settings(_env_file=None, google_api_key="k", grader_decision_model="")) is None
    made = default_decision_grader(Settings(_env_file=None, google_api_key="k", grader_decision_model="gemini-decision-1", grader_fail_at=0.9, grader_hears_call=False))
    assert (made.model, made.fail_at, made.hears) == ("gemini-decision-1", 0.9, False)


# --- the two graders ---


async def test_the_decision_grader_checks_the_model_rules_and_the_grader_model_is_not_asked():
    decider, _ = grader({"answers": answers(name_asked_once__asked_again_after_handoff=yes(0.95))})
    model = scripted("should not be used")

    made = await grade(CALL, visited=VISITED, summary_tries=1, model=model, decider=decider, audio=AUDIO)

    assert model.calls == [] and [score.rule for score in made.scores] == [rule.key for rule in RULES]
    assert {score.rule for score in made.scores if score.verdict == FAIL and rule_of(score.rule).check} == {"name_asked_once"}
    assert made.readings["agent_voice"] == "natural"


async def test_when_the_decision_grader_gives_no_answer_the_grader_model_reads_the_transcript():
    class Broken:
        async def grade(self, *args, **kwargs):
            raise RuntimeError("down")

    text = '{"name_asked_once": {"verdict": "fail", "cause": "asked_again_after_handoff"}, "nothing_asked_again": "pass", "question_answered": "na", "caller_memory_right": "na", "facts_from_profile": "pass"}'
    for decider in (grader(RuntimeError("HTTP 500"))[0], Broken()):
        model = scripted(text)
        made = await grade(CALL, visited=VISITED, summary_tries=1, model=model, decider=decider, audio=AUDIO)
        assert len(model.calls) == 1 and made.by == "the grader model" and made.readings == {}
        assert [score.rule for score in made.scores] == [rule.key for rule in RULES if not rule.hears]


def test_the_differences_of_two_graders_name_only_the_rules_that_both_checked():
    first = [Score("name_asked_once", FAIL), Score("question_answered", PASS), Score("no_talk_over", FAIL)]
    second = [Score("name_asked_once", PASS), Score("question_answered", PASS)]
    assert differences(first, second) == ["name_asked_once: fail and pass"]


# --- the record, the report, and the fix request ---


@pytest.fixture
def db():
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    yield connection
    connection.close()


def call(**over) -> CallAnalytics:
    base = {"call_id": "call-1", "day": "2026-11-21", "channel": "voice", "version": VERSION, "test": False, "duration_secs": 60, "agents": VISITED, "outcome": "none"}
    return CallAnalytics(**{**base, **over})


async def test_the_readings_are_stored_and_the_public_report_does_not_show_them(db):
    decider, _ = grader({"answers": answers(caller_mood=choice("frustrated"), tone_fits_caller__cheerful_to_upset_caller=yes(0.9))})

    record = await record_call(db, call(), CALL, decider=decider, audio=AUDIO)

    stored = AnalyticsStore(db).calls()[0]
    assert stored.readings == record.readings == {"caller_mood": "frustrated", "caller_mood_at_end": "positive", "agent_voice": "natural"}
    report = public_report([stored])
    assert "frustrated" not in json.dumps(report) and "readings" not in json.dumps(report)
    assert report["calls"][0]["scores"]["tone_fits_caller"] == FAIL
    by = {rule["key"]: rule["checked_by"] for rule in report["rules"]}
    assert (by["no_talk_over"], by["name_asked_once"], by["handoff_intro"]) == ("a model that hears the call", "a model", "code")


def test_a_database_from_before_the_readings_gets_the_column(db):
    db.execute(
        "CREATE TABLE call_analytics (seq INTEGER PRIMARY KEY AUTOINCREMENT, call_id TEXT NOT NULL UNIQUE, day TEXT NOT NULL, channel TEXT NOT NULL, version TEXT NOT NULL,"
        " test INTEGER NOT NULL, duration_secs INTEGER NOT NULL, agents TEXT NOT NULL, outcome TEXT NOT NULL, caller_turns INTEGER NOT NULL, failovers INTEGER NOT NULL,"
        " tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL, cost_usd REAL NOT NULL, handoff_secs TEXT NOT NULL)"
    )
    db.execute("INSERT INTO call_analytics (call_id, day, channel, version, test, duration_secs, agents, outcome, caller_turns, failovers, tokens_in, tokens_out, cost_usd, handoff_secs) VALUES ('old', '2026-11-20', 'voice', 'v', 0, 5, 'classifier', 'none', 1, 0, 0, 0, 0, '[]')")
    store = AnalyticsStore(db)
    store.save(call(readings={"caller_mood": "sad"}))
    assert [found.readings for found in store.calls()] == [{}, {"caller_mood": "sad"}]


def test_an_audio_rule_that_no_prompt_can_repair_starts_no_fix_request():
    scores = [Score("no_talk_over", FAIL, cause="did_not_stop"), Score("speech_clear", FAIL, cause="cut_off"), Score("tone_fits_caller", FAIL, agent="contact", cause="cheerful_to_upset_caller")]
    assert [request.rule for request in fix_requests(call(scores=scores), [])] == ["tone_fits_caller"]
    env = {"FIX_RULE": "no_talk_over", "FIX_CAUSE": "did_not_stop", "FIX_VERSION": VERSION}
    with pytest.raises(Refused):
        request_from(env)
    assert request_from({**env, "FIX_RULE": "tone_fits_caller", "FIX_CAUSE": "cheerful_to_upset_caller"}).rule == "tone_fits_caller"


# --- the call audio ---


def test_the_parts_of_a_call_join_and_a_delete_leaves_nothing():
    store = CallAudioStore()
    assert store.get("a") is None
    store.add("a", b"\x01\x00" * 4000, 8000)
    store.add("a", b"\x02\x00" * 4000, 8000)
    store.add("a", b"\x03\x00" * 4000, 16000)  # another sample rate: not kept
    store.add("b", b"\x04\x00" * 8, 8000)
    audio = store.get("a")
    assert audio.seconds == 1.0 and audio.pcm == b"\x01\x00" * 4000 + b"\x02\x00" * 4000
    assert read_wav(audio.wav()) == audio
    store.drop("a")
    assert store.get("a") is None and store.get("b") is not None


def test_audio_past_the_limit_is_not_kept():
    store = CallAudioStore()
    store.add("a", bytes(2 * 8000 * (MAX_SECS - 1)), 8000)
    store.add("a", bytes(2 * 8000 * 5), 8000)
    store.add("a", bytes(2 * 8000 * 5), 8000)
    assert store.get("a").seconds == MAX_SECS


async def test_audio_kept_after_a_transfer_is_deleted_later_unless_the_call_goes_on():
    store = CallAudioStore()
    store.add("gone", b"\x01\x00" * 80, 8000)
    store.add("back", b"\x01\x00" * 80, 8000)
    store.drop_later("gone", 0.01)
    store.drop_later("back", 0.01)
    store.keep("back")  # the call came back: a new media stream started
    await asyncio.sleep(0.05)
    assert store.get("gone") is None and store.get("back") is not None


async def test_the_recorder_keeps_a_part_and_a_forgotten_caller_loses_the_whole_call():
    store, forgotten = CallAudioStore(), [False]
    buffer = recorder("a", sample_rate=8000, forgotten=lambda: forgotten[0], store=store)

    await buffer._call_event_handler("on_audio_data", b"\x01\x00" * 80, 8000, 1)
    await asyncio.sleep(0)
    assert store.get("a").pcm == b"\x01\x00" * 80

    forgotten[0] = True
    await buffer._call_event_handler("on_audio_data", b"\x02\x00" * 80, 8000, 1)
    await asyncio.sleep(0)
    assert store.get("a") is None


async def test_in_a_real_pipeline_the_recorder_has_both_sides_in_the_store_when_the_stream_ends():
    from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
    from pipecat.tests.utils import run_test

    store = CallAudioStore()
    buffer = recorder("a", sample_rate=8000, forgotten=lambda: False, store=store)
    caller = [InputAudioRawFrame(audio=(1000).to_bytes(2, "little", signed=True) * 160, sample_rate=8000, num_channels=1) for _ in range(25)]
    agent = [OutputAudioRawFrame(audio=(-2000).to_bytes(2, "little", signed=True) * 160, sample_rate=8000, num_channels=1) for _ in range(25)]

    await run_test(buffer, frames_to_send=[*caller, *agent])

    audio = store.get("a")
    assert audio is not None and audio.sample_rate == 8000
    samples = {int.from_bytes(audio.pcm[at : at + 2], "little", signed=True) for at in range(0, len(audio.pcm), 2)}
    assert 1000 in samples and -2000 in samples  # the caller's sound and the agent's sound
    assert 0.9 <= audio.seconds <= 1.1  # 25 frames of 20 ms for each side, one after the other


def test_an_agent_line_that_denies_the_memory_gets_its_own_question_and_names_the_agent():
    lines = [
        "agent: I remember callers, so first: who am I speaking with?",  # no "not": no question
        "caller: Sam.",
        "agent: Welcome back, Sam. Last time you asked about a store.",
        "caller: I never called before.",
        "agent: I do not look up earlier calls.",
        "agent: Hi, I'm the tour agent, running on Amazon Nova. The line keeps notes for each phone number. It is not a secret.",
    ]
    asked = [line for line in asked_lines(lines, ["classifier", "tour"]) if DENIED in line.kinds]
    assert [(line.number, line.agent) for line in asked] == [(5, "classifier"), (6, "tour")]
    made = questions(visited=["classifier", "tour"], owner="Ann", heard=False, lines=asked)
    assert "In line 5, does the agent deny" in made["line_5__denied"]["instructions"]
    reply = answers(heard=False, line_5__denied=yes(0.93), line_6__denied=yes(0.04))
    top = fail_probability(rule_of("caller_memory_right"), reply, asked)
    assert (top.cause, top.agent, round(top.chance, 2)) == ("denied_the_memory", "classifier", 0.93)
    assert fail_probability(rule_of("nothing_asked_again"), reply, asked).chance < 0.1  # the line is not a question of that rule


def test_the_memory_rule_fails_with_its_cause_and_is_na_when_no_one_spoke_about_an_earlier_call():
    wrong = answers(heard=False, **{cause_key("caller_memory_right", "notes_of_another_caller"): yes(0.93), "caller_memory_right" + AGENT: choice("classifier")})
    got = {score.rule: score for score in read_answers(wrong, heard=False, fail_at=0.8).scores}["caller_memory_right"]
    assert (got.verdict, got.agent, got.cause) == (FAIL, "classifier", "notes_of_another_caller")
    quiet = answers(heard=False, **{"caller_memory_right" + APPLIES: yes(0.04)})
    assert {score.rule: score.verdict for score in read_answers(quiet, heard=False, fail_at=0.8).scores}["caller_memory_right"] == NA
    assert rule_of("caller_memory_right").fix_request and set(ASKS["caller_memory_right"].causes) == {code for code, _ in rule_of("caller_memory_right").causes}
