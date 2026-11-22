"""The analytics store, and the public report that must not leak a call."""

import json

from ai_call_center.analytics import MESSAGE, NO_OUTCOME, TEXT, TRANSFER, VOICE, AnalyticsStore, Answer, CallAnalytics, call_day, forget_call, harness_version, public_report, record_call
from ai_call_center.grading import FAIL, NA, PASS, RULES, Score
from ai_call_center.graph.tools import forget
from ai_call_center.leads import LeadStore
from datetime import UTC, datetime
from tests.conftest import scripted

SECRET_NOTE = "The agent asked Dana from Acme for her name two times."
CALL_ID = "v2:secret-call-id"


def a_call(call_id: str = CALL_ID, **over) -> CallAnalytics:
    call = CallAnalytics(
        call_id=call_id,
        day="2026-11-20",
        channel=VOICE,
        version="abc123def456",
        test=False,
        duration_secs=127,
        agents=["classifier", "tour", "contact"],
        outcome=MESSAGE,
        caller_turns=9,
        failovers=1,
        tokens_in=11_030,
        tokens_out=2_551,
        cost_usd=0.0417,
        answers=[Answer("openai_realtime", 0.612, "classifier"), Answer("openai_realtime", 0.9, "tour"), Answer("gemini_live", 1.4, "contact")],
        handoff_secs=[1.9, 2.1],
        scores=[Score("name_asked_once", FAIL, SECRET_NOTE), Score("handoff_intro", PASS), Score("question_answered", NA)],
    )
    return CallAnalytics(**{**call.__dict__, **over})


def test_a_record_comes_back_as_it_was_saved_with_a_number():
    store = AnalyticsStore(":memory:")
    store.save(a_call())
    [back] = store.calls()
    assert back.seq == 1
    assert back.scores[0].rule == "handoff_intro"  # in the order of RULES
    assert {**back.__dict__, "seq": 0, "scores": sorted(back.scores, key=lambda s: s.rule)} == {**a_call().__dict__, "scores": sorted(a_call().scores, key=lambda s: s.rule)}


def test_a_call_saved_again_keeps_its_number_and_gets_the_new_figures():
    # A call that comes back from a transfer nobody answered is recorded a second time.
    store = AnalyticsStore(":memory:")
    store.save(a_call("call-1", outcome=TRANSFER))
    store.save(a_call("call-2"))
    store.save(a_call("call-1", outcome=NO_OUTCOME, duration_secs=300, answers=[Answer("gemini_live", 2.0, "contact")], scores=[Score("handoff_intro", FAIL, "new")]))
    first, second = store.calls()
    assert (first.call_id, first.seq, second.seq) == ("call-1", 1, 2)
    assert (first.outcome, first.duration_secs, len(first.answers)) == (NO_OUTCOME, 300, 1)
    assert first.scores == [Score("handoff_intro", FAIL, "new")]


def test_forget_deletes_the_whole_record_of_one_call(tmp_path):
    db = tmp_path / "calls.sqlite3"
    store = AnalyticsStore(db)
    store.save(a_call("call-1"))
    store.save(a_call("call-2"))
    forget_call(db, "call-1")
    assert [call.call_id for call in store.calls()] == ["call-2"]
    for table in ("call_analytics", "call_answers", "call_scores"):
        assert store._conn.execute(f"SELECT COUNT(*) FROM {table} WHERE call_id = 'call-1'").fetchone()[0] == 0


def test_the_public_report_holds_no_call_id_no_note_and_no_outcome_of_one_call():
    report = public_report([a_call()])
    text = json.dumps(report)
    assert CALL_ID not in text and "secret" not in text
    assert "Dana" not in text and "Acme" not in text and SECRET_NOTE not in text
    [call] = report["calls"]
    assert "outcome" not in call and "call_id" not in call and "test" not in call
    # The allow-list. A new field must be added here on purpose, after a check that it says nothing about a caller.
    assert set(call) == {"n", "day", "channel", "version", "duration_secs", "caller_turns", "agents", "failovers", "tokens_in", "tokens_out", "cost_cents", "answers", "handoff_secs", "scores"}
    assert call["scores"] == {"name_asked_once": "fail", "handoff_intro": "pass", "question_answered": "na"}
    assert call["answers"][0] == {"agent": "classifier", "provider": "OpenAI", "secs": 0.61} and call["answers"][2]["provider"] == "Google Gemini"
    assert (call["duration_secs"], call["cost_cents"], call["day"]) == (130, 4.2, "2026-11-20")  # the length is rounded to 10 s


def test_a_score_for_a_rule_that_is_gone_stays_out_of_the_report():
    [call] = public_report([a_call(scores=[Score("an_old_rule", FAIL, "x"), Score("handoff_intro", PASS)])])["calls"]
    assert call["scores"] == {"handoff_intro": "pass"}


def test_outcomes_are_totals_and_only_from_five_calls_on():
    calls = [a_call(f"call-{n}", outcome=TRANSFER if n == 0 else MESSAGE) for n in range(4)]
    assert public_report(calls)["totals"]["outcomes"] is None
    calls.append(a_call("call-4", outcome=NO_OUTCOME))
    assert public_report(calls)["totals"]["outcomes"] == {"message": 3, "transfer": 1, "none": 1}


def test_test_calls_are_left_out_unless_asked_for():
    calls = [a_call("real"), a_call("mine", test=True, channel=TEXT)]
    assert public_report(calls)["totals"]["calls"] == 1
    assert public_report(calls, tests=True)["totals"]["calls"] == 2


def test_the_report_gives_a_score_for_each_version_in_the_order_the_versions_took_calls():
    old = [Score("name_asked_once", FAIL), Score("handoff_intro", PASS)]
    new = [Score("name_asked_once", PASS), Score("handoff_intro", PASS), Score("question_answered", NA)]
    calls = [
        a_call("c1", version="v-old", scores=old, answers=[Answer("openai_realtime", 1.0, "tour")]),
        a_call("c2", version="v-old", scores=old, answers=[Answer("openai_realtime", 2.0, "tour")]),
        a_call("c3", version="v-new", scores=new, answers=[Answer("openai_realtime", 0.5, "tour")]),
    ]
    report = public_report(calls)
    assert report["versions"] == [
        {"version": "v-old", "calls": 2, "passed": 2, "failed": 2, "score": 0.5, "median_answer_secs": 1.5},
        {"version": "v-new", "calls": 1, "passed": 2, "failed": 0, "score": 1.0, "median_answer_secs": 0.5},
    ]
    assert report["totals"]["score"] == 0.67 and report["totals"]["median_answer_secs"] == 1.0
    by_key = {rule["key"]: rule for rule in report["rules"]}
    assert [rule["key"] for rule in report["rules"]] == [rule.key for rule in RULES]
    assert (by_key["name_asked_once"]["pass"], by_key["name_asked_once"]["fail"], by_key["question_answered"]["na"]) == (1, 2, 1)
    assert by_key["name_asked_once"]["checked_by"] == "a model" and by_key["handoff_intro"]["checked_by"] == "code"


def test_an_empty_store_gives_an_empty_report():
    report = public_report([])
    assert report["calls"] == [] and report["versions"] == [] and report["totals"]["score"] is None and report["totals"]["median_answer_secs"] is None


async def test_record_call_grades_the_call_counts_the_turns_and_saves_the_record():
    store = AnalyticsStore(":memory:")
    lines = ["agent: Who am I speaking with?", "caller: Dana.", "caller: How does this work?", "agent: Hi, I'm the tour agent, running on OpenAI."]
    model = scripted('{"name_asked_once": {"verdict": "pass"}, "question_answered": {"verdict": "fail", "note": "No answer."}}')
    record = await record_call(store._conn, a_call(agents=["classifier", "tour"], caller_turns=0, scores=[]), lines, summary_tries=2, grader=model)
    assert record.caller_turns == 2
    assert {score.rule: score.verdict for score in record.scores} == {
        "tool_name_not_spoken": PASS,
        "handoff_intro": PASS,
        "summary_first_try": FAIL,
        "name_asked_once": PASS,
        "question_answered": FAIL,
    }
    assert store.calls() == [CallAnalytics(**{**record.__dict__, "seq": 1})]
    # The transcript was read for the grade. It is in no table.
    dump = "\n".join(store._conn.iterdump())
    assert "How does this work" not in dump and "Dana" not in dump


def test_forget_caller_deletes_the_notes_of_the_callers_earlier_calls(ctx, leads):
    # The analytics tables are in the same database as the call log.
    analytics = AnalyticsStore(leads.connection)
    analytics.save(a_call("old-call"))
    analytics.save(a_call("other-person"))
    leads.start_call("old-call", ctx.caller_number)
    leads.finish_call("old-call", agents=["classifier"], summary="Asked about a project.", caller_name="Dana")
    ctx.caller_name = "Dana"
    forget(ctx)
    notes = {call.call_id: [score.note for score in call.scores if score.note] for call in analytics.calls()}
    assert notes == {"old-call": [], "other-person": [SECRET_NOTE]}
    assert [score.verdict for score in analytics.calls()[0].scores] == [PASS, FAIL, NA]  # the verdicts stay


def test_the_day_of_a_call_is_the_us_eastern_date():
    assert call_day(datetime(2026, 11, 21, 2, 30, tzinfo=UTC)) == "2026-11-20"  # 9:30 PM Eastern, the day before


def test_a_configured_version_wins_and_git_gives_one_without_it():
    assert harness_version(" gh-abc123 ") == "gh-abc123"
    assert harness_version("")  # a commit, or "unknown" outside a git checkout: never empty


# --- the end of a phone call ---


def _session(tmp_path, **over):
    from ai_call_center.config import Settings
    from ai_call_center.voice.call_session import CallSession

    profile = tmp_path / "profile.md"
    profile.write_text("# Summary\nTest Owner builds AI systems.\n")
    settings = Settings(_env_file=None, **{"leads_db": tmp_path / "leads.sqlite3", "grader_model": "", "harness_version": "gh-abc123", "owner_phone": "+15555550199", "profile_path": profile, "respellings": "", **over})
    return CallSession(websocket=None, call_data={"call_id": "v2:phone-call"}, settings=settings, telephony=None, specs={})  # type: ignore[arg-type]


def _stats():
    from pipecat.metrics.metrics import LLMTokenUsage

    from ai_call_center.voice.stats import CallStats

    stats = CallStats()
    stats.bind({}, lambda: "openai_realtime", lambda: "tour")
    stats.timer.spoke, stats.timer.last_speech = True, stats.clock() - 0.8
    stats.agent_started()
    stats.note_handoff(1.9)
    stats.failovers = 1
    stats.add_usage("openai_realtime", LLMTokenUsage(prompt_tokens=20_000, completion_tokens=2_000, total_tokens=22_000, input_audio_tokens=500, output_audio_tokens=1_500))
    return stats


MESSAGES = [
    {"role": "assistant", "content": "Who am I speaking with?"},
    {"role": "user", "content": "This is Dana."},
    {"role": "assistant", "content": "Hi, I'm the tour agent, running on OpenAI."},
]


async def test_the_end_of_a_phone_call_saves_the_figures_and_the_code_rules_scores(tmp_path, ctx):
    session = _session(tmp_path)
    await session._record_analytics(ctx, _stats(), MESSAGES, visited=["classifier", "tour"], outcome=MESSAGE, summary_tries=1)
    [call] = AnalyticsStore(session.settings.leads_db).calls()
    assert (call.call_id, call.channel, call.version, call.test, call.outcome) == ("v2:phone-call", VOICE, "gh-abc123", False, MESSAGE)
    assert (call.agents, call.caller_turns, call.failovers, call.handoff_secs) == (["classifier", "tour"], 1, 1, [1.9])
    [answer] = call.answers
    assert (answer.agent, answer.leg) == ("tour", "openai_realtime") and 0.7 < answer.seconds < 1.0
    assert call.tokens_in == 20_000 and call.tokens_out == 2_000 and call.cost_usd > 0
    assert {score.rule: score.verdict for score in call.scores} == {"tool_name_not_spoken": PASS, "handoff_intro": PASS, "summary_first_try": PASS}


async def test_a_call_from_the_owners_phone_is_marked_as_a_test(tmp_path, ctx):
    session = _session(tmp_path, owner_phone=ctx.caller_number)
    await session._record_analytics(ctx, _stats(), MESSAGES, visited=["classifier"], outcome=NO_OUTCOME, summary_tries=0)
    [call] = AnalyticsStore(session.settings.leads_db).calls()
    assert call.test and public_report([call])["calls"] == []


async def test_a_caller_who_asked_to_be_forgotten_gets_no_record_and_loses_an_earlier_one(tmp_path, ctx):
    session = _session(tmp_path)
    await session._record_analytics(ctx, _stats(), MESSAGES, visited=["classifier"], outcome=TRANSFER, summary_tries=1)  # before a transfer that came back
    ctx.forgotten = True
    await session._record_analytics(ctx, _stats(), MESSAGES, visited=["classifier"], outcome=NO_OUTCOME, summary_tries=0)
    assert AnalyticsStore(session.settings.leads_db).calls() == []


async def test_a_failure_in_the_analytics_does_not_stop_the_end_of_a_call(tmp_path, ctx):
    session = _session(tmp_path)
    await session._record_analytics(ctx, object(), MESSAGES, visited=["classifier"], outcome=NO_OUTCOME, summary_tries=0)  # type: ignore[arg-type]
    assert AnalyticsStore(session.settings.leads_db).calls() == []


async def test_the_end_of_a_call_publishes_the_report_but_not_after_a_test_call(tmp_path, ctx, monkeypatch):
    from ai_call_center.voice import call_session

    published = []

    async def publish(settings):
        published.append(len(AnalyticsStore(settings.leads_db).calls()))

    monkeypatch.setattr(call_session, "publish_report", publish)
    session = _session(tmp_path)
    await session._record_analytics(ctx, _stats(), MESSAGES, visited=["classifier"], outcome=MESSAGE, summary_tries=1)
    assert published == [1]  # after the record is saved
    ctx.forgotten = True
    await session._record_analytics(ctx, _stats(), MESSAGES, visited=["classifier"], outcome=NO_OUTCOME, summary_tries=0)
    assert published == [1, 0]  # the record is gone from the report too
    owner = _session(tmp_path, owner_phone=ctx.caller_number)
    ctx.forgotten = False
    await owner._record_analytics(ctx, _stats(), MESSAGES, visited=["classifier"], outcome=NO_OUTCOME, summary_tries=0)
    assert published == [1, 0]
