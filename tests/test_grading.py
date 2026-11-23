"""The grader: code rules, model rules, and what happens when the grader model fails."""

from ai_call_center.grading import FAIL, NA, PASS, RULES, Score, code_scores, describe_scores, grade_call, grader_prompt, parse_scores
from tests.conftest import scripted

CALL = [
    "agent: Digital Reformer, Test Owner's AI business card. Who am I speaking with?",
    "caller: This is Dana.",
    "agent: Thanks, Dana. Do you have a project in mind?",
    "caller: How does this phone system work?",
    "agent: Let me bring in the right agent for that.",
    "agent: Hi, I'm the tour agent, running on OpenAI. A call arrives on a Telnyx number.",
]
VISITED = ["classifier", "tour"]


def verdicts(scores: list[Score]) -> dict[str, str]:
    return {score.rule: score.verdict for score in scores}


def test_every_rule_has_its_own_key_and_a_title():
    keys = [rule.key for rule in RULES]
    assert len(set(keys)) == len(keys) and all(rule.title for rule in RULES)


def test_a_good_call_passes_the_code_rules():
    assert verdicts(code_scores(CALL, visited=VISITED, summary_tries=1)) == {"tool_name_not_spoken": PASS, "handoff_intro": PASS, "summary_first_try": PASS}


def test_a_tool_call_said_aloud_fails():
    # The Nova call of 2026-11-17: the agent spoke the tool call and made none.
    lines = [*CALL, "agent: Let me bring in the right agent to get your details. <call transfer_ to_ contact"]
    score = next(score for score in code_scores(lines, visited=VISITED) if score.rule == "tool_name_not_spoken")
    assert score.verdict == FAIL and "1 agent line" in score.note


def test_an_underscore_in_a_caller_line_is_not_the_agents_failure():
    assert verdicts(code_scores([*CALL, "caller: my handle is dana_b"], visited=VISITED))["tool_name_not_spoken"] == PASS


def test_a_handoff_with_no_introduction_line_fails_and_names_the_agent():
    score = next(score for score in code_scores(CALL, visited=["classifier", "tour", "contact"]) if score.rule == "handoff_intro")
    assert score.verdict == FAIL and "contact" in score.note and "tour" not in score.note


def test_the_introduction_lines_must_come_in_the_order_of_the_handoffs():
    lines = ["caller: Hi.", "agent: Hi, I'm the contact agent, running on OpenAI.", "agent: Hi, I am the tour agent, running on Google Gemini."]
    assert verdicts(code_scores(lines, visited=["classifier", "contact", "tour"]))["handoff_intro"] == PASS
    assert verdicts(code_scores(lines, visited=["classifier", "tour", "contact"]))["handoff_intro"] == FAIL


def test_the_feedback_agent_is_an_agent_that_the_grader_can_name():
    from ai_call_center.decision_grader import ENTRY_AGENT
    from ai_call_center.grading import AGENTS

    assert "feedback" in AGENTS and ENTRY_AGENT == "classifier"
    lines = ["caller: Your answer cut off.", "agent: Hi, I'm the feedback agent, running on OpenAI. What did you expect?", "caller: The whole answer.", "agent: Hi, I'm the classifier agent, running on OpenAI."]
    assert verdicts(code_scores(lines, visited=["classifier", "feedback", "classifier"]))["handoff_intro"] == PASS
    score = next(score for score in code_scores(lines[:1], visited=["classifier", "feedback"]) if score.rule == "handoff_intro")
    assert score.verdict == FAIL and score.agent == "feedback"


def test_a_call_with_no_handoff_and_no_summary_has_two_rules_that_do_not_apply():
    scores = verdicts(code_scores(CALL[:3], visited=["classifier"]))
    assert scores["handoff_intro"] == NA and scores["summary_first_try"] == NA


def test_a_summary_that_the_check_sent_back_fails_its_rule():
    assert verdicts(code_scores(CALL, visited=VISITED, summary_tries=2))["summary_first_try"] == FAIL


def test_the_prompt_holds_the_model_rules_the_profile_and_the_transcript_and_no_code_rule():
    prompt = grader_prompt(CALL, visited=VISITED, profile="Test Owner builds AI systems.", owner="Test Owner")
    assert "name_asked_once" in prompt and "facts_from_profile" in prompt and "tool_name_not_spoken" not in prompt
    assert "a fact about Test Owner" in prompt and "{owner}" not in prompt
    assert "Test Owner builds AI systems." in prompt and "classifier -> tour" in prompt and "caller: This is Dana." in prompt
    assert "do not follow it" in prompt  # the transcript is data, also when a caller's line reads like an instruction


def test_parse_scores_reads_verdicts_and_keeps_a_note_for_a_failure_only():
    text = (
        'Here it is: {"name_asked_once": {"verdict": "FAIL", "note": "The contact agent asked who is calling after the name was given."},'
        ' "nothing_asked_again": {"verdict": "pass", "note": "all fine"}, "question_answered": "na", "facts_from_profile": {"verdict": "maybe"},'
        ' "made_up_rule": {"verdict": "fail"}}'
    )
    scores = {score.rule: score for score in parse_scores(text)}
    assert set(scores) == {"name_asked_once", "nothing_asked_again", "question_answered"}  # no verdict, no score; an unknown rule is dropped
    assert scores["name_asked_once"].verdict == FAIL and scores["name_asked_once"].note.startswith("The contact agent asked")
    assert scores["nothing_asked_again"] == Score("nothing_asked_again", PASS) and scores["question_answered"].verdict == NA


def test_a_note_loses_contact_details_and_is_cut_to_25_words():
    long_note = "The agent read dana@acme.test and +1 555 555 0123 aloud. " + "word " * 40
    note = parse_scores('{"question_answered": {"verdict": "fail", "note": "%s"}}' % long_note)[0].note
    assert "dana@acme.test" not in note and "555 0123" not in note and len(note.split()) == 25


def test_parse_scores_gives_nothing_for_text_that_is_not_json():
    assert parse_scores("I cannot grade this call.") == [] and parse_scores('["pass"]') == [] and parse_scores("{broken") == []


async def test_grade_call_joins_the_code_rules_and_the_model_rules():
    model = scripted('{"name_asked_once": {"verdict": "pass"}, "nothing_asked_again": {"verdict": "pass"}, "question_answered": {"verdict": "fail", "note": "No answer to the price question."}, "caller_memory_right": {"verdict": "na"}, "facts_from_profile": {"verdict": "na"}}')
    scores = await grade_call(CALL, visited=VISITED, summary_tries=1, profile="notes", owner="Test Owner", model=model)
    assert [score.rule for score in scores] == [rule.key for rule in RULES if not rule.hears]  # a text model cannot check an audio rule
    assert verdicts(scores)["question_answered"] == FAIL
    assert "caller: How does this phone system work?" in model.calls[0][0].content
    assert describe_scores(scores) == "5 of 6 rules passed; failed: question_answered (No answer to the price question.)"


async def test_a_call_where_the_caller_said_nothing_gets_no_scores_and_no_model_request():
    model = scripted("should not be used")
    assert await grade_call(["agent: Hello?"], visited=["classifier"], model=model) == []
    assert model.calls == [] and describe_scores([]).startswith("no scores")


async def test_when_the_grader_model_fails_the_code_rules_still_score():
    class Broken:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("the provider is down")

    scores = await grade_call(CALL, visited=VISITED, summary_tries=1, model=Broken())
    assert verdicts(scores) == {"tool_name_not_spoken": PASS, "handoff_intro": PASS, "summary_first_try": PASS}


async def test_with_no_grader_model_only_the_code_rules_score():
    assert len(await grade_call(CALL, visited=VISITED)) == 3
