"""The fault report: what went wrong in a call, with each private word removed by code."""

from __future__ import annotations

import json

from ai_call_center.fault_report import LIMITS, REMOVED, FaultReport, public_words, safe, said_names, scrub, summary_report, words_of, write_report
from ai_call_center.grading import rule_of

LINES = [
    "agent: Hello. Who am I speaking with?",
    "caller: This is Zofia Kowalczyk from Brightwater Dental. My number is five five five zero one two three.",
    "agent: Welcome back, Zofia. Last time you asked about a booking platform for Brightwater.",
    "caller: I never called before. Why did you say that? Write to zofia@brightwater.example or call 555 0123.",
    "agent: I do not look up earlier calls.",
]
CALL = words_of(LINES)


def clean(text: str, **over) -> str:
    return scrub(text, call_words=CALL, **over)


def test_a_word_of_the_call_stays_only_when_it_is_public_already():
    assert "caller" in public_words() and "zofia" not in public_words() and "brightwater" not in public_words()
    got = clean("The agent welcomed Zofia Kowalczyk of Brightwater Dental back and named a booking platform.")
    assert "Zofia" not in got and "Kowalczyk" not in got and "Brightwater" not in got and REMOVED in got
    assert got.startswith("The agent welcomed") and "booking platform" in got
    # A word that the model adds itself is not a word of the call, so it stays.
    assert clean("The agent denied a capability of the system.") == "The agent denied a capability of the system."


def test_digits_addresses_links_and_a_row_of_number_words_are_removed():
    got = clean("The caller gave 555 0123, then zofia@brightwater.example, then www.brightwater.example/contact.")
    assert not any(ch.isdigit() for ch in got) and "@" not in got and "example" not in got
    got = clean("The caller said five five five zero one two three, and asked one question.")
    assert "five" not in got and "three" not in got and "asked one question" in got


def test_a_known_name_and_a_capital_letter_inside_a_sentence_are_removed():
    call = words_of(["caller: this is mark rose", "agent: thanks mark, the system will call back"])
    public = frozenset({"mark", "rose", "thanks", "the", "system", "will", "call", "back", "caller", "agent", "said", "his", "name", "is", "this"})
    # "mark" and "rose" are plain words too. As a name inside a sentence they have a capital letter.
    assert scrub("The caller said his name is Mark Rose.", call_words=call, public=public) == f"The caller said his name is {REMOVED}."
    # A name that the server knows for this caller goes even in lower case.
    assert "mark" not in scrub("The caller, mark, asked again.", call_words=call, public=public, private=frozenset({"mark"}))
    # "I" and a short form in capital letters stay, and so does the first word of a sentence.
    assert scrub("The agent said: I will call back. Thanks.", call_words=call, public=public) == "The agent said: I will call back. Thanks."


def test_code_finds_the_name_that_the_caller_said_even_when_the_server_does_not_know_it():
    lines = ["agent: Tell me your first name.", "caller: mark", "agent: Welcome back.", "caller: My name is Mark Rose and I never called before.", "caller: no"]
    assert said_names(lines) == {"mark", "rose"}  # "no" is not after the name question; a plain base word is never a name
    assert said_names(["agent: What is your name?", "caller: I do not want to say that to you now"]) == frozenset()
    assert words_of(["Mark", None, ""]) == {"mark"}  # the server has no name for a new caller


def test_a_plain_form_of_a_public_word_stays_and_curly_quotes_become_plain():
    call = words_of(["agent: I don\u2019t look up records. That seemed wrong, and the matching issues stay."])
    public = frozenset({"i", "t", "don", "look", "up", "records", "seem", "match", "issue", "the", "stay", "agent", "said"})
    assert scrub("The agent said: I don\u2019t look up records.", call_words=call, public=public) == "The agent said: I don't look up records."
    assert scrub("seemed, matching, issues", call_words=call, public=public) == "seemed, matching, issues"


def test_a_report_is_cut_to_its_limit_and_holds_only_safe_characters():
    got = clean("The agent said \u201cno\u201d \u2014 then <b>stopped</b> `rm -rf` {x}. " + "It asked again. " * 60, limit=120)
    assert len(got) <= 128 and safe(got, 120) and got.endswith("...")
    assert not safe("call 555", 100) and not safe("a/b", 100) and not safe("x" * 200, 100) and safe("", 100)


class Model:
    def __init__(self, text: str) -> None:
        self.text, self.asked = text, []

    async def ainvoke(self, messages, config=None):
        self.asked.append(messages[0].content)
        return type("Reply", (), {"text": self.text})()


async def test_the_model_writes_the_report_and_code_removes_what_the_model_let_through():
    answer = {
        "happened": "The caller gave a first name. The agent welcomed Zofia back and read notes of an earlier call. The caller said that this was a first call.",
        "agent_said": "Welcome back, Zofia. Last time you asked about a booking platform for Brightwater.",
        "expected": "The agent says that it matched the name to an earlier caller.",
        "extra": "ignored",
    }
    model = Model("Here it is:\n" + json.dumps(answer))
    report = await write_report(LINES, rule=rule_of("question_answered"), agent="classifier", note="0.87", model=model)  # the server keeps no name for this call
    assert "2. caller: This is Zofia" in model.asked[0] and "Never write the name of a person" in model.asked[0]
    assert report.happened.startswith("The caller gave a first name.") and "Zofia" not in "".join(report.fields().values())
    assert report.agent_said == f"Welcome back, {REMOVED}. Last time you asked about a booking platform for {REMOVED}."
    assert all(safe(text, LIMITS[key]) for key, text in report.fields().items())


async def test_no_usable_answer_gives_an_empty_report_and_never_an_error():
    assert not await write_report(LINES, rule=rule_of("question_answered"), agent="", note="", model=Model("I cannot do that."))
    assert not await write_report(LINES, rule=rule_of("question_answered"), agent="", note="", model=None)


def test_the_summary_rule_gets_a_report_from_fixed_texts():
    report = summary_report(['- The "summary" has more than 40 words.', '- The "topic" holds a number.'])
    assert "longer than the word limit" in report.happened and "The topic held a number." in report.happened
    assert safe(report.happened, LIMITS["happened"]) and safe(report.expected, LIMITS["expected"])
    assert not summary_report([]) and not summary_report(["- something new"]) and not FaultReport()
