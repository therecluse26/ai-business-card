"""A handoff tool name that the agent said aloud in place of a tool call."""

import asyncio
from types import SimpleNamespace

from pipecat.frames.frames import BotStoppedSpeakingFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection

from ai_call_center.voice.call_session import NoToolCall
from ai_call_center.voice.spoken_tool import SpokenToolWatch, spoken_handoff

AGENTS = ("classifier", "profile", "tour", "consultant", "contact")
DOWN = FrameDirection.DOWNSTREAM
# What the consultant agent said on the real call of 2026-11-17.
LEAK = "Let me bring in the right agent to get your details. <call transfer_ to_ contact"


def test_the_real_leak_names_the_contact_agent():
    assert spoken_handoff(LEAK, AGENTS) == "contact"


def test_other_spellings_of_a_tool_name():
    assert spoken_handoff("transfer_to_tour", AGENTS) == "tour"
    assert spoken_handoff("Transfer _ to _ Profile", AGENTS) == "profile"
    assert spoken_handoff("transfer to_consultant", AGENTS) == "consultant"


def test_plain_speech_is_not_a_tool_name():
    assert spoken_handoff("I can transfer to contact you later.", AGENTS) == ""
    assert spoken_handoff("Let me bring in the right agent for that.", AGENTS) == ""
    assert spoken_handoff("transfer_to_nobody", AGENTS) == ""
    assert spoken_handoff(LEAK, ()) == ""


def test_the_last_tool_name_wins():
    assert spoken_handoff("transfer_to_tour, no: transfer_to_contact", AGENTS) == "contact"


def test_watch_reports_when_the_agent_stops_speaking():
    watch = SpokenToolWatch(AGENTS, None)
    assert watch.observe(LLMFullResponseStartFrame(), DOWN) == ""
    assert watch.observe(LLMTextFrame("Let me bring in the right agent."), DOWN) == ""
    assert watch.observe(LLMTextFrame(" <call transfer_ to_"), DOWN) == ""
    assert watch.observe(LLMTextFrame(" contact"), DOWN) == ""
    assert watch.observe(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM) == "contact"
    assert watch.observe(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM) == ""  # one report for one leak


def test_watch_forgets_the_text_of_an_earlier_answer():
    watch = SpokenToolWatch(AGENTS, None)
    watch.observe(LLMTextFrame(LEAK), DOWN)
    watch.observe(LLMFullResponseStartFrame(), DOWN)
    watch.observe(LLMTextFrame("What does your company do?"), DOWN)
    assert watch.observe(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM) == ""


def test_no_tool_call_runs_the_switch_step():
    ran = []

    async def switch() -> None:
        ran.append(True)

    async def go() -> None:
        await NoToolCall().result_callback("Handed off.", properties=SimpleNamespace(on_context_updated=switch))
        await NoToolCall().result_callback("No handoff was made.")  # a refusal has no switch step

    asyncio.run(go())
    assert ran == [True]
