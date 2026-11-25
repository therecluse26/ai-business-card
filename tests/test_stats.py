"""Live call figures: answer time, tokens, and the cost estimate."""

from pipecat.metrics.metrics import LLMTokenUsage

from ai_call_center.analytics import Answer
from ai_call_center.voice.stats import PER_MINUTE, PRICES, AnswerTimer, CallStats, Price, Usage

LOUD, QUIET = 0.1, 0.0001


def speak(timer: AnswerTimer, start: float, secs: float) -> float:
    """Feed caller speech in 20 ms frames. Returns the time of the last frame."""
    now = start
    for i in range(int(secs / 0.02)):
        now = start + i * 0.02
        timer.caller_audio(LOUD, now)
    return now


def test_answer_time_runs_from_the_end_of_the_caller_speech_to_the_agent_audio():
    timer = AnswerTimer()
    end = speak(timer, 10.0, 1.0)
    for i in range(40):  # the caller is quiet for 0.8 s
        timer.caller_audio(QUIET, end + 0.02 * (i + 1))
    gap = timer.agent_started(end + 0.8)
    assert gap is not None and abs(gap - 0.8) < 0.001


def test_agent_audio_with_no_caller_speech_before_it_is_not_an_answer():
    timer = AnswerTimer()
    assert timer.agent_started(5.0) is None  # the greeting
    end = speak(timer, 10.0, 0.5)
    assert timer.agent_started(end + 0.5) is not None
    assert timer.agent_started(end + 3.0) is None  # the agent goes on; the caller said nothing new


def test_a_click_on_the_line_is_not_speech():
    timer = AnswerTimer()
    timer.caller_audio(LOUD, 1.00)
    timer.caller_audio(LOUD, 1.02)
    timer.caller_audio(QUIET, 1.04)
    assert timer.agent_started(2.0) is None


def test_a_very_long_gap_is_not_counted():
    timer = AnswerTimer()
    end = speak(timer, 10.0, 0.5)
    assert timer.agent_started(end + 60) is None


def test_usage_splits_text_audio_and_cached_tokens():
    usage = Usage()
    # The input count holds the audio tokens and the cached tokens.
    usage.add(LLMTokenUsage(prompt_tokens=1341, completion_tokens=132, total_tokens=1473, cache_read_input_tokens=1216, input_audio_tokens=46, output_audio_tokens=50, cache_read_input_audio_tokens=0))
    assert (usage.text_in, usage.text_in_cached, usage.audio_in, usage.audio_in_cached) == (79, 1216, 46, 0)
    assert (usage.text_out, usage.audio_out) == (82, 50)
    assert usage.tokens_in == 1341 and usage.tokens_out == 132


def test_usage_with_no_breakdown_counts_as_text():
    usage = Usage()
    usage.add(LLMTokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120))
    assert (usage.text_in, usage.text_out, usage.audio_in, usage.audio_out) == (100, 20, 0, 0)


def test_cost_uses_one_price_for_each_kind_of_token():
    price = Price(text_in=1.0, text_in_cached=0.1, audio_in=10.0, audio_in_cached=0.5, text_out=2.0, audio_out=20.0)
    usage = Usage(text_in=1_000_000, text_in_cached=1_000_000, audio_in=100_000, audio_in_cached=200_000, text_out=500_000, audio_out=50_000)
    assert abs(usage.cost(price) - (1.0 + 0.1 + 1.0 + 0.1 + 1.0 + 1.0)) < 1e-9


def test_describe_reports_speed_handoffs_tokens_and_cost():
    stats = CallStats()
    stats.answers += [Answer("openai_realtime", 0.6), Answer("openai_realtime", 1.0), Answer("gemini_live", 1.4)]
    stats.note_handoff(1.5)
    stats.note_handoff(2.5)
    stats.add_usage("openai_realtime", LLMTokenUsage(prompt_tokens=20_000, completion_tokens=2_000, total_tokens=22_000, input_audio_tokens=500, output_audio_tokens=1_500))
    stats.add_usage("gemini_live", LLMTokenUsage(prompt_tokens=5_000, completion_tokens=800, total_tokens=5_800, input_audio_tokens=600, output_audio_tokens=600))

    text = stats.describe()

    assert "over 3 answers" in text and "1.0 seconds after the caller stopped talking" in text and "slowest answer took 1.4 seconds" in text
    assert "OpenAI 0.8 seconds over 2" in text and "Google Gemini 1.4 seconds over 1" in text
    assert "Handoffs: 2" in text and "2.0 seconds on average" in text
    assert "25,000 in and 2,800 out" in text
    dollars, unknown = stats.cost()
    assert unknown == []
    expected = (19_500 * 0.60 + 500 * 10 + 500 * 2.40 + 1_500 * 20) / 1e6 + (4_400 * 0.75 + 600 * 3 + 200 * 4.50 + 600 * 12) / 1e6
    assert abs(dollars - expected) < 1e-9
    assert f"about {expected * 100:.0f} cents" in text


def test_describe_before_anything_is_measured():
    text = CallStats().describe()
    assert "no answer has been timed yet" in text and "no usage has been reported yet" in text


def test_a_small_cost_is_said_as_under_one_cent_and_an_unknown_provider_is_named():
    stats = CallStats()
    stats.add_usage("openai_realtime", LLMTokenUsage(prompt_tokens=1_000, completion_tokens=50, total_tokens=1_050))
    stats.add_usage("a_new_leg", LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    text = stats.describe()
    assert "under one cent" in text and "No price is known for a_new_leg" in text


def test_every_built_voice_leg_has_a_price():
    from ai_call_center.voice.legs import LEG_BUILDERS

    assert set(LEG_BUILDERS) <= set(PRICES) | set(PER_MINUTE)


def test_phonic_minutes_are_in_the_cost_though_phonic_reports_no_tokens():
    """Phonic bills by the minute and reports no tokens. Its minutes must not drop out of the total."""
    held = {"secs": 60.0}
    stats = CallStats()
    stats.bind({}, lambda: "phonic", lambda: "classifier", [("phonic", lambda: held["secs"])])

    dollars, unknown = stats.cost()
    assert unknown == [] and abs(dollars - 1.0 * PER_MINUTE["phonic"]) < 1e-9
    text = stats.describe()
    assert "Tokens: no usage has been reported yet. Phonic bills by the minute, not by the token: 1.0 minutes so far." in text
    assert "about 7 cents" in text

    # A new pipeline (a restarted media stream) brings new services; the minutes of the old ones stay.
    stats.add_usage("openai_realtime", LLMTokenUsage(prompt_tokens=20_000, completion_tokens=2_000, total_tokens=22_000, input_audio_tokens=500, output_audio_tokens=1_500))
    stats.bind({}, lambda: "openai_realtime", lambda: "tour", [("phonic", lambda: 30.0)])
    assert stats.seconds_held() == {"phonic": 90.0}
    dollars, _ = stats.cost()
    tokens = (19_500 * 0.60 + 500 * 10 + 500 * 2.40 + 1_500 * 20) / 1e6
    assert abs(dollars - (tokens + 1.5 * PER_MINUTE["phonic"])) < 1e-9
    assert "Tokens so far: 20,000 in and 2,000 out. Phonic bills by the minute" in stats.describe()


def test_a_service_that_held_no_session_adds_nothing():
    stats = CallStats()
    stats.bind({}, lambda: "openai_realtime", None, [("phonic", lambda: 0.0)])
    assert stats.seconds_held() == {} and "no usage has been reported yet" in stats.describe() and "minute" not in stats.describe()
