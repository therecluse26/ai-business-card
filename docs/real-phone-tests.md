# Run book: tests on a real phone

These tests cover the parts that no real phone call has used yet (status on 2026-11-18).
Three calls cover all of them. Each call costs a few cents.

| Call | What it tests |
|---|---|
| A | The Gemini leg, the Nova Sonic leg, and LangSmith voice traces |
| B | The spoken fallback when no voice provider answers |
| C | A call that a guardrail limit turns away |

Already confirmed on a real phone, not repeated here: call screening on a transfer, and failover
from one provider to the other.

## Before you start

1. Start the server:

   ```sh
   sudo tailscale up
   tailscale funnel --bg 8000
   uv run serve
   ```

2. Read the start of the log. These three lines must be there:

   - `voice providers at start: ...` with `consultant=nova_sonic/...` in it.
     The same line must hold `classifier=gemini_live/...`.
   - `LangSmith tracing is on: project voice-business-card`
   - `ready: webhooks at ...`

3. Run `uv run check-nova` and `uv run check-phonic`. Each must print what the provider said, not an error.
4. Make a copy of your settings. Calls B and C change them.

   ```sh
   cp .env .env.backup
   ```

The server writes everything to `data/server.log`. After each call, the line
`call <id>: figures: ...` gives the answer times, the tokens and the cost.

## Call A: four providers and the trace

Use the normal `.env`. Dial `TELNYX_PHONE_NUMBER`.

| Step | You say | You must hear |
|---|---|---|
| 1 | Your name, when the agent asks | The greeting and the welcome, in the classifier's voice (Gemini) |
| 2 | "Tell me about the owner's background." | A new voice: "Hi, I'm the profile agent, running on Google Gemini." |
| 3 | One more question about the owner | An answer in the same voice |
| 4 | "I run a small business. How could this help me?" | A new voice: "Hi, I'm the consultant agent, running on Amazon Nova." |
| 5 | Answer its question, then ask one of your own | Two or more answers from Nova. This is the first real call on Nova. |
| 6 | "How does this phone system work?" | A new voice: the tour agent, running on OpenAI |
| 7 | "What has this call cost so far?" | Figures: the time, the providers, "about" a cost |
| 8 | "Is this call traced?" | The agent names LangSmith |
| 9 | "That's all, goodbye." | A goodbye. The agent hangs up. |

Pass:

- Each handoff starts with the intro line, in a different voice.
- No silence longer than about 3 seconds.
- The log has no `ERROR` line for this call.

Then check LangSmith, project `voice-business-card`, tab **Threads**:

- One thread for this call. Its name is the Telnyx call id (it starts with `v3:`).
- The trace has a span for each turn.
- Your words are there, not only the agents' words.
- The `get_call_stats` tool call is there.
- A `call-summary` run is in the same thread.
- Known limit: the Nova part shows turns and timing, not words or tool calls.

Write down how the handoffs look in the trace. That is the one open question.

## Call B: no voice provider answers

Use a second phone for this call if you can. The fallback transfers to `OWNER_PHONE`, and your own
phone cannot ring while it is the caller.

1. In `.env`, make the two keys wrong and take Nova away:

   ```
   OPENAI_API_KEY=wrong
   GOOGLE_API_KEY=wrong
   NOVA_AWS_ACCESS_KEY_ID=
   NOVA_AWS_SECRET_ACCESS_KEY=
   PHONIC_API_KEY=ph_wrong
   ```

2. Restart `uv run serve`.
3. Call the number.

You must hear, in Telnyx's built-in voice, within about 2 seconds of the answer:

> "Sorry, the AI agents on this line have a technical problem. I am connecting you to <owner> now."

Then:

- Inside the transfer hours (`TRANSFER_HOURS`, `TRANSFER_DAYS`, US Eastern): `OWNER_PHONE` rings.
  If nobody takes the call, the caller hears "Sorry, <owner> could not take the call. Please call
  again later. Goodbye."
- Outside the transfer hours: the voice asks the caller to call again, and the call ends.

Pass: the caller never sits in silence, and the log has
`no voice provider is answering (...); using the spoken fallback`.

4. Put the settings back and restart:

   ```sh
   cp .env.backup .env
   uv run serve
   ```

## Call C: a call that is turned away

`OWNER_PHONE` has no per-number limit, so this test uses the daily minutes cap. Calls A and B
already used more than one minute today.

1. In `.env`, set `DAILY_MINUTES_CAP=1`.
2. Restart `uv run serve`.
3. Call the number.

You must hear one sentence in Telnyx's built-in voice, then the line closes:

> "This demo line has reached its limit for today. Please call again tomorrow. Goodbye."

Pass:

- No AI agent speaks.
- The server hangs up by itself when the sentence ends.
- The log has `call <id> from <number> turned away: daily minutes cap`.
- LangSmith has no new thread: no AI session opened.

4. Put the settings back and restart:

   ```sh
   cp .env.backup .env
   uv run serve
   ```

## After the tests

- Check that `.env` is the same as `.env.backup`, then delete `.env.backup`. It holds your keys.
- Note what failed, with the time of the call. The log and the LangSmith thread for that time show
  the cause.
