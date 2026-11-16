# AI business card

A phone number that works as an interactive audio portfolio. A caller is greeted by an AI
voice agent, routed between specialist agents that run on different voice providers, and
can be transferred to a human.

Stack: Telnyx (phone), Pipecat (audio), LangGraph + LangChain (routing, state, tools).

## Status

M1 to M5 are done: the five agents run as a text chat and on a phone call.
On the phone, each agent has its own voice. Two agents run on Google Gemini Live and three on OpenAI
Realtime; a handoff changes the voice and the provider in under a second. The tools (save a lead,
call stats, transfer to the owner) run from the voice model's requests. The line recognises a
returning caller, looks up a new caller's carrier and area, and limits calls, call length, and
transfer hours. When a voice provider stops answering, the call moves to the other provider.

The tour agent can tell the caller how fast the agents answered and what the call has cost so far.

Not built yet: the local pipeline leg (M7). Its timing spike is done; see "Pipeline leg spike (M7)".

| Agent | Job |
|---|---|
| `classifier` | Greets, asks why the caller called, routes |
| `profile` | Answers questions about the owner from `profile.md` |
| `tour` | Explains how this system works, with live facts from the call |
| `consultant` | Asks about the caller's business and suggests uses |
| `contact` | Connects the caller to the owner, or takes a message: a name and one line on what they want |

## Run

```sh
uv sync
cp .env.example .env        # set LLM_MODEL, the matching API key, OWNER_NAME, OWNER_PHONE
$EDITOR profile.md          # replace the placeholder facts with yours
uv run chat                 # text conversation with the agents
uv run pytest               # routing and tool tests; no API key needed
```

`LLM_MODEL` takes `<provider>:<model>`, for example `anthropic:claude-opus-5-5`,
`openai:gpt-5.5`, or `google_genai:gemini-3.8-pro`.

## Telnyx setup (for the phone milestones)

```sh
bash scripts/telnyx-setup.sh
```

The wizard opens each Telnyx portal page, says what to copy, and writes `TELNYX_API_KEY`,
`TELNYX_PUBLIC_KEY`, `TELNYX_PHONE_NUMBER`, and `TELNYX_APP_ID` into `.env`. It creates the
Voice API application and connects the number by API. Re-running keeps saved values.

## Run the phone server (M2)

```sh
sudo tailscale up                 # once per boot
tailscale funnel --bg 8000        # once; publishes https://<node>.<tailnet>.ts.net -> localhost:8000
uv run serve                      # FastAPI on 127.0.0.1:8000
```

At startup the server sets the Telnyx Voice API application's webhook URL to `PUBLIC_URL/webhooks/telnyx`.
Telnyx sends call events there; the server answers each call with bidirectional PCMU media streaming
to `wss://.../media`, where Pipecat bridges the audio to the active agent's voice leg.

### How a voice call runs (M4)

Every agent gets its own OpenAI Realtime session, opened when the call starts, behind Pipecat's
`LLMSwitcher`. OpenAI fixes a session's voice once it has spoken, so a voice change needs a new
session; pre-opening them turns a 3-4 s reconnect gap into a switch under 1 s. A handoff tool call
settles in the shared context, the switcher activates the target agent's session, and that session
replays the context and answers. The session the caller left is reset in the background, so a
handoff back to it is clean.

### Live transfer to the owner

A transfer is an outbound call from the Telnyx application to `OWNER_PHONE`. Telnyx refuses it
(error 10010) until the application has an outbound voice profile. Run this once:

```bash
uv run setup-transfer
```

It creates a profile named `ai-business-card` (US destinations only, 2 calls at a time, 5 USD per day)
and assigns it to the application. Without a profile the server logs `LIVE TRANSFER IS OFF` at start,
and the contact agent takes a message and offers no transfer.

How a transfer runs (`telephony/transfers.py`):

1. A caller who asks for the owner gives only a name. `request_owner_transfer` saves the name and the
   reason as a lead, so a call that is not answered still leaves a record.
2. A gate waits for the agent's goodbye line to finish. Then the media stream stops and the transfer
   starts. Telnyx refuses a transfer while streaming is on (error 90045).
3. The owner's phone rings for `TRANSFER_RING_SECS` (25). If the owner picks up, the two are connected.
4. If the owner does not pick up, or Telnyx refuses the transfer, the media stream starts again. The
   contact agent comes back with the whole conversation and says the owner will call back.
5. Telnyx's built-in voice speaks only when the stream cannot start again.

### One loudness for every voice

Voices differ in level. Measured on the phone side, Gemini's Charon was 7 dB louder than OpenAI's marin
and 10 dB louder than coral. `voice/leveler.py` sits after the voice services and scales each voice
toward `VOICE_LEVEL_DBFS` (-22). It starts from the measured levels in `VOICE_LEVELS_DBFS`
(`agents/specs.py`) and keeps measuring during the call. With it, the five default voices measured
within 1 dB of each other. A new voice needs a line in `VOICE_LEVELS_DBFS`; a test fails without it.

Turn detection is a setting. `OPENAI_TURN_DETECTION=server` with `OPENAI_VAD_SILENCE_MS=400` is the
fastest; `semantic` lets the model judge when the caller is done, which costs time.

### Two voice providers on one call (M3)

`DEFAULT_LEGS` in `agents/specs.py` sets the voice leg of each agent: `profile` and `consultant` run on
Gemini Live, the others on OpenAI Realtime. Without `GOOGLE_API_KEY` every agent runs on OpenAI.
All services share one conversation context, so the new agent knows what the caller said to the old one.

What differs between the providers is in `voice/legs.py`:

- Gemini fixes a session's tools when the session opens, so each Gemini agent gets its tools at build time.
- Gemini answers every tool result. A handoff result is therefore not sent to the old Gemini session.
- A Gemini session resets in about 0.4 s; an OpenAI session in about 2.4 s.
- Gemini reconnects by itself after an error. The `ServiceWatchdog` reconnects OpenAI sessions only.

Measured with caller audio at 8 kHz: the new agent's first audio came 0.63 to 0.98 s after a switch.

### Caller memory and number lookup

- At hangup a text model (`SUMMARY_MODEL`) writes a summary of at most 40 words. It is stored with the
  caller's number in the `calls` table. No audio is stored.
- On the next call from that number, the classifier gets one thing: the first name the earlier caller gave.
  It asks only "Am I speaking with <name>?". A phone number can be shared, so nothing else is said yet.
- After a yes, the classifier calls `confirm_returning_caller`. Only then do the company, the saved
  interest, and the last three summaries enter the conversation. After a no, they never do, and no agent
  will say who called before.
- A confirmed caller does not give their details again: `save_lead` reuses the email address on file.
  The email address and the phone number never enter a prompt or a tool result.
- `CALLER_LOOKUP` asks Telnyx Number Lookup for a new caller's carrier, line type, area, and registered
  name. Telnyx bills each lookup; the result is cached, so one number is looked up once. The agents are
  told not to open the call with it and not to treat the name as confirmed.

### Guardrails (M5)

Each limit is a setting in `.env`; `0` turns a limit off.

| Setting | Default | Effect |
|---|---|---|
| `MAX_CALL_MINUTES` | 10 | The agent warns the caller one minute before. The server ends the call at the limit. |
| `MAX_CALLS_PER_NUMBER_PER_DAY` | 5 | Calls from one number in 24 hours. `OWNER_PHONE` is exempt. Hidden numbers share one allowance. |
| `MAX_CONCURRENT_CALLS` | 3 | Calls in progress at one time. |
| `DAILY_MINUTES_CAP` | 120 | Call minutes in 24 hours, all callers together. This bounds the daily model cost. |
| `TRANSFER_HOURS`, `TRANSFER_DAYS` | `09:00-18:00`, `mon-fri` | US Eastern. Outside these hours the agent takes a message and makes no transfer. |

A call over a limit is answered with no media stream, so no AI session opens. Telnyx speaks one
sentence with the reason, and the server hangs up when the speech ends. The server never records a call.

### When a voice provider fails (M5)

A caller must not sit in silence. The server treats a voice service as dead in these cases:

- The provider refuses the account: a bad API key, or no credit.
- A reconnect fails, or the error comes back within 10 seconds of a reconnect, or two reconnects were not enough.
- Pipecat gives up on the service after repeated connection failures.
- The agent is asked to speak (greeting, handoff) and no audio starts within 8 seconds.

What happens then:

1. The provider is marked down for 5 minutes (`voice/health.py`). New calls build every agent on the other provider.
2. If the dead service is the active agent, the server restarts the call's media stream. The new AI
   session builds the same agent on the other provider, with the whole conversation. The agent says:
   "Sorry about that pause. I'm the profile agent, now running on OpenAI."
3. If a handoff reaches an agent whose provider is down, the same restart happens. The caller hears a normal handoff.
4. If no provider is left, or the call has moved twice already, Telnyx's built-in voice apologises.
   Inside the transfer hours it then transfers the caller to `OWNER_PHONE`. Outside them, or with
   `TRANSFER_ON_FAILURE=false`, it asks the caller to call again, and hangs up.

Measured with a simulated Telnyx stream on 2026-11-16: a bad OpenAI key moved the greeting to Gemini
in 0.3 s plus the stream restart; with both keys bad, the built-in voice spoke 1.4 s after the call connected.

### Live figures for the tour agent (M6)

`get_call_stats` gives the tour agent measured figures for the call in progress (`voice/stats.py`).
The server writes the same figures to the log when the call ends.

| Figure | How it is measured |
|---|---|
| Answer time | From the last 20 ms of caller audio that held speech to the first agent audio, on the phone side. The same for both providers. Median and slowest, and the median for each provider. |
| Handoff time | From the handoff tool call to the switch to the next agent's voice service. |
| Tokens | What each provider reports for each answer: text and audio, in and out, cached and not. |
| Cost | Tokens times the list prices in `PRICES`. Voice models only: not Telnyx minutes, the number lookup, or the summary model. |

The prices were read on 2026-11-16. The OpenAI prices are from OpenAI's model page. For
`gemini-3.8-live` the sources disagree on the text prices, so the table holds the higher ones and
the estimate is on the high side. Check `PRICES` before you quote a cost to anyone.

Measured on one simulated call of 129 s with three agents: median answer time 1.3 s, handoffs
1.9 s on average, 11,030 tokens in and 2,551 out, about 4 cents.

### Check the lead email

After you fill in `OWNER_EMAIL` and the `SMTP_*` settings, send one sample lead email:

```
uv run test-email
```

It uses the same code path as a real lead and writes nothing to the lead table. On a failure it
prints the server's answer and what to check.

### Pipeline leg spike (M7)

The pipeline leg is planned as a local speech-to-text model, Claude through LangGraph, and local
Kokoro text-to-speech. `scripts/spike_pipeline_leg.py` times each step on the CPU with phone-quality
audio. The leg itself is not built. Run the spike:

```
uv run --with kokoro-onnx --with moonshine-voice --with faster-whisper python scripts/spike_pipeline_leg.py
```

Measured on 2026-11-16 on a Ryzen 7 7840HS, no GPU. The caller audio was synthetic and clean, so
the word errors are a best case.

| Step | Time |
|---|---|
| Moonshine base, speech-to-text, 1.8 s to 7.4 s of speech | 58 ms to 247 ms, 0 word errors in 45 |
| Moonshine tiny | 41 ms to 191 ms, 0 word errors |
| Whisper base.en (int8) | 281 ms to 389 ms, 0 word errors |
| Whisper small.en (int8) | 927 ms to 1087 ms |
| Claude, first full sentence (median of 3) | haiku-5-5 1251 ms; opus-5-5 1660 ms; sonnet-5-5 2664 ms |
| Kokoro, first audio | 171 ms for 1 word; 827 ms for 11 words; 931 ms for 18 words |

What this means:

- Kokoro makes a whole sentence before it gives any audio. A short first clause starts in about 0.2 s; a full sentence takes about 0.8 s.
- Speech-to-text plus first audio stays under the 500 ms target only when the first clause is short.
- Claude's first sentence takes 1.3 s or more. With the 0.4 s wait for the end of the caller's turn, the
  pipeline leg would answer about 2.0 s to 2.7 s after the caller stops. The speech-to-speech legs measured 1.2 s to 1.4 s.

### Text messages (off by default)

With `SMS_ENABLED=true` and `PORTFOLIO_URL` set, an agent can text that one link to the calling number,
once per call, and the owner gets a text just before a live transfer. US carriers block texts from an
unregistered number: the Telnyx number needs a messaging profile and a registered 10DLC campaign first.

This machine runs NixOS. `uv.toml` pins `python-preference = "only-managed"`, because prebuilt wheels
(numpy, onnxruntime) load only under a Python that runs through nix-ld.

## Layout

```
profile.md                       the only source for claims about the owner
src/ai_call_center/
  config.py                      settings from .env
  agents/specs.py                one AgentSpec per agent: instructions, tools, handoffs, voice leg
  graph/state.py                 CallState (checkpointed) and CallContext (per call)
  graph/handoffs.py              transfer_to_<agent> tools
  graph/tools.py                 save_lead, request_owner_transfer, confirm_returning_caller, get_call_stats, text_portfolio_link
  graph/build.py                 the agent loop; middleware swaps prompt and tools per active agent
  leads.py                       SQLite store: leads, the call log, cached number lookups
  callers.py                     returning-caller memory: first-name question, history after a yes, summary at hangup
  guardrails.py                  call limits at admission and the transfer hours
  notify.py                      lead emails to the owner over SMTP
  test_email.py                  `uv run test-email`: one sample lead email, to check the SMTP settings
  telephony/                     Telnyx client, webhooks, media WebSocket; a recording fake for tests
  telephony/lookup.py            Telnyx Number Lookup for a new caller, cached
  telephony/transfers.py         live transfer: call the owner, bring the contact agent back on no answer
  telephony/outbound.py          one-time setup of the outbound voice profile (uv run setup-transfer)
  voice/leveler.py               one loudness for every agent voice
  voice/health.py                which voice providers are answering; a failed one is avoided for 5 minutes
  voice/stats.py                 live figures for a call: answer time, handoff time, tokens, cost estimate
  voice/legs.py                  builds one speech-to-speech service per agent (OpenAI or Gemini) and drives it
  voice/tools.py                 the LangGraph tools as Pipecat function schemas
  voice/call_session.py          one phone call: pipeline, switcher, tool handlers, transfer gate, time limit
  app.py                         FastAPI server (uv run serve)
  chat.py                        terminal simulator
tests/                           scripted fake model, no network
```

## How routing works

Every caller turn is one graph run on the same `thread_id`. The checkpointed state holds
`active_agent`. Middleware reads it and gives the model that agent's prompt and tools. A
handoff tool changes `active_agent`; the loop calls the model again as the new agent. The
next turn goes straight to that agent.
