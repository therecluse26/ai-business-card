# AI business card

A phone number that works as an interactive audio portfolio. A caller is greeted by an AI
voice agent, routed between specialist agents that run on different voice providers, and
can be transferred to a human.

Stack: Telnyx (phone), Pipecat (audio), LangGraph + LangChain (routing, state, tools).

## Status

M1 to M4 are done, and the guardrails of M5: the five agents run as a text chat and on a phone call.
On the phone, each agent has its own voice. Two agents run on Google Gemini Live and three on OpenAI
Realtime; a handoff changes the voice and the provider in under a second. The tools (save a lead,
call stats, transfer to the owner) run from the voice model's requests. The line recognises a
returning caller, looks up a new caller's carrier and area, and limits calls, call length, and
transfer hours.

Not built yet: a spoken fallback when every voice provider fails (M5), live cost and latency
figures for the tour agent (M6), and the local pipeline leg (M7).

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
  telephony/                     Telnyx client, webhooks, media WebSocket; a recording fake for tests
  telephony/lookup.py            Telnyx Number Lookup for a new caller, cached
  telephony/transfers.py         live transfer: call the owner, bring the contact agent back on no answer
  telephony/outbound.py          one-time setup of the outbound voice profile (uv run setup-transfer)
  voice/leveler.py               one loudness for every agent voice
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
