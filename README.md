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
returning caller by the name they give, looks up a new caller's carrier and area, and limits calls, call length, and
transfer hours. When a voice provider stops answering, the call moves to the other provider.

The tour agent can tell the caller how fast the agents answered and what the call has cost so far.
With LangSmith set up, every call is traced; see "Tracing with LangSmith (M6)".

Dropped: the local pipeline leg (M7). Its timing spike showed it would answer about 1 s slower; see "Pipeline leg spike (M7)".

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

### Handoff rules in voice mode

- **The agent that gets the call answers the caller.** A handoff is refused while the caller has not spoken to the new agent. Without this rule, two agents passed one caller back and forth.
- **An agent cannot hand the call to itself.**
- **The caller's audio is silence until the greeting starts.** A click on the line at the start of a call was answered with a second greeting.
- **A session gets only the tool results of its own tool calls.** When the call returns to an OpenAI agent, the results that other agents wrote are marked as known, and the audio item of its old session is forgotten. OpenAI rejects both, and the session then restarts.
- **A tool result goes to Amazon Nova as a JSON object.** Bedrock rejects a plain sentence.
- **The next agent reads the caller's last words.** Pipecat can commit a caller's turn more than 1.5 s after a Nova agent calls a handoff tool. The handoff waits 0.5 s, then adds the words to the shared context itself.
- **The next agent reads the call so far.** All agents share one conversation, and a handoff replays it. An OpenAI Realtime session gets it from Pipecat as one JSON text, called "a previously saved conversation", and a small model uses that badly: the contact agent asked for a project that the caller had described to the consultant agent. So the handoff message repeats the last 16 turns word for word (`call_so_far`).
- **The next agent reads the caller's name.** When the caller gave or confirmed a name, the handoff message says it, and says never to ask for it again.

### Noise on the line

OpenAI's voice detection decides when a caller's turn starts, and a start cuts the agent off. No detection mode tells a cough from a word: server detection at `OPENAI_VAD_THRESHOLD=0.8`, semantic detection (`OPENAI_TURN_DETECTION=semantic`) and a local Silero model all took recorded coughs for speech. Semantic detection also answered slower on real calls (a median of 1.4 s against 0.7 s). The default is server detection at 0.8.

So the server handles the recovery (`voice/cutoff.py`):

- When a caller's sound starts, the server puts a note into the active agent's session. The note says what to do if the turn has no clear words.
- A sound that cut the agent off: the agent says the cut sentence again and goes on.
- A sound in silence: the agent asks its last question again.
- A turn with clear words: the agent forgets the note and answers the words.
- A note lives for one answer. The server deletes it when the agent next starts to speak.

Limits: the notes work for agents on OpenAI Realtime only; Gemini Live and Nova Sonic do not report the start of a caller's turn. The results come from simulated calls with recorded coughs, not from a tuned production system.

`SAVE_CALLER_AUDIO=true` writes the caller's side of each call to `data/caller_audio/` as a WAV file, so you can hear what the detector heard. It holds callers' voices: it is off by default and the folder is git-ignored.

### A tool name said aloud

A speech-to-speech model sometimes speaks a tool call as text. On a real call a Nova Sonic agent said "call transfer to underscore underscore contact", made no tool call, and went silent. The server reads each agent's text (`voice/spoken_tool.py`). If an agent says the name of a handoff tool and no tool call follows in 0.4 s, the server makes the handoff itself.

### The same words two times in one answer

An OpenAI Realtime answer can hold more than one audio item, and sometimes the second item is a copy of the first. On a real call the caller heard the full greeting two times, 18 seconds in all. The audio of an answer arrives much faster than it plays, so the server holds each later item until its first 24 letters are known (`voice/repeat.py`). A copy is dropped; an item with new words plays in order. An item that gets no text within about 3 seconds of audio plays, so the guard can never make an agent silent.

### The greeting

The first words of a call are exact (`greeting_words` in `agents/specs.py`); the first message of the call gives them to the greeting agent word for word.

- `OWNER_NAME_PRONUNCIATION` tells every agent how to say the owner's name, in plain words. It is a hint in the prompt: the voice models take no phonetic markup.
- Every caller hears the same first words. With `BRAND_NAME` set: "Hi, you've reached <brand>. I'm one of five
  AI agents <owner> built, and this call is a live sample of the work <first name> does for clients. I remember
  callers, so first: who am I speaking with?" Without a brand, the first sentence is "Hi, you've reached
  <owner>'s AI business card." The words say nothing about an earlier call, so they leak nothing.
- The caller gives a name. The classifier says "Thanks, <name>." and calls `identify_caller`. Code checks the
  name (see "Caller memory" below), and the tool result gives the next words exactly.
- A new caller, or a name with no match: "Nice to meet you, <name>. <first name> is a software engineer and
  architect, available for contract work or a full-time role. <first name>'s work covers new platforms and APIs,
  cloud systems, modernizing old code, and AI agents like the ones on this call. Right now you're talking to a
  graph of AI agents on models from <n> companies, with caller memory and RAG search over <first name>'s writing.
  Interrupt me any time. Try asking me <tip>. So, do you have a project or a role in mind, or would you like
  to look around first?" The company count comes from the legs that the call runs on (`companies_phrase`).
- A name that matches an earlier caller on the number: "Welcome back, <name>. Last time you asked about <topic>.
  Want to pick up there, or start something new?" Without a stored topic: "Welcome back, <name>. Good to hear
  from you again. What can I do for you today?"
- The tip is picked at random per call, and each one starts a real feature: "what this call has cost so far",
  "which AI model you're talking to right now", and "to forget you". With the RAG search on, also
  "what <first name> thinks about microservices" and "... about technical debt". Edit `CALL_TIPS` and `WRITING_TIPS`.
- The consultant agent covers software engineering in general (products, platforms, APIs, cloud, architecture,
  modernizing old systems) as well as AI, and it says so when plain software is the better tool.

### Live transfer to the owner

A transfer is an outbound call from the Telnyx application to `OWNER_PHONE`. Telnyx refuses it
(error 10010) until the application has an outbound voice profile. Run this once:

```bash
uv run setup-transfer
```

It creates a profile named `ai-business-card` (US destinations only, 2 calls at a time, 5 USD per day)
and assigns it to the application. Without a profile the server logs `LIVE TRANSFER IS OFF` at start,
and the contact agent takes a message and offers no transfer.

A message is the normal case, and a transfer needs the caller's own words. "Leave a message with
the owner", "email the owner", and "have the owner call me" are messages: the contact agent takes the
message with `save_lead`, and a saved message never leads to a transfer. Before a transfer, the server
reads the caller's last four turns in the transcript. If none of them asks to speak, talk, or be
connected (`LIVE_REQUEST` in `graph/tools.py`), no transfer is made and nothing is saved. The agent then
asks one question: a message, or a live connection. The answer decides.

How a transfer runs (`telephony/transfers.py`):

1. A caller who asks to speak to the owner gives only a name. `request_owner_transfer` saves the name and the
   reason as a lead, so a call that is not answered still leaves a record.
   This happens only when the owner's phone will ring. A transfer that is refused (outside
   `TRANSFER_HOURS`, or not possible on this server) sends no email at that moment: the agent offers
   to take a message, and that message is the one email. If the caller hangs up without a message,
   the "asked to speak to you" email goes out when the call ends. A message that the caller leaves
   after a transfer nobody answered is a new record with its own email.
2. A gate waits for the agent's goodbye line to finish. Then the media stream stops and the transfer
   starts. Telnyx refuses a transfer while streaming is on (error 90045).
3. The owner's phone rings for `TRANSFER_RING_SECS` (25). If the owner picks up, the two are connected.
4. If the owner does not pick up, or Telnyx refuses the transfer, the media stream starts again. The
   contact agent comes back with the whole conversation and says the owner will call back.
5. Telnyx's built-in voice speaks only when the stream cannot start again.

### Call screening: the owner accepts with a key

`TRANSFER_SCREENING=true` is the default. The server then does not use the Telnyx transfer command.

1. The agent says "I am calling Brad now. Please hold." The server stops the media stream.
2. The server plays a ring tone to the caller, in a loop. It calls the owner's phone as a separate call.
3. The owner answers. A Telnyx voice says: "Voice card call from Dana Reyes. <the reason>. Press 1 to accept.
   Press 2 to decline." The caller still hears the ring tone.
4. On key 1, the server stops the ring tone and connects the two calls.
5. On key 2, or no key, the server hangs up the owner's call. The contact agent comes back to the caller.

The caller never hears the owner's phone, so the caller never hears a voicemail greeting. A voicemail cannot
press 1. The question is asked twice, and each time the owner has `TRANSFER_ACCEPT_SECS` (6) seconds for a key.
With `TRANSFER_VOICEMAIL_DETECTION=true`, a detected machine ends the wait a few seconds after the answer.

- The ring tone is made in code (`telephony/hold_audio.py`) and served at `/audio/ringback.wav`.
- `TRANSFER_HOLD_AUDIO_URL` replaces it with any WAV or MP3 file, for example hold music you have a licence for.
- Screening needs `TELNYX_APP_ID` and `PUBLIC_URL`. Without the app id, the server uses the plain transfer below.

### When the owner's voicemail answers (without call screening)

A voicemail answers a transfer like a person does. Without a check, the caller lands in the owner's voicemail.

- `TRANSFER_VOICEMAIL_DETECTION=true` (the default) adds Telnyx answering machine detection to the transfer.
- Telnyx listens to the first seconds after the answer and sends `call.machine.detection.ended`.
- On the result `machine`, the server hangs up the owner's leg. The contact agent comes back to the caller, as
  for a phone that rang out. The caller can hear the first seconds of the voicemail greeting.
- On `human` or `not_sure`, the call goes on. Answer with a short "Hello": a long greeting can sound like a machine.
- The same option parks the caller's leg when the owner's leg ends. So after a normal conversation, the
  server hangs up the caller's leg when the owner hangs up.

### The agent hangs up after a goodbye

Every agent has the `end_call` tool. Without it, a call stayed open after the goodbye until the caller hung up.

- The agent calls `end_call` when the caller says goodbye, or says they need nothing else.
- The server checks the caller's own words first. It reads the caller's last two turns for a goodbye
  ("bye", "that's all", "no thanks", "I'm good", "I have to go"). If none is there, the call stays open and
  the agent is told to go on. The log line is `end_call held back`.
- If the check passes, the agent says one goodbye line. The server hangs up when the agent stops speaking,
  through the same gate that a transfer uses. If the agent says nothing, the server hangs up after 4 seconds.
- The call log and the summary are written as for any other call.

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

### A third provider: Amazon Nova Sonic

An agent can run on `amazon.nova-2-5-sonic`, on Amazon Bedrock. It is off until the server has AWS keys.

1. Create an IAM user in your own AWS account with one permission: `bedrock:InvokeModel` on
   `arn:aws:bedrock:*::foundation-model/amazon.nova-*`. The two-way stream is authorized under that action.
2. Put its key in `.env` as `NOVA_AWS_ACCESS_KEY_ID` and `NOVA_AWS_SECRET_ACCESS_KEY`. The names start with
   `NOVA_` on purpose: `AWS_` variables in a shell can belong to another account.
3. Run `uv run check-nova`. It opens one short session and prints what Nova said, or the AWS error.
4. Move an agent: `AGENT_LEGS=consultant=nova_sonic`. Restart the server.

How this leg differs from the other two:

- It connects when a handoff reaches its agent, and disconnects when the agent is left. AWS closes a stream
  that carries no audio, so a stream opened at call start would be dead when it is needed.
- Without the keys, or when Nova is marked down, the agent runs on OpenAI or Gemini. Nova is the last fallback.
- Nova has few English voices (`tiffany`, `matthew`, `amy`), so agents share voices when several run on Nova.
- Bedrock answers "Invalid input request" when the request to speak arrives in the same instant as a long
  conversation history. The leg waits `NOVA_SETTLE_SECS` (0.3 s) between the two.
- Nova answers only while microphone audio arrives. `check-nova` sends silence for that reason.
- Status: `check-nova` passes, and simulated calls ran a handoff into Nova, an answer, and a handoff out.
  It has not carried a real phone call yet.

### Caller memory and number lookup

- At hangup a text model (`SUMMARY_MODEL`) writes a summary of at most 40 words. It is stored with the
  caller's number in the `calls` table. No audio is stored.
- The summary model also writes a `topic` of at most 8 words ("a new booking platform"), with no names
  or numbers, and an `interest`: `project`, `role`, or `none`. Code cuts the topic to 8 words.
- No agent knows if a number called before. Every caller is asked who they are, and the name goes to
  `identify_caller`. Code, never a model, compares it with each name on record for the number
  (`names.same_person`): the same first name after cleanup, a short or long form of it (Bob and Robert),
  or the same sound code and first letter (Steven and Stephen). With any doubt, the caller counts as new.
- Only on a match do the company, the saved interest, and the last three summaries enter the conversation.
  With no match, they never do, and no agent will say who called before.
- After `identify_caller`, who the caller is is settled for the whole call. No agent asks for the name again.
- The history holds one person's calls and leads only. What Steve left on a number is not released to
  Bill on the same number, and it is not reused for him.
- "Forget me": after one yes, `forget_caller` deletes that person's leads and blanks the summary, name,
  and topic of their calls. The call rows stay, because the per-number call limits count them. The call in
  progress gets no summary, no transcript in the log, no warm-lead email, and its caller audio file is deleted.
- Warm lead: when the summary says `project` or `role`, no message was saved, the caller was not forgotten,
  and the number is not hidden, the owner gets an email "Warm lead: <name> talked about a project, no message left".
- The summary model also writes `owner_note` (at most 120 words: every specific the caller gave about the
  work), `timeline`, and `size`. The end-of-call emails use it: the warm lead, and the note for a refused
  transfer with no message. So the owner gets the details even when no agent saved them. The summary runs
  before those emails for this reason.
- A missed save: when the caller says yes to the message read-back and the agent answers without calling
  `save_lead`, the server tells the agent "Call save_lead now" (`missed_save`), once per yes.
- The consultant asks two qualifying questions: "When would you want this to start?" and "Is this a few weeks
  of work, or a longer engagement?". `save_lead` stores the answers as `timeline` and `size`, and the lead email
  shows them first. Existing databases get the new columns at start.
- A confirmed caller does not give their details again: `save_lead` reuses the email address that the
  same person left. The email address and the phone number never enter a prompt or a tool result.
- `CALLER_LOOKUP` asks Telnyx Number Lookup for a new caller's carrier, line type, area, and registered
  name. Telnyx bills each lookup; the result is cached, so one number is looked up once. The agents are
  told not to open the call with it. The registered name goes to the lead email only, never to a prompt.

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

### Tracing with LangSmith (M6)

Off unless `.env` sets `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`. `LANGSMITH_PROJECT` names
the project (default `default`). The code is in `tracing.py`.

Setup:

1. Make a free account at smith.langchain.com, and an API key under Settings.
2. Put `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...` and `LANGSMITH_PROJECT=voice-business-card` in `.env`.
3. Restart the server. The log says `LangSmith tracing is on: project voice-business-card`.

Each call is one LangSmith thread, keyed by the Telnyx call id. The thread holds:

| Run | From |
|---|---|
| One trace per media stream: each turn, the caller's words, the agent's words, each tool call, the timing | Pipecat's OpenTelemetry spans, through LangSmith's Pipecat integration |
| A second trace after a stream restart (failover, or a transfer nobody answered) | The same, with `call.restart_reason` set |
| `call-summary`: the summary written at hangup | LangChain's LangSmith client |

`uv run chat` traces too: a `chat-turn` run per caller turn, and the `call-summary`, in one thread
named `chat-<id>`.

Limits:

- Pipecat 1.16 has no trace spans for Amazon Nova Sonic. An agent on Nova shows turns and timing only,
  not its words or tool calls.
- No audio is sent. LangSmith can attach a recording of each call, but this project stores no call
  audio.
- Call text is sent in full: names, numbers and what the caller said.
- Spans are sent from a background thread, so tracing does not delay an answer. When LangSmith
  refuses them (a wrong key, for example), the log shows one error and the call goes on.
- The free Developer plan keeps up to 5,000 traces a month for 14 days.

Tests never send traces: `tests/conftest.py` sets `LANGSMITH_TRACING=false` before the project loads.

### A message is saved only after the caller's yes

A save emails the owner at once, so the voice session checks the caller's own words first
(`message_confirm_question` in `graph/tools.py`). The agent must have asked if the message is right,
and the caller's answer must be a short yes. If not, the first `save_lead` call is held back and the
agent says the message back. The check holds a save back once only. If the caller hangs up before a
yes, the held message is saved when the call ends, with the note "The call ended before the caller
confirmed this message."

### The lead email

When a caller confirms a message, or asks to speak to the owner, the owner gets one email
(`notify.py`). It has an HTML part and a plain-text part with the same content:

- The subject says who and what: "Message from <name> (<company>)" or "<name> asked to speak to you".
- The caller's message comes first, with the details the caller gave.
- "Reach them": the number to call back as a link, and the email address and company if given.
- "About the call": the time, the call length, the agents the caller spoke with, what else they asked,
  the number of earlier calls, and the carrier, line type, and area from the number lookup.
- A field with nothing in it is left out. Every value is escaped, because it comes from a caller's speech.

A free Zoho Mail account with its own domain sends through `smtp.zoho.com`, port 465.
`smtppro.zoho.com` is for paid plans and answers a free account with `Access Restricted`.

After you fill in `OWNER_EMAIL` and the `SMTP_*` settings, send one sample lead email:

```
uv run test-email
```

It uses the same code path as a real lead and writes nothing to the lead table. On a failure it
prints the server's answer and what to check.

### Pipeline leg spike (M7)

The pipeline leg (dropped on 2026-11-18) was planned as a local speech-to-text model, Claude through LangGraph, and local
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

### Website (off by default)

With `WEBSITE_URL` set, every agent knows the owner's website, says it when a caller asks, and spells it
on request. The goodbye line names it too. A voice says the address in words: a part of the host name that
equals `BRAND_NAME` without spaces is said as the brand's words, so `digitalreformer.dev` with
`BRAND_NAME="Digital Reformer"` is said "digital reformer dot dev".

### Published writing: a RAG tool (off by default)

With `WRITING_FEED_URL` set to an RSS feed, the profile agent answers questions about the owner's views
with `search_writing`, a RAG search. It says so aloud first ("Let me run a quick RAG search..."),
because the line is a showpiece. The tour agent can explain how the search works.

- Each post is cut into passages of about 900 characters, one section of a post each.
- A local embedding model ranks the passages by meaning: `WRITING_EMBEDDING_MODEL`, by default
  `BAAI/bge-small-en-v1.5` through fastembed (ONNX on the CPU, about 65 MB, about 5 ms a query, no API cost).
- BM25 ranks the passages by keyword. Reciprocal rank fusion merges the two lists.
- A passage under a cosine similarity of 0.55 is dropped, so an off-topic question gets no passage.
- The model reads at most 3 passages (about 400 tokens), never whole posts. The prompt holds only the
  number of posts and the feed's description. So the cost per call stays the same as posts are added.

The server reads the feed once at start, with a 5 s limit, and saves a copy in `data/writing.json`.
When the feed cannot be read, the server uses that copy. Passage vectors are saved in
`data/writing_vectors.npz` under a hash of each passage, so a restart embeds only new or changed passages.
Restart the server to pick up a new post. The first start downloads the model to `data/models/`;
a container image should include that folder. With `WRITING_EMBEDDING_MODEL=` empty, the search is BM25 only.

This machine runs NixOS. `uv.toml` pins `python-preference = "only-managed"`, because prebuilt wheels
(numpy, onnxruntime) load only under a Python that runs through nix-ld.

## Layout

```
profile.md                       the only source for claims about the owner (with the writing feed)
src/ai_call_center/
  config.py                      settings from .env
  agents/specs.py                one AgentSpec per agent: instructions, tools, handoffs, voice leg
  graph/state.py                 CallState (checkpointed) and CallContext (per call)
  graph/handoffs.py              transfer_to_<agent> tools
  graph/tools.py                 save_lead, request_owner_transfer, identify_caller, forget_caller, get_call_stats, text_portfolio_link, end_call
  graph/build.py                 the agent loop; middleware swaps prompt and tools per active agent
  leads.py                       SQLite store: leads, the call log, cached number lookups
  callers.py                     returning-caller memory: first-name question, history after a yes, summary at hangup
  guardrails.py                  call limits at admission and the transfer hours
  notify.py                      lead emails to the owner over SMTP
  writing.py                     RAG over the owner's posts: RSS feed, passages, hybrid search, saved vectors
  names.py                       same_person: does a spoken first name match a name on record (spelling, short form, sound)
  test_email.py                  `uv run test-email`: one sample lead email, to check the SMTP settings
  check_nova.py                  `uv run check-nova`: one short Nova Sonic session, to check the AWS keys
  telephony/                     Telnyx client, webhooks, media WebSocket; a recording fake for tests
  telephony/lookup.py            Telnyx Number Lookup for a new caller, cached
  telephony/transfers.py         live transfer: call the owner, ask for a key, bring the contact agent back on no answer
  telephony/hold_audio.py        the ring tone a caller hears while the owner's phone rings
  telephony/outbound.py          one-time setup of the outbound voice profile (uv run setup-transfer)
  voice/leveler.py               one loudness for every agent voice
  voice/health.py                which voice providers are answering; a failed one is avoided for 5 minutes
  voice/stats.py                 live figures for a call: answer time, handoff time, tokens, cost estimate
  voice/legs.py                  builds one speech-to-speech service per agent (OpenAI, Gemini, or Nova) and drives it
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
