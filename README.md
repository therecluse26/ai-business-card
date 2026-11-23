# AI business card

A phone number that works as an interactive audio portfolio. A caller is greeted by an AI
voice agent, routed between specialist agents that run on different voice providers, and
can be transferred to a human.

Stack: Telnyx (phone), Pipecat (audio), LangGraph + LangChain (routing, state, tools).

## Status

M1 to M5 are done: the agents run as a text chat and on a phone call.
On the phone, each agent has its own voice. Two agents run on Google Gemini Live and four on OpenAI
Realtime; a handoff changes the voice and the provider in under a second. The tools (save a lead,
call stats, transfer to the owner) run from the voice model's requests. The line recognises a
returning caller by the name they give, looks up a new caller's carrier and area, and limits calls, call length, and
transfer hours. When a voice provider stops answering, the call moves to the other provider.

The tour agent can tell the caller how fast the agents answered and what the call has cost so far.
A small model on the server hears how the caller sounds; see "The server hears how the caller sounds (Pipecat Voice Reader)".
With LangSmith set up, every call is traced; see "Tracing with LangSmith (M6)".
After hangup a grader scores each call against eleven rules, and the figures and the scores are stored for a public report;
see "A grader scores each call after hangup" and "Call analytics and the public report".
A failed rule starts a job on GitHub that opens a pull request with a fix, with no action from the owner;
see "The improvement loop: a failed rule opens a pull request".

Dropped: the local pipeline leg (M7). Its timing spike showed it would answer about 1 s slower; see "Pipeline leg spike (M7)".

| Agent | Job |
|---|---|
| `classifier` | Greets, asks why the caller called, routes |
| `profile` | Answers questions about the owner from `profile.md` |
| `tour` | Explains how this system works, with live facts from the call |
| `consultant` | Asks about the caller's business and suggests uses |
| `contact` | Connects the caller to the owner, or takes a message: a name and one line on what they want |
| `feedback` | Takes a bug report or a feature idea about the line itself, for a GitHub issue after the call |

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
This check, and the check for an outbound voice profile, run in the background: the server takes calls
about 0.3 s after its imports end, and the first call waits for the checks only if they are still running.
Telnyx sends call events there; the server answers each call with bidirectional PCMU media streaming
to `wss://.../media/<token>`, where Pipecat bridges the audio to the active agent's voice leg.

The token protects the media endpoint. A webhook carries Telnyx's signature, but a media stream carries
none. So the server puts a random one-time token in each stream URL that it gives Telnyx, and the media
endpoint refuses a stream with no valid token (`telephony/stream_tokens.py`). Without this, anyone could
open a paid AI session with no phone call, outside the guardrails.

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
- A sound in silence: the agent says "I didn't catch that. Could you say it again?" It does not ask its last question again: a greeting agent asked it after each "um", and a feedback agent asked "anything else?" again after the caller said no.
- A turn with clear words: the agent forgets the note and answers the words. A short "yes" or "no" is clear words.
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
- `RESPELLINGS` is the stronger tool when a voice still says a name wrong. It holds `<written>=<respelling>` pairs, for example `Lovelace=Luv-layss`. A voice model says a word the way the word is written in the text it reads. So the server puts the respelling in everything a voice model reads: the instructions, the greeting, each tool result. The lead email, the call summary, the log and the text chat keep the written word (`voice/respell.py`).
- Every caller hears the same first words. With `BRAND_NAME` set: "Hi, you've reached <brand>. I'm one of six
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
asks one question: a message, or a live connection. The answer decides. This word list is the fallback.
With a Jev key, a decision model reads the caller's words for their meaning: see the next section.

### A decision model reads the caller's request (Jev)

`LIVE_REQUEST` is a word list, and a word list reads words, not meaning. It fails in two directions:

- "Is he around? I'd love a quick word with him" asks for a live call, and holds none of the words.
- "Have him call me back. I can't talk right now" asks for a message, and holds the word "talk".

With `TYPESAFE_API_KEY` set, Jev decides, and the word list is only the fallback. Jev is the decision model
of TypeSafe AI (`decisions.py`). It writes no text. It answers one typed yes or no question with the
probability of a yes.

1. The server sends Jev the end of the call, from the caller's fourth-last turn on, and one question:
   "Did the caller ask, in their own words, to speak with the owner on this call?"
   The request also says whose business card this is, so Jev knows who "him" is.
2. The probability decides (`LIVE_CALL_YES` and `LIVE_CALL_NO` in `graph/tools.py`):

| Probability | What happens |
|---|---|
| 0.8 or more | The transfer goes ahead. |
| Above 0.3, under 0.8 | No transfer. The agent asks one question: a message, or a live connection. |
| 0.3 or less | No transfer. The agent takes a message. |

3. With no decision, the word list decides, as it does without a key.

- Jev can never stop a call. After `JEV_TIMEOUT_MS` (500), or on any error, there is no decision. The request
  has no retry.
- The yes limit is 0.8, above the 0.7 that TypeSafe gives as a start, because a wrong yes rings the owner's phone.
- A wrong no costs one question: the agent asks, and the caller's answer lets the transfer go ahead.
- The time: every first transfer request waits for Jev's answer, at most `JEV_TIMEOUT_MS`. When the word
  list has no match, the handler already waits 0.8 s for the transcript of the caller's last turn, and the
  server opens the connection to Jev during that wait.
- A caller who only says yes to the agent's offer has not asked in their own words. The question says so.
- A caller who asked to be forgotten is not sent to Jev. The word list decides for that caller.
- The log line is `Jev read the transfer request: 0.93 in 212 ms (jev-1.14.0; word list: miss): the transfer goes ahead`.
  It names the word list's answer too, so the log shows each case where the two differ.
  `get_call_stats` gives the tour agent the probability and the time, so it can say them on the call.
- `uv run check-jev` sends 16 made-up samples and prints each probability and time. Run it after you
  set the key, and after each change to the question.
- Measured on 2026-11-19 with `jev-latest`, 16 samples: each request for a live call got 0.96 or more, and
  each other line got 0.09 or less. No sample came near a limit. A request took 71 to 165 ms on an open
  connection. The first request opens the connection, and took 141 to 377 ms in four runs.
- Context decides the result. With no line that names the owner, "Can I speak to him?" got 0.74, under the
  yes limit. With that line it got 0.97. So `conversation_state` always sends it.
- Not measured: a real call. The samples are short and clean, and a real transcript is not.

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

- At hangup a text model (`SUMMARY_MODEL`) writes a summary of at most 40 words. Code checks it, and a
  summary with a fault goes back to the model one time: see "A grader scores each call after hangup". It is stored with the
  caller's number in the `calls` table. No audio is stored.
- The summary model also writes a `topic` of at most 8 words ("a new booking platform"), with no names
  or numbers, and an `interest`: `project`, `role`, or `none`. Code cuts the topic to 8 words.
- No agent knows if a number called before. Every caller is asked who they are, and the name goes to
  `identify_caller`. Code, never a model, compares it with each name on record for the number
  (`names.same_person`): the same first name after cleanup, a short or long form of it (Bob and Robert),
  or the same sound code and first letter (Steven and Stephen). With any doubt, the caller counts as new.
- A match makes the caller "recognised": the agent says "Welcome back", and gets no notes. The company,
  the saved interest, and the last three summaries enter the conversation only after the passphrase
  check (below). With no match, they never do, and no agent will say who called before.
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
- A verified caller does not give their details again: `save_lead` reuses the email address that the
  same person left. The email address and the phone number never enter a prompt or a tool result.
- `CALLER_LOOKUP` asks Telnyx Number Lookup for a new caller's carrier, line type, area, and registered
  name. Telnyx bills each lookup; the result is cached, so one number is looked up once. The agents are
  told not to open the call with it. The registered name goes to the lead email only, never to a prompt.

### The passphrase check: recognised is not verified

A phone number can be faked, and a name is no secret. Both are on a business card. So a match of
number and name is not proof of who calls. The line has two levels (`passphrase.py`):

| Level | What the caller gave | What the line does |
|---|---|---|
| Recognised | A phone number and a name that match an earlier caller | Says "Welcome back". No agent gets a note of an earlier call. |
| Verified | The passphrase of that earlier caller too | The agents get the notes, and the first agent names the last topic. |

- **The offer.** After a saved message, the contact agent offers one time: "I can remember your details
  for next time, if you give me a short word or phrase to identify you. Make it a made-up one, not a
  password you use anywhere else." The agent says the passphrase back, and keeps it after a yes.
  `CALLER_PASSPHRASE_OFFER=false` turns the offer off.
- **The check.** On the next call the first agent says: "Welcome back, <name>. I believe we have talked
  before. Just to verify, may I have the short passphrase that you gave me?" The answer goes to the tool
  `verify_passphrase`. Code compares it. The voice model gets a yes or a no, and never the stored passphrase.
- **Speech is not exact.** The comparison ignores capital letters, spaces and punctuation, and it uses
  how the words sound (Metaphone). "Purple lanturn" is "purple lantern".
- **Storage.** The table `caller_phrases` holds a salted scrypt hash, not the passphrase.
- **Tries.** Two tries in one call. Then the call goes on with no notes. After 5 wrong tries in a row,
  over any number of calls, the line stops asking: that passphrase is locked for good.
- **A passphrase opens only the calls made with it:** the call where it was set, and each later call
  where the caller said it. A call where the caller was only recognised is opened by no passphrase. So a
  person with a faked number who sets a passphrase gets nothing of the real caller's calls.
- **A new passphrase takes the place of the old one,** and opens only its own calls. So "I forgot it"
  never opens the old notes. The owner still has them.
- **Calls from before this check** have no passphrase. No agent can use their notes.
- **A caller with no passphrase who asks about an earlier call** hears why: a phone number can be faked,
  so the line uses those notes only after a passphrase check.
- **The contact details on file** (`save_lead` with email "on file") are reused only for a verified caller.
- **The lead email** says which level the caller had: "Verified: they said the passphrase of that caller",
  or "Not verified: they gave no passphrase, and a phone number can be faked".
- **"Forget me"** deletes the passphrase row too.

Where the passphrase is, and is not:

- It is in the private records of the call: the transcript in the server log, the traces, and the audio
  that the grader hears. A person who can read those can read the notes too, so the passphrase adds no
  risk there. This is why the offer asks for a made-up phrase and not a password that opens other accounts.
- It is not in the call summary (code changes it to `[passphrase]`), and not in the two public texts:
  a fault report and a GitHub issue. The scrub removes its words.

What the check cannot promise:

- The words "may I have the passphrase" tell a person with a faked number that this name called before
  and set a passphrase.
- A short spoken phrase has few possible values. The limit on tries protects it on the phone. The hash
  does not protect it from a person who has the database file.
- A person with a faked number can use up the tries and lock the real caller's passphrase, or set a new
  one in its place. The notes then stay closed to everyone on the phone, which is the safe result. The
  real caller can set a new passphrase for later calls.

Measured on 2026-11-22: 4 made-up calls were added to `evals/grader_calls.json` (a right passphrase, two
wrong tries, a new passphrase after a message, a question about an earlier call with no passphrase).
No rule fails in them. The decision grader agreed: no false alarm, and "I can't use the notes from your
earlier calls" is not read as a denial of the memory (the set is now 63 calls: 28 of 30 planted failures
found, 1 false alarm of 285). Not measured: a phone call with a voice model.

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

### The server hears how the caller sounds (Pipecat Voice Reader)

Pipecat 1.16 has a feature in alpha, expressive conversation. It has two halves, and one of them fits this server:

- **Not used: tags that a TTS service performs.** The LLM writes a tag such as `<<sympathetic>>`, and the TTS
  service speaks in that way. Each agent here is one speech-to-speech model. It has no TTS service, and it
  chooses its own delivery.
- **Used: a model that hears the caller's voice.** Pipecat Voice Reader is a small model that runs on this
  server's CPU. It reads the sound of a turn, not the words: the tone of voice (positive, neutral, sad,
  frustrated), how activated the voice is, laughter, and the speech rate.

Pipecat's own processor puts a reading at the head of the user's message, for a text LLM to read. A
speech-to-speech model gets the caller's audio and no such message. So `voice/voice_reader.py` runs the same
model in a processor of its own, directly after the transport's input:

1. Silero, a local speech detector, finds each stretch of the caller's speech. It sends no frame, so the
   voice services still decide when a turn starts and ends.
2. A turn is every stretch from the agent's last words to its next words. Each time the caller pauses
   (0.2 s), the model reads the turn, at most its last 8 seconds. A turn under 0.8 s is not read.
3. The read runs in a thread. No audio frame waits for it, so it adds no time to an answer.

What a voice reading does:

- **The log.** One line for each read: `the caller's voice in turn 3: neutral 0.87, arousal 0.49, ...`.
- **The tour agent.** `get_call_stats` gives the counts, the speech rate, and the read time, so the agent
  can say what the model heard on this call.
- **The lead email.** A row "Caller's voice" gives the counts. It says that a model measured them.
- **A note to the agent.** When a turn reads "frustrated" at 0.6 or more (`FRUSTRATED_AT`), the agent that
  holds the call gets one note: show that you understood, then help directly. The note stays for two
  caller turns, then the server takes it out. Only an OpenAI agent takes a note; `VOICE_STEER=false`
  turns the note off.

Rules and limits:

- A reading is a measurement of one turn, not a fact about a person. The texts give counts, not a verdict.
- The reader sends no audio and no reading out of the server, except the row in the owner's lead email.
  The decision grader is another part: it sends the call audio to Google after hangup; see "A grader
  scores each call after hangup". The readings are in memory for the length of the call. A caller who asks to be forgotten loses them, and the reader
  hears no more of that call.
- A phone call is 8 kHz, and the model hears 16 kHz. Each clip is resampled.
- `uv run check-voice` reads saved caller audio (`SAVE_CALLER_AUDIO=true`) with the same parts, and prints
  each turn's reading and read time.
- Measured on 2026-11-19, on 13 saved calls from tests: 88 turns read, read time 45 to 147 ms with a median
  of 58 ms on a laptop. 87 turns read neutral and 1 positive, so no turn would have sent a note. In a real
  Pipecat pipeline the reader passed 2039 audio frames at 0.29 ms a frame.
- Not measured: a real call with the reader on, a caller who is frustrated, the read time on the Fly.io
  CPU, and what the note does to an agent's answer.
- `VOICE_READER=false` turns the reader off. The model is about 28 MB. The image holds it; on a laptop
  the first start downloads it from Hugging Face into `VOICE_MODEL_DIR`.

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
- No audio is sent to LangSmith. LangSmith can attach a recording of each call, but this project writes
  no call audio to a file.
- Call text is sent in full: names, numbers and what the caller said.
- Spans are sent from a background thread, so tracing does not delay an answer. When LangSmith
  refuses them (a wrong key, for example), the log shows one error and the call goes on.
- The free Developer plan keeps up to 5,000 traces a month for 14 days.

Tests never send traces: `tests/conftest.py` sets `LANGSMITH_TRACING=false` before the project loads.

### A grader scores each call after hangup

A grader is a check of a result against a list of rules. On a phone call, a grader cannot check an
answer before the caller hears it: the voice models speak while they think, and a check would add to
the answer time. So this line has two graders, and both run after hangup.

**The summary check, with one more try** (`callers.summarize_call`). The summary is stored, read by the
next call's agents, and emailed, and no caller waits for it. Code reads each answer of the summary model:

- The summary has at most 40 words, and it is not empty.
- No text holds an email address or a phone number. A phone number is a run of 9 digits or more.
- The topic holds no number.

An answer that breaks a rule goes back to the model one time, with the broken rules named. What is
still wrong after the second answer, code removes: it cuts the summary to whole sentences inside 40
words, writes `[removed]` in place of an address or a number, and drops a topic that is not safe to say.

**The call grader, with no retry** (`grading.py`). The call is over, so a score cannot repair it. The
score shows if a change to a prompt or a tool made the calls better.

| Rule | Checked by |
|---|---|
| `tool_name_not_spoken`: no agent said the name of a tool aloud | Code |
| `handoff_intro`: after each handoff, the new agent said which agent it is | Code |
| `summary_first_try`: the call summary passed its check on the first try | Code |
| `name_asked_once`: the agents asked for the caller's name one time at most | A model |
| `nothing_asked_again`: no agent asked for something the caller had said already | A model |
| `question_answered`: each question got an answer, an honest "I do not know", or the right agent | A model |
| `caller_memory_right`: the agents were right and honest about the caller's earlier calls | A model |
| `facts_from_profile`: each fact about the owner came from the profile notes | A model |
| `tone_fits_caller`: the agent's tone fitted how the caller sounded | A model that hears the call |
| `no_talk_over`: no agent talked over the caller | A model that hears the call |
| `speech_clear`: the agents' speech was clear and complete | A model that hears the call |

- A verdict is `pass`, `fail`, or `na`. `na` means that the call had no moment where the rule applies.
- A failure also names the agent at fault and a cause code. Each is a word from a fixed list in `grading.py`.
- With no model, or when each model fails, the code rules still give their scores.

Two models can check the model rules:

**The decision grader** (`decision_grader.py`) is the first choice for a voice call. It is Gemini Decision 1,
a decision model of Google. A decision model writes no text: it answers typed questions, and each answer
is one option from a fixed list, with a probability for each option.

1. During the call, the server records both sides in memory, in one channel (`voice/call_audio.py`).
2. After hangup, one request holds the audio, the profile notes, the agents on the call, and the transcript.
   The request says that the recording is right where the two differ.
3. Each cause code of a model rule has one yes/no question about one event. An example: "Did an agent
   ask the caller for a fact that the caller had already told that same agent?" The answer is a probability.
4. A rule fails when the model gives one of its events a probability of `GRADER_FAIL_AT` (0.8) or more.
   That event is the cause code. A failure starts a paid job that proposes a fix, so the limit is high.
   Under the limit the rule passes, and a note keeps a failure that the model thought likely.
   One more question for each rule gives the agent at fault.
5. Three rules have questions about single lines too. Each rule needs a search through the call, so code
   does the search, and the model judges one line at a time.
   - `nothing_asked_again`: code finds each agent line that asks something. For each one, the model answers:
     "Had the caller already given the answer in lines 1 to 8?" Two more small questions take a line out of
     the count: a question for the name, and a read-back.
   - `question_answered`: code finds each question of the caller, but not one in the caller's last turn. For
     each one, the model answers: "Did the agents leave that question without a response?"
   - `caller_memory_right`: code finds each agent line that speaks about the memory of calls and has a
     "not". For each one, the model answers: "Does the agent deny that it knows about earlier calls?"
   - Code knows which agent said a line. So for these failures, code names the agent at fault.
6. The same request asks for three readings: how the caller sounded over the call, how the caller sounded
   at the end, and how the agents' voices sounded. A reading is not a rule. It is stored, and it is private.

**The grader model** (`GRADER_MODEL`) is a text model that reads the transcript. It checks a call when the
decision grader is off or gives no answer, and it is the first choice in the text mode. It cannot check an
audio rule. It writes a note of at most 25 words for a failure; the note is for the owner only. With
doubt, it must pass a rule.

- With `GRADER_COMPARE=true` (the default), the grader model checks each voice call too. The log shows where
  the two differ; the record keeps the decision grader's verdicts. Set it to `false` when the two agree
  on real calls: it is one more request for each call.
- `GRADER_DECISION_MODEL=` (empty) turns the decision grader off. It also needs `GOOGLE_API_KEY`.
- `GRADER_HEARS_CALL=false` keeps the call audio on the server. The decision grader then reads the transcript.

The call audio:

- It is in memory only. The server never writes it to a file.
- It leaves the server one time: after hangup, in the request to Google's Decisions API.
- It is deleted after the grade. After a transfer it stays for 5 minutes, because a call comes back
  when nobody answers the transfer.
- A caller who asks to be forgotten loses it at once, and the rest of that call is not recorded.
- Use a paid tier for `GOOGLE_API_KEY`: on the free tier, Google's terms let it use the data to improve its products.
- This is a recording of a phone call, held for a short time. Some places want the consent of each person
  on a call. The greeting does not say that the call is recorded.

Measured on 2026-11-21:

- The test set is `evals/grader_calls.json`: 41 made-up calls, as transcripts. They hold 24 planted
  failures and 140 checks that must not fail. `uv run check-grader --set` runs it.
- The decision grader found 22 of the 24 at the limit of 0.8. It gave no false alarm. Each found failure got
  the right agent and a right cause. The two misses got 0.64 and 0.74: in both, a later agent asked for what
  the caller told the first agent.
- The highest probability of a check that must not fail was 0.64: an agent that said "I do not know his rates".
  At a limit of 0.6 the grader finds 24 of 24, but gives 1 false alarm. So the limit stays at 0.8.
- The grader model found 9 of 9 on the first 13 calls, with no false alarm. It did not read the other 28.
  So the grader model is first in the text mode.
- The structure of a question matters more than its words. Three steps were measured on the first 13 calls,
  which hold 9 planted failures. One question with the whole rule text and the options pass, fail, na: 6 found.
  One short yes/no question for each cause: 7 found. With the questions about single lines: 9 found.
- A shorter question was better. Two questions had a list of what does not count. Without the list, the lowest
  planted failure went from 0.66 to 0.75, and no false alarm came back.
- `nothing_asked_again` was the hard rule. With questions about the whole call, the grader found 4 of its 7
  planted failures. 14 wordings of those questions did not repair that. The model gave 0.86 or more when
  the caller complained ("as I said"), and 0.68 to 0.79 when the caller only said the answer again.
- The questions about single lines repaired most of it. For each of the two rules, 8 calls were written after
  the change and graded one time. No caller complains or asks again in those calls. The grader found 4 of 4
  for `nothing_asked_again` (0.85 or more) and 4 of 4 for `question_answered` (0.91 or more), with no false alarm.
- Not kept: the agent's own words in the small question. It raised the two misses to 0.95, but it lowered
  another planted failure to 0.71 and raised a read-back of one detail to 0.79.
- A probability near the limit is not stable. The line numbers in the transcript moved one answer from 0.90 to 0.78,
  before the questions about single lines replaced it. Read a probability of 0.7 to 0.9 as "likely".
- A question of the caller is found by its question mark. Speech recognition can leave the mark out. Such a
  question is left to the questions about the whole call.
- Each call here is made up, in clean text. A real call has speech recognition errors. So these counts are a start.

Measured on 2026-11-22, after the rule `caller_memory_right` came in:

- A real call showed two faults that no rule covered. The first agent welcomed the caller back after a first
  name only, and the caller said that the earlier call was not theirs. Then the agent said that it does not
  look up earlier calls, which is false.
- 10 made-up calls were added: 5 with a planted failure of the new rule, and 5 with no failure. The 5 with no
  failure hold what must not fail: a returning caller, an agent that found no earlier call, a caller who asks
  to be forgotten, and an agent that will not share another person's call.
- The set is now 51 calls with 29 planted failures and 226 checks that must not fail. The decision grader
  found 27 of the 29. The new rule: 5 of 5 found (0.96 or more), and no false alarm in the 51 calls. Its
  highest check that must not fail was 0.60: an agent that says "I have no notes about you now" after it
  deleted them.
- 1 false alarm of 226, at 0.81, in the old rule `nothing_asked_again`. It is in a new call: an agent uses
  the notes of another caller after a handoff, and the caller must say the need again. The agent asked nothing.
- Two of the 10 calls were changed after the first run. Each had a second fault that was not planted: an
  agent's question, and a caller's question that got a false answer. The second run counted them as they are now.
- The first wording of one question missed one planted failure (0.67): "keeps no notes of earlier calls".
  The agent in that call said "nothing about a call is stored". A wider wording found it (0.96).
- On the real call, the question about the whole call gave the denial 0.37. The questions about single lines
  gave four denial lines 0.86 to 0.97, and the true answers of a later agent 0.09 or less.
  The first wording of the small question gave those four lines 0.38 to 0.52. It named "this phone line";
  the agent spoke about itself ("I do not look up past callers").
- The same request two times gave the same probabilities.
- Two saved recordings of the caller's side of test calls, at 8 kHz: 86 s took 1.1 s and 6,827 input tokens;
  601 s took 6.1 s and 23,299 tokens. At the published price of 0.10 USD for a million input tokens, a
  10-minute call costs about 0.2 cents.
- Those recordings hold no agent. The reading of the agents' voices came back at 0.37, so it was not kept.
- Not measured: a real call with both sides. The audio rules and the readings have no check against a known answer.

More notes:

- The transcript is data for a grader: each request says not to follow a line that reads like an instruction.
- The server log has one line for each call: `grade (version 6e77d41c0a1b): 5 of 6 rules passed; failed: name_asked_once (...)`.
- With LangSmith on, the grader's request is the run `call-grade` in the call's thread.
- A call where the caller said nothing gets no scores. A caller who asked to be forgotten is not graded.
- To add a rule, add a `Rule` to `RULES`. Never use an old key for a new rule: the scores are stored by key.

Limits: a model rule is a model's opinion, so one verdict can be wrong. Read the scores over many
calls, not one. No rule checks how an agent says the owner's name: the model does not know the right
sound. The time before an answer is not a rule either: the server measures it (`voice/stats.py`).

### Call analytics and the public report

When a call ends, its figures and its scores go to three tables in the leads database (`analytics.py`):
one row for each call, one for each timed answer, and one for each score.

- **Each call:** the US Eastern date, the channel (`voice` or `text`), the version, the length, the agents,
  the caller's turns, the outcome, the failovers, the tokens, the cost estimate, and the readings.
- **Each timed answer:** the agent, the voice provider, and the seconds from the end of the caller's speech
  to the agent's first audio.
- **Each score:** the rule, the verdict, and the grader's note.

The version is the git commit of the build. `scripts/fly-deploy.sh` puts it in the image as
`HARNESS_VERSION`; on a laptop git gives it, with `-dirty` for a change that is not committed. So the
scores of one version of the prompts can be compared with the next.

What the tables do not hold: the phone number, the name, the summary, the transcript, and the time of
day. Three fields are private all the same: the call id, which joins a record to its LangSmith trace,
the grader's notes, and the readings of how a call sounded.

`uv run call-report` prints the public report as JSON. It is made for a public page:

- `rules`: each rule, with its count of `pass`, `fail`, and `na`.
- `totals`: the calls, the score, the median answer time, and the outcomes.
- `versions`: the calls, the score, and the median answer time of each version, in the order the versions took calls.
- `calls`: one entry for each call, with the fields of an allow-list (`_public_call`).

The score is the passed verdicts as a share of the passed and the failed ones. `na` does not count.

What the report leaves out, and why:

- The call id, the grader's notes, and the readings: they are private.
- The outcome of one call (a message, a transfer, or none): it can say something about one caller. The
  report gives totals only, and only from 5 calls on. Limit: a reader who keeps two reports can still
  see the change between them.
- Test calls: a call from `OWNER_PHONE`, and each call of the terminal simulator. `--tests` puts them in.
- A field that is not on the allow-list. A new field is private until someone adds it to `_public_call`.

`uv run call-report --notes` prints each call's scores with the notes, as text. That output is private.

`uv run check-grader` sends one made-up call to each grader. The call has three failures on purpose.
The run prints each rule with the verdict of each grader and the verdict it must get. For the decision
grader it prints the probability of the most probable cause too. Run it after you change a rule,
`GRADER_MODEL`, or `GRADER_DECISION_MODEL`. It is two paid requests; `--only decision` or `--only model` sends one.
`--audio FILE` sends a WAV file to the decision grader, to see what it hears.

`uv run check-grader --set` measures a grader. It sends each made-up call of `evals/grader_calls.json`.
Each call there has known verdicts. The run counts the failures found and the false alarms. For the
decision grader it shows the same counts at other limits too. Run it before you trust a change to a
question (`ASKS` in `decision_grader.py`), to a rule, or to `GRADER_FAIL_AT`. Write a new call and its
verdicts before you run a grader on it. `--only decision` costs less than one cent for the 41 calls;
the grader model costs about 1 to 2 cents for each call. The job that proposes a fix cannot change this file.

**The public page.** After each call, the server writes the public report as `calls.json` to a
Cloudflare R2 bucket (`publish.py`). The page `/calls` on the owner's site reads that file, so a
visitor to the page never starts this server. A test call does not start an upload. When a caller
asks to be forgotten, the file is written again without that call.

- The upload is one signed PUT to R2's S3 API. The signing code is in `publish.py`, with no AWS package.
  A test checks it against the example signature in the AWS documentation.
- A failed upload is one line in the log. It cannot stop the end of a call.
- `uv run call-report --publish` writes the file now. Use it for the first upload, and after you change a rule's title.

To set it up:

1. Make the bucket: `pnpm wrangler r2 bucket create voice-card-report`, in the site's repository.
2. In the Cloudflare dashboard, open **R2 object storage**, then **API Tokens**. Make a token with the
   permission **Object Read & Write**, for this bucket only.
3. Put the four values in `.env`: `REPORT_R2_ACCOUNT_ID`, `REPORT_R2_BUCKET`, `REPORT_R2_ACCESS_KEY_ID`,
   and `REPORT_R2_SECRET_ACCESS_KEY`. With one of them empty, there is no upload.
4. Run `uv run call-report --publish`. Then run `bash scripts/fly-secrets.sh` to copy the values to Fly.

"Forget me" and the analytics: the call in progress gets no record, and a record from earlier in the
same call is deleted. The notes of that person's earlier calls are deleted too. The figures and the
verdicts of those earlier calls stay, because they say nothing about a person.

### A fixed score for a change to a prompt or a word list

A change to a prompt or a word list is good only if a fixed measure says so. `uv run score-harness`
is that measure (`score.py`). It reads a made-up test set, the folder `evals/`. No line in it is from
a real call.

- `evals/live_call.json`: caller lines, each with the answer to "does this caller want a live call?".
  The word list of the transfer check must give that answer. This part makes no model request, and it
  gives the same result each time. `uv run check-jev` reads the same file.
- `evals/calls.json`: made-up calls. Each one has the caller's lines and what the call must do: the
  agents that take part, the outcome, and if the caller is forgotten. The agents answer the lines in
  the text mode. Then the grader scores the call against its rules. This part makes model requests
  that cost money, so it runs only with `--calls`.

```
uv run score-harness                                  # the word list only: no model request
uv run score-harness --out before.json                # keep the result of the version before a change
uv run score-harness --compare before.json            # after the change; exit 1 when a check that passed now fails
uv run score-harness --calls --only live-call --show  # one made-up call, with its text (paid)
uv run score-harness --calls                          # each made-up call (paid)
```

- In a made-up call the caller's lines are fixed: the caller does not react to the agent. No email
  is sent, no phone rings, and the lead store is in memory. Jev is off, so the word list decides a transfer.
- The agents' answers change from run to run. Thus one check of a made-up call can pass in one run
  and fail in the next. Read the checks that changed before you trust a small difference.
- A rule that does not apply to a call gives no check. A model rule with no verdict is a failed check.
- Add a new sample or a new call at the end of its file. Do not change or remove an old one: two
  results are compared by the name of each check.

### The improvement loop: a failed rule opens a pull request

No person starts this loop. A call ends, and the rest runs with no action from the owner:

1. **The grader scores the call** against its rules (`grading.py`). For a failed rule it also names
   the agent at fault and a cause code.
2. **The server writes a fault report** (`fault_report.py`): what the caller wanted, what the agent
   did, and what the agent had to do. Code removes each private word before the report leaves.
3. **The server sends a fix request** (`improve.py`) with the fault report. It starts the workflow
   `.github/workflows/improve.yml` with GitHub's REST API.
4. **A coding agent makes one change** (the Claude Code GitHub Action). It reads the fault report, the
   rule and the agent's instructions. It adds one made-up call to `evals/calls.json` that shows the
   fault. Then it changes one prompt or one word list.
5. **Fixed code measures the change** (`proposal.py`, `score.py`). It checks which files changed, runs
   the unit tests, and runs the scorer before the change and after it.
6. **The job opens a pull request**. It starts with what went wrong in the call. Then it shows the
   rule, the change, and both scores.
7. **The owner reviews it.** A merge to `main` deploys the server. The next calls have a new harness
   version, and their scores show if the change helped.

What a fix request holds: the rule, the agent, the cause code, the harness version, two counts, and
the fault report. Each name is a word from a fixed list in `grading.py`; a value that is not in its
list is not sent, and the job refuses it again. The job has no transcript and no access to the traces.

**The fault report** is the one part that comes from the call. The repository is public, and so are
the logs of the job, so the report is made in two steps:

1. The grader model (`GRADER_MODEL`) reads the transcript on the server and describes the fault in
   three short texts. Its instructions: write "the caller" and "the agent", name the kind of a detail
   and never the detail, and do not copy a sentence of the caller.
2. Code removes what an instruction cannot promise (`scrub`):
   - A word of the call stays only when it says nothing about a person: the word is in the source text
     of this package in lower case, or it is a plain English word (`english_words.txt`, from the SCOWL
     spelling dictionary). A word that the dictionary also has with a capital letter ("John", "Rose")
     is not in that list, and the table of name forms in `names.py` does not count as source text.
     Each other word of the call becomes `[removed]`.
   - Each name of the caller becomes `[removed]`. The names come from the server's record of the call
     and from the transcript: the answer to the name question, and the words after "my name is".
   - A word of the call with a capital letter inside a sentence becomes `[removed]`. Names that are no
     secret stay: the voice models, languages ("Spanish"), and well-known products (`KNOWN_NAMES`).
   - Each digit, each email address, each link, and a row of three or more number words is removed.
   - A text has a length limit (1500, 600 and 500 characters) and a fixed set of characters.
     The job checks both again and refuses a fix request that breaks them.

The rule `summary_first_try` needs no model: its fault report is a fixed text for each fault that the
summary check found ("The summary was longer than the word limit of the prompt.").

What the scrub cannot promise:

- A name of another person that the source text also uses as a plain word ("Mark", "Will") can stay,
  when the model writes it in lower case or as the first word of a sentence. So can a rare name that
  the dictionary has only as a plain word ("Sky"). The model's instructions forbid names.
- A plain word can say what the call was about ("dental", "clinic"). Before 2026-11-22 such a word
  became `[removed]`; that made a caller's feature request unreadable ("the line speaks [removed]").
- A caller can influence the report. The job gives it to the coding agent as a description of a fault
  and tells the agent not to follow it. The limits on the coding agent below apply all the same.
- A report with many `[removed]` marks is less clear. With no grader model, or after an error, the fix
  request goes out with no fault report, and the pull request says so.

Limits:

- One fix request for each (rule, agent, cause) at one harness version. The same failure on the next
  call starts no second job. After a deploy the version is new, so the failure can ask again.
- At most `IMPROVE_DAILY_LIMIT` fix requests in one day (default 2).
- Only a server whose version is a clean git commit sends a fix request. `uv run chat` sends none.
- A call from the owner's phone can send one. That shows the loop with no real caller.
- A caller who asked to be forgotten is not graded, so that call sends nothing.
- The audio rules `no_talk_over` and `speech_clear` send none. No prompt and no word list can repair them:
  the voice provider's turn detection and its audio decide them. Their scores are in the report all the same.

What the coding agent can and cannot do:

- It has a token that can only read the repository. Its change leaves its job as a patch.
- It can change the agents' instructions (`agents/specs.py`), the handoff text (`graph/handoffs.py`),
  the tool words and the word lists (`graph/tools.py`), the summary prompt (`callers.py`), the tests,
  and the test set. The job refuses a patch that changes another file: the grader, the scorer, a
  workflow, or a script.
- It can add an entry at the end of a test set file. It cannot change or remove an old entry.
- This list is the scope `prompts`: a failed rule, and an issue with the label `auto-fix`. The triage
  agent's label `auto-build` and the owner's label `build` on an issue give the wider scope `code` (see
  "From an issue to a pull request").

What the pull request shows, and its limits:

- **Better:** a check that failed now passes, and no check got worse. The pull request is ready for review.
- **Same:** no check changed. The pull request is a draft, and it says that the scorer does not show
  the change. The scorer runs the agents in the text mode. A fault that only a voice model makes,
  for example a tool name said aloud, does not happen there.
- **Worse:** a check that passed now fails. The job opens no pull request; its summary shows the scores.
- The agents' answers change from run to run, so one check of a made-up call can change with no cause.
- The grader can be wrong about a call. Then the coding agent is told to change nothing when it finds no reason.

To turn the loop on:

1. Give the repository one secret for the coding agent. For a Claude subscription: run `claude setup-token`
   on your computer, then `gh secret set CLAUDE_CODE_OAUTH_TOKEN` in this repository. For the API account:
   `gh secret set ANTHROPIC_API_KEY`. With both, the subscription pays for the coding agent and the API key
   pays for the scorer. With no `ANTHROPIC_API_KEY`, the scorer runs no made-up call.
2. In **Settings**, **Actions**, **General**, turn on "Allow GitHub Actions to create and approve pull requests".
3. Make a fine-grained personal access token: this repository only, with the repository
   permission **Actions** (read and write). It can start a workflow. It cannot change code.
   For issues from a caller's feedback (above), add the permission **Issues** (read and write).
4. Put `IMPROVE_GITHUB_TOKEN` and `IMPROVE_REPO` (as `owner/name`) in `.env`. Then run
   `bash scripts/fly-secrets.sh` to copy them to Fly.
5. Make a second fine-grained personal access token: this repository only, with the repository
   permission **Pull requests** (read and write) and nothing else. Then run `gh secret set IMPROVE_PR_TOKEN`
   in this repository. The fix job opens its pull request with this token, so the checks of the pull request
   (`test.yml`) start at once. GitHub holds each run of a pull request that a job's own token opened, until
   a person selects "Approve workflows to run". With no `IMPROVE_PR_TOKEN`, or with one that is not valid
   any more, the job's own token opens the pull request, and that approval is necessary again.
   Only the last job of the workflow gets this token. That job runs no line of the coding agent's change.

Optional repository variables: `IMPROVE_MODEL` (the coding agent's model), `IMPROVE_SCORE_CALLS`
(`all`, `new` or `none`: which made-up calls the scorer runs; these cost money), and `OWNER_NAME`.
The workflow runs on a public repository, where the standard runners use no paid minutes.

### Feedback from a caller: a GitHub issue with a count

A caller can say a fault aloud ("the line was silent for a long time, that is a bug"), or ask for a
feature ("this line should speak Spanish"). No rule of the grader fails for such a sentence. This step
keeps it (`feedback.py`):

1. **The decision grader answers two more questions** in its one request after hangup: did the caller
   report a fault of the phone line, and did the caller ask for a feature. With no yes at
   `GRADER_FAIL_AT`, nothing more happens, unless the feedback agent filed a report (below).
2. **The grader model writes each item** in three short texts: a title, what the caller reported or
   asked for, and what the line did at that moment. At most 3 items for one call.
3. **Code removes each private word** with the scrub of the fault report (below). The transcript
   stays on the server.
4. **The duplicate check.** Before the model writes, the server reads the issues that callers opened
   before (the labels `caller-fault` and `caller-idea`). The model gets their titles and names the
   issue that is about the same thing, also when the words are different.
   - No such issue: a new issue opens, with its label.
   - An open issue is about the same thing: no new issue. The count in the issue text goes up by one,
     and a comment keeps the new caller's words.
   - The owner closed that issue as "not planned": the count goes up, with no comment and no new issue.
   - The owner closed that issue as completed: a new issue opens, because the fault is back after a fix.

**The feedback agent.** The other agents hand such a caller to the `feedback` agent. A view about the
owner, a plain question, or one answer that an agent can correct stays with the agent. The feedback
agent asks at most two short questions, one at a time, and only for what the caller did not say yet:
for a fault, what happened and what the caller expected; for an idea, what the line should do and why.
It says the report back in one sentence. After the caller's yes, it calls `file_feedback`.

- `file_feedback(kind, summary)` only keeps the report in the call context, at most 3 for one call.
  No request leaves the server during the call. Its result gives the agent the words to say, and the
  words promise no fix and no date.
- After hangup, a kind that the agent filed counts as certain, as if the decision grader said yes. The
  agent's summary goes to the grader model's prompt as a hint, marked as data. The public text still
  comes from the grader model and the scrub; the words of the summary are scrubbed as words of the call.
- The feedback agent has no transfer tool, so a report never rings the owner's phone.
- A caller who asked to be forgotten files nothing: "forget me" clears the reports, and the call is not graded.
- A text call (`uv run chat`) keeps the reports but opens no issue.

An issue starts with two lines that code keeps current:

```
**Callers who asked for this: 10**
Last report: 2026-11-22
```

The count is a count of callers, not of calls. A table on the server keeps a hash of the phone number
for each issue, so one caller counts one time for one issue. A call with a hidden number counts as its
own caller. "Forget me" deletes the caller's rows from that table; the count in the issue stays.

Limits:

- The number of new issues in one day has no limit. An issue costs nothing. Before 2026-11-22 a limit
  of 3 new issues in one day dropped each later report, and such a report was lost. The cost limits
  are now on the jobs that an issue can start (the next section).
- At most 3 items from one call.
- A caller who asked to be forgotten gets no issue: that call is not graded.
- The two models must agree on the kind: the grader model's item is dropped when the decision grader
  did not say yes for that kind, and the feedback agent filed no report of that kind.
- `FEEDBACK_ISSUES=false` turns the step off. With no token or no repository, nothing is sent.
- When the decision grader gives no answer for a call, only a report that the feedback agent filed
  is read from that call.

What this step cannot promise:

- The issue text is public, and a model wrote it from a caller's words. The scrub removes names and
  numbers; it cannot judge whether a sentence is polite or true. Read an issue before you act on it.
- A caller can say the same thing from two phone numbers, and that counts two times.
- Two calls that end in the same minute with the same new feedback can open two issues. Close one, and
  the count goes on in the other.

Measured on 2026-11-22: the set is now 59 calls (28 of 30 planted failures found, 1 false alarm of 265).
8 made-up calls were added to `evals/grader_calls.json` (5 with feedback, 3 that
must give none: a question about a feature, a complaint about the owner, a correction of one answer).
The decision grader found 6 of 6 kinds of feedback (0.97 or more), with no false alarm in 112 checks
(the highest was 0.46). The whole step ran one time with the grader model and a GitHub in memory:
the 5 calls opened 6 issues; two more calls said an old thing in new words ("dead air when you switch
agents", "offer this in her language") and each raised the right count; one new idea opened a new issue.
One label was corrected after the first run: in `caller_reports_a_wrong_agent` the caller asks for the
contact agent and gets the tour agent. The grader failed `question_answered` there (0.87), and it was right.

Set up: add the repository permission **Issues** (read and write) to the token `IMPROVE_GITHUB_TOKEN`,
and make the two labels one time: `gh label create caller-fault` and `gh label create caller-idea`.
Without the permission, the server log says "the caller's feedback did not get into GitHub: HTTPError:
HTTP Error 403".

### From an issue to a pull request: the queue, the triage agent, and two jobs

An issue is the one place where work waits. A label says where an issue is, and a label is the only
thing that starts a job (`issue_queue.py`, `.github/workflows/issues.yml`).

| An issue with | What happens | Who decided |
|---|---|---|
| `caller-fault`, no other label | The triage agent sorts it (`triage.yml`). | The queue |
| `caller-idea`, no other label | It gets `needs-owner`. No agent is asked. | Code: a feature is the owner's decision |
| `auto-fix` | The fix job starts (`improve.yml`, scope `prompts`). | The triage agent, or the owner |
| `auto-build` | The build job starts (`improve.yml`, scope `code`). | The triage agent, or the owner |
| `needs-owner` | It waits. | The triage agent, or code |
| `not-actionable` | It waits. No job can act on the text. | The triage agent |
| `build` | The build job starts for any issue, also for a feature. | Only the owner |
| `fix-started` | A job started. No second job starts. | The queue |

**The triage agent** reads one issue and the prompts of the agents. Its only output is one label from
the list `auto-fix`, `auto-build`, `needs-owner`, `not-actionable`, and one short reason. A plain
fault starts a job with no word of the owner. The agent chooses `auto-fix` when it can name the text
that the change is in: a prompt, a handoff text, a tool's words, or a word list. It chooses
`auto-build` when the report is clear and the repair needs code; the job finds the cause. It chooses
`needs-owner` only for three cases: a large change (a weaker passphrase check, a change of what the
line keeps of a caller or of when it rings the owner's phone, a new cost, service or dependency, an
ability that goes), a report that two different repairs fit, or a fault that no change in this
repository can repair. A fault is not a case for the owner only because it needs code. Code reads the
answer: a label that is not in the list becomes `needs-owner`, a label that starts a job on an issue
that is not a fault becomes `needs-owner`, and code sets the label and writes the comment. The agent's token can only read, and
it has no tool that changes a file or runs a command.

**The two jobs** are one workflow with two scopes:

- **The fix job (scope `prompts`)** is the job that a failed rule starts, with the issue in place of
  the fault report. It can change the same files, and the same guard checks it.
- **The build job (scope `code`)** can change the code of the phone line (`src/ai_call_center/`), the
  tests and this file. It cannot change the code that measures or judges a change: the scorer, the
  guard, the queue, the grader and its rules, the scrub of the fault report, and the step that makes an
  issue from a caller's words (`CODE_REFUSED` in `proposal.py`). It cannot change a workflow, a script,
  or a dependency. The job also runs the formal check of caller memory. Its pull request is always a
  draft, because the scorer measures the agents' words and not new code.

The pull request says `Closes #N`, so a merge closes the issue. A job that opens no pull request
writes the reason as a comment on the issue, removes `fix-started` and the label that started it, and
sets `needs-owner`. To run the build job again, put `build` on the issue again.

**The owner's label wins.** The owner can set `auto-fix`, `auto-build` or `build` on an issue at any time, also on
an issue that the owner wrote by hand. The queue reads the labels again on each run, and it sorts no
issue a second time. To have an issue sorted again, remove its triage label.

Who can start a job:

- Only an issue that the owner of the repository wrote counts. The server opens a caller's issue with
  the owner's token (`IMPROVE_GITHUB_TOKEN`). An issue that another person opens on GitHub starts
  nothing, with each label. A comment on an issue is never read.
- Only a person with rights on the repository can set a label.
- Each job reads the issue itself and checks it again: the author, the label for its scope, and, for a
  caller's issue, that each text holds only letters and plain punctuation. So a person who starts
  `improve.yml` by hand with the number of an issue gets no more than the labels allow.
- The text of a caller's issue is data for both agents. Each task says so, and the guard of the files
  and the owner's review hold the result.

Limits, each for one day (US Eastern). An issue past a limit waits for the next day; it is never dropped.

- `TRIAGE_DAILY_LIMIT` triage runs (default 10).
- `ISSUE_FIX_DAILY_LIMIT` jobs for issues, in both scopes (default 2). Each job uses the owner's Claude
  subscription, and a build job can run for 60 minutes; that is the reason for the limit. The owner's
  `build` always starts its job, and that job counts in the limit of the day.
- The triage agent gets 3 tries for one issue, one in a day. After 3 failed runs the issue gets
  `needs-owner`, with a comment.

The queue runs on each new issue, on each change of a label by a person, and each morning. It reads
all open issues and does only what is not done, so a lost event loses no work. It needs no new secret:
the token of its own run sets the labels and starts the two other workflows. So the actor of those
runs is the bot of GitHub Actions, and each agent step names it in `allowed_bots`; the agent action
refuses a run that another bot started. Optional repository
variables: `TRIAGE_MODEL`, `BUILD_MODEL`, `TRIAGE_DAILY_LIMIT`, `ISSUE_FIX_DAILY_LIMIT`.

What this does not promise:

- The triage agent can be wrong. A wrong `auto-fix` or `auto-build` costs one job, whose pull request
  the owner reviews; no job merges its own change. A wrong `needs-owner` costs the owner one label.
- Each fix adds a made-up call at the end of `evals/calls.json`. Two open pull requests then change
  the same lines, and the second one needs a merge of `main` before it can merge.
- A pull request that the owner closes with no merge leaves `fix-started` on its issue. Remove the
  label to run a job again, or close the issue.

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

The server keeps a copy of the feed in `data/writing.json`. At start it uses that copy at once and
reads the feed in the background, so a slow feed cannot delay the start. When the feed has changed,
the next call uses the new posts. Only the first run, which has no copy, waits for the feed (5 s limit).
When the feed cannot be read, the server stays on the copy. Passage vectors are saved in
`data/writing_vectors.npz` under a hash of each passage, so a restart embeds only new or changed passages.
Restart the server to pick up a new post. The first start downloads the model to `data/models/`;
a container image should include that folder. With `WRITING_EMBEDDING_MODEL=` empty, the search is BM25 only.

This machine runs NixOS. `uv.toml` pins `python-preference = "only-managed"`, because prebuilt wheels
(numpy, onnxruntime) load only under a Python that runs through nix-ld.

## Hosting on Fly.io

The server can run on one Fly.io machine that sleeps between calls and wakes for a call. The files are
`Dockerfile`, `fly.toml` and `scripts/fly-secrets.sh`. The run book is [docs/hosting-fly.md](docs/hosting-fly.md):
setup, the checks after the first deploy, the cost, the backups, and how to move the phone line between
the laptop and Fly.io.

A push to `main` that changes the server deploys it (`.github/workflows/fly-deploy.yml`). The deploy runs
the tests first, and it waits until no call is in progress (`scripts/fly-deploy.sh`).

In a container the server must listen on all addresses, so the image sets `HOST=0.0.0.0`. On the laptop
the default stays `127.0.0.1`.

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
  decisions.py                   Jev, a decision model: the probability that a caller asked for a live call; the word list is the fallback
  grading.py                     the grader: the rules, the code checks, and the request to the grader model, after hangup
  decision_grader.py             the decision grader: Gemini Decision 1 hears the call audio and answers typed questions; the grader model is the fallback
  analytics.py                   the figures, answer times and scores of each call; the public report (`uv run call-report`)
  publish.py                     the upload of the public report to a Cloudflare R2 bucket, for the page on the owner's site
  check_grader.py                `uv run check-grader`: one made-up call to each grader, to check the keys, the questions and the verdicts
  caller_voice.py                a voice reading for each caller turn, and the texts for the tour agent and the lead email
  score.py                       `uv run score-harness`: the fixed score of the prompts and the word lists, against the made-up test set in evals/
  improve.py                     a fix request: a failed rule starts the GitHub workflow that proposes a fix
  fault_report.py                what went wrong in a call, with each private word removed by code
  english_words.txt              the plain English words that the scrub keeps (SCOWL; licence in english_words.LICENSE.txt)
  feedback.py                    a caller's fault report or feature idea becomes a GitHub issue, with a count of the callers
  passphrase.py                  the passphrase check: a match of number and name opens no notes of an earlier call; the caller's passphrase does
  proposal.py                    `uv run proposal`: the fixed steps of that workflow (the task, the guard on changed files, the pull request text)
  check_jev.py                   `uv run check-jev`: made-up caller lines to Jev, to check the key, the question and the time limit
  check_voice.py                 `uv run check-voice`: saved caller audio through the voice reader, with each turn's reading and read time
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
  voice/call_audio.py            both sides of a call, in memory until the decision grader has heard them
  voice/voice_reader.py          Pipecat Voice Reader on the caller's audio: a voice reading for each turn, and the note when a caller sounds frustrated
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
