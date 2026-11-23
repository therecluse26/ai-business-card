---
name: verify-business-card
description: Drive the AI voice business card (this repo, package `ai_call_center`) the way a caller does, and capture proof that a change works. Covers the text call (`uv run chat`), the local phone server (`uv run serve`, HTTP and WebSocket), and the evidence of a real phone call on the live line. Use it after a change to an agent prompt, a tool, caller memory, the passphrase check, a guard of the server, or before a claim that "it works" that unit tests do not prove.
---

# Verify the AI voice business card

A caller uses this product by phone. An agent cannot place a phone call, so a proof uses one of three surfaces. Use the first one that can show the behavior.

| Surface | What it runs | Cost | What it proves |
|---|---|---|---|
| Local phone server | `uv run serve` on `127.0.0.1:8011`, with no link to Telnyx | Free | The guards of the server: health, webhook signature, media stream token |
| Text call | `uv run chat`: the same graph of agents, the same tools, the same store, with a small text model | Paid: model requests, about $0.01 for a call | Routing, tool calls, saved leads, caller memory, the passphrase check |
| Real phone call | The live line on Fly.io (`voice-business-card`). The owner calls it. | Paid: voice models | Voices, handoffs, timing, the grader, feedback issues |

The text call does not prove voice behavior: it has no audio, no voice model and no Telnyx. A change to `voice/` or `telephony/` needs the local server, a unit test, or a real phone call.

The harness is one file, `control.py`, next to this file. Run every command from the repository root. The feature map in [`features/`](features/README.md) has one recipe for each feature. Read its index first.

## Rules before you start

- **A text call costs money, and the owner's budget is small.** Before each text call, tell the owner how many calls you want, why, and the cost. Wait for a yes. One text call of 5 turns is about 10 model requests of about 4,000 input tokens each. Take the prices from the `claude-api` skill, not from memory.
- **A text call runs on a small model: `PROOF_MODEL` in `control.py`, not `LLM_MODEL` of `.env`.** The live line runs small realtime voice models. A large text model follows a prompt that a small voice model does not, so a pass on a large model proves less about the line. No vendor publishes a text model that is equal to its realtime model, so this is a match of size, not a measured match.
- **Never run `uv run serve` or `uv run chat` by hand for a proof.** The harness sets the safe settings. Without them, `serve` can move the webhook of the live phone line to this computer, and a saved lead sends a real email to the owner.
- **Never send a test request to the live server.** `GET https://voice-business-card.fly.dev/health` is the only request that is allowed.
- **This workspace blocks `curl` and `wget` in the shell.** Use `control.py http` for an HTTP request.
- **The evidence can hold private data** (the owner's phone number is in `server.out`). The folder `data/verify/` is in `.gitignore`. Do not copy evidence into a commit, an issue or a pull request of this public repository.

## Launch

```sh
C=.claude/skills/verify-business-card/control.py
RUN=$(uv run python $C new)          # makes data/verify/<date-time>/ and prints its path
```

Each run has its own folder and its own empty database (`$RUN/leads.sqlite3`), so a run starts from zero: no leads, no earlier calls, no passphrase. The tables are made at first use; there is no migration step and no seed data.

- **Text call:** there is no process to keep alive. Each `control.py chat` command is one call. Ready means the transcript shows the greeting after `agent>`.
- **Local phone server:** `uv run python $C server --run $RUN`. Ready is the line `The server is up on http://127.0.0.1:8011`. The harness waits for `GET /health` to answer, up to 60 s.

Two runs can drive text calls side by side: each has its own database. Only one run can have a server, because the port is fixed at 8011. The `server` command refuses to start when another process answers on that port.

## Doctor

```sh
uv run python $C doctor --run $RUN
```

It changes nothing. It loads the settings the way the app does and prints one line for each check. Every line must start with `ok` or `info`; a line that starts with `FAIL` means: do not drive. It checks that the database is in the run folder, that no email, Telnyx request, number lookup, GitHub job or report upload can happen, the commit of the code and if `src/` has changes that are not committed, the two models that a text call uses, and, when the run has a server, that the process group of this run answers on port 8011. It fails when something else owns the port.

## Drive

The handles below are the app's own output. Each feature file gives the exact commands for one feature.

**Text call.** One command is one call. Each `--say` is one caller turn, in order. The harness types a line only after the prompt `caller (<agent>)> ` is back, then types `/quit`.

```sh
uv run python $C chat --run $RUN --name call-1 --from +15555550142 \
  --say "Hi, this is Dana Reyes." --say "I would like to leave a message."
```

- `agent> ...` is what the active agent says.
- `caller (<agent>)> ` names the active agent. A change of this name is a handoff.
- `  [tool] <name>(<args>)` is a tool call, and `  [result] ...` is what the code gave back to the model.
- `[call summary saved for <number>: ...]`, `[grade: ...]` and `[hung up] active agent: <agent>; leads saved this process: <n>` end a call.
- `--from` sets the caller's number. Use the same number for two calls to test caller memory. The default `+15555550142` is made up. Do not use `+10000000000`: the app treats it as a hidden number.
- `--model <provider>:<name>` runs the agents on another model than `PROOF_MODEL`. Say in the proof which model ran. A larger model needs a reason and the owner's yes.

The caller's lines are fixed, and the model's answers change from run to run. Write caller lines that still fit when the agent asks its questions in another order. When an agent asks for something that no line gives, the run does not prove the feature: report that, and do not count it as a pass.

**Local phone server.**

```sh
uv run python $C http GET http://127.0.0.1:8011/health
uv run python $C http POST http://127.0.0.1:8011/webhooks/telnyx --json '{"data":{"event_type":"call.initiated"}}'
uv run python $C ws ws://127.0.0.1:8011/media/not-a-token
```

Each prints one line: the request, then `-> HTTP <status>, <type>, <n> bytes`, or how the server answered the WebSocket. The server's own view of each request is in `$RUN/server.out`.

**Stored state.**

```sh
uv run python $C rows --run $RUN
```

It reads the run's database (read-only) and prints the leads, the calls, and the passphrase rows without the salt and the hash.

## Evidence

A proof is these files in `$RUN`, and they stay after cleanup:

- `chat-<name>.txt`: the full transcript of a text call, with the caller's number, the time and the model.
- `server.out`: the output of the local server, with one line for each request it refused.
- `rows-at-cleanup.txt`: the stored rows, written by cleanup. Run `rows` yourself for the state between two calls.
- Save the output of `http`, `ws`, `rows` and `doctor` with `| tee $RUN/<name>.txt` when a feature file says so.

The proof standards:

- Drive the real user path: the caller's words in a text call, a real request to the server. A unit test or a direct call of a tool function is not a proof by this skill.
- Capture the action and the result: the `[tool]` line and its `[result]` line, not only the last thing the agent said.
- Check the side effect with a second view: after a saved message, `rows` must show the lead. After a passphrase is set, `rows` must show one row in `caller_phrases`.
- The safe settings turn off real boundaries: email, Telnyx, Jev, the grader model, tracing, GitHub. A text call therefore proves that a lead was stored and that an email was recorded. It does not prove that the email arrives. `uv run test-email` proves that, and it sends a real email.
- An agent's sentence is not proof of a fact. "I have saved your message" counts only with the `save_lead` result and the row.

## Cleanup

```sh
uv run python $C cleanup --run $RUN
```

It stops the server that this run started (by its process group, never by name), writes `rows-at-cleanup.txt`, deletes the run's database, and lists the evidence that stays. Run it after a failed attempt too. To remove the evidence itself, delete the run folder by hand when the proof is no longer needed.

## Helpers

`control.py` is the only helper. `uv run python .claude/skills/verify-business-card/control.py --help` lists the commands: `new`, `doctor`, `chat`, `server`, `http`, `ws`, `rows`, `cleanup`. The dictionary `SAFE` at its top is the list of safe settings; a new outside service of the app needs a new line there.

## A real phone call

Only the owner can make one. Use it when the text call cannot show the behavior. [`features/real-phone-call.md`](features/real-phone-call.md) says how to check the deployed version first and which log lines are the evidence.
