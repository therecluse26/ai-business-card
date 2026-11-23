# AI voice business card: verification map

This folder is the maintained source for verifying what a caller can do with the AI voice business card. Read this index before you drive the app, then use the matching feature file as the recipe.

## Baseline preconditions

- Run every command from the repository root, with `C=.claude/skills/verify-business-card/control.py`.
- Make a run folder: `RUN=$(uv run python $C new)`. A run starts with an empty database: no leads, no earlier calls, no passphrase.
- `uv run python $C doctor --run $RUN` prints no line that starts with `FAIL`.
- A text call costs money. The owner said yes to the number of text calls in this run, after a cost quote.
- A text call runs on the small model `PROOF_MODEL` of `control.py`, because the live line runs small realtime voice models. A larger model needs a reason and the owner's yes.
- Never drive a server that this run did not start, and never send a test request to the live server.

## Driving conventions

- One `control.py chat` command is one call. Each `--say` is one caller turn.
- Use `--from +15555550142` unless a recipe gives another number. The same number in two calls is the same phone.
- Use made-up names, companies and email addresses (`@example.test`). Never use a real person's data.
- The model's words change from run to run. Assert the `[tool]` lines, the `[result]` lines, the name in the prompt `caller (<agent>)> `, and the stored rows. Do not assert an agent's exact sentence, except words that a `[result]` line told the agent to say.
- When the agent asks for something that no `--say` line gives, the run proves nothing. Report it as not proven, with the transcript.
- Do not use `curl`; this workspace blocks it. Use `control.py http`.

## Proof and skip reporting

- A proof names the feature ID, the run folder, the commit from `doctor`, and the model that ran.
- A text call proof is the transcript file plus the output of `rows`.
- A server proof is the output line of `http` or `ws` plus the matching line in `$RUN/server.out`.
- A path that was not driven is reported as not driven, with the reason (for example: the owner gave no yes for a paid call). A feature that was driven through the text call is not reported as verified on the phone.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph on what the caller gets. It then has exactly four H2 sections in this order: `Sub-features`, `How to get to it (user POV)`, `Driving it with control.py`, `Gotchas`.

## Features

| Feature | Surface | Cost | Last driven end to end |
|---|---|---|---|
| [Guards of the phone server](./phone-server-guards.md) | Local phone server | Free | 2026-11-22, commit `1e750bb`: all four sub-features as written |
| [Leave a message](./leave-a-message.md) | Text call | Paid | Its own recipe: not driven yet. 2026-11-22, commit `1e750bb`, `anthropic:claude-haiku-5-5`: `message-route`, `message-confirm`, `message-save` and `message-offer` held in call 1 of the caller-memory recipe, with no email address. `message-once` not driven. |
| [Caller memory and the passphrase check](./caller-memory.md) | Two text calls | Paid | 2026-11-22, commit `1e750bb`, `anthropic:claude-haiku-5-5`: `memory-new`, `memory-set`, `memory-ask` and `memory-verified` as written. `memory-wrong` and `memory-recognised` not driven. |
| [Ask to speak to the owner](./speak-to-the-owner.md) | Text call | Paid | Not driven yet. The recipe comes from the code, not from a run. |
| [A real phone call on the live line](./real-phone-call.md) | Real phone call | Paid, by the owner | Not driven by an agent: only the owner can call. |

Not in the map yet: the tour agent's live figures, the consultant agent, the search over the owner's writing, "forget me", feedback from a caller that becomes a GitHub issue, and the failover between voice providers.
