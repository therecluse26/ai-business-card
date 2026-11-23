# Caller memory and the passphrase check

The line knows a returning caller by their phone number and the name they give. That makes the caller "recognised": a welcome back, and no notes. The agents get the notes of an earlier call only after the caller says the passphrase that they set on that call ("verified").

## Sub-features

- `memory-new` a first call from a number gets no "welcome back".
- `memory-set` after a saved message, the caller sets a passphrase; the agent says it back one time, and the caller confirms it.
- `memory-ask` on the next call with the same number and name, the first agent asks for the passphrase.
- `memory-verified` with the right passphrase, the agent gets the notes and names the topic of the earlier call.
- `memory-wrong` with a wrong passphrase two times, the agent gets no notes and does not ask again.
- `memory-recognised` with no passphrase on file, the same name gets "Welcome back" with no topic.

## How to get to it (user POV)

- Call 1: give your name, leave a message, say yes to "Want to set one?", say a made-up phrase, and say yes to the read-back.
- Call 2, from the same phone: give the same name, and say the phrase when the agent asks for it.
- In a text call: the same words, typed, with the same `--from` number in both calls.

## Driving it with control.py

Preconditions:

- The baseline of the index holds, and the owner said yes to two text calls (three for `memory-wrong`).
- `uv run python $C rows --run $RUN` shows no call and no passphrase row.

- **Call 1: leave a message and set a passphrase.** Run:

  ```sh
  uv run python $C chat --run $RUN --name memory-1 --from +15555550142 \
    --say "Hi, this is Dana Reyes." \
    --say "I would like to leave a message for the owner." \
    --say "My message is: I need help with a data pipeline for my company, Example Freight." \
    --say "Yes, that is right." \
    --say "Yes. My passphrase is copper kettle." \
    --say "Yes, that is right." \
    --say "No, that is all. Goodbye."
  ```

  The transcript has no `Welcome back`. It has `[tool] save_lead(`, then two `[tool] set_passphrase(passphrase='copper kettle')` lines: the first result holds `I heard: copper kettle. Is that right?`, the second holds `Done. On your next call, I will ask for that passphrase`. Neither result holds `Not kept`.
- **State between the calls.** Run `uv run python $C rows --run $RUN | tee $RUN/memory-rows-1.txt`. `## caller_phrases` has one row for `+15555550142` with `failed` = `0`. `## phrase_calls` has one row: that passphrase and the id of call 1. `## calls` shows a topic and a summary for call 1.
- **Call 2: the passphrase check.** Run:

  ```sh
  uv run python $C chat --run $RUN --name memory-2 --from +15555550142 \
    --say "Hi, it is Dana Reyes again." \
    --say "It is copper kettle." \
    --say "Please remind me what my last call was about." \
    --say "Thanks, that is all. Goodbye."
  ```

  The result of `[tool] identify_caller(` holds `Just to verify, may I have the short passphrase`. Before the next step, no `agent>` line names the data pipeline or Example Freight. Then `[tool] verify_passphrase(passphrase='copper kettle')` has a result that starts with `Say exactly these words to the caller, and nothing else: "Thank you, that matches.` and holds `Notes from Dana's earlier calls`. The `agent>` line after it holds `Last time you asked about` and the topic of call 1. That line is the proof of `memory-verified`; the third caller line is a probe, not a check.
- **State after call 2.** Run `uv run python $C rows --run $RUN | tee $RUN/memory-rows-2.txt`. `## phrase_calls` now has two rows for the same passphrase: call 1 and call 2.
- **Wrong passphrase (a third paid call).** Run call 2 again with `--name memory-wrong`, with `--say "It is wooden ladle."` two times in place of the right passphrase. The second `verify_passphrase` result holds `You have no notes of the caller's earlier calls`. No `agent>` line names the data pipeline or Example Freight. `rows` then shows `failed` = `2` for the passphrase row.
- **Proof.** `$RUN/chat-memory-1.txt`, `$RUN/chat-memory-2.txt`, `$RUN/memory-rows-1.txt`, `$RUN/memory-rows-2.txt`.

## Gotchas

- The comparison uses how the passphrase sounds, so `copper kettle` and `Copper Kettle.` are the same passphrase. Use a clearly different phrase for the wrong try.
- A passphrase must have 4 letters or more and cannot be the caller's name. With a shorter one the result holds `Not kept`, and the recipe proves nothing.
- A passphrase opens only the calls that were made with it. A call where the caller was only recognised stays closed for every passphrase. So call 2 names call 1 only when call 1 set the passphrase.
- Five wrong tries in a row, over all calls, lock the passphrase for good. Start a new run for a new attempt; do not reuse a run with a locked row.
- The passphrase is in the transcript file. That is accepted for private evidence. Never copy it to a public text.
- On 2026-11-22, on `anthropic:claude-haiku-5-5`, the first agent answered the third caller line of call 2 with "I can't share anything about other calls", after a right passphrase. The notes were in its context. That is a fault of the agent's answer, not of the passphrase check, and the owner knows it. Report what that turn gives; do not fail the recipe on it.
- The grade line of call 1 can show `failed: handoff_intro`: on the same run the contact agent gave no introduction line. That rule is not part of this feature.
- `leads saved this process` counts every lead in the run's database, so call 2 prints `1` with no `save_lead`.
- The formal check of this area is the skill `formal-verify-caller-memory`. It proves a property over every order of actions; this recipe proves that a model takes the path on a call.
