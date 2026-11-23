# Ask to speak to the owner

A caller who asks, in their own words, to speak to the owner now gets a live transfer: the owner's phone rings, and the call moves to it. A caller who only wants a follow-up gets a message, and no phone rings.

## Sub-features

- `transfer-asked` the caller's own words start the transfer: `request_owner_transfer` runs.
- `transfer-note` the owner gets a record that says who asked and why, before the transfer.
- `transfer-not-asked` a caller who wants to leave a message is never transferred.

## How to get to it (user POV)

- Call the line, give your name, and say "I want to speak to the owner now" (with the owner's name or "the owner").
- In a text call: the same words, typed. No phone rings: the text call records the transfer.

## Driving it with control.py

Preconditions:

- The baseline of the index holds, and the owner said yes to one text call.
- `doctor` shows no `FAIL`: the safe settings turn off Jev and allow a transfer at any hour.

- **The call.** Run:

  ```sh
  uv run python $C chat --run $RUN --name transfer \
    --say "Hi, this is Dana Reyes from Example Freight." \
    --say "I want to speak to the owner right now, please. It is about a contract role." \
    --say "Yes, please connect me now."
  ```

- **Transfer.** The transcript has `[tool] request_owner_transfer(` with a `reason`. One result starts with `Transfer to the owner requested (`. The chat then prints `[call transferred to <the owner's number>; hanging up]` and ends by itself.
- **Note for the owner.** Run `uv run python $C rows --run $RUN | tee $RUN/transfer-rows.txt`. `## leads` has one row for the call with `Dana Reyes`.
- **Proof.** `$RUN/chat-transfer.txt` and `$RUN/transfer-rows.txt`.

## Gotchas

- The first `request_owner_transfer` result can be a question for the caller ("connect now, or take a message?"). The third `--say` line answers it. Without that line the run can end with no transfer; that is not a failure of the product.
- The safe settings turn off Jev, so the word list alone reads the caller's request. A phone call on the live line uses Jev too. Report the proof as "word list only".
- The safe settings empty `TRANSFER_HOURS`. The refusal outside the transfer hours is not covered here.
- No phone rings and no call moves in a text call. Call screening, voicemail detection and the return to the agents need a real phone call.
- The transcript holds the owner's real phone number. Keep it out of every public text.
