# Leave a message

A caller who wants a follow-up gives a name and one line on what they want. The contact agent says the message back, the caller says yes, and the message is stored as a lead and sent to the owner.

## Sub-features

- `message-route` the first agent hands the caller to the contact agent.
- `message-confirm` the contact agent says the message back before it saves it.
- `message-save` `save_lead` stores one lead for the call, with the caller's name and message.
- `message-once` a second `save_lead` with nothing new stores no second lead.
- `message-offer` after the save, the agent offers a passphrase for the next call.

## How to get to it (user POV)

- Call the line, give your name, and say that you want to leave a message.
- Say what you want and, if you like, an email address. Say yes when the agent says the message back.
- In a text call: the same words, typed.

## Driving it with control.py

Preconditions:

- The baseline of the index holds, and the owner said yes to one text call.
- `uv run python $C rows --run $RUN` shows no lead.

- **The call.** Run:

  ```sh
  uv run python $C chat --run $RUN --name message \
    --say "Hi, this is Dana Reyes." \
    --say "I would like to leave a message for the owner." \
    --say "My message is: I need help with a data pipeline for my company, Example Freight. My email is dana@example.test." \
    --say "Yes, that is right." \
    --say "No thanks, no passphrase." \
    --say "No, that is all. Goodbye."
  ```

- **Route.** The transcript has `[tool] identify_caller(name='Dana Reyes')` and `[tool] transfer_to_contact()`, and then the prompt `caller (contact)> `.
- **Confirm, then save.** A `[tool] save_lead(` line with `name='Dana Reyes'` comes after the caller's yes, not before it. Its result starts with `[result] Lead saved as record 1.` and says `do not say an email was sent`, because the safe settings record the email.
- **Offer.** The same result holds the offer: `I can remember your details for next time`.
- **Stored state.** Run `uv run python $C rows --run $RUN | tee $RUN/message-rows.txt`. `## leads` has exactly one row, with `Dana Reyes`, `dana@example.test`, `Example Freight` and the caller's number. `## calls` has one row with `ended` = `1`.
- **End of the call.** The transcript ends with `[hung up] active agent: contact; leads saved this process: 1` and `[1 lead email(s) recorded, not sent:`.
- **Proof.** `$RUN/chat-message.txt` and `$RUN/message-rows.txt`.

## Gotchas

- The agent can ask its questions in another order, or save after the third line. Check the order of the `[tool]` lines against the caller's yes; do not check line numbers.
- The email is recorded, not sent. This recipe does not prove that an email reaches the owner.
- When the agent calls `end_call`, the chat ends before the last `--say` line is typed. The harness then stops; this is not a failure.
- `leads saved this process` counts every lead in the run's database, not only this call's lead.
