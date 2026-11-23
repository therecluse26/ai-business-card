---
name: formal-verify-caller-memory
description: Check, over every order of caller actions, that caller memory never gives one person's data to another person. Covers the passphrase check, the reuse of earlier-call data and the CRM lookup - `passphrase.py`, `recall` in `callers.py`, `identify`, `verify`, `set_phrase`, `save_lead_record`, `forget` in `graph/tools.py`, and `request_status`, `sync_call` in `crm.py`. Use it before a commit that changes one of those, the fields of `CallContext` they use, or the tables `leads`, `calls`, `caller_phrases`, `phrase_calls`, `crm_calls`; and when someone asks "can a faked number get the notes?".
---

# Formal check of caller memory

**The property.** On a call, the agents and a new lead get no data that another person left on an earlier call.

**The harm when it breaks.** A person who fakes a phone number, or who shares a phone with the real caller, hears or reuses the real caller's project notes, email address or company, or hears where the real caller's request stands in the owner's CRM.

**The check.** `check.py`, next to this file, is a bounded exhaustive search. It runs the real functions on a real SQLite store, for every order of caller actions over three calls, and looks for a secret word of one person in what the code gives the agents on a call of the other person. No test framework and no extra tool is needed: Python and the project's own packages.

Why this tool and not a TLA+ model: the question is "can the steps happen in a bad order", which a model checker answers. But a model is a hand-made copy, and it goes stale when the code changes. Here each step is a small Python function with no network, so the search runs the code itself. A change to the code is checked with no change to a model.

## Does your change touch the verified code?

The coverage record. The check runs these, and nothing else of the product:

| File | What the check runs |
|---|---|
| `src/ai_call_center/graph/tools.py` | `identify`, `verify`, `set_phrase`, `save_lead_record`, `forget`, and the texts `NO_NOTES`, `PHRASE_ASKED`, `PHRASE_CLOSED` |
| `src/ai_call_center/passphrase.py` | `PhraseStore` (`find`, `check`, `add_call`, `set`, `forget`), `sound_key`, `usable`, `_hash`, `TRIES`, `LOCK_AT` |
| `src/ai_call_center/callers.py` | `recall`, `history_text`, `UNKNOWN_NUMBERS` |
| `src/ai_call_center/leads.py` | `LeadStore`: `save`, `leads_for`, `lead_for_call`, `start_call`, `finish_call`, `calls_for`, `forget` |
| `src/ai_call_center/crm.py` | `request_status` (the CRM lookup), `sync_call`, `entry_for_call`, `write_waiting` and the write steps under it (`_find_or_make_person`, `_fill_person`, `_open_deal`), `same_crm_person`, `CrmStore` |
| `src/ai_call_center/names.py` | `same_person` |
| `src/ai_call_center/graph/state.py` | the fields of `CallContext` in `CTX_FIELDS` of `check.py` |
| `src/ai_call_center/agents/specs.py` | `Welcome.returning_caller`, `Welcome.new_caller`, `passphrase_question`, `passphrase_right`, `passphrase_read_back` |

**Sync commit: `57e4b45`.** At this commit the check was confirmed to describe the code above. The search and `--mutants` passed on its code on 2026-11-23.

```sh
git diff --stat 57e4b45 -- src/ai_call_center/graph/tools.py src/ai_call_center/crm.py src/ai_call_center/passphrase.py src/ai_call_center/callers.py src/ai_call_center/leads.py src/ai_call_center/names.py src/ai_call_center/graph/state.py src/ai_call_center/agents/specs.py
```

No output: the verified code is the same, and the last result still holds. Any output: run the check, then read "When a change is intended" below. Move the sync commit forward only after you confirmed that `check.py` still describes the code at the new commit: its actions, its `CTX_FIELDS`, its `TABLES`, its `PLACES` and its `MUTANTS`.

## Run it

From the repository root:

```sh
uv run python .claude/skills/formal-verify-caller-memory/check.py            # the search: about 90 seconds
uv run python .claude/skills/formal-verify-caller-memory/check.py --mutants  # can the search fail? about 10 minutes
uv run python .claude/skills/formal-verify-caller-memory/check.py --calls 2  # a quick look: about 5 seconds, a weaker result
```

The exit code is 0 for a pass and 1 for a fail. The tool versions are fixed by `uv.lock` and by `requires-python` in `pyproject.toml`. The search makes no network request and costs no money. It changes no file: its store is in memory.

Run the search after each change to the files in the coverage record. Run `--mutants` after each change to `check.py`, and when a mutant is reported as `STALE`.

## Read the result

The search prints one block for each finding, with the shortest list of steps that gives it, in the words of the code. The last line is the verdict.

| Line | Meaning | What to do |
|---|---|---|
| `PASS: no data of one person reached a call of the other person.` | The property holds inside the bounds. | Nothing. |
| `PASS, with the known findings above. No new finding.` | Only faults that are known and reported. It shows only when `KNOWN_FINDINGS` has an entry; see "Known findings". | Nothing new. Do not call the property proven for the kind of data in a known finding. |
| `NEW FINDING: a <kind> of the other person is in the <place>.` | The property is broken, by the steps shown. | Decide: real bug or modeling error (next section). |
| `A KNOWN FINDING NO LONGER HAPPENS` | The code changed and a known fault is gone. | When the fix was intended, remove that entry from `KNOWN_FINDINGS` in `check.py` and from "Known findings" here. |
| `NOT REACHED, so the search is too weak: ...` | The search never reached a state where data is released. A pass would mean nothing. | A change made a path dead, or an action of `check.py` no longer fits the code. Fix `check.py`, not the list of witnesses. |
| `The code wrote to [...], which the search does not snapshot.` | The code now writes a table that the search does not carry between states. | Add the table and its columns to `TABLES`. |
| `STALE` (with `--mutants`) | The text that a mutant replaces is no longer in the code. | Update that entry of `MUTANTS` to the new code, so that it still removes the same check. |
| `NOT CAUGHT` (with `--mutants`) | The search passes on broken code. | The search lost power. Find what `check.py` no longer drives or no longer watches. |

A kind is `note word` (a message, an email address, a company, a call summary or a topic), `surname`, or `passphrase word`. A place is `tool result` (the text that a tool gives back to the voice model), `caller_history` (the notes in the call context), `caller_name` (the name that every agent gets in its handoff message, `known_name_note` in `voice/call_session.py`), `new lead` (the lead that a save writes on this call), or `CRM record` (a record of the owner's CRM that `crm_lookup` read on this call; the agent gets its status and stage).

## A real bug, or a modeling error?

Read the steps of the finding and answer in this order:

1. **Can a caller do each step on a phone call?** Each step is one tool call that the voice model can make in that state, or a hangup. If a step is impossible in the product (not only unlikely), the search is wrong: fix `actions` or `apply` in `check.py`.
2. **Is the word in that place a release of data to that person?** The place must be something an agent reads or a row that the call stores. If the word is in the place by design, and the design was decided by the owner, the property text here is wrong: change it with the owner, never alone.
3. **Otherwise it is a real bug.** Report it with the steps as printed. Do not change `KNOWN_FINDINGS`, the secret words, the bounds or the places to get a pass. Do not change product code unless the owner asks for the fix.

An unlikely order of steps is still a real bug: a person with a faked number chooses the order.

## When a change is intended

| The change | What to update in `check.py` |
|---|---|
| A new tool, or a new caller action, that reads or writes caller memory | A new entry in `WORDS`, and a branch in `World.apply` that calls the real function |
| A new field of `CallContext` that this code reads or writes | `CTX_FIELDS` (a field that is missing there is lost between two steps of the search) |
| A new table, or a new column that this code writes | `TABLES` |
| A new way for earlier-call data to reach an agent (a new prompt note, a new tool result) | `PLACES`, and the dictionary `places` at the end of `World.apply` |
| A check was moved or reworded | The matching entry of `MUTANTS`; run `--mutants` |
| A fix of a known finding | Remove it from `KNOWN_FINDINGS` and from "Known findings" below |

Then run the search and `--mutants`, and move the sync commit in this file.

`MUTANTS` has eight broken copies of the code, and the search must fail on each one. Two copies are left out on purpose: `save_lead_record` without its `ctx.identity_verified` test, and `request_status` without its `ctx.identity_verified` test. Each test is a second layer. Only `verify` fills `ctx.open_calls`, and only after a right passphrase, so the `open_calls` filter holds the property alone. If a change lets other code fill `ctx.open_calls`, those tests become the only guard: add the broken copies to `MUTANTS` then.

## What a pass covers, and what it does not

Inside these bounds, a pass covers every order of actions:

- **Three calls** from one phone number (`--calls 4` is deeper and takes much longer). Two people. The second person is, in one run, a person with the same first name and another surname (a shared phone); and in a second run, a person who gives the real caller's full name (a faked number).
- **The actions of a call:** give the name (`identify_caller`); say the right passphrase, a wrong one, or none (`verify_passphrase`); set a passphrase (`set_passphrase`, two times for the read-back); leave a message in three forms (`save_lead` with a new email address, with the email "on file", with "on file" and no name); ask where their request stands (`crm_lookup`); ask to be forgotten (`forget_caller`); hang up. A hangup runs the real CRM write (`sync_call`).
- **At most one message on a call** (`SAVES_IN_A_CALL`).

The assumptions:

- Neither person says the other person's passphrase. Two passphrases that sound the same (`sound_key`) are not covered.
- Each tool call is one step. That is true for `identify`, `verify`, `set_phrase` and `forget`: they have no `await`. `save_lead_record` waits for the owner's email after it writes the lead; a second tool call during that wait is not covered.
- The summary of a call holds only what that caller said on that call. The summary model does not run.
- Each call is about a project, so each call that is not forgotten leaves a lead: a person and a deal in the CRM. A call with no message and no talk about work leaves nothing in the CRM, and is not covered; the lookup has nothing to read for it.
- The CRM is a stand-in with no memory (`SearchCrm`): the ID of a record is its content. It always answers, each deal is open, and it has no person that the owner added by hand. So these paths of the write are not covered: an entry that waits, a person found by the CRM search, a person found by an email address, a closed deal, and what `_fill_person` sends to a person that exists.
- Three settings differ from the product, to make the search finite and fast: the cost of the passphrase hash (`_SCRYPT`), the lock limit (`LOCK_AT` is 2, not 5), and the clock (one second for each record).

Not covered at all:

- The voice handlers and the text tools that wrap these functions (`voice/call_session.py`, the `@tool` functions), and the return of a call after a transfer to the owner (`telephony/transfers.py` copies the passphrase fields into the resumed call).
- What a model says. The check proves what data the code gives an agent, not what the agent then does with it.
- The email to the owner, the server log, the traces, the fault report and the GitHub issue.
- What the owner sees in the CRM. A person with a faked number who gives the real caller's full name gets their entry on the real caller's person, marked "No passphrase check, so it is not proof". That is a record for the owner, not a release to a caller.
- A caller with a hidden number.
- That "forget me" deletes everything. The search runs `forget` and looks for leaks after it; it does not check what stays in the store.
- The number of passphrase guesses that a person gets.

## Known findings

None is open. `KNOWN_FINDINGS` in `check.py` is empty, so the search passes only with no finding at all.

One finding is closed. Keep it here: it shows what a real finding of this check looks like.

**Closed on 2026-11-22: the surname of an earlier caller reached the agents after a name match, before any passphrase check.**

1. The real caller called, and gave the name "Sam Carter". The call ended.
2. A second person called from the same number, and gave the name "Sam Pike", or only "Sam".
3. `identify` matched the two names by the first name, and set `ctx.caller_name = memory.name`: "Sam Carter".

Every agent then read "The caller's name is Sam Carter" in its handoff message, and the second person's call was stored under that name. The fix: `identify` sets `ctx.caller_name` to the name that the caller said on this call. `verify` sets the name on record, and only after a right passphrase. One broken copy in `MUTANTS` puts the old line back, and the search must catch it.

## A second target, when this one is done

`telephony/transfers.py`: the live transfer to the owner (call screening, voicemail detection, the return of the caller to the agents) reacts to Telnyx webhooks that can arrive in any order. A promise such as "a caller is never left with no agent and no owner" is a question about the order of events, so a TLA+ model fits it. If someone builds that model, read every result with the pstack `/statechart` skill.
