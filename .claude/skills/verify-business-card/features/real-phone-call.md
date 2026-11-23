# A real phone call on the live line

The live line runs on Fly.io as the app `voice-business-card`. A real call is the only proof of what a caller hears: the voices, the handoffs, the timing, the grade after hangup, and the feedback that becomes a GitHub issue. Only the owner can make the call. An agent prepares it and reads the evidence.

## Sub-features

- `live-version` the server runs the commit that holds the change.
- `live-call` the owner's call takes the expected path.
- `live-grade` the grader scored the call after hangup.

## How to get to it (user POV)

- Dial the phone number of the line from a real phone, and speak.

## Driving it with control.py

Preconditions:

- The change is on `main` and the deploy run of `fly-deploy.yml` ended with success (`gh run list --workflow fly-deploy.yml --limit 1`).
- The owner agreed to make one call, and knows what to say. Give the owner the lines as a short numbered list.

- **Version.** Run `uv run python $C http GET https://voice-business-card.fly.dev/health`. The line ends with `-> HTTP 200`; this request also wakes the machine. Then read the version from the server: `nix shell nixpkgs#flyctl -c fly ssh console -a voice-business-card -C "python -c 'from ai_call_center.analytics import harness_version; print(harness_version())'"`. It must be the first 12 characters of the commit on `main`.
- **The call.** The owner calls and says the agreed lines. Wait for the owner's word that the call ended.
- **Evidence.** Read the end of the server log: `nix shell nixpkgs#flyctl -c fly ssh console -a voice-business-card -C "tail -n 400 /data/server.log"`. Find the lines of the call by its call id. The handles:
  - `caller identified (name given: True, earlier caller: <True|False>)`
  - `passphrase check (try <n>, verified: <True|False>)`
  - `the caller set a passphrase`
  - `handoff <agent> -> <agent> (leg=... voice=...) switched <n> ms after the tool call`
  - `grade (version <commit>): ...`
  - `feedback of call <id>: ...`
- **Proof.** The matching log lines, quoted in the report to the owner, with the times in US Eastern time.

## Gotchas

- The server log holds what callers said. It is private. Quote only the lines of the owner's own test call, and never put a log line in a commit, an issue or a pull request.
- The machine sleeps between calls. The first request after a sleep can take some seconds.
- A real call costs money (voice models, Telnyx minutes, the grader). One call by the owner is the budget; do not ask for a second call before the first one is read.
- A call from the owner's own number is not limited by the daily call limit, and the grader still scores it.
- Never send a webhook or a media stream to the live server as a test.
