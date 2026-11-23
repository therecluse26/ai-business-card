# Guards of the phone server

The phone server answers Telnyx, not a person. Its guards decide who can start a paid AI session: only a signed webhook from Telnyx can start a call, and only a media stream with a one-time token from this server gets audio.

## Sub-features

- `guard-health` the server says that it is up.
- `guard-webhook` a webhook with no Telnyx signature is refused.
- `guard-media` a media stream with an unknown token is refused before it opens.
- `guard-ringback` the ring tone that a caller hears during a transfer is served as a WAV file.

## How to get to it (user POV)

- Telnyx sends each call event to `POST /webhooks/telnyx`.
- Telnyx opens the audio of a call at `WS /media/<token>`.
- Telnyx fetches `GET /audio/ringback.wav` while the owner's phone rings.
- Fly.io and the deploy script read `GET /health`.

## Driving it with control.py

Preconditions:

- The baseline of the index holds.
- The server of this run is up: `uv run python $C server --run $RUN` printed `The server is up on http://127.0.0.1:8011`.
- `$RUN/server.out` has the line `PUBLIC_URL or TELNYX_APP_ID not set; the Telnyx webhook URL was not checked`. This line proves that the run did not touch the live phone line.

- **Health.** Run `uv run python $C http GET http://127.0.0.1:8011/health | tee $RUN/guard-health.txt`. The line ends with `-> HTTP 200, application/json, 11 bytes: {"ok":true}`.
- **Unsigned webhook.** Run `uv run python $C http POST http://127.0.0.1:8011/webhooks/telnyx --json '{"data":{"event_type":"call.initiated"}}' | tee $RUN/guard-webhook.txt`. The line ends with `-> HTTP 400`. `$RUN/server.out` has a new line `rejected webhook: Missing required header: Telnyx-Signature-Ed25519`.
- **Media stream with an unknown token.** Run `uv run python $C ws ws://127.0.0.1:8011/media/not-a-token | tee $RUN/guard-media.txt`. The line ends with `-> REFUSED before the stream opened: HTTP 403`. `$RUN/server.out` has a new line `media stream refused: its token is unknown, used, or too old`.
- **Ring tone.** Run `uv run python $C http GET http://127.0.0.1:8011/audio/ringback.wav | tee $RUN/guard-ringback.txt`. The line ends with `-> HTTP 200, audio/wav, 96044 bytes`.
- **Proof.** The four `guard-*.txt` files and `$RUN/server.out`.

## Gotchas

- The server closes a refused stream with WebSocket code 1008 before it accepts it, so a client sees HTTP 403, not a close code.
- A signed webhook cannot be made here: only Telnyx has the signing key. The accepted path of a webhook, and a media stream with a valid token, need a real phone call. Do not report them as verified by this recipe.
- A valid token comes only from the server's answer to a signed `call.initiated` webhook. There is no test endpoint that gives one, and none must be added for a proof.
- The size of the ring tone changes when `ringback_wav()` changes. A new size with status 200 and type `audio/wav` is not a failure; update this file.
