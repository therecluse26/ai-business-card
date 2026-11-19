# Run book: the server on Fly.io

The server runs on one Fly.io machine that sleeps between calls. A call wakes it.

| Part | Where it is |
|---|---|
| The server and its packages | The container image (`Dockerfile`) |
| The embedding model for the writing search | The container image |
| The database, the saved feed, the server log | A 1 GB volume at `/data` |
| The keys and the owner's settings | Fly secrets, copied from `.env` |
| The public URL and the paths on the volume | `fly.toml` |

Measured on the laptop on 2026-11-19, in a container with one CPU and 2 GB:

- The image is 798 MB.
- A start with a filled volume takes 3.4 s. The server then uses 352 MiB.
- The first start on an empty volume takes about 40 s. It makes the search vectors for all posts, and
  it uses 1.1 GiB.

## What it costs

Prices from Fly's price page on 2026-11-18, for the region `iad`.

| Item | Price | For this line |
|---|---|---|
| The machine while it runs (`performance-1x`, 2 GB) | $33.00 a month if it never sleeps | About $0.046 an hour |
| The machine while it sleeps | $0.15 per GB of image a month | About $0.12 a month |
| The volume, 1 GB | $0.15 per GB a month | $0.15 a month |
| Volume snapshots | The first 10 GB are free | $0 |

At 2.5 running hours a month, the total is about $0.40.

The machine size is two lines in `fly.toml`. `shared-cpu-1x` with `1gb` costs $6.70 a month if it never
sleeps. Do not use it for the first start on an empty volume: that start needs 1.1 GiB.

## Before you start

**The phone line follows the server that started last.** At each start, the server sets the Telnyx
webhook URL to its own `PUBLIC_URL`. So the first deploy moves the phone line from the laptop to Fly.io.
See "Move the line" below.

**The database on Fly.io starts empty.** The caller memory on the laptop stays on the laptop.

## One-time setup

1. Get the `fly` command. On NixOS:

   ```sh
   nix shell nixpkgs#flyctl
   ```

2. Make an account or log in. Fly.io asks for a payment card.

   ```sh
   fly auth signup    # or: fly auth login
   ```

3. Choose the app name. It must be unique on Fly.io. If `voice-business-card` is taken, change two
   lines in `fly.toml`: `app` and `PUBLIC_URL`.

4. Make the app:

   ```sh
   fly apps create voice-business-card
   ```

5. Copy the settings from `.env` to Fly secrets. The script prints the names, never the values.

   ```sh
   bash scripts/fly-secrets.sh
   ```

   The script does not send `PUBLIC_URL`, the paths or the port. `fly.toml` and the image own those.
   The image also sets `SAVE_CALLER_AUDIO=false`: no caller audio is stored on the host.

6. Deploy. The first deploy also makes the volume.

   ```sh
   fly deploy --ha=false
   ```

   `--ha=false` makes one machine. Without it Fly.io makes two, and two machines cannot share the database.

## Check the first start

1. Read the log:

   ```sh
   fly logs
   ```

   These lines must be there. The first start takes about 40 s.

   - `writing: embedded ... new passages`
   - `voice providers at start: ...` with the same providers as on the laptop
   - `ready: webhooks at https://<app>.fly.dev/webhooks/telnyx`
   - `Telnyx webhook URL updated: https://<app>.fly.dev/webhooks/telnyx`

2. Restart once. This frees the memory that the first start used for the search vectors.

   ```sh
   fly apps restart voice-business-card
   ```

3. Check the address:

   ```sh
   curl https://<app>.fly.dev/health
   ```

   It must print `{"ok":true}`.

## Check the sleep

This check answers the open question: does the machine sleep by itself?

1. Do not call and do not open the address for 10 minutes.
2. Run `fly status`. The machine's state must be `suspended`.
3. Call the number. The call must be answered as fast as on the laptop.
4. Read `fly logs`. The call must have no `ERROR` line.

If the state stays `started` with no calls, something sends requests to the server. `fly logs` shows
each request with its path. Tell Claude what the log shows.

## Check the cost in the first week

- `fly machine status <machine id>` shows the last events. Many `start` events with no calls mean
  bots are waking the machine.
- The Fly dashboard, **Billing**, shows the cost of the month so far.
- 20 wakes a day with no call add about $2.30 a month on this machine size.

## Deploy a change

**A push to `main` deploys by itself.** GitHub Actions runs `.github/workflows/fly-deploy.yml`:

1. It runs the tests. A failed test stops the deploy.
2. It builds the new image. The running server is not touched.
3. It waits until the machine sleeps, so no call is in progress. It checks each 15 seconds.
4. It deploys.

The workflow runs only when the server's files change: `src/`, `Dockerfile`, `fly.toml`,
`pyproject.toml`, `uv.lock`, `profile.md`. A change to the docs or the tests deploys nothing.

If the machine does not sleep in 15 minutes, the workflow stops with an error. Nothing is deployed and
no call is cut. Start it again in GitHub: **Actions**, **Deploy to Fly.io**, **Run workflow**.

To deploy by hand, with the same wait:

```sh
nix shell nixpkgs#flyctl -c bash scripts/fly-deploy.sh
```

`SKIP_WAIT=1` before the command deploys at once. That cuts a call in progress.

Know these effects of each deploy:

- **The phone line moves to Fly.io.** If the laptop has the line for a test, a push to `main` takes it away.
- **The first call after a deploy can meet a cold start.** That is about 9 s. The machine sleeps about
  2 minutes later, and a wake from the sleep takes 0.26 s.
- **A call can still be cut in a gap of a few seconds,** between the last check and the machine's stop.
- **The settings are not deployed.** `.env` is not in the repository. After a change to `.env`, run
  `bash scripts/fly-secrets.sh`. It restarts the machine at once, with no wait for a call.

The workflow needs the repository secret `FLY_API_TOKEN`. To make a new one:

```sh
fly tokens create deploy -n github-actions | gh secret set FLY_API_TOKEN
```

## Move the line

- **To the laptop:** start the server on the laptop (`uv run serve`). It takes the line.
- **Back to Fly.io:** `fly apps restart voice-business-card`. The start on Fly.io takes the line.

Only one server has the line at one time. A sleeping machine does not take the line back by itself.

## Backups

- Fly.io takes one snapshot of the volume each day and keeps it for 5 days (`snapshot_retention` in `fly.toml`).
- Each lead is also sent to the owner by email at the moment the caller confirms it.
- At worst, one day of caller memory and of the call log is lost.

List the snapshots:

```sh
fly volumes list
fly volumes snapshots list <volume id>
```

Restore: make a new volume from a snapshot, then deploy. Fly's guide:
<https://fly.io/docs/volumes/snapshots/>.

Not built: a copy within seconds (Litestream to Cloudflare R2). The reasons are in the hosting notes
of 2026-11-19: it adds a second program that can stop the start, and the daily snapshot is enough for now.

## Not verified yet

Verified on 2026-11-19, on the real host:

- A real call runs on Fly.io: answer time 1.4 s (the median), 2 handoffs, no `ERROR` line.
- The machine suspends by itself. Fly's proxy suspended it 1 minute 49 seconds after a call ended.
- Fly's health check does not keep the machine awake.
- A cold start (after a deploy or a restart) takes about 9 s from the machine's start to `ready`.

- A call wakes a suspended machine with no error. The machine resumed in 0.26 s, and the greeting
  started 2.4 s after the call reached Fly.io. A call to the awake machine took 3.1 s for the same steps.

Still open:

- How many requests from bots wake the machine.
- The CPU that one call uses, and three calls at one time (`MAX_CONCURRENT_CALLS=3`).
