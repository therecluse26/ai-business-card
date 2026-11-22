#!/usr/bin/env bash
# Copy the settings in .env to the Fly.io app as secrets. Run it again after you change .env.
# It prints the names it sends, never the values.
#
# Not sent: the settings that fly.toml and the Dockerfile own (the public URL, the paths, the port),
# and empty values.
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP="PUBLIC_URL HOST PORT LEADS_DB PROFILE_PATH SERVER_LOG SAVE_CALLER_AUDIO CALLER_AUDIO_DIR WRITING_CACHE_PATH WRITING_MODEL_DIR VOICE_MODEL_DIR HARNESS_VERSION"

uv run python - "$SKIP" <<'PY' | fly secrets import "$@"
import sys
from dotenv import dotenv_values

skip = set(sys.argv[1].split())
sent = []
for name, value in dotenv_values(".env").items():
    if name in skip or not value:
        continue
    if "\n" in value:
        sys.exit(f"{name} has more than one line; set it by hand with `fly secrets set`")
    print(f"{name}={value}")
    sent.append(name)
print(f"sending {len(sent)} settings: {' '.join(sent)}", file=sys.stderr)
PY
