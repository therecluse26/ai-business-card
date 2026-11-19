#!/usr/bin/env bash
# Deploy to Fly.io without cutting a call in progress. GitHub Actions runs this on a push to main
# (.github/workflows/fly-deploy.yml). You can run it by hand too:
#
#   nix shell nixpkgs#flyctl -c bash scripts/fly-deploy.sh
#
# A deploy stops the machine, and that would end a call. So the script deploys only when the machine
# sleeps. Fly's proxy suspends the machine about 2 minutes after the last traffic, and a call keeps its
# media stream open, so a sleeping machine has no call.
#
#   WAIT_SECS=900   how long to wait for the sleep (default: 15 minutes = the longest call, the idle
#                   time before the sleep, and a margin). After that the script stops, with no deploy.
#   SKIP_WAIT=1     deploy at once. This cuts a call in progress.
#
# What stays open: a call can arrive in the few seconds between the last check and the machine's stop.
set -euo pipefail
cd "$(dirname "$0")/.."

WAIT_SECS="${WAIT_SECS:-900}"
LABEL="${1:-manual-$(date -u +%Y%m%d%H%M%S)}"
APP="$(sed -n 's/^app = "\(.*\)".*/\1/p' fly.toml)"
IMAGE="registry.fly.io/$APP:$LABEL"

echo "== 1/4 Build the image $IMAGE. The running server is not touched."
flyctl deploy --build-only --push --image-label "$LABEL"

echo "== 2/4 Wait until no call is in progress."
states() {
    flyctl machine list --json | python3 -c 'import json, sys; print(" ".join(m["state"] for m in json.load(sys.stdin)))'
}
all_asleep() {
    local state
    for state in $1; do
        case "$state" in
            suspended | stopped) ;;
            *) return 1 ;;
        esac
    done
}
deadline=$(($(date +%s) + WAIT_SECS))
while :; do
    now="$(states)"
    if [ "${SKIP_WAIT:-}" = "1" ]; then
        echo "SKIP_WAIT=1: no wait. Machine state: ${now:-no machine}."
        break
    fi
    if all_asleep "$now"; then
        echo "Machine state: ${now:-no machine}. No call is in progress."
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "The machine did not sleep in ${WAIT_SECS} s (state: $now). Nothing was deployed, and no call was cut." >&2
        echo "Run the deploy again later. If no call is in progress, \`fly logs\` shows what sends requests." >&2
        exit 1
    fi
    echo "Machine state: $now. A call can be in progress. The next check is in 15 s."
    sleep 15
done

echo "== 3/4 Deploy $IMAGE."
flyctl deploy --image "$IMAGE" --ha=false

# A deploy leaves a machine that slept in the state "stopped". The next call would then wait for a full
# start of the server (about 9 s). So start it now: the server starts with no caller on the line, and
# Fly's proxy suspends it again after some minutes with no traffic. A wake from that sleep is fast.
echo "== 4/4 Start the machine, so the next call does not wait for the server's start."
flyctl machine list --json | python3 -c 'import json, sys; print("\n".join(m["id"] for m in json.load(sys.stdin) if m["state"] != "started"))' | while read -r id; do
    [ -n "$id" ] && flyctl machine start "$id"
done
