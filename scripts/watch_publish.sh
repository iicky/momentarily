#!/usr/bin/env bash
# Poll the live snapshot until the Worker has published the expected commit.
# Success is keyed off provenance.code_sha matching the deployed commit, not a
# freshness age — a missing/zero timestamp must never read as "ancient" and
# trip a false rollback.
set -euo pipefail

EXPECTED_SHA="${1:-$(git rev-parse HEAD)}"
URL="${2:-https://feed.momentarily.nyc/v1/snapshot.json}"
TIMEOUT_SECS="${WATCH_TIMEOUT_SECS:-420}"   # ~7 min
INTERVAL_SECS="${WATCH_INTERVAL_SECS:-20}"

short() { printf '%.7s' "$1"; }
deadline=$(( $(date +%s) + TIMEOUT_SECS ))
iter=0

while :; do
  iter=$((iter + 1))

  body="$(curl -fsS --max-time 15 "$URL" 2>/dev/null || true)"
  if [ -z "$body" ]; then
    echo "iter $iter: snapshot unreachable — retrying"
  else
    # Parse code_sha + generated_at in one pass; emit "sha<TAB>age_or_NA".
    read -r live_sha age < <(printf '%s' "$body" | python3 -c '
import json, sys, time
try:
    d = json.load(sys.stdin)
except Exception:
    print("PARSE_ERR NA"); sys.exit()
sha = (d.get("provenance") or {}).get("code_sha") or "unknown"
g = d.get("generated_at")
age = str(int(time.time()) - int(g)) if isinstance(g, (int, float)) and g > 0 else "NA"
print(sha, age)
')
    if [ "$live_sha" = "PARSE_ERR" ]; then
      echo "iter $iter: snapshot did not parse — retrying"
    elif [ "$(short "$live_sha")" = "$(short "$EXPECTED_SHA")" ]; then
      echo "PUBLISHED: $(short "$EXPECTED_SHA") is live (generated ${age}s ago)"
      exit 0
    else
      echo "iter $iter: live=$(short "$live_sha") want=$(short "$EXPECTED_SHA") (age ${age}s) — waiting"
    fi
  fi

  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "TIMEOUT: $(short "$EXPECTED_SHA") did not publish within ${TIMEOUT_SECS}s (last live=${live_sha:-none})"
    exit 1
  fi
  sleep "$INTERVAL_SECS"
done
