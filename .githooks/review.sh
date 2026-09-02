#!/usr/bin/env bash
# Codex adversarial pre-commit review — tracked. Invoked from the top of
# .githooks/pre-commit, before the lint/typecheck gates (see that file's header).
#
# Behavior: runs Codex as an adversarial reviewer over the STAGED diff and
# blocks the commit ONLY on an explicit `VERDICT: FAIL`.
#
# Escape hatches:
#   git commit --no-verify              # skip all hooks
#   CODEX_REVIEW_SKIP=1 git commit ...  # skip just this Codex gate
#
# Tunables:
#   CODEX_REVIEW_MODEL=<model>          # override Codex model (default: codex's)
#   CODEX_REVIEW_EFFORT=<effort>        # reasoning effort, default medium
#                                       # (low|medium|high|xhigh — pinned so the
#                                       #  gate does NOT inherit a slow global xhigh)
#   CODEX_REVIEW_TIMEOUT=<seconds>      # default 300
#
# Fail-OPEN by design: any infra problem (codex missing / not authed / error /
# timeout / no clear verdict) ALLOWS the commit. Only a real FAIL verdict blocks.

set -uo pipefail

[ "${CODEX_REVIEW_SKIP:-}" = "1" ] && exit 0

if ! command -v codex >/dev/null 2>&1; then
  echo "codex-review: 'codex' not on PATH — skipping (fail-open)." >&2
  exit 0
fi

# Only the staged changes are being committed; review exactly those.
DIFF="$(git diff --cached --no-color 2>/dev/null)"
[ -z "$DIFF" ] && exit 0   # nothing staged (e.g. merge commit) — nothing to review

# Bounded so the gate can never hang the full historical 5 min: this account is
# locked to gpt-5.5 (no faster model accepted), so low effort + a tight timeout
# are the only speed levers. Fail-open means a review that overruns just skips.
TIMEOUT="${CODEX_REVIEW_TIMEOUT:-180}"
EFFORT="${CODEX_REVIEW_EFFORT:-low}"

# Pin reasoning effort so the gate doesn't inherit a slow global xhigh.
CODEX_ARGS=(--skip-git-repo-check -c sandbox_mode="read-only" -c model_reasoning_effort="$EFFORT")
[ -n "${CODEX_REVIEW_MODEL:-}" ] && CODEX_ARGS+=(-m "$CODEX_REVIEW_MODEL")

read -r -d '' PROMPT <<'EOF'
You are an adversarial code reviewer acting as a BLOCKING pre-commit gate.
The staged git diff for this commit is provided in the <stdin> block below.

Review ONLY the staged changes. Hunt for real, blocking problems:
correctness bugs, security holes, data-loss / corruption risks, broken
invariants, leaked secrets or credentials, and clear footguns. Be concrete,
skeptical, and terse. Do NOT flag pure style/formatting — linters/formatters handle that.

Decision rule:
- If the staged changes are safe to commit, make the LAST line of your reply
  EXACTLY:  VERDICT: PASS
- If there is at least one real blocking problem, briefly explain each one,
  then make the LAST line of your reply EXACTLY:  VERDICT: FAIL

The verdict line must be the final line of your output.
EOF

TMP_DIFF="$(mktemp)"; TMP_OUT="$(mktemp)"; TMP_ERR="$(mktemp)"
trap 'rm -f "$TMP_DIFF" "$TMP_OUT" "$TMP_ERR"' EXIT
printf '%s' "$DIFF" > "$TMP_DIFF"

echo "codex-review: adversarial review of staged changes (timeout ${TIMEOUT}s)…" >&2

# stdout carries the reviewer's message and ENDS with the verdict line; codex's
# own telemetry ("tokens used", counts) goes to stderr. Keep them apart: merging
# them (2>&1) leaves trailing telemetry after the verdict, which makes it
# impossible to require the verdict be the final line — and a gate that cannot
# check that has to accept an early PASS followed by truncated or noisy output.
run_codex() {
  codex exec "${CODEX_ARGS[@]}" "$PROMPT" < "$TMP_DIFF" > "$TMP_OUT" 2> "$TMP_ERR"
}

# Portable timeout: prefer gtimeout/timeout, else a bash watchdog.
# (codex's own redirections are applied by this shell, so the timeout binary
#  just needs the bare command.)
RC=0
TBIN=""
command -v gtimeout >/dev/null 2>&1 && TBIN="gtimeout"
[ -z "$TBIN" ] && command -v timeout >/dev/null 2>&1 && TBIN="timeout"
if [ -n "$TBIN" ]; then
  "$TBIN" "$TIMEOUT" codex exec "${CODEX_ARGS[@]}" "$PROMPT" < "$TMP_DIFF" > "$TMP_OUT" 2> "$TMP_ERR"; RC=$?
else
  run_codex & CMD_PID=$!
  ( sleep "$TIMEOUT"; kill -TERM "$CMD_PID" 2>/dev/null ) & WATCH_PID=$!
  wait "$CMD_PID" 2>/dev/null; RC=$?
  kill -TERM "$WATCH_PID" 2>/dev/null; wait "$WATCH_PID" 2>/dev/null || true
fi

OUT="$(cat "$TMP_OUT" 2>/dev/null)"
ERR="$(cat "$TMP_ERR" 2>/dev/null)"

if [ "$RC" -ne 0 ]; then
  echo "codex-review: codex exec failed/timed out (rc=$RC) — skipping (fail-open)." >&2
  printf '%s\n' "$ERR" | tail -3 >&2
  exit 0
fi

# The prompt requires the verdict to be the FINAL line, so that is what gets
# checked. Two asymmetric rules, because the two mistakes are not equally bad:
#
#   - A FAIL anywhere in the message blocks, final line or not. Missing a real
#     failure is the one outcome this gate exists to prevent.
#   - A PASS is honoured ONLY as the last non-empty line, exactly. An early PASS
#     followed by caveats or a truncated response is not a clean pass, and is
#     treated as no verdict rather than silently accepted as one.
#
# Anything unparseable still fails OPEN, per this script's documented design: a
# flaky reviewer must not become a wall that trains everyone to use --no-verify.
LAST_LINE="$(printf '%s\n' "$OUT" | grep -v '^[[:space:]]*$' | tail -1)"
VERDICT=""
if printf '%s\n' "$OUT" | grep -Eq '^[[:space:]]*VERDICT: FAIL[[:space:]]*$'; then
  VERDICT="FAIL"
elif [ "$(printf '%s' "$LAST_LINE" | tr -d '[:space:]')" = "VERDICT:PASS" ]; then
  VERDICT="PASS"
fi

case "$VERDICT" in
  FAIL)
    {
      echo
      echo "──────────────────────────────────────────────────────"
      echo "  Codex adversarial review: ✗ BLOCKED this commit"
      echo "──────────────────────────────────────────────────────"
      printf '%s\n' "$OUT"
      echo "──────────────────────────────────────────────────────"
      echo "Bypass once:  git commit --no-verify    (or fix & re-commit)"
    } >&2
    exit 1
    ;;
  PASS)
    echo "codex-review: ✓ PASS" >&2
    exit 0
    ;;
  *)
    echo "codex-review: no clear final verdict from codex — skipping (fail-open)." >&2
    printf '%s\n' "$OUT" | tail -3 >&2
    exit 0
    ;;
esac
