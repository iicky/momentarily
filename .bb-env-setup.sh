#!/usr/bin/env bash
# Provision a fresh worktree so it can read the committed .murk vault.
#
# murk locates its age identity by hashing the vault's absolute path, so every
# worktree looks in a key-store slot of its own and only the checkout where
# `murk init` ran has one. Without this step every secret read in a new worktree
# fails with "MURK_KEY not set", and the viz Models tab, the training tools, and
# every wrangler script are unusable.
#
# murk-wt hardlinks the primary checkout's key into this worktree's slot. Same
# inode, so re-keying propagates with no resync. After it runs, plain `murk get`
# and `murk exec` work with no repo-local .env, no MURK_KEY_FILE export, and no
# direnv hook.
#
# Every step is non-fatal: bb deletes the worktree if this script exits non-zero.

set -uo pipefail

warn() { echo "warning: $*" >&2; }

# Hooks inherit a sanitized environment, so don't trust PATH alone.
murk_wt=$(command -v murk-wt || echo "$HOME/.local/bin/murk-wt")
if [ ! -f .murk ]; then
    :
elif [ -x "$murk_wt" ]; then
    "$murk_wt" link || warn "murk-wt link failed — secrets unavailable until you run it by hand"
else
    warn "murk-wt not on PATH — this worktree cannot decrypt .murk"
fi

if command -v uv >/dev/null 2>&1; then
    uv sync || warn "uv sync failed — run it by hand before using 'uv run'"
fi

# Point git at the tracked hooks (adversarial review, tracker-leak guard, and
# the scoped lint/typecheck gate). Idempotent; .githooks/pre-commit chains on to
# any tool-managed hook when one is present. Without this a fresh clone commits
# with no gate at all.
if git rev-parse --git-dir >/dev/null 2>&1; then
    git config core.hooksPath .githooks || warn "could not set core.hooksPath — pre-commit gate inactive"
fi

exit 0
