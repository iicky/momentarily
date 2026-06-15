"""Code-provenance stamp for published artifacts.

Every artifact the offline tooling writes (params.json, calibration.json,
eval.json) carries a `provenance` block answering "which code produced this":

    {"code_sha": "<git commit>", "dirty": bool|None, "producer": str}

`code_sha` is git's own commit hash, embedded verbatim — not a hash we choose.
(Content/lineage hashing we control uses BLAKE3; that lives elsewhere.)

Resolution order, so local dev, CI, and the trainer container all work:
  1. MOMENTARILY_CODE_SHA env var — authoritative; injected at build/CI time.
  2. .build-sha file next to the package — the container path, since the image
     excludes .git (so `git rev-parse` can't run there). Written by the deploy
     script and COPYed into the image.
  3. `git rev-parse HEAD` — local dev, where the working tree is present.
  4. "unknown" — nothing else resolved.

`dirty` is only known when computed from a live git tree (step 3) or passed via
MOMENTARILY_CODE_DIRTY; otherwise None (a clean-checkout build leaves it unset
rather than asserting a value it can't verify).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TypedDict

_BUILD_SHA_FILE = Path(__file__).resolve().parent.parent / ".build-sha"


class Provenance(TypedDict):
    code_sha: str
    dirty: bool | None
    producer: str


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _env_dirty() -> bool | None:
    raw = os.environ.get("MOMENTARILY_CODE_DIRTY")
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes"}


def code_provenance() -> Provenance:
    """Resolve the code provenance of the current process."""
    producer = os.environ.get("MOMENTARILY_PRODUCER") or "local"

    sha = os.environ.get("MOMENTARILY_CODE_SHA")
    if sha:
        return {"code_sha": sha.strip(), "dirty": _env_dirty(), "producer": producer}

    if _BUILD_SHA_FILE.is_file():
        try:
            file_sha = _BUILD_SHA_FILE.read_text().strip()
        except OSError:
            file_sha = ""
        if file_sha:
            return {"code_sha": file_sha, "dirty": _env_dirty(), "producer": producer}

    git_sha = _git("rev-parse", "HEAD")
    if git_sha:
        status = _git("status", "--porcelain")
        dirty = bool(status) if status is not None else _env_dirty()
        return {"code_sha": git_sha, "dirty": dirty, "producer": producer}

    return {"code_sha": "unknown", "dirty": _env_dirty(), "producer": producer}
