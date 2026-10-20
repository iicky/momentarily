"""The model package runs from its built wheel, with no checkout on the path.

`pyproject.toml` packages only `src/momentarily`, so an import of a top-level
repo module (`training`, `collector`, `scripts`) passes every in-checkout test
and still fails for anyone who installs the wheel. This builds the wheel,
installs it in a fresh venv outside the checkout, and runs the network CLI.

uv runs with `--offline`: the build backend and the runtime dependencies come
from uv's cache, which `uv sync` fills. With a cold cache the test is skipped
rather than reaching for the network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Heavy optional dependencies that must stay out of the model's import path.
# The wheel installs them, so only sys.modules can show a stray import.
_PROBE = """
import sys
from importlib.resources import files
import momentarily
import momentarily.model.network.build
import momentarily.model.network.day_feed
import momentarily.model.observations.columns
import momentarily.model.observations.contracts
import momentarily.model.observations.decode
import momentarily.model.observations.keys
import momentarily.model.observations.legacy.alerts
import momentarily.model.observations.legacy.alerts_liveness
import momentarily.model.observations.legacy.common
import momentarily.model.observations.legacy.ene
import momentarily.model.observations.legacy.fresh_feeds
import momentarily.model.observations.legacy.health
import momentarily.model.observations.legacy.predictions
import momentarily.model.observations.legacy.trace
decode = files("momentarily.model.observations.decode")
assert decode.joinpath("nyct-subway.proto").is_file()
assert "site-packages" in momentarily.__file__, momentarily.__file__
stray = sorted(m for m in ("boto3", "botocore", "pyarrow", "training") if m in sys.modules)
assert not stray, stray
"""

# Phrases uv prints when an offline run needs something its cache lacks.
_COLD_CACHE = ("not found in the cache", "network connectivity is disabled")


def _uv(uv: str, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [uv, *args, "--offline"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if any(p in proc.stderr for p in _COLD_CACHE):
            pytest.skip(f"uv cache cannot satisfy an offline {args[0]}")
        pytest.fail(f"uv {' '.join(args)} failed:\n{proc.stderr}")
    return proc


def test_network_layer_runs_from_installed_wheel(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    dist = tmp_path / "dist"
    _uv(uv, "build", "--wheel", "--out-dir", str(dist), str(REPO), cwd=REPO)
    (wheel,) = dist.glob("momentarily-*.whl")

    venv = tmp_path / "venv"
    _uv(uv, "venv", "--python", sys.executable, str(venv), cwd=tmp_path)
    python = venv / "bin" / "python"
    _uv(uv, "pip", "install", "--python", str(python), str(wheel), cwd=tmp_path)

    # A clean environment: no PYTHONPATH or VIRTUAL_ENV leaking the checkout in.
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    for argv in (
        [str(python), "-m", "momentarily.model.network", "--help"],
        [str(python), "-c", _PROBE],
    ):
        proc = subprocess.run(
            argv, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr
