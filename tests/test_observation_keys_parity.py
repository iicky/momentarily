"""Cross-language guard for the observations key layout.

The Worker recorder formats its R2 keys in worker/src/observation_keys.ts;
the offline derive job finds them through
src/momentarily/model/observations/keys.py. The committed fixture
(tests/fixtures/parity_observation_keys.json) pins both to the same strings.
This guards the Python side; worker/test/recorder.test.ts replays the same
fixture against the TypeScript side.
"""

from __future__ import annotations

from scripts.gen_observation_keys_fixture import FIXTURE_PATH, render


def test_fixture_is_current() -> None:
    """If this fails, run: uv run python -m scripts.gen_observation_keys_fixture"""
    assert FIXTURE_PATH.read_text() == render()
