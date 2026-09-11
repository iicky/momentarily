"""Drift guards for package metadata.

The repo went through a period where pyproject said MIT while LICENSE was
Apache 2.0. These two structural checks fail loudly if it happens again.
Whether NOTICE or README *describe* the license correctly is a wording
question, not one a test can enforce without also failing on a legitimate
rewrite of that prose.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_license_file_is_apache_2() -> None:
    """LICENSE must be Apache 2.0 — sets the intent the other places agree with."""
    license_text = (REPO_ROOT / "LICENSE").read_text()
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text


def test_pyproject_license_points_at_license_file() -> None:
    """pyproject must reference the LICENSE file so it can't drift textually."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    license_spec = pyproject["project"]["license"]
    assert license_spec == {"file": "LICENSE"}, (
        f"pyproject license must be {{ file = 'LICENSE' }}, got {license_spec!r}"
    )
