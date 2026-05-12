"""Drift guards for package metadata.

The repo went through a period where pyproject said MIT, NOTICE said MIT, and
LICENSE was Apache 2.0. These checks fail loudly if it happens again.
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


def test_notice_describes_apache_not_mit() -> None:
    """NOTICE prose must match the LICENSE file's license."""
    notice = (REPO_ROOT / "NOTICE").read_text()
    assert "Apache License, Version 2.0" in notice or "Apache 2.0" in notice
    assert "MIT" not in notice, "NOTICE references MIT but the repo is Apache 2.0"


def test_readme_describes_apache_not_mit() -> None:
    """README's license section must agree with LICENSE."""
    readme = (REPO_ROOT / "README.md").read_text()
    assert "Apache License 2.0" in readme or "Apache 2.0" in readme
    # README may legitimately reference other projects' licenses in prose,
    # so don't blanket-ban MIT — just check our own license claim block.
    license_section_start = readme.find("## License")
    if license_section_start != -1:
        license_section = readme[license_section_start:]
        assert "Apache" in license_section
