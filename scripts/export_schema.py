"""Export the Snapshot JSON Schema from the Pydantic model.

Pydantic (src/momentarily/schema.py) is the source of truth for the published
contract; the TypeScript Worker mirrors it by hand. Committing the generated
schema turns drift into a test failure — tests/test_parity.py regenerates and
diffs against the committed copy.

Run:  uv run python -m scripts.export_schema
"""

from __future__ import annotations

import json
from pathlib import Path

from momentarily.schema import Snapshot

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "snapshot.schema.json"


def render_schema() -> str:
    """The committed schema text: sorted keys + trailing newline so the diff
    is stable and reviewable."""
    schema = Snapshot.model_json_schema()
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(render_schema())
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
