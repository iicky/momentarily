"""Export the PublishedRouteShapes JSON Schema from the Pydantic model.

Same pattern as export_schema.py — Pydantic is the source of truth; the
committed JSON turns drift into a test failure.

Run:  uv run python -m scripts.export_route_shapes_schema
"""

from __future__ import annotations

import json
from pathlib import Path

from momentarily.schema import PublishedRouteShapes

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schema" / "route_shapes.schema.json"
)


def render_schema() -> str:
    """The committed schema text: sorted keys + trailing newline so the diff
    is stable and reviewable."""
    schema = PublishedRouteShapes.model_json_schema()
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(render_schema())
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
