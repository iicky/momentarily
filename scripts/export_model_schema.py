"""Export the data-model JSON Schemas (schema/model/s1/) from Pydantic.

Pydantic (src/momentarily/model/) is the source of truth. The TypeScript types
in shared/model/s1/ are generated from these files by
worker/scripts/model_types.ts, so the chain is Pydantic -> JSON Schema -> TS
with no hand-written mirror. tests/test_model_network.py fails on a stale
schema; worker/test/model_network.test.ts fails on stale generated TS.

Run:  uv run python -m scripts.export_model_schema
      (cd worker && bun run gen:model)
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from momentarily.model.network.contracts import (
    DayFeedMap,
    NetworkIndex,
    NetworkSnapshot,
)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema" / "model" / "s1"

DOCUMENTS: dict[str, type[BaseModel]] = {
    "network": NetworkSnapshot,
    "network_index": NetworkIndex,
    "day_feed_map": DayFeedMap,
}


def render_schemas() -> dict[str, str]:
    """File name -> committed text: sorted keys + trailing newline."""
    return {
        f"{name}.schema.json": json.dumps(
            model.model_json_schema(), indent=2, sort_keys=True
        )
        + "\n"
        for name, model in DOCUMENTS.items()
    }


def main() -> int:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    rendered = render_schemas()
    for stale in SCHEMA_DIR.glob("*.schema.json"):
        if stale.name not in rendered:
            stale.unlink()
    for name, text in rendered.items():
        (SCHEMA_DIR / name).write_text(text)
        print(f"wrote {SCHEMA_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
