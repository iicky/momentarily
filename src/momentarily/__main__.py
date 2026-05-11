"""CLI entrypoint: fetch → derive → publish.

Invoked by GitHub Actions cron (and locally for manual runs / development).
Reads configuration from environment variables; fails loudly if anything is
missing.
"""

from __future__ import annotations

import logging
import os
import sys

_LOGGER = logging.getLogger("momentarily")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    missing = [
        name
        for name in (
            "MTA_API_KEY",
            "R2_ACCOUNT_ID",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET",
        )
        if not os.environ.get(name)
    ]
    if missing:
        _LOGGER.error("Missing required env vars: %s", ", ".join(missing))
        return 2

    _LOGGER.info("Momentarily publisher: not yet wired up to live feeds")
    _LOGGER.info("Wire fetch → derive → publish here once R2 + MTA key are ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
