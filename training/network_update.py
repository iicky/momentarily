"""Keep the layer-1 day map and network builds current in R2 (the nightly step).

    murk exec -- uv run python -m training.network_update [--through YYYY-MM-DD]

Runs ``momentarily.model.network.update`` with the bucket as both the source
(recorder ticks, ``archive/gtfs/``) and the output (``model/network/s1/``).
The rules, the printed lines and the exit status are those of
``python -m momentarily.model.network update`` (docs/model-network.md); only
the store differs. Credentials come from ``training.r2_client``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from botocore.exceptions import ClientError

from momentarily.model.network.build import current_code
from momentarily.model.network.update import PublishConflictError, run, yesterday_utc
from training.derive_observations import R2Storage
from training.r2_client import load_config, make_client


class NetworkR2Storage(R2Storage):
    """``R2Storage`` writing JSON and gzip objects with their content types.

    ``update`` writes a build only where none exists, so a build is put with
    ``If-None-Match: *``: one that appeared since the check is a conflict
    rather than replaced."""

    def put(self, key: str, data: bytes) -> None:
        if not key.endswith(".json.gz"):
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType="application/json",
            )
            return
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType="application/gzip",
                IfNoneMatch="*",
            )
        except ClientError as err:
            code = err.response.get("Error", {}).get("Code")
            if code in ("PreconditionFailed", "412"):
                raise PublishConflictError([key]) from err
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.network_update")
    parser.add_argument("--through", help="last UTC day to add; default yesterday")
    args = parser.parse_args(argv)
    through: str = args.through or yesterday_utc(datetime.now(UTC))
    config = load_config()
    store = NetworkR2Storage(make_client(config), config.bucket)
    return run(store, store, through, current_code())


if __name__ == "__main__":
    raise SystemExit(main())
