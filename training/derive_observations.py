"""Derive one UTC day of recorded observations in R2 (the nightly step).

    murk exec -- uv run python -m training.derive_observations --day YYYY-MM-DD \\
        [--hours H-H] [--overwrite] [--report report.json]

Reads the recorder's ticks, bundles and blobs from the bucket and writes the
derived parquet files back under the same ``model/observations/s1/`` keys.
The arguments, JSON report, overwrite rule and exit status are those of
``python -m momentarily.model.observations.derive`` (docs/model-observations.md
§4.2); only the store differs. Credentials come from ``training.r2_client``.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

from momentarily.model.observations.derive.cli import add_arguments, execute
from training.r2_client import get_object_bytes, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class R2Storage:
    """The derive job's ``Storage`` over one R2 bucket."""

    def __init__(self, client: S3Client, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", []) if "Key" in o]
        return sorted(keys)

    def get(self, key: str) -> bytes:
        try:
            return get_object_bytes(self._client, self._bucket, key)
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise KeyError(key) from err
            raise

    def put(self, key: str, data: bytes) -> None:
        # The derive job writes only parquet files.
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType="application/vnd.apache.parquet",
        )

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.derive_observations")
    add_arguments(parser)
    args = parser.parse_args(argv)
    config = load_config()
    store = R2Storage(make_client(config), config.bucket)
    return execute(args, store, store)


if __name__ == "__main__":
    raise SystemExit(main())
