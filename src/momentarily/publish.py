"""Reference-only: R2 upload via boto3.

The live publish path is the TypeScript Cloudflare Worker (see bead c72.8) using
the R2 binding, not boto3. This module is kept as a comparison reference while
the Worker is being built; remove once c72.8 ships.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config

from momentarily.schema import Snapshot

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


# Default cache-control. Tune per object once the publisher is running live.
DEFAULT_CACHE_CONTROL = "public, max-age=60, s-maxage=300"


def s3_client(
    *,
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
) -> S3Client:
    """Build a boto3 S3 client targeted at the user's R2 endpoint."""
    return boto3.client(  # pyright: ignore[reportUnknownMemberType]
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def upload_snapshot(
    snapshot: Snapshot,
    *,
    client: S3Client,
    bucket: str,
    key: str = "v1/snapshot.json",
    cache_control: str = DEFAULT_CACHE_CONTROL,
) -> None:
    """Serialize the snapshot to JSON and PUT it to the configured R2 bucket+key."""
    body = json.dumps(snapshot.model_dump(mode="json"), separators=(",", ":")).encode(
        "utf-8"
    )
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        CacheControl=cache_control,
    )
