"""Shared R2 (S3-compatible) client for Python training tools.

Credentials come from the process environment first, then the murk vault.
Locally, `murk exec -- python -m training.<tool>` injects them from the
age-encrypted `.murk` vault (decrypted in-process via MURK_KEY). In the
Cloudflare trainer container there is no vault — the Worker passes R2_* as
plain env vars at container start, and those take precedence.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


def _require(key: str) -> str:
    """Fetch a required key from the environment, falling back to the vault."""
    value = os.environ.get(key)
    if value is None:
        # murk is a local-only dep — import lazily so the trainer container
        # (which always has R2_* in the environment) never needs it installed.
        import murk

        value = murk.get(key)
    if value is None:
        raise KeyError(f"{key} not in environment or murk vault")
    return value


def load_config() -> R2Config:
    """Read R2 credentials from the environment or the murk vault.

    Raises whatever murk raises if a key is absent from both and MURK_KEY is
    missing — those errors are clear enough we don't need to wrap them.
    """
    return R2Config(
        account_id=_require("R2_ACCOUNT_ID"),
        access_key_id=_require("R2_ACCESS_KEY_ID"),
        secret_access_key=_require("R2_SECRET_ACCESS_KEY"),
        bucket=_require("R2_BUCKET"),
    )


def make_client(config: R2Config | None = None) -> S3Client:
    """Build a boto3 S3 client targeting Cloudflare R2."""
    cfg = config or load_config()
    return boto3.client(  # pyright: ignore[reportUnknownMemberType]
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


# Under concurrent GETs, R2 intermittently answers with NoSuchKey (or
# MethodNotAllowed) for an object that exists — HEAD and the listing both return
# its size and etag, and an immediate retry succeeds. botocore's retry mode
# treats those codes as definitive client errors, so they are never retried for
# us. Swallowing them instead would silently drop real data from an eval window.
_SPURIOUS_GET_CODES = frozenset({"NoSuchKey", "MethodNotAllowed", "404", "405"})


def get_object_bytes(
    client: S3Client, bucket: str, key: str, *, attempts: int = 5
) -> bytes:
    """GET an object body, retrying R2's spurious not-found answers.

    A key that is genuinely absent still raises, after the retries are spent.
    """
    for attempt in range(attempts):
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in _SPURIOUS_GET_CODES or attempt == attempts - 1:
                raise
            time.sleep(0.1 * 2**attempt)
    raise AssertionError("unreachable")
