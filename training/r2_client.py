"""Shared R2 (S3-compatible) client for Python training tools.

Credentials come from the process environment first, then the murk vault.
Locally, `murk exec -- python -m training.<tool>` injects them from the
age-encrypted `.murk` vault (decrypted in-process via MURK_KEY). In the
Cloudflare trainer container there is no vault — the Worker passes R2_* as
plain env vars at container start, and those take precedence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config

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
