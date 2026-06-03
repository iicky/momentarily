"""Shared R2 (S3-compatible) client for Python training tools.

Credentials live in the murk vault at the repo root (`.murk`). The vault is
age-encrypted; `murk.get()` decrypts in-process using MURK_KEY from the
environment (set by your shell or `direnv` after `murk env`).

Run with: `python -m training.<tool>` from a shell where MURK_KEY is set,
or via `murk exec -- python -m training.<tool>` for ephemeral injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import boto3
import murk
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
    """Fetch a required key from the vault, raising if it's missing."""
    value = murk.get(key)
    if value is None:
        raise KeyError(f"{key} not found in murk vault")
    return value


def load_config() -> R2Config:
    """Read R2 credentials from the murk vault.

    Raises whatever murk raises if MURK_KEY is missing or a key isn't in the
    vault — those errors are clear enough we don't need to wrap them.
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
