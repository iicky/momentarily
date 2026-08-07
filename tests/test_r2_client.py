"""Spurious-not-found retry in get_object_bytes (training/r2_client.py).

Fake S3 clients — no R2 access.
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from training.r2_client import get_object_bytes


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeClient:
    """Fails the first `fail_times` GETs with `code`, then serves the payload."""

    def __init__(self, *, code: str, fail_times: int, payload: bytes = b"ok") -> None:
        self._code = code
        self._fail_times = fail_times
        self._payload = payload
        self.calls = 0

    def get_object(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise ClientError({"Error": {"Code": self._code}}, "GetObject")
        return {"Body": _Body(self._payload)}


@pytest.mark.parametrize("code", ["NoSuchKey", "MethodNotAllowed", "404", "405"])
def test_retries_spurious_not_found(code: str) -> None:
    client = _FakeClient(code=code, fail_times=2)
    assert get_object_bytes(client, "b", "k", attempts=5) == b"ok"  # pyright: ignore[reportArgumentType]
    assert client.calls == 3


def test_raises_when_retries_are_exhausted() -> None:
    """A genuinely absent key must still surface, not read as empty."""
    client = _FakeClient(code="NoSuchKey", fail_times=99)
    with pytest.raises(ClientError):
        get_object_bytes(client, "b", "k", attempts=3)  # pyright: ignore[reportArgumentType]
    assert client.calls == 3


def test_does_not_retry_other_errors() -> None:
    client = _FakeClient(code="AccessDenied", fail_times=1)
    with pytest.raises(ClientError):
        get_object_bytes(client, "b", "k", attempts=5)  # pyright: ignore[reportArgumentType]
    assert client.calls == 1


def test_returns_immediately_when_the_first_get_succeeds() -> None:
    client = _FakeClient(code="NoSuchKey", fail_times=0, payload=b"payload")
    assert get_object_bytes(client, "b", "k") == b"payload"  # pyright: ignore[reportArgumentType]
    assert client.calls == 1
