"""Where the derive job reads raw objects and writes derived files.

A store holds objects under the R2 key layout of ``keys``. The derive job
needs four operations: list the keys under a prefix, and read, write or
delete one key. ``LocalStorage`` maps keys to paths under a directory (a local
mirror of the bucket, or a test fixture); the R2 store lives in ``training``
so the model package never imports boto3.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def list(self, prefix: str) -> list[str]:
        """Every key that starts with ``prefix``, sorted."""
        ...

    def get(self, key: str) -> bytes:
        """The bytes of ``key``. Raises ``KeyError`` when it does not exist."""
        ...

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``, replacing what was there."""
        ...

    def delete(self, key: str) -> None:
        """Remove ``key``; nothing happens when it does not exist."""
        ...


class LocalStorage:
    """Keys as paths under ``root``: key ``a/b/c`` is the file ``root/a/b/c``.
    Files whose name starts with ``.`` are temporary and never listed."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        parts = key.split("/")
        if not key or key.startswith("/") or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"{key!r} is not an object key")
        return self.root.joinpath(*parts)

    def list(self, prefix: str) -> list[str]:
        head, _, _ = prefix.rpartition("/")
        base = self.root.joinpath(*head.split("/")) if head else self.root
        if not base.is_dir():
            return []
        found: list[str] = []
        for dirpath, _, files in os.walk(base):
            for name in files:
                if name.startswith("."):
                    continue
                key = (Path(dirpath) / name).relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    found.append(key)
        return sorted(found)

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as err:
            raise KeyError(key) from err

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
