"""Small deterministic integrity primitives shared by persistence boundaries."""

from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | PathLike[str]) -> str:
    """Return the lowercase SHA-256 digest of one regular file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Integrity input is not a file: {source}.")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file_set(paths: Iterable[str | PathLike[str]]) -> str:
    """Hash the identities and complete contents of an ordered file set."""

    resolved = sorted({Path(path).expanduser().resolve() for path in paths})
    if not resolved:
        raise ValueError("An input-file signature requires at least one file.")
    digest = hashlib.sha256()
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"Integrity input is not a file: {path}.")
        encoded = str(path).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
        digest.update(path.stat().st_size.to_bytes(8, byteorder="big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


__all__ = ["sha256_file", "sha256_file_set"]
