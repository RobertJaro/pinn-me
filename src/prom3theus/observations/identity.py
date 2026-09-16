"""Preparation implementation identity, independent of runtime read settings."""

import hashlib
import importlib.metadata
import sys
from pathlib import Path


def implementation_signature():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    # Only preparation dependencies: trainer/plot edits cannot alter arrays.
    paths = set()
    for package in ("observations", "instruments", "config", "core", "preprocess"):
        paths.update((root / package).rglob("*.py"))
    paths.update(
        root / name
        for name in (
            "components/observations.py",
            "application/joint_contracts.py",
            "rt/geometry.py",
            "rt/units.py",
        )
        if (root / name).exists()
    )
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(str(sys.version_info[:3]).encode())
    for package in ("torch", "numpy", "pytorch-lightning"):
        digest.update(importlib.metadata.version(package).encode())
    return digest.hexdigest()
