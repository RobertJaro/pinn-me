"""Versioned model artifacts and export utilities."""

from typing import TYPE_CHECKING, Any

from .model import ARTIFACT_SCHEMA_VERSION, ArtifactManifest

if TYPE_CHECKING:
    from .checkpoint import (
        P3S_FORMAT,
        P3S_VERSION,
    )
    from .errors import ArtifactExportError
    from .export import export_artifact
    from .loader import P3SLoader


def __getattr__(name: str) -> Any:
    if name == "P3SLoader":
        from .loader import P3SLoader

        return P3SLoader
    if name in {
        "P3S_FORMAT",
        "P3S_VERSION",
    }:
        from .checkpoint import (
            P3S_FORMAT,
            P3S_VERSION,
        )

        return {
            "P3S_FORMAT": P3S_FORMAT,
            "P3S_VERSION": P3S_VERSION,
        }[name]
    if name == "ArtifactExportError":
        from .errors import ArtifactExportError

        return ArtifactExportError
    if name == "export_artifact":
        from .export import export_artifact

        return export_artifact
    raise AttributeError(name)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactExportError",
    "ArtifactManifest",
    "P3S_FORMAT",
    "P3SLoader",
    "P3S_VERSION",
    "export_artifact",
]
