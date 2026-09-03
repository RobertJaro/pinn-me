"""Versioned model artifacts and export utilities."""

from typing import TYPE_CHECKING, Any

from .model import ARTIFACT_SCHEMA_VERSION, ArtifactManifest

if TYPE_CHECKING:
    from .errors import ArtifactExportError
    from .export import export_artifact


def __getattr__(name: str) -> Any:
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
    "export_artifact",
]
