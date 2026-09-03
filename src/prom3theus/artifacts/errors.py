"""Errors raised at the artifact export boundary."""


class ArtifactExportError(ValueError):
    """Raised when an artifact cannot be exported without guessing."""


__all__ = ["ArtifactExportError"]
