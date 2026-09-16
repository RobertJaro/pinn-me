"""Evaluation snapshots and scientific exports."""


def __getattr__(name):
    if name == "P3SLoader":
        from .loader import P3SLoader

        return P3SLoader
    if name in {"P3S_FORMAT", "P3S_VERSION"}:
        from . import checkpoint

        return getattr(checkpoint, name)
    if name == "export_save_state":
        from .export import export_save_state

        return export_save_state
    if name == "ArtifactExportError":
        from .errors import ArtifactExportError

        return ArtifactExportError
    raise AttributeError(name)


__all__ = [
    "P3SLoader",
    "P3S_FORMAT",
    "P3S_VERSION",
    "export_save_state",
    "ArtifactExportError",
]
