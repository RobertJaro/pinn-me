"""Portable environment and path resolution for inversion configuration."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re


_ENVIRONMENT_REFERENCE = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


class EnvironmentResolutionError(ValueError):
    """Raised when a required environment placeholder cannot be resolved."""


def expand_environment(
    value: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Expand ``${NAME}`` and shell-like ``${NAME:-default}`` placeholders.

    Unlike :func:`os.path.expandvars`, an unset variable without a default is a
    configuration error instead of silently remaining in the resulting path.
    Empty values use the default for the ``:-`` form, matching shell semantics.
    """

    environment = os.environ if environ is None else environ

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        resolved = environment.get(name)
        if resolved:
            return resolved
        if default is not None:
            return default
        raise EnvironmentResolutionError(
            f"environment variable {name!r} is required by configuration"
        )

    expanded = value
    # Permit an environment value or default to contain another placeholder,
    # while detecting accidental self-referential expansion.
    for _ in range(20):
        updated = _ENVIRONMENT_REFERENCE.sub(replace, expanded)
        if updated == expanded:
            break
        expanded = updated
    else:  # pragma: no cover - an intentionally pathological environment
        raise EnvironmentResolutionError("environment expansion exceeded 20 passes")

    if "${" in expanded:
        raise EnvironmentResolutionError(
            f"malformed or unresolved environment placeholder in {value!r}"
        )
    return expanded


def resolve_path(
    value: str,
    *,
    base_directory: Path,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one configured path without requiring the target to exist.

    Relative values are anchored to the directory containing the YAML file,
    never to the process working directory.
    """

    if not value.strip():
        raise ValueError("configured path must not be empty")
    expanded = expand_environment(value, environ=environ)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve(strict=False)


__all__ = ["EnvironmentResolutionError", "expand_environment", "resolve_path"]
