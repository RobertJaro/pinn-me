"""Build common LTE atomic metadata and the STiC thermodynamic lookup."""

from __future__ import annotations

from pathlib import Path

from ._shared import _copy_reviewed_resources, _generate_stic_table, _write_json


REVIEWED_RESOURCES = (
    "common/abundances.json",
    "common/lines.json",
)
GENERATED_RESOURCES = ("common/stic_continuum_table.json",)


def build(output: Path, payloads: dict[str, bytes], source_manifest: dict) -> dict:
    """Build and return the shared atomic and thermodynamic resources."""

    _copy_reviewed_resources(output, source_manifest, REVIEWED_RESOURCES)
    table = _generate_stic_table(payloads)
    _write_json(output / GENERATED_RESOURCES[0], table, compact=True)
    return table
