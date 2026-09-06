"""Build common LTE atomic metadata and the STiC thermodynamic lookup."""

from __future__ import annotations

from pathlib import Path

from ._shared import (
    _copy_reviewed_resources,
    _generate_falc_reference_atmosphere,
    _generate_chianti_eos_table,
    _generate_stic_table,
    _write_json,
)


REVIEWED_RESOURCES = (
    "common/abundances.json",
    "common/lines.json",
)
GENERATED_RESOURCES = (
    "common/stic_continuum_table.json",
    "common/chianti_thermodynamic_table.json",
    "common/falc_reference_atmosphere.json",
)


def build(output: Path, payloads: dict[str, bytes], source_manifest: dict) -> dict:
    """Build and return the shared atomic and thermodynamic resources."""

    _copy_reviewed_resources(output, source_manifest, REVIEWED_RESOURCES)
    stic_table = _generate_stic_table(payloads)
    chianti_eos_table = _generate_chianti_eos_table(payloads)
    falc_reference = _generate_falc_reference_atmosphere(payloads)
    _write_json(output / GENERATED_RESOURCES[0], stic_table, compact=True)
    _write_json(output / GENERATED_RESOURCES[1], chianti_eos_table)
    _write_json(output / GENERATED_RESOURCES[2], falc_reference)
    return stic_table
