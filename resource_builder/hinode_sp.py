"""Build the complete Hinode/SOT-SP resource namespace."""

from __future__ import annotations

from pathlib import Path

from ._shared import (
    FTS_SOURCE_ID,
    _copy_reviewed_resources,
    _solar_reference,
    _write_json,
)


REVIEWED_RESOURCES = (
    "hinode_sp/blend_inventory.json",
    "hinode_sp/instrument_hinode_sp.json",
)
GENERATED_RESOURCES = ("hinode_sp/solar_reference_630nm.json",)
SOLAR_REFERENCE_WINDOW_ANGSTROM = (6300.0, 6304.0)


def build(output: Path, payloads: dict[str, bytes], source_manifest: dict) -> None:
    """Build every resource owned by the Hinode/SP adapter."""

    _copy_reviewed_resources(output, source_manifest, REVIEWED_RESOURCES)
    _write_json(
        output / GENERATED_RESOURCES[0],
        _solar_reference(payloads[FTS_SOURCE_ID], *SOLAR_REFERENCE_WINDOW_ANGSTROM),
    )
