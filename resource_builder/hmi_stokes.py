"""Build the complete SDO/HMI Stokes resource namespace."""

from __future__ import annotations

from pathlib import Path

from ._shared import (
    FTS_SOURCE_ID,
    _copy_reviewed_resources,
    _solar_reference,
    _write_json,
)


REVIEWED_RESOURCES = ("hmi_stokes/instrument_hmi.json",)
GENERATED_RESOURCES = ("hmi_stokes/solar_reference_617nm.json",)
SOLAR_REFERENCE_WINDOW_ANGSTROM = (6172.5, 6174.2)


def build(output: Path, payloads: dict[str, bytes], source_manifest: dict) -> None:
    """Build every resource owned by the HMI Stokes adapter."""

    _copy_reviewed_resources(output, source_manifest, REVIEWED_RESOURCES)
    _write_json(
        output / GENERATED_RESOURCES[0],
        _solar_reference(payloads[FTS_SOURCE_ID], *SOLAR_REFERENCE_WINDOW_ANGSTROM),
    )
