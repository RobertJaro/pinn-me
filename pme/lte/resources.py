"""Prepared, reusable LTE runtime-resource bundles."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


REQUIRED_HINODE_PRODUCTION_FILES = {
    "abundances.json",
    "blend_inventory.json",
    "instrument_hinode_sp.json",
    "lines.json",
    "solar_reference_630nm.json",
    "stic_continuum_table.json",
}
OPTIONAL_LTE_REFERENCE_FILES = {
    "barklem_collet_2016_ReadMe.txt",
    "barklem_collet_2016_table4.dat",
    "barklem_collet_2016_table8.dat",
    "eos_table.json",
    "hminus_continuum.json",
    "partition_functions.json",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resource_bundle(directory) -> dict:
    """Validate a prepared offline bundle and return checkpoint-safe metadata."""

    root = Path(directory).expanduser().resolve()
    bundle_path = root / "bundle.json"
    manifest_path = root / "sources.json"
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared LTE resources are missing in {root}. Run "
            "pinn-me-lte-fetch-data before starting training."
        )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1 or bundle.get("bundle_type") != (
        "pinn-me-lte-runtime-resources"
    ):
        raise RuntimeError(f"Unsupported LTE resource bundle in {bundle_path}.")
    manifest_digest = _file_sha256(manifest_path)
    if manifest_digest != bundle.get("source_manifest_sha256"):
        raise RuntimeError(
            "Prepared LTE source manifest does not match bundle.json; rerun preparation."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported LTE source manifest in {manifest_path}.")
    file_records = {}
    for group in ("vendored_files", "generated_files", "reviewed_files"):
        records = manifest.get(group)
        if not isinstance(records, dict):
            raise RuntimeError(f"LTE source manifest has no valid {group!r} inventory.")
        overlap = set(file_records).intersection(records)
        if overlap:
            raise RuntimeError(f"LTE source manifest repeats files: {sorted(overlap)}")
        file_records.update(records)

    missing_records = sorted(REQUIRED_HINODE_PRODUCTION_FILES - set(file_records))
    if missing_records:
        repair = (
            " Rebuild it by running "
            f"`pinn-me-lte-fetch-data --output-dir {root}`."
        )
        raise RuntimeError(
            f"Prepared LTE bundle is from an incomplete runtime contract; manifest "
            f"does not record {missing_records}." + repair
        )

    runtime_files = bundle.get("runtime_files")
    if not isinstance(runtime_files, list) or not runtime_files:
        raise RuntimeError("Prepared LTE bundle has no runtime file inventory.")
    missing = [name for name in runtime_files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared LTE bundle is incomplete: {missing}")
    expected_runtime_files = {*file_records, "sources.json"}
    if set(runtime_files) != expected_runtime_files:
        raise RuntimeError(
            "LTE bundle runtime inventory does not match its source manifest."
        )
    if set(bundle.get("required_production_files", ())) != (
        REQUIRED_HINODE_PRODUCTION_FILES
    ):
        raise RuntimeError(
            "LTE bundle does not declare the complete production resource contract."
        )
    declared_reference_files = set(bundle.get("optional_reference_files", ()))
    recorded_reference_files = OPTIONAL_LTE_REFERENCE_FILES.intersection(file_records)
    if declared_reference_files != recorded_reference_files:
        raise RuntimeError(
            "LTE bundle optional-reference inventory disagrees with its source manifest."
        )
    for filename, record in file_records.items():
        expected_digest = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(expected_digest, str):
            raise RuntimeError(f"LTE resource {filename!r} has no SHA256 digest.")
        actual_digest = _file_sha256(root / filename)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"LTE resource {filename!r} failed SHA256 verification: "
                f"expected {expected_digest}, got {actual_digest}."
            )
    stic_table = json.loads(
        (root / "stic_continuum_table.json").read_text(encoding="utf-8")
    )
    solar_reference = json.loads(
        (root / "solar_reference_630nm.json").read_text(encoding="utf-8")
    )
    if solar_reference.get("schema_version") != 1 or solar_reference.get(
        "units", {}
    ).get("continuum_radiance") != "W m^-3 sr^-1":
        raise RuntimeError("Prepared LTE bundle has no valid absolute solar reference.")
    continuum_contract = stic_table.get("continuum_contract")
    top_boundary = stic_table.get("falc_top_boundary")
    reference_solver = stic_table.get("reference_solver", {})
    axes = stic_table.get("axes", {})
    if stic_table.get("schema_version") != 2 or not isinstance(
        continuum_contract, dict
    ):
        raise RuntimeError("Prepared LTE bundle has no valid STiC continuum contract.")
    required_stic_thermodynamics = (
        "log10_mass_density_kg_m3",
        "log10_electron_density_m3",
        "log10_neutral_hydrogen_density_m3",
        "log10_fe_i_population_over_partition_m3",
    )
    if any(
        not isinstance(stic_table.get(field), list)
        for field in required_stic_thermodynamics
    ) or not isinstance(
        continuum_contract.get("thermodynamic_population_contract"), str
    ):
        raise RuntimeError(
            "Prepared LTE bundle lacks coherent pinned-STiC thermodynamic populations."
        )
    if continuum_contract.get("physical_height_qualified") is not True or (
        continuum_contract.get("runtime_fallback")
        != "none; missing or invalid STiC table is a hard error"
    ):
        raise RuntimeError(
            "Prepared LTE bundle is not qualified for physical geometric height."
        )
    if reference_solver.get("commit") != (
        "18cda77d038a97f007a783dcb61ea9a9a1244bf7"
    ) or reference_solver.get("runtime_iterations") != 0:
        raise RuntimeError(
            "Prepared LTE bundle is not the pinned offline-only STiC reference."
        )
    try:
        temperature_axis = [float(value) for value in axes["log10_temperature_k"]]
        pressure_axis = [float(value) for value in axes["log10_gas_pressure_pa"]]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Prepared LTE bundle has invalid STiC thermodynamic axes.") from error
    if (
        len(temperature_axis) < 2
        or len(pressure_axis) < 2
        or not all(math.isfinite(value) for value in (*temperature_axis, *pressure_axis))
        or any(right <= left for left, right in zip(temperature_axis, temperature_axis[1:]))
        or any(right <= left for left, right in zip(pressure_axis, pressure_axis[1:]))
    ):
        raise RuntimeError("Prepared LTE bundle has invalid STiC thermodynamic axes.")
    try:
        top_pressure_pa = float(top_boundary["gas_pressure_pa"])
        gravity_cm_s2 = float(top_boundary["gravity_cm_s2"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Prepared LTE bundle lacks the pinned FALC/STiC top-boundary "
            "pressure and gravity."
        ) from error
    if (
        not isinstance(top_boundary, dict)
        or top_boundary.get("target_log10_tau500") != -5.0
        or not math.isfinite(top_pressure_pa)
        or top_pressure_pa <= 0.0
        or not math.isfinite(gravity_cm_s2)
        or gravity_cm_s2 <= 0.0
    ):
        raise RuntimeError(
            "Prepared LTE bundle lacks a finite, positive pinned FALC/STiC "
            "top-boundary pressure and gravity."
        )
    if bundle.get("physical_depth_contract") != continuum_contract or (
        bundle.get("falc_top_boundary") != top_boundary
    ):
        raise RuntimeError(
            "Prepared LTE bundle metadata disagrees with its verified STiC table."
        )
    return {
        "directory": str(root),
        "bundle_schema_version": 1,
        "bundle_type": bundle["bundle_type"],
        "instrument": bundle.get("instrument"),
        "source_manifest_sha256": manifest_digest,
        "resource_sha256": {
            filename: record["sha256"] for filename, record in file_records.items()
        },
        "training_contract": bundle.get("training_contract"),
        "required_production_files": sorted(REQUIRED_HINODE_PRODUCTION_FILES),
        "optional_reference_files": sorted(recorded_reference_files),
        "physical_depth_contract": continuum_contract,
        "falc_top_boundary": top_boundary,
        "stic_lookup_bounds": {
            "temperature_log10_k": [temperature_axis[0], temperature_axis[-1]],
            "gas_pressure_log10_pa": [pressure_axis[0], pressure_axis[-1]],
        },
        "solar_reference": {
            "file": "solar_reference_630nm.json",
            "source": solar_reference.get("source"),
            "units": solar_reference.get("units"),
            "wavelength_range_air_angstrom": solar_reference.get(
                "wavelength_range_air_angstrom"
            ),
            "limb_darkening": solar_reference.get("limb_darkening"),
        },
    }


__all__ = [
    "REQUIRED_HINODE_PRODUCTION_FILES",
    "OPTIONAL_LTE_REFERENCE_FILES",
    "validate_resource_bundle",
]
