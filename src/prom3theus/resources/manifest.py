"""Integrity and scientific-contract checks for packaged LTE resources."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from importlib.resources import files
import json
import math
from pathlib import PurePosixPath


BUNDLE_TYPE = "prom3theus-lte-runtime-resources"
SUPPORTED_INSTRUMENTS = ("hinode_sp", "hmi_stokes")
TRAINING_RESOURCE_CONTRACT = "offline-read-only; no downloads or table conversion"
REQUIRED_COMMON_PRODUCTION_FILES = frozenset(
    {
        "common/abundances.json",
        "common/lines.json",
        "common/stic_continuum_table.json",
    }
)
HINODE_PRODUCTION_FILES = frozenset(
    {
        "hinode_sp/blend_inventory.json",
        "hinode_sp/instrument_hinode_sp.json",
        "hinode_sp/solar_reference_630nm.json",
    }
)
HMI_PRODUCTION_FILES = frozenset(
    {
        "hmi_stokes/instrument_hmi.json",
        "hmi_stokes/solar_reference_617nm.json",
    }
)
REQUIRED_HINODE_PRODUCTION_FILES = (
    REQUIRED_COMMON_PRODUCTION_FILES | HINODE_PRODUCTION_FILES
)
REQUIRED_HMI_PRODUCTION_FILES = REQUIRED_COMMON_PRODUCTION_FILES | HMI_PRODUCTION_FILES
HMI_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM = (6160.0, 6170.0, 6180.0, 6190.0)
HINODE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM = (6290.0, 6300.0, 6310.0, 6320.0)
REFERENCE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM = (5000.0,)
HMI_STIC_REQUIRED_DOMAIN_ANGSTROM = (6160.0, 6190.0)
HINODE_STIC_REQUIRED_DOMAIN_ANGSTROM = (6290.0, 6320.0)


def resource_path(resource_name: str | None = None):
    """Return the bundle root or one safe bundle-relative resource path."""

    root = files("prom3theus.resources").joinpath("data")
    if resource_name is None:
        return root
    if not isinstance(resource_name, str) or not resource_name or "\\" in resource_name:
        raise ValueError(
            "A resource name must be a safe non-empty relative POSIX path."
        )
    relative = PurePosixPath(resource_name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(
            "A resource name must be a safe non-empty relative POSIX path."
        )
    return root.joinpath(*relative.parts)


def _open_binary(resource):
    return resource.open("rb") if hasattr(resource, "open") else open(resource, "rb")


def _open_text(resource):
    return (
        resource.open("r", encoding="utf-8")
        if hasattr(resource, "open")
        else open(resource, encoding="utf-8")
    )


def file_sha256(resource) -> str:
    """Calculate a resource SHA256 digest without loading it all into memory."""

    digest = hashlib.sha256()
    with _open_binary(resource) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_resource(
    resource, manifest: Mapping, *, resource_name: str, kind: str
) -> str:
    """Verify one scientific resource against a pinned source manifest."""

    resource_path(resource_name)
    matches = [
        manifest.get(section, {}).get(resource_name)
        for section in ("vendored_files", "generated_files", "reviewed_files")
        if isinstance(manifest.get(section), Mapping)
        and manifest[section].get(resource_name) is not None
    ]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise RuntimeError(
            f"{kind} resource {resource_name!r} is not recorded exactly once in the "
            "LTE source manifest."
        )
    expected = matches[0].get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise RuntimeError(
            f"{kind} resource {resource_name!r} has no valid SHA256 digest."
        )
    actual = file_sha256(resource)
    if actual != expected:
        raise RuntimeError(
            f"{kind} resource {resource_name!r} failed SHA256 verification: "
            f"expected {expected}, got {actual}."
        )
    return actual


def _load_json(resource) -> dict:
    with _open_text(resource) as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise RuntimeError(f"Expected a JSON object in {resource}.")
    return document


def _packaged_file_names(root, prefix: PurePosixPath = PurePosixPath()) -> set[str]:
    names: set[str] = set()
    for child in root.iterdir():
        relative = prefix / child.name
        if child.is_file():
            names.add(relative.as_posix())
        elif child.is_dir():
            names.update(_packaged_file_names(child, relative))
    return names


def load_verified_source_manifest() -> dict:
    """Load the source manifest after checking its bundle-pinned digest."""

    root = resource_path()
    bundle_resource = root.joinpath("bundle.json")
    manifest_resource = root.joinpath("sources.json")
    if not bundle_resource.is_file() or not manifest_resource.is_file():
        raise FileNotFoundError(f"LTE resource bundle is incomplete in {root}.")
    bundle = _load_json(bundle_resource)
    if bundle.get("schema_version") != 2 or bundle.get("bundle_type") != BUNDLE_TYPE:
        raise RuntimeError(f"Unsupported LTE resource bundle in {bundle_resource}.")
    expected = bundle.get("source_manifest_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise RuntimeError("bundle.json has no valid source-manifest SHA256 digest.")
    if file_sha256(manifest_resource) != expected:
        raise RuntimeError("The LTE source manifest does not match bundle.json.")
    manifest = _load_json(manifest_resource)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported source manifest in {manifest_resource}.")
    return manifest


def _validate_stic_coverage(table: Mapping) -> dict:
    axes = table.get("axes", {})
    try:
        wavelengths = [float(value) for value in axes["wavelength_vacuum_angstrom"]]
        domains = [
            tuple(float(value) for value in interval)
            for interval in table["validated_wavelength_domains_angstrom"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid STiC wavelength-coverage metadata.") from error
    if (
        len(wavelengths) < 2
        or not all(math.isfinite(value) for value in wavelengths)
        or any(right <= left for left, right in zip(wavelengths, wavelengths[1:]))
        or not domains
        or any(
            len(interval) != 2
            or not all(math.isfinite(value) for value in interval)
            or interval[1] < interval[0]
            for interval in domains
        )
    ):
        raise RuntimeError("Invalid STiC wavelength coverage.")
    required_nodes = (
        *REFERENCE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM,
        *HMI_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM,
        *HINODE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM,
    )
    missing_nodes = [
        node
        for node in required_nodes
        if not any(
            math.isclose(value, node, rel_tol=0.0, abs_tol=1.0e-8)
            for value in wavelengths
        )
    ]
    required_domains = {
        "tau500_reference": (5000.0, 5000.0),
        "hmi_stokes": HMI_STIC_REQUIRED_DOMAIN_ANGSTROM,
        "hinode_sp": HINODE_STIC_REQUIRED_DOMAIN_ANGSTROM,
    }
    missing_domains = [
        instrument
        for instrument, (lower, upper) in required_domains.items()
        if not any(a <= lower and b >= upper for a, b in domains)
    ]
    if missing_nodes or missing_domains:
        raise RuntimeError(
            "The packaged STiC table does not cover the exact Hinode SP and HMI "
            f"production contract (missing nodes={missing_nodes}, "
            f"missing domains={missing_domains})."
        )
    return {
        "wavelength_vacuum_angstrom": wavelengths,
        "validated_wavelength_domains_angstrom": [
            list(interval) for interval in domains
        ],
    }


def validate_resource_bundle() -> dict:
    """Validate the immutable packaged LTE bundle and return its metadata."""

    root = resource_path()
    bundle_resource = root.joinpath("bundle.json")
    manifest_resource = root.joinpath("sources.json")
    bundle = _load_json(bundle_resource)
    manifest_digest = file_sha256(manifest_resource)
    manifest = load_verified_source_manifest()

    file_records: dict[str, Mapping] = {}
    for group in ("vendored_files", "generated_files", "reviewed_files"):
        records = manifest.get(group)
        if not isinstance(records, dict):
            raise RuntimeError(f"Source manifest has no valid {group!r} inventory.")
        overlap = set(file_records).intersection(records)
        if overlap:
            raise RuntimeError(f"Source manifest repeats files: {sorted(overlap)}")
        file_records.update(records)

    supported_instruments = bundle.get("supported_instruments")
    if supported_instruments != list(SUPPORTED_INSTRUMENTS):
        raise RuntimeError(
            "LTE bundle does not declare the supported instruments exactly."
        )
    common_files = bundle.get("common_production_files")
    instrument_files = bundle.get("instrument_production_files")
    if common_files != sorted(REQUIRED_COMMON_PRODUCTION_FILES):
        raise RuntimeError("LTE bundle does not declare its exact common contract.")
    if not isinstance(instrument_files, dict) or set(instrument_files) != set(
        SUPPORTED_INSTRUMENTS
    ):
        raise RuntimeError("LTE bundle has no exact per-instrument contract.")
    expected_instrument_files = {
        "hinode_sp": HINODE_PRODUCTION_FILES,
        "hmi_stokes": HMI_PRODUCTION_FILES,
    }
    if any(
        instrument_files[name] != sorted(expected_instrument_files[name])
        for name in SUPPORTED_INSTRUMENTS
    ):
        raise RuntimeError("LTE bundle has an invalid per-instrument contract.")
    if bundle.get("training_contract") != TRAINING_RESOURCE_CONTRACT:
        raise RuntimeError("LTE bundle has an invalid offline-training contract.")
    declared = set(common_files)
    for names in instrument_files.values():
        declared.update(map(str, names))
    if declared - set(file_records):
        raise RuntimeError("Production resources are absent from the source manifest.")
    if declared != set(file_records):
        raise RuntimeError(
            "Source manifest contains resources outside the production contract."
        )

    runtime_files = bundle.get("runtime_files")
    expected_runtime_files = sorted([*file_records, "sources.json"])
    if runtime_files != expected_runtime_files:
        raise RuntimeError("LTE runtime inventory does not match the source manifest.")
    packaged_files = _packaged_file_names(root)
    expected_packaged_files = {*expected_runtime_files, "bundle.json"}
    if packaged_files != expected_packaged_files:
        missing_packaged_files = sorted(expected_packaged_files - packaged_files)
        unexpected_packaged_files = sorted(packaged_files - expected_packaged_files)
        raise RuntimeError(
            "LTE bundle directory does not match its exact production inventory; "
            f"missing={missing_packaged_files}, "
            f"unexpected={unexpected_packaged_files}. Remove stale resource files "
            "or deploy the complete packaged bundle."
        )
    missing_files = [
        filename for filename in runtime_files if not resource_path(filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"LTE resource bundle is incomplete: {missing_files}")
    digests = {
        filename: verify_manifest_resource(
            resource_path(filename),
            manifest,
            resource_name=filename,
            kind="LTE",
        )
        for filename in file_records
    }

    table = _load_json(resource_path("common/stic_continuum_table.json"))
    raw_contract = table.get("continuum_contract")
    top_boundary = table.get("falc_top_boundary")
    reference_solver = table.get("reference_solver", {})
    axes = table.get("axes", {})
    if table.get("schema_version") != 2 or not isinstance(raw_contract, dict):
        raise RuntimeError("LTE bundle has no valid STiC continuum contract.")
    contract_fields = (
        "tau500_quantity",
        "scattering_is_stored_separately",
        "physical_height_qualified",
        "runtime_fallback",
        "source_function_limitation",
        "thermodynamic_population_contract",
    )
    try:
        contract = {field: raw_contract[field] for field in contract_fields}
    except KeyError as error:
        raise RuntimeError(
            "STiC table has an incomplete continuum contract."
        ) from error
    required_populations = (
        "log10_mass_density_kg_m3",
        "log10_electron_density_m3",
        "log10_neutral_hydrogen_density_m3",
        "log10_fe_i_population_over_partition_m3",
    )
    if any(not isinstance(table.get(field), list) for field in required_populations):
        raise RuntimeError("STiC table lacks coherent thermodynamic populations.")
    if not isinstance(contract.get("thermodynamic_population_contract"), str):
        raise RuntimeError("STiC table lacks its population contract.")
    if (
        contract.get("physical_height_qualified") is not True
        or contract.get("runtime_fallback")
        != "none; missing or invalid STiC table is a hard error"
    ):
        raise RuntimeError("STiC table is not qualified for physical height synthesis.")
    if (
        reference_solver.get("commit") != "18cda77d038a97f007a783dcb61ea9a9a1244bf7"
        or reference_solver.get("runtime_iterations") != 0
    ):
        raise RuntimeError("LTE bundle is not the pinned offline STiC reference.")
    try:
        temperature_axis = [float(value) for value in axes["log10_temperature_k"]]
        pressure_axis = [float(value) for value in axes["log10_gas_pressure_pa"]]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid STiC thermodynamic axes.") from error
    if (
        len(temperature_axis) < 2
        or len(pressure_axis) < 2
        or not all(
            math.isfinite(value) for value in (*temperature_axis, *pressure_axis)
        )
        or any(b <= a for a, b in zip(temperature_axis, temperature_axis[1:]))
        or any(b <= a for a, b in zip(pressure_axis, pressure_axis[1:]))
    ):
        raise RuntimeError("Invalid STiC thermodynamic axes.")
    if not isinstance(top_boundary, dict):
        raise RuntimeError("LTE bundle lacks its FALC top boundary.")
    try:
        top_pressure = float(top_boundary["gas_pressure_pa"])
        gravity = float(top_boundary["gravity_cm_s2"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid FALC top boundary.") from error
    if (
        top_boundary.get("target_log10_tau500") != -5.0
        or not math.isfinite(top_pressure)
        or top_pressure <= 0
        or not math.isfinite(gravity)
        or gravity <= 0
    ):
        raise RuntimeError("Invalid FALC top boundary.")
    if (
        bundle.get("physical_depth_contract") != contract
        or bundle.get("falc_top_boundary") != top_boundary
    ):
        raise RuntimeError("Bundle metadata disagrees with the STiC table.")

    coverage = _validate_stic_coverage(table)
    return {
        "bundle_schema_version": 2,
        "bundle_type": BUNDLE_TYPE,
        "supported_instruments": list(SUPPORTED_INSTRUMENTS),
        "source_manifest_sha256": manifest_digest,
        "resource_sha256": digests,
        "training_contract": TRAINING_RESOURCE_CONTRACT,
        "common_production_files": sorted(REQUIRED_COMMON_PRODUCTION_FILES),
        "instrument_production_files": {
            name: sorted(expected_instrument_files[name])
            for name in SUPPORTED_INSTRUMENTS
        },
        "physical_depth_contract": contract,
        "falc_top_boundary": top_boundary,
        "stic_lookup_bounds": {
            "temperature_log10_k": [temperature_axis[0], temperature_axis[-1]],
            "gas_pressure_log10_pa": [pressure_axis[0], pressure_axis[-1]],
        },
        "stic_wavelength_coverage": coverage,
    }


def validate_instrument_resource_bundle(instrument: str) -> dict:
    """Validate and return the complete packaged contract for one instrument."""

    if instrument not in SUPPORTED_INSTRUMENTS:
        raise ValueError(
            f"Unsupported LTE resource instrument {instrument!r}; expected one of "
            f"{list(SUPPORTED_INSTRUMENTS)}."
        )
    bundle = validate_resource_bundle()
    production_files = sorted(
        REQUIRED_COMMON_PRODUCTION_FILES
        | (
            HINODE_PRODUCTION_FILES
            if instrument == "hinode_sp"
            else HMI_PRODUCTION_FILES
        )
    )
    return {
        "instrument": instrument,
        "bundle_schema_version": bundle["bundle_schema_version"],
        "bundle_type": bundle["bundle_type"],
        "training_contract": bundle["training_contract"],
        "production_files": production_files,
        "resource_sha256": {
            name: bundle["resource_sha256"][name] for name in production_files
        },
        "physical_depth_contract": bundle["physical_depth_contract"],
        "falc_top_boundary": bundle["falc_top_boundary"],
        "stic_lookup_bounds": bundle["stic_lookup_bounds"],
        "stic_wavelength_coverage": bundle["stic_wavelength_coverage"],
    }


__all__ = [
    "BUNDLE_TYPE",
    "HINODE_PRODUCTION_FILES",
    "HINODE_STIC_REQUIRED_DOMAIN_ANGSTROM",
    "HINODE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM",
    "HMI_PRODUCTION_FILES",
    "HMI_STIC_REQUIRED_DOMAIN_ANGSTROM",
    "HMI_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM",
    "REQUIRED_COMMON_PRODUCTION_FILES",
    "REQUIRED_HINODE_PRODUCTION_FILES",
    "REQUIRED_HMI_PRODUCTION_FILES",
    "REFERENCE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM",
    "SUPPORTED_INSTRUMENTS",
    "TRAINING_RESOURCE_CONTRACT",
    "file_sha256",
    "load_verified_source_manifest",
    "resource_path",
    "validate_resource_bundle",
    "validate_instrument_resource_bundle",
    "verify_manifest_resource",
]
