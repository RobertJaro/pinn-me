"""Independent, versioned scientific-resource set registry.

The legacy LTE bundle deliberately retains its exact ``resources/data``
inventory and validator.  New resource sets live beside it under
``resources/sets`` and are sealed independently, so an HMI-only run neither
loads nor validates EUV resources.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from importlib.resources import files
import json
import math
from pathlib import PurePosixPath
import re

from .manifest import validate_resource_bundle


LEGACY_LTE_RESOURCE_SET_ID = "legacy_lte_v2"
AIA_RESOURCE_SET_ID = "aia_euv_v1"
AIA_TEMPERATURE_RESPONSE_RESOURCE = f"{AIA_RESOURCE_SET_ID}:aia_temperature_response"
RESOURCE_SET_IDS = (LEGACY_LTE_RESOURCE_SET_ID, AIA_RESOURCE_SET_ID)
_RESOURCE_SET_ROOTS = {
    AIA_RESOURCE_SET_ID: PurePosixPath("euv/aia_euv_v1"),
}
_RESOURCE_REFERENCE = re.compile(r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SET_TYPE = "prom3theus-scientific-resource-set"
_TRAINING_CONTRACT = "offline-read-only; no downloads or table conversion"


def _safe_relative_path(resource_name: str) -> PurePosixPath:
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
    return relative


def resource_set_path(set_id: str, resource_name: str | None = None):
    """Return one safe path in a registered non-legacy resource set."""

    if set_id not in _RESOURCE_SET_ROOTS:
        if set_id == LEGACY_LTE_RESOURCE_SET_ID:
            raise ValueError(
                "Use prom3theus.resources.resource_path() for legacy_lte_v2."
            )
        raise ValueError(
            f"Unknown resource set {set_id!r}; expected one of {list(RESOURCE_SET_IDS)}."
        )
    relative_root = _RESOURCE_SET_ROOTS[set_id]
    root = files("prom3theus.resources").joinpath("sets", *relative_root.parts)
    if resource_name is None:
        return root
    relative = _safe_relative_path(resource_name)
    return root.joinpath(*relative.parts)


def _open_binary(resource):
    return resource.open("rb") if hasattr(resource, "open") else open(resource, "rb")


def _open_text(resource):
    return (
        resource.open("r", encoding="utf-8")
        if hasattr(resource, "open")
        else open(resource, encoding="utf-8")
    )


def _file_sha256(resource) -> str:
    digest = hashlib.sha256()
    with _open_binary(resource) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(resource, label: str) -> dict:
    try:
        with _open_text(resource) as handle:
            document = json.load(handle)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {resource}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"{label} must be a JSON object: {resource}")
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


def _valid_digest(value) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _canonical_sha256(document: Mapping) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_aia_table(table: Mapping, instrument: Mapping, sources: Mapping) -> dict:
    channels = ["171", "193", "211"]
    try:
        axis = [float(value) for value in table["axes"]["log10_temperature_k"]]
        response = [[float(value) for value in row] for row in table["response"]]
        reference_density = float(table["reference_log10_electron_density_cm3"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid AIA temperature-response arrays.") from error
    expected_axis = [4.0 + 0.05 * index for index in range(101)]
    if (
        table.get("schema_version") != 1
        or table.get("resource_type") != "prom3theus-aia-temperature-response"
        or table.get("channels") != channels
        or len(axis) != len(expected_axis)
        or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(axis, expected_axis, strict=True)
        )
        or len(response) != len(channels)
        or any(len(row) != len(axis) for row in response)
        or any(
            not math.isfinite(value) or value < 0.0 for row in response for value in row
        )
        or any(max(row) <= 0.0 for row in response)
        or table.get("response_unit") != "DN s^-1 pixel^-1 cm^5"
        or table.get("emission_measure_convention") != "electron_density_squared"
        or not math.isclose(reference_density, 9.0, rel_tol=0.0, abs_tol=0.0)
        or table.get("density_dependence")
        != (
            "total response evaluated at fixed reference electron density; "
            "runtime electron density enters only through n_e^2"
        )
        or table.get("components")
        != {
            "available": False,
            "reason": "the pinned converted source stores total response only",
        }
        or table.get("interpolation")
        != {
            "coordinate": "log10_temperature_k",
            "method": "piecewise_linear",
            "values": "linear_response",
        }
        or table.get("outside_domain")
        != {
            "analytic_high_temperature_tail": False,
            "behavior": "exact_zero",
            "edge_taper": {
                "endpoint_value": "exact_zero",
                "method": "quintic_smootherstep",
                "width_log10_temperature_k": 0.05,
            },
            "temperature_is_modified": False,
        }
    ):
        raise RuntimeError("Invalid AIA temperature-response scientific contract.")

    derivation = table.get("derivation")
    converted = sources.get("converted_source_artifact")
    upstream = table.get("upstream_provenance")
    if (
        not isinstance(derivation, Mapping)
        or not isinstance(converted, Mapping)
        or not isinstance(upstream, Mapping)
        or derivation.get("source_artifact_sha256") != converted.get("sha256")
        or derivation.get("source_response_id") != converted.get("response_id")
        or upstream.get("response_id") != converted.get("response_id")
        or upstream.get("spectral_emissivity", {})
        .get("atomic_database", {})
        .get("version")
        != "11.0.2"
        or upstream.get("spectral_emissivity", {}).get("abundance", {}).get("name")
        != "sun_coronal_2021_chianti"
    ):
        raise RuntimeError("AIA response provenance is incomplete or inconsistent.")

    calibration = instrument.get("calibration_contract")
    convention_id = table.get("calibration_convention_id")
    if (
        instrument.get("schema_version") != 1
        or instrument.get("resource_type") != "prom3theus-aia-euv-instrument"
        or instrument.get("instrument") != "SDO/AIA"
        or instrument.get("channels") != channels
        or instrument.get("response_resource")
        != f"{AIA_RESOURCE_SET_ID}:aia_temperature_response"
        or not isinstance(calibration, Mapping)
        or not isinstance(convention_id, str)
        or convention_id != instrument.get("calibration_convention_id")
        or convention_id != f"sha256:{_canonical_sha256(calibration)}"
    ):
        raise RuntimeError("Invalid AIA instrument calibration contract.")
    return {
        "channels": channels,
        "temperature_log10_k": [axis[0], axis[-1]],
        "response_unit": table["response_unit"],
        "emission_measure_convention": table["emission_measure_convention"],
        "reference_log10_electron_density_cm3": reference_density,
        "calibration_convention_id": convention_id,
        "response_source_sha256": converted["sha256"],
        "response_id": converted["response_id"],
        "components_available": False,
        "outside_domain": "exact_zero",
        "edge_taper": "quintic_smootherstep over 0.05 dex at both boundaries",
    }


def validate_resource_set(set_id: str) -> dict:
    """Validate one registered resource set and return its runtime metadata."""

    if set_id == LEGACY_LTE_RESOURCE_SET_ID:
        return validate_resource_bundle()
    root = resource_set_path(set_id)
    manifest_resource = root.joinpath("manifest.json")
    sources_resource = root.joinpath("sources.json")
    if not manifest_resource.is_file() or not sources_resource.is_file():
        raise FileNotFoundError(f"Resource set {set_id!r} is incomplete in {root}.")
    manifest = _load_json(manifest_resource, "resource-set manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("resource_set_type") != _SET_TYPE
        or manifest.get("resource_set_id") != set_id
        or manifest.get("training_contract") != _TRAINING_CONTRACT
    ):
        raise RuntimeError(f"Unsupported resource-set manifest for {set_id!r}.")
    expected_source_digest = manifest.get("source_manifest_sha256")
    if not _valid_digest(expected_source_digest):
        raise RuntimeError(f"Resource set {set_id!r} has no valid source digest.")
    actual_source_digest = _file_sha256(sources_resource)
    if actual_source_digest != expected_source_digest:
        raise RuntimeError(f"Resource set {set_id!r} source manifest is corrupted.")

    resource_digests = manifest.get("resource_sha256")
    logical_resources = manifest.get("logical_resources")
    runtime_files = manifest.get("runtime_files")
    if (
        not isinstance(resource_digests, dict)
        or not isinstance(logical_resources, dict)
        or not isinstance(runtime_files, list)
        or runtime_files != sorted([*resource_digests, "sources.json"])
        or set(logical_resources.values()) != set(resource_digests)
        or any(not _valid_digest(value) for value in resource_digests.values())
    ):
        raise RuntimeError(f"Resource set {set_id!r} has an invalid inventory.")
    packaged = _packaged_file_names(root)
    expected_packaged = {*runtime_files, "manifest.json"}
    if packaged != expected_packaged:
        raise RuntimeError(
            f"Resource set {set_id!r} inventory mismatch: "
            f"missing={sorted(expected_packaged - packaged)}, "
            f"unexpected={sorted(packaged - expected_packaged)}."
        )
    for filename, expected in resource_digests.items():
        resource = resource_set_path(set_id, filename)
        if _file_sha256(resource) != expected:
            raise RuntimeError(
                f"Resource set {set_id!r} file {filename!r} failed SHA256 verification."
            )

    sources = _load_json(sources_resource, "resource-set sources")
    table = _load_json(
        resource_set_path(set_id, logical_resources["aia_temperature_response"]),
        "AIA temperature response",
    )
    instrument = _load_json(
        resource_set_path(set_id, logical_resources["instrument_aia_euv"]),
        "AIA instrument contract",
    )
    scientific = _validate_aia_table(table, instrument, sources)
    compatibility = manifest.get("compatibility")
    if compatibility != {
        "chianti_database_release": "11.0.2",
        "emission_measure_convention": "electron_density_squared",
    }:
        raise RuntimeError("AIA resource-set compatibility metadata is invalid.")
    return {
        "resource_set_id": set_id,
        "resource_set_schema_version": 1,
        "source_manifest_sha256": actual_source_digest,
        "resource_sha256": dict(resource_digests),
        "logical_resources": dict(logical_resources),
        "training_contract": _TRAINING_CONTRACT,
        "compatibility": dict(compatibility),
        "scientific_contract": scientific,
    }


def validate_resource_sets(set_ids: Iterable[str]) -> dict[str, dict]:
    """Validate an exact requested set union and cross-check shared releases."""

    requested = tuple(set_ids)
    if not requested:
        raise ValueError("At least one resource set must be requested.")
    if len(set(requested)) != len(requested):
        raise ValueError("Resource set IDs must be unique.")
    unknown = sorted(set(requested).difference(RESOURCE_SET_IDS))
    if unknown:
        raise ValueError(f"Unknown resource sets: {unknown}")
    validated = {set_id: validate_resource_set(set_id) for set_id in requested}
    if AIA_RESOURCE_SET_ID in validated and LEGACY_LTE_RESOURCE_SET_ID in validated:
        legacy_release = validated[LEGACY_LTE_RESOURCE_SET_ID]["thermodynamic_eos"][
            "reference_ionization_equilibrium"
        ]["database_release"]
        aia_release = validated[AIA_RESOURCE_SET_ID]["compatibility"][
            "chianti_database_release"
        ]
        if legacy_release != aia_release:
            raise RuntimeError(
                "LTE thermodynamics and AIA emissivity use different CHIANTI releases."
            )
    return validated


def resolve_resource_reference(reference: str):
    """Resolve ``resource_set:logical_name`` after full set validation."""

    if (
        not isinstance(reference, str)
        or _RESOURCE_REFERENCE.fullmatch(reference) is None
    ):
        raise ValueError(
            "A resource reference must use 'resource_set:logical_name' syntax."
        )
    set_id, logical_name = reference.split(":", 1)
    if set_id == LEGACY_LTE_RESOURCE_SET_ID:
        raise ValueError("Logical legacy_lte_v2 resource references are not supported.")
    metadata = validate_resource_set(set_id)
    filename = metadata["logical_resources"].get(logical_name)
    if filename is None:
        raise KeyError(
            f"Resource set {set_id!r} has no logical resource {logical_name!r}."
        )
    return resource_set_path(set_id, filename)


__all__ = [
    "AIA_RESOURCE_SET_ID",
    "AIA_TEMPERATURE_RESPONSE_RESOURCE",
    "LEGACY_LTE_RESOURCE_SET_ID",
    "RESOURCE_SET_IDS",
    "resolve_resource_reference",
    "resource_set_path",
    "validate_resource_set",
    "validate_resource_sets",
]
