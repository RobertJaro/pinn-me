"""Publish the audited compact AIA EUV response resource set.

This first AIA resource is deliberately a conversion step, not an atomic-data
calculation.  It selects the three initial coronal channels from one pinned,
versioned SuNeRF response artifact and preserves that artifact's complete
provenance.  In particular, the output remains a total-only response evaluated
at the artifact's fixed reference density; this builder does not invent
component decompositions or extrapolation tails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


RESOURCE_SET_ID = "aia_euv_v1"
SOURCE_ARTIFACT_SHA256 = (
    "90e893725f1cef0340347f901ff4da82b95ae2d1cf01132565592f6839178281"
)
SOURCE_RESPONSE_ID = (
    "sha256:73a29fd3468b187c054bd986fe41b4d029c1e7a6b8dec6f10897b74badeb80fd"
)
SOURCE_SCHEMA = "sunerf.euv-temperature-response"
SOURCE_SCHEMA_VERSION = 1
SOURCE_CHANNELS = ("A94", "A131", "A171", "A193", "A211", "A335")
SELECTED_SOURCE_CHANNELS = ("A171", "A193", "A211")
SELECTED_CHANNELS = ("171", "193", "211")
EXPECTED_RESPONSE_UNIT = "cm5 DN / (pix s)"
OUTPUT_RESPONSE_UNIT = "DN s^-1 pixel^-1 cm^5"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            json.dumps(
                document,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )


def _scalar(archive: np.lib.npyio.NpzFile, field: str):
    value = np.asarray(archive[field])
    if value.shape != ():
        raise RuntimeError(f"Source AIA field {field!r} must be scalar.")
    return value.item()


def _load_source(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned AIA source artifact is missing: {path}")
    digest = _file_sha256(path)
    if digest != SOURCE_ARTIFACT_SHA256:
        raise RuntimeError(
            "AIA source artifact failed SHA256 verification: "
            f"expected {SOURCE_ARTIFACT_SHA256}, got {digest}."
        )
    required = {
        "schema",
        "schema_version",
        "channels",
        "log_temperature",
        "log_density",
        "response",
        "response_unit",
        "emission_measure_convention",
        "provenance_json",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"AIA source artifact is missing fields: {missing}")
        schema = str(_scalar(archive, "schema"))
        schema_version = int(_scalar(archive, "schema_version"))
        channels = tuple(np.asarray(archive["channels"]).astype(str).tolist())
        log_temperature = np.asarray(archive["log_temperature"], dtype=np.float64)
        log_density = np.asarray(archive["log_density"], dtype=np.float64)
        response = np.asarray(archive["response"], dtype=np.float64)
        response_unit = str(_scalar(archive, "response_unit"))
        convention = str(_scalar(archive, "emission_measure_convention"))
        try:
            provenance = json.loads(str(_scalar(archive, "provenance_json")))
        except json.JSONDecodeError as error:
            raise RuntimeError("AIA source provenance is not valid JSON.") from error

    expected_axis = np.linspace(4.0, 9.0, 101, dtype=np.float64)
    if (
        schema != SOURCE_SCHEMA
        or schema_version != SOURCE_SCHEMA_VERSION
        or channels != SOURCE_CHANNELS
        or not np.array_equal(log_temperature, expected_axis)
        or not np.array_equal(log_density, np.asarray([9.0]))
        or response.shape != (len(channels), 1, log_temperature.size)
        or not np.isfinite(response).all()
        or np.any(response < 0.0)
        or response_unit != EXPECTED_RESPONSE_UNIT
        or convention != "ne2"
        or not isinstance(provenance, dict)
        or provenance.get("response_id") != SOURCE_RESPONSE_ID
    ):
        raise RuntimeError("Pinned AIA source artifact violates its audited contract.")
    return {
        "channels": channels,
        "log_temperature": log_temperature,
        "log_density": log_density,
        "response": response,
        "response_unit": response_unit,
        "emission_measure_convention": convention,
        "provenance": provenance,
    }


def _source_inventory(provenance: Mapping[str, Any], source_name: str) -> dict:
    spectral = provenance["spectral_emissivity"]
    instrument = provenance["instrument_throughput"]
    return {
        "schema_version": 1,
        "resource_set_id": RESOURCE_SET_ID,
        "converted_source_artifact": {
            "filename": source_name,
            "schema": SOURCE_SCHEMA,
            "schema_version": SOURCE_SCHEMA_VERSION,
            "sha256": SOURCE_ARTIFACT_SHA256,
            "response_id": SOURCE_RESPONSE_ID,
        },
        "scientific_sources": {
            "chianti_atomic_database": dict(spectral["atomic_database"]),
            "chianti_abundance": dict(spectral["abundance"]),
            "chianti_ionization_equilibrium": dict(
                spectral["ionization_equilibrium"]
            ),
            "aia_full_instrument_response": dict(instrument["calibration"]),
            "aia_degradation_correction": dict(instrument["degradation"]),
        },
        "builder": {
            "name": "resource_builder.aia_euv",
            "algorithm_version": 1,
            "operation": "verified channel selection without numerical resampling",
        },
    }


def _calibration_contract(provenance: Mapping[str, Any]) -> dict:
    instrument = provenance["instrument_throughput"]
    return {
        "schema": "prom3theus.aia-reference-calibration",
        "schema_version": 1,
        "reference_epoch": provenance["calibration_epoch"],
        "sensitivity_convention": provenance["sensitivity_convention"],
        "degradation": {
            "algorithm": instrument["degradation"]["algorithm"],
            "calibration_version": instrument["degradation"][
                "calibration_version"
            ],
            "source_sha256": instrument["degradation"]["sha256"],
            "observation_requirement": (
                "divide observed count rates by the degradation factor relative "
                "to the response reference epoch"
            ),
        },
        "eve_normalization": dict(instrument["eve_normalization"]),
        "crosstalk": dict(instrument["crosstalk"]),
        "measurement_semantics": provenance["measurement_semantics"],
        "native_pixel_solid_angle_sr": provenance["native_pixel_solid_angle_sr"],
        "native_pixel_solid_angle_relative_tolerance": provenance[
            "native_pixel_solid_angle_relative_tolerance"
        ],
        "response_unit": OUTPUT_RESPONSE_UNIT,
        "emission_measure_convention": "electron_density_squared",
    }


def build(source_response: Path, output_directory: Path) -> None:
    """Build one complete, independently sealed AIA runtime resource set."""

    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    source = _load_source(source_response)
    provenance = source["provenance"]
    indices = [source["channels"].index(channel) for channel in SELECTED_SOURCE_CHANNELS]
    selected_response = source["response"][indices, 0, :]
    if not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in selected_response.reshape(-1)
    ):
        raise RuntimeError("Selected AIA response contains invalid values.")

    calibration_contract = _calibration_contract(provenance)
    calibration_convention_id = "sha256:" + _sha256_bytes(
        _canonical_json(calibration_contract)
    )
    table = {
        "schema_version": 1,
        "resource_type": "prom3theus-aia-temperature-response",
        "channels": list(SELECTED_CHANNELS),
        "axes": {
            "log10_temperature_k": source["log_temperature"].tolist(),
        },
        "response": selected_response.tolist(),
        "response_unit": OUTPUT_RESPONSE_UNIT,
        "emission_measure_convention": "electron_density_squared",
        "reference_log10_electron_density_cm3": float(source["log_density"][0]),
        "density_dependence": (
            "total response evaluated at fixed reference electron density; "
            "runtime electron density enters only through n_e^2"
        ),
        "components": {
            "available": False,
            "reason": "the pinned converted source stores total response only",
        },
        "interpolation": {
            "coordinate": "log10_temperature_k",
            "values": "linear_response",
            "method": "piecewise_linear",
        },
        "outside_domain": {
            "behavior": "exact_zero",
            "temperature_is_modified": False,
            "analytic_high_temperature_tail": False,
            "edge_taper": {
                "method": "quintic_smootherstep",
                "width_log10_temperature_k": 0.05,
                "endpoint_value": "exact_zero",
            },
        },
        "calibration_convention_id": calibration_convention_id,
        "derivation": {
            "operation": "exact channel selection without resampling",
            "source_channels": list(SELECTED_SOURCE_CHANNELS),
            "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
            "source_response_id": SOURCE_RESPONSE_ID,
        },
        "upstream_provenance": provenance,
    }
    instrument = {
        "schema_version": 1,
        "resource_type": "prom3theus-aia-euv-instrument",
        "instrument": "SDO/AIA",
        "channels": list(SELECTED_CHANNELS),
        "calibration_convention_id": calibration_convention_id,
        "calibration_contract": calibration_contract,
        "response_resource": f"{RESOURCE_SET_ID}:aia_temperature_response",
    }
    sources = _source_inventory(provenance, source_response.name)

    runtime_documents = {
        "aia_temperature_response.json": table,
        "instrument_aia_euv.json": instrument,
        "sources.json": sources,
    }
    for filename, document in runtime_documents.items():
        _write_json(output_directory / filename, document)

    source_manifest_sha256 = _file_sha256(output_directory / "sources.json")
    resource_sha256 = {
        filename: _file_sha256(output_directory / filename)
        for filename in ("aia_temperature_response.json", "instrument_aia_euv.json")
    }
    manifest = {
        "schema_version": 1,
        "resource_set_type": "prom3theus-scientific-resource-set",
        "resource_set_id": RESOURCE_SET_ID,
        "training_contract": "offline-read-only; no downloads or table conversion",
        "runtime_files": sorted([*resource_sha256, "sources.json"]),
        "logical_resources": {
            "aia_temperature_response": "aia_temperature_response.json",
            "instrument_aia_euv": "instrument_aia_euv.json",
        },
        "source_manifest_sha256": source_manifest_sha256,
        "resource_sha256": resource_sha256,
        "compatibility": {
            "chianti_database_release": provenance["spectral_emissivity"][
                "atomic_database"
            ]["version"],
            "emission_measure_convention": "electron_density_squared",
        },
    }
    _write_json(output_directory / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-response", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    build(
        args.source_response.expanduser().resolve(),
        args.output_directory.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
