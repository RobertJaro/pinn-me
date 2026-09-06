import json

import pytest

import prom3theus.resources.manifest as resource_manifest

from prom3theus.resources import (
    resource_path,
    validate_instrument_resource_bundle,
    validate_resource_bundle,
    verify_manifest_resource,
)


def test_packaged_resource_bundle_contains_only_production_inputs():
    metadata = validate_resource_bundle()
    data = resource_path()
    root_files = {child.name for child in data.iterdir() if child.is_file()}
    root_directories = {child.name for child in data.iterdir() if child.is_dir()}
    assert metadata["bundle_schema_version"] == 2
    assert metadata["bundle_type"] == "prom3theus-lte-runtime-resources"
    assert metadata["supported_instruments"] == ["hinode_sp", "hmi_stokes"]
    assert metadata["common_production_files"] == [
        "common/abundances.json",
        "common/chianti_thermodynamic_table.json",
        "common/falc_reference_atmosphere.json",
        "common/lines.json",
        "common/stic_continuum_table.json",
    ]
    assert metadata["instrument_production_files"] == {
        "hinode_sp": [
            "hinode_sp/blend_inventory.json",
            "hinode_sp/instrument_hinode_sp.json",
            "hinode_sp/solar_reference_630nm.json",
        ],
        "hmi_stokes": [
            "hmi_stokes/instrument_hmi.json",
            "hmi_stokes/solar_reference_617nm.json",
        ],
    }
    assert root_files == {"bundle.json", "sources.json"}
    assert root_directories == {"common", "hinode_sp", "hmi_stokes"}
    assert not {
        "eos_table.json",
        "hminus_continuum.json",
        "partition_functions.json",
    } & {
        child.name
        for folder in root_directories
        for child in data.joinpath(folder).iterdir()
    }


def test_bundle_inventory_error_identifies_stale_and_missing_files(monkeypatch):
    packaged = resource_manifest._packaged_file_names(resource_path())
    monkeypatch.setattr(
        resource_manifest,
        "_packaged_file_names",
        lambda root: (packaged - {"common/lines.json"}) | {"lines.json"},
    )

    with pytest.raises(RuntimeError) as caught:
        validate_resource_bundle()

    message = str(caught.value)
    assert "missing=['common/lines.json']" in message
    assert "unexpected=['lines.json']" in message


def test_packaged_resource_bundle_covers_both_instrument_windows():
    metadata = validate_resource_bundle()
    coverage = metadata["stic_wavelength_coverage"]
    assert coverage["wavelength_vacuum_angstrom"] == [
        5000.0,
        6160.0,
        6170.0,
        6180.0,
        6190.0,
        6290.0,
        6300.0,
        6310.0,
        6320.0,
    ]
    assert coverage["validated_wavelength_domains_angstrom"] == [
        [5000.0, 5000.0],
        [6160.0, 6190.0],
        [6290.0, 6320.0],
    ]


def test_hinode_resource_view_contains_full_atomic_and_instrument_contract():
    metadata = validate_instrument_resource_bundle("hinode_sp")
    assert metadata["instrument"] == "hinode_sp"
    assert metadata["production_files"] == [
        "common/abundances.json",
        "common/chianti_thermodynamic_table.json",
        "common/falc_reference_atmosphere.json",
        "common/lines.json",
        "common/stic_continuum_table.json",
        "hinode_sp/blend_inventory.json",
        "hinode_sp/instrument_hinode_sp.json",
        "hinode_sp/solar_reference_630nm.json",
    ]
    assert set(metadata["resource_sha256"]) == set(metadata["production_files"])
    eos = metadata["thermodynamic_eos"]
    assert eos["lookup_log10_temperature_k"] == [4.0, 9.0]
    assert eos["reference_ionization_equilibrium"]["database_release"] == "11.0.2"
    assert eos["runtime_contract"]["stic_to_chianti_log_blend_temperature_k"] == [
        1.0e4,
        10.0**4.5,
    ]
    assert eos["runtime_contract"][
        "chianti_to_fully_ionized_log_blend_temperature_k"
    ] == [1.0e7, 2.0e7]
    assert eos["runtime_contract"]["ideal_transition_criterion"] == (
        "CHIANTI remains exact through the native 10 MK node; a C1 "
        "log-temperature blend then reaches the fully ionized limit at exactly "
        "20 MK, and starts with both CHIANTI-derived mu and mu_e within 1e-3 "
        "relative error of that limit"
    )
    assert eos["runtime_contract"]["lower_transition"] == {
        "electron_coordinate": (
            "logit(free_electrons_per_H / fully_ionized_electrons_per_H)"
        ),
        "non_electron_coordinate": "log10(non_electron_particles_per_H)",
        "interior_weight": "quintic smootherstep in log10(T)",
        "endpoint_tangent_match_width_log10_temperature": 0.05,
        "endpoint_tangent_ramp_power": 5,
        "lower_endpoint_ramp": "1 - (1 - u)^p",
        "upper_endpoint_ramp": "u^p",
        "integrated_tangent_change": "slope * width / p",
        "closure": ("particles_per_H = non_electron_particles_per_H + electrons_per_H"),
    }
    assert eos["runtime_contract"]["stic_interpolation"] == {
        "coordinates": "tensor product in log10(T/K) and log10(Pgas/Pa)",
        "uniform_axis_coordinate": "(x - x_0) * (N - 1) / (x_(N-1) - x_0)",
        "cells": "Catmull-Rom cubic with endpoint-index clamping",
        "boundary_derivative": (
            "one half of the outer one-sided secant, matching the established "
            "STiC runtime interpolant"
        ),
    }
    assert eos["runtime_contract"]["pressure_continuation"] == {
        "domain": "outside the packaged STiC log10(Pgas/Pa) axis",
        "coordinates": (
            "logit(electrons_per_H / fully_ionized_electrons_per_H) and "
            "log10(non_electron_particles_per_H)"
        ),
        "edge_derivative": (
            "derived from the clamped-Catmull boundary derivatives of log10(rho) "
            "and log10(n_e)"
        ),
        "tangent_decay_width_log10_pressure": 0.05,
        "tangent_decay_power": 5,
        "derivative_ramp": "(1 - u)^p",
        "integrated_coordinate_change": (
            "signed slope * width * [1 - (1 - u)^(p + 1)] / (p + 1)"
        ),
        "asymptotic_closure": (
            "constant particles_per_H and electrons_per_H; rho and n_e are linear in P"
        ),
    }


def _falc_validation_documents():
    table = json.loads(
        resource_path("common/falc_reference_atmosphere.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(resource_path("sources.json").read_text(encoding="utf-8"))
    continuum = json.loads(
        resource_path("common/stic_continuum_table.json").read_text(encoding="utf-8")
    )
    return table, manifest, continuum["falc_top_boundary"]


def test_packaged_bundle_exposes_complete_native_falc_reference():
    falc = validate_resource_bundle()["falc_reference_atmosphere"]

    assert falc["native_depth_count"] == 82
    assert falc["native_top"] == {
        "height_m": 2073502.4593743724,
        "temperature_k": 100000.0,
        "gas_pressure_pa": 0.031934421263776124,
        "microturbulence_m_per_s": 10680.960000000001,
    }
    assert falc["log10_tau500_range"] == [
        -5.405870771873386,
        1.3948466796994528,
    ]
    assert falc["line_formation_log10_tau500_range"] == [-5.0, 1.0]


def test_native_falc_contract_rejects_corrupted_physical_array():
    table, manifest, top_boundary = _falc_validation_documents()
    table["gas_pressure_pa"][10] *= 1.0001

    with pytest.raises(RuntimeError, match="physical contract"):
        resource_manifest._validate_falc_reference_atmosphere(
            table, manifest, top_boundary
        )


@pytest.mark.parametrize("corruption", ["source digest", "source provenance"])
def test_native_falc_contract_rejects_corrupted_source(corruption):
    table, manifest, top_boundary = _falc_validation_documents()
    if corruption == "source digest":
        table["source"]["source_sha256"]["stic_falc_82"] = "0" * 64
    else:
        table["source"]["commit"] = "unreviewed"

    with pytest.raises(RuntimeError, match="provenance or construction contract"):
        resource_manifest._validate_falc_reference_atmosphere(
            table, manifest, top_boundary
        )


def test_atomic_resource_checksum_rejects_modified_payload(tmp_path):
    manifest = json.loads(resource_path("sources.json").read_text(encoding="utf-8"))
    altered = tmp_path / "lines.json"
    altered.write_bytes(resource_path("common/lines.json").read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="failed SHA256 verification"):
        verify_manifest_resource(
            altered,
            manifest,
            resource_name="common/lines.json",
            kind="atomic-line",
        )


def test_resource_paths_and_manifest_digests_are_strict():
    with pytest.raises(ValueError, match="relative POSIX path"):
        resource_path("../lines.json")

    manifest = json.loads(resource_path("sources.json").read_text(encoding="utf-8"))
    manifest["reviewed_files"]["common/lines.json"]["sha256"] = manifest[
        "reviewed_files"
    ]["common/lines.json"]["sha256"].upper()
    with pytest.raises(RuntimeError, match="no valid SHA256"):
        verify_manifest_resource(
            resource_path("common/lines.json"),
            manifest,
            resource_name="common/lines.json",
            kind="atomic-line",
        )


def test_chianti_thermodynamic_contract_rejects_inconsistent_composition():
    table = json.loads(
        resource_path("common/chianti_thermodynamic_table.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(resource_path("sources.json").read_text(encoding="utf-8"))
    table["fully_ionized_limit"]["mean_molecular_weight"] *= 1.01

    with pytest.raises(RuntimeError, match="composition closure"):
        resource_manifest._validate_chianti_eos_table(table, manifest)


def test_chianti_thermodynamic_contract_requires_the_native_grid():
    table = json.loads(
        resource_path("common/chianti_thermodynamic_table.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(resource_path("sources.json").read_text(encoding="utf-8"))
    table["axes"]["log10_temperature_k"][50] += 1.0e-4

    with pytest.raises(RuntimeError, match="lookup arrays"):
        resource_manifest._validate_chianti_eos_table(table, manifest)
