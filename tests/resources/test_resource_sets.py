from __future__ import annotations

import json

import pytest

from prom3theus.resources import (
    AIA_RESOURCE_SET_ID,
    AIA_TEMPERATURE_RESPONSE_RESOURCE,
    LEGACY_LTE_RESOURCE_SET_ID,
    resolve_resource_reference,
    resource_path,
    resource_set_path,
    validate_resource_set,
    validate_resource_sets,
)


def test_aia_resource_set_is_independent_and_scientifically_explicit():
    metadata = validate_resource_set(AIA_RESOURCE_SET_ID)

    assert metadata["resource_set_id"] == "aia_euv_v1"
    assert metadata["logical_resources"] == {
        "aia_temperature_response": "aia_temperature_response.json",
        "instrument_aia_euv": "instrument_aia_euv.json",
    }
    assert metadata["compatibility"] == {
        "chianti_database_release": "11.0.2",
        "emission_measure_convention": "electron_density_squared",
    }
    contract = metadata["scientific_contract"]
    assert contract["channels"] == ["171", "193", "211"]
    assert contract["temperature_log10_k"] == [4.0, 9.0]
    assert contract["reference_log10_electron_density_cm3"] == 9.0
    assert contract["components_available"] is False
    assert contract["outside_domain"] == "exact_zero"
    assert contract["edge_taper"] == (
        "quintic_smootherstep over 0.05 dex at both boundaries"
    )
    assert contract["response_source_sha256"] == (
        "90e893725f1cef0340347f901ff4da82b95ae2d1cf01132565592f6839178281"
    )


def test_aia_resources_do_not_change_the_exact_legacy_inventory():
    assert {child.name for child in resource_path().iterdir() if child.is_dir()} == {
        "common",
        "hinode_sp",
        "hmi_stokes",
    }
    assert resource_set_path(AIA_RESOURCE_SET_ID).is_dir()
    validated = validate_resource_sets(
        (LEGACY_LTE_RESOURCE_SET_ID, AIA_RESOURCE_SET_ID)
    )
    assert tuple(validated) == (LEGACY_LTE_RESOURCE_SET_ID, AIA_RESOURCE_SET_ID)


def test_aia_response_reference_resolves_only_declared_logical_resources():
    path = resolve_resource_reference(AIA_TEMPERATURE_RESPONSE_RESOURCE)
    assert path.name == "aia_temperature_response.json"
    with pytest.raises(KeyError, match="no logical resource"):
        resolve_resource_reference("aia_euv_v1:not_declared")
    with pytest.raises(ValueError, match="syntax"):
        resolve_resource_reference("aia_euv_v1:../sources")


def test_aia_table_retains_full_upstream_provenance_and_exact_source_nodes():
    path = resolve_resource_reference(AIA_TEMPERATURE_RESPONSE_RESOURCE)
    table = json.loads(path.read_text(encoding="utf-8"))

    assert table["density_dependence"] == (
        "total response evaluated at fixed reference electron density; "
        "runtime electron density enters only through n_e^2"
    )
    assert table["outside_domain"] == {
        "analytic_high_temperature_tail": False,
        "behavior": "exact_zero",
        "edge_taper": {
            "endpoint_value": "exact_zero",
            "method": "quintic_smootherstep",
            "width_log10_temperature_k": 0.05,
        },
        "temperature_is_modified": False,
    }
    assert table["derivation"]["operation"] == (
        "exact channel selection without resampling"
    )
    upstream = table["upstream_provenance"]
    assert upstream["spectral_emissivity"]["atomic_database"]["version"] == "11.0.2"
    assert upstream["spectral_emissivity"]["abundance"]["name"] == (
        "sun_coronal_2021_chianti"
    )
    assert upstream["instrument_throughput"]["provider"] == {
        "name": "aiapy",
        "version": "0.12.1",
    }
    axis = table["axes"]["log10_temperature_k"]
    expected_peaks = {
        "171": (5.9, 7.687749865854099e-25),
        "193": (6.15, 3.878407656902718e-25),
        "211": (6.25, 1.2304057670130418e-25),
    }
    for channel, row in zip(table["channels"], table["response"], strict=True):
        index = max(range(len(row)), key=row.__getitem__)
        expected_log_temperature, expected_value = expected_peaks[channel]
        assert axis[index] == pytest.approx(expected_log_temperature, abs=1.0e-12)
        assert row[index] == pytest.approx(expected_value, rel=1.0e-14)
