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
        "common/lines.json",
        "common/stic_continuum_table.json",
        "hinode_sp/blend_inventory.json",
        "hinode_sp/instrument_hinode_sp.json",
        "hinode_sp/solar_reference_630nm.json",
    ]
    assert set(metadata["resource_sha256"]) == set(metadata["production_files"])


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
