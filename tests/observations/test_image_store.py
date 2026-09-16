"""Versioned persistence tests for scalar image observations."""

from __future__ import annotations

import json

import pytest
import torch

from prom3theus.core import sha256_file_set
from prom3theus.observations import (
    IMAGE_OBSERVATION_STORE_FORMAT,
    IMAGE_OBSERVATION_STORE_VERSION,
    ImageObservationRaster,
    ImageObservationStore,
    image_observation_store_signature,
)


SOLAR_RADIUS_M = 6.957e8


def _raster(channel: int, shape=(2, 3), group="group-0"):
    ray = torch.zeros(*shape, 3, dtype=torch.float32)
    ray[..., 2] = -1
    surface = torch.zeros_like(ray)
    surface[..., 2] = SOLAR_RADIUS_M
    return ImageObservationRaster(
        intensity=torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape),
        uncertainty=torch.full(shape, 0.5),
        ray_direction=ray,
        surface_position_m=surface,
        valid_mask=torch.ones(shape, dtype=torch.bool),
        absolute_tai_seconds=1_700_000_000.125 + channel,
        channel_angstrom=channel,
        exposure_group=group,
        metadata={
            "intensity_unit": "DN s-1 pixel-1",
            "ray_geometry": {"solar_radius_m": SOLAR_RADIUS_M},
            "provenance": {"record": f"AIA-{channel}"},
        },
    )


def test_image_store_round_trip_is_json_npy_and_memory_mappable(tmp_path):
    rasters = [_raster(171, (2, 3)), _raster(193, (3, 2))]
    root = ImageObservationStore.save_sequence(
        tmp_path / "prepared-images",
        rasters,
        raster_names=["aia-171", "aia-193"],
        source_signature="a" * 64,
        metadata={"observation": "aia_euv", "revision": 1},
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format"] == IMAGE_OBSERVATION_STORE_FORMAT
    assert manifest["version"] == IMAGE_OBSERVATION_STORE_VERSION
    assert not list(root.glob("*.pt")) and not list(root.glob("*.pkl"))
    assert all(
        record["file"].endswith(".npy")
        for raster in manifest["rasters"]
        for record in raster["arrays"].values()
    )

    loaded, names, metadata = ImageObservationStore.load_sequence(
        root, expected_signature="a" * 64
    )
    assert names == ["aia-171", "aia-193"]
    assert metadata == {"observation": "aia_euv", "revision": 1}
    assert [raster.spatial_shape for raster in loaded] == [(2, 3), (3, 2)]
    torch.testing.assert_close(loaded[0].intensity, rasters[0].intensity)
    assert loaded[1].absolute_tai_seconds == rasters[1].absolute_tai_seconds
    assert loaded[1].metadata["provenance"]["record"] == "AIA-193"


def test_image_store_rejects_checksum_corruption_and_unsafe_paths(tmp_path):
    corrupt_root = ImageObservationStore.save(
        tmp_path / "corrupt", _raster(171), source_signature="b" * 64
    )
    manifest = json.loads((corrupt_root / "manifest.json").read_text())
    array_path = corrupt_root / manifest["rasters"][0]["arrays"]["intensity"]["file"]
    array_path.write_bytes(array_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        ImageObservationStore.load(corrupt_root)

    unsafe_root = ImageObservationStore.save(
        tmp_path / "unsafe", _raster(171), source_signature="c" * 64
    )
    unsafe_manifest_path = unsafe_root / "manifest.json"
    unsafe_manifest = json.loads(unsafe_manifest_path.read_text())
    unsafe_manifest["rasters"][0]["arrays"]["intensity"]["file"] = "../escape.npy"
    unsafe_manifest_path.write_text(json.dumps(unsafe_manifest))
    with pytest.raises(ValueError, match="Unsafe"):
        ImageObservationStore.load(unsafe_root)


def test_image_store_signature_scopes_preparation_dependencies(tmp_path):
    source = tmp_path / "aia-prepared.fits"
    source.write_bytes(b"calibrated image v1")
    source_digest = sha256_file_set([source])
    store_configuration = {
        "type": "aia_euv",
        "directory": str(tmp_path),
        "channels_angstrom": [171, 193, 211],
    }
    first = image_observation_store_signature(
        store_configuration,
        {
            "directory": "/install/one",
            "preprocessing_bundle_sha256": "degradation-v10",
        },
        source_files_sha256=source_digest,
    )
    relocated_dependency = image_observation_store_signature(
        store_configuration,
        {
            "directory": "/install/two",
            "preprocessing_bundle_sha256": "degradation-v10",
        },
        source_files_sha256=source_digest,
    )
    assert first == relocated_dependency

    changed_preparation = image_observation_store_signature(
        store_configuration,
        {"preprocessing_bundle_sha256": "degradation-v11"},
        source_files_sha256=source_digest,
    )
    assert changed_preparation != first

    # Loader/scheduling and CHIANTI response resources are intentionally not
    # inputs to the adapter-filtered image store configuration.
    changed_runtime_only = image_observation_store_signature(
        dict(store_configuration),
        {"preprocessing_bundle_sha256": "degradation-v10"},
        source_files_sha256=source_digest,
    )
    assert changed_runtime_only == first

    source.write_bytes(b"calibrated image v2")
    changed_source = image_observation_store_signature(
        store_configuration,
        {"preprocessing_bundle_sha256": "degradation-v10"},
        source_files_sha256=sha256_file_set([source]),
    )
    assert changed_source != first


def test_image_store_rejects_duplicate_group_channel_identity(tmp_path):
    root = ImageObservationStore.save_sequence(
        tmp_path / "duplicate",
        [_raster(171), _raster(171)],
        raster_names=["first", "second"],
        source_signature="d" * 64,
    )
    with pytest.raises(ValueError, match="only one raster"):
        ImageObservationStore.load_sequence(root)
