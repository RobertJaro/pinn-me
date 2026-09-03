"""Focused contracts for the clean observation boundary."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from prom3theus.observations import (
    ObservationPixelDataset,
    ObservationRaster,
    ObservationSpec,
    ObservationStore,
    StoredObservationDataModule,
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    collate_observation_samples,
    observation_store_signature,
)
from prom3theus.core import sha256_file_set


def _raster() -> ObservationRaster:
    height, width, wavelength = 2, 3, 4
    valid = torch.tensor([[True, False, True], [True, True, False]])
    ray = torch.zeros(height, width, 3)
    ray[..., 2] = -1.0
    surface = torch.zeros_like(ray)
    surface[..., 2] = 6.957e8
    basis = torch.eye(3).expand(height, width, 3, 3).clone()
    return ObservationRaster(
        stokes=torch.arange(
            height * width * 4 * wavelength, dtype=torch.float32
        ).reshape(height, width, 4, wavelength),
        wavelength_angstrom=torch.linspace(6301.0, 6301.3, wavelength),
        coordinates=torch.zeros(height, width, 3),
        ray_direction=ray,
        surface_position_m=surface,
        stokes_basis=basis,
        valid_mask=valid,
        metadata={
            "fixture": "canonical-raster",
            "nested": {"revision": 1},
            "ray_geometry": {"solar_radius_m": 6.957e8},
        },
        auxiliary={"removed_solar_los_velocity_m_per_s": torch.ones(height, width)},
    )


def test_canonical_raster_dataset_and_batch_are_instrument_neutral():
    raster = _raster()
    assert raster.spatial_shape == (2, 3)
    torch.testing.assert_close(raster.mu[raster.valid_mask], torch.ones(4, 1))

    dataset = ObservationPixelDataset(
        raster,
        include_surface_position=True,
        auxiliary_fields=["removed_solar_los_velocity_m_per_s"],
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert set(sample) == {
        "coordinates",
        "ray_direction",
        "stokes_basis",
        "stokes",
        "surface_position_m",
        "removed_solar_los_velocity_m_per_s",
        "pixel_index",
    }
    batch = collate_observation_samples([dataset[0], dataset[1]])
    assert batch["stokes"].shape == (2, 4, 4)
    assert batch["pixel_index"].shape == (2, 2)


def test_observation_spec_records_explicit_velocity_gauge():
    spec = ObservationSpec(
        observation_id="fixture",
        observation_type="hinode_sp",
        instrument_type="hinode_sp",
        wavelength_angstrom=torch.tensor([6301.0, 6301.1]),
        continuum_indices=(0,),
        radiance_scale_w_m3_sr=1.0,
        velocity_synthesis_mode=CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    )
    assert (
        spec.metadata()["velocity_synthesis_mode"]
        == CARRINGTON_REGISTERED_RELATIVE_VELOCITY
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        ObservationSpec(
            observation_id="bad",
            observation_type="x",
            instrument_type="x",
            wavelength_angstrom=torch.tensor([2.0, 1.0]),
            continuum_indices=(0,),
            radiance_scale_w_m3_sr=1.0,
            velocity_synthesis_mode=CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
        )

    with pytest.raises(ValueError, match="line identifiers must be unique"):
        replace(spec, required_line_ids=("FeI_6301", "FeI_6301"))
    with pytest.raises(ValueError, match="finite, increasing"):
        replace(spec, line_support_angstrom=(6300.0, float("nan")))


def test_raster_rejects_degenerate_valid_surface_geometry():
    raster = _raster()
    surface = raster.surface_position_m.clone()
    surface[0, 0] = 0
    with pytest.raises(ValueError, match="0 < mu <= 1"):
        ObservationRaster(
            stokes=raster.stokes,
            wavelength_angstrom=raster.wavelength_angstrom,
            coordinates=raster.coordinates,
            ray_direction=raster.ray_direction,
            surface_position_m=surface,
            stokes_basis=raster.stokes_basis,
            valid_mask=raster.valid_mask,
            metadata=raster.metadata,
        )


def test_raster_rejects_invalid_physical_values_at_valid_pixels():
    raster = _raster()
    empty_mask = torch.zeros_like(raster.valid_mask)
    with pytest.raises(ValueError, match="at least one valid pixel"):
        replace(raster, valid_mask=empty_mask)

    stokes = raster.stokes.clone()
    stokes[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="Stokes values must be finite"):
        replace(raster, stokes=stokes)

    auxiliary = dict(raster.auxiliary)
    auxiliary["removed_solar_los_velocity_m_per_s"] = torch.full(
        raster.spatial_shape, float("inf")
    )
    with pytest.raises(ValueError, match="Auxiliary array.*finite"):
        replace(raster, auxiliary=auxiliary)


def test_raster_requires_stokes_basis_to_share_the_ray_orientation():
    raster = _raster()
    basis = raster.stokes_basis.clone()
    basis[..., 1, :] *= -1
    basis[..., 2, :] *= -1
    with pytest.raises(ValueError, match="surface to the observer"):
        replace(raster, stokes_basis=basis)


def test_dataset_rejects_lossy_indices_and_reserved_auxiliary_names():
    raster = _raster()
    with pytest.raises(TypeError, match="integer dtype"):
        ObservationPixelDataset(raster, pixel_indices=torch.tensor([[0.5, 0.0]]))
    with pytest.raises(ValueError, match="unique"):
        ObservationPixelDataset(
            raster,
            auxiliary_fields=[
                "removed_solar_los_velocity_m_per_s",
                "removed_solar_los_velocity_m_per_s",
            ],
        )
    with pytest.raises(ValueError, match="collides"):
        replace(raster, auxiliary={"stokes": torch.ones(raster.spatial_shape)})


def test_sampling_bounds_require_explicit_consistent_solar_geometry():
    raster = _raster()
    missing = replace(raster, metadata={"fixture": "missing-radius"})
    data = StoredObservationDataModule([missing], ["missing"])
    with pytest.raises(ValueError, match="declare ray_geometry.solar_radius_m"):
        data.setup()

    inconsistent = replace(
        raster,
        metadata={**raster.metadata, "ray_geometry": {"solar_radius_m": 7.0e8}},
    )
    data = StoredObservationDataModule([inconsistent], ["inconsistent"])
    with pytest.raises(ValueError, match="declared solar radius"):
        data.setup()


def test_store_is_json_and_npy_with_checksum_and_signature(tmp_path):
    raster = _raster()
    signature = observation_store_signature(
        {"slice": [0, 2]},
        {"atomic": "v1"},
        source_files_sha256="a" * 64,
    )
    root = ObservationStore.save(
        tmp_path / "prepared", raster, source_signature=signature
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source_signature"] == signature
    assert all(
        record["file"].endswith(".npy")
        for record in manifest["rasters"][0]["arrays"].values()
    )
    assert not list(root.glob("*.pt")) and not list(root.glob("*.pkl"))

    loaded = ObservationStore.load(root, expected_signature=signature)
    torch.testing.assert_close(loaded.stokes, raster.stokes)
    torch.testing.assert_close(
        loaded.auxiliary["removed_solar_los_velocity_m_per_s"], torch.ones(2, 3)
    )
    with pytest.raises(ValueError, match="source signature"):
        ObservationStore.load(root, expected_signature="d" * 64)

    first_array = root / manifest["rasters"][0]["arrays"]["stokes"]["file"]
    first_array.write_bytes(first_array.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        ObservationStore.load(root)


def test_multi_raster_store_rehydrates_training_and_hmi_response(tmp_path):
    base = _raster()
    spectral = torch.full((*base.spatial_shape, 6, 3), 0.2)
    continuum = torch.full((*base.spatial_shape, 6), 0.4)
    raster = replace(
        base,
        auxiliary={
            **base.auxiliary,
            "instrument_response:spectral_weights": spectral,
            "instrument_response:continuum_weights": continuum,
        },
    )
    root = ObservationStore.save_sequence(
        tmp_path / "sequence",
        [raster, raster],
        raster_names=["first", "second"],
        source_signature="b" * 64,
        metadata={"validation_raster_index": 1},
    )
    rasters, names, metadata = ObservationStore.load_sequence(root)
    assert names == ["first", "second"]
    assert metadata["validation_raster_index"] == 1
    assert "instrument_response:spectral_weights" in rasters[0].auxiliary
    data = StoredObservationDataModule(
        rasters, names, validation_raster_index=1, batch_size=2, num_workers=0
    )
    data.setup()
    batch = next(iter(data.train_dataloader()))
    assert batch["instrument_response"]["spectral_weights"].shape == (2, 6, 3)
    assert batch["instrument_response"]["continuum_weights"].shape == (2, 6)


def test_store_signature_excludes_resource_installation_directory():
    config = {"type": "hinode_sp", "directory": "/scientific/input"}
    first = observation_store_signature(
        config,
        {"directory": "/install/a", "source_manifest_sha256": "abc"},
        source_files_sha256="c" * 64,
    )
    second = observation_store_signature(
        config,
        {"directory": "/install/b", "source_manifest_sha256": "abc"},
        source_files_sha256="c" * 64,
    )
    assert first == second


def test_store_signature_changes_when_source_file_content_changes(tmp_path):
    source = tmp_path / "observation.fits"
    source.write_bytes(b"first scientific payload")
    config = {"type": "hinode_sp", "directory": str(tmp_path)}
    resources = {"source_manifest_sha256": "abc"}
    first = observation_store_signature(
        config,
        resources,
        source_files_sha256=sha256_file_set([source]),
    )
    source.write_bytes(b"other scientific payload")
    second = observation_store_signature(
        config,
        resources,
        source_files_sha256=sha256_file_set([source]),
    )
    assert first != second
