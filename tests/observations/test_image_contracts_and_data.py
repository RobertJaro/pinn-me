"""Contracts and deterministic loading for native-grid image observations."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from prom3theus.observations import (
    ImageObservationRaster,
    ImageObservationSpec,
    ImagePixelDataset,
    StoredImageDataModule,
    collate_image_samples,
    reconstruct_image,
)


SOLAR_RADIUS_M = 6.957e8
INTENSITY_UNIT = "DN s-1 pixel-1"


def _raster(
    channel: int,
    shape: tuple[int, int],
    *,
    time: float,
    group: str = "exposure-0",
    invalid_last: bool = False,
) -> ImageObservationRaster:
    height, width = shape
    ray = torch.zeros(height, width, 3, dtype=torch.float32)
    ray[..., 2] = -1.0
    surface = torch.zeros_like(ray)
    surface[..., 2] = SOLAR_RADIUS_M
    valid = torch.ones(shape, dtype=torch.bool)
    if invalid_last:
        valid[-1, -1] = False
    return ImageObservationRaster(
        intensity=torch.arange(height * width, dtype=torch.float32).reshape(shape)
        + channel,
        uncertainty=torch.full(shape, 0.25, dtype=torch.float32),
        ray_direction=ray,
        surface_position_m=surface,
        valid_mask=valid,
        absolute_tai_seconds=time,
        channel_angstrom=channel,
        exposure_group=group,
        metadata={
            "intensity_unit": INTENSITY_UNIT,
            "ray_geometry": {"solar_radius_m": SOLAR_RADIUS_M},
            "wcs": {"native_grid": True},
        },
    )


def _spec(groups=("exposure-0",)) -> ImageObservationSpec:
    return ImageObservationSpec(
        observation_id="aia-euv-fixture",
        observation_type="aia_euv",
        instrument_type="aia_temperature_response",
        intensity_unit=INTENSITY_UNIT,
        channels_angstrom=(171, 193, 211),
        exposure_groups=groups,
        calibration_convention={"degradation_correction": "applied_once"},
        geometry_convention={"ray_direction": "observer_to_sun"},
        required_resource_sets=("aia_euv_v1",),
    )


def test_image_contract_preserves_channel_time_units_and_geometry():
    raster = _raster(171, (2, 3), time=1_711_234_567.25, invalid_last=True)
    assert raster.spatial_shape == (2, 3)
    assert raster.channel_angstrom == 171
    assert raster.absolute_tai_seconds == 1_711_234_567.25
    assert raster.exposure_group == "exposure-0"
    torch.testing.assert_close(
        raster.mu[raster.valid_mask], torch.ones(5, dtype=torch.float32)
    )

    spec = _spec()
    assert spec.observation_kind == "image"
    assert spec.metadata()["channels_angstrom"] == [171, 193, 211]
    assert spec.required_resource_sets == ("aia_euv_v1",)


def test_image_contract_rejects_nonphysical_values_and_fake_ray_orientation():
    raster = _raster(171, (2, 2), time=1.0)
    uncertainty = raster.uncertainty.clone()
    uncertainty[0, 0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        replace(raster, uncertainty=uncertainty)

    reversed_ray = -raster.ray_direction
    with pytest.raises(ValueError, match="observer-to-Sun"):
        replace(raster, ray_direction=reversed_ray)

    with pytest.raises(ValueError, match="declared solar radius"):
        replace(
            raster,
            metadata={
                **raster.metadata,
                "ray_geometry": {"solar_radius_m": SOLAR_RADIUS_M + 1.0e6},
            },
        )


def test_image_dataset_emits_physical_fields_and_derived_indices():
    raster = _raster(193, (2, 3), time=1_711_234_568.5, invalid_last=True)
    dataset = ImagePixelDataset(raster, image_index=4, channel_index=1)
    assert len(dataset) == 5
    batch = collate_image_samples([dataset[0], dataset[1]])
    assert set(batch) == {
        "intensity",
        "uncertainty",
        "ray_direction",
        "surface_position_m",
        "absolute_tai_seconds",
        "channel_angstrom",
        "channel_index",
        "image_index",
        "pixel_index",
    }
    assert batch["absolute_tai_seconds"].dtype is torch.float64
    assert batch["channel_angstrom"].tolist() == [193, 193]
    assert batch["channel_index"].tolist() == [1, 1]
    assert batch["image_index"].tolist() == [4, 4]

    reconstructed = reconstruct_image(
        raster.intensity[raster.valid_mask], dataset.pixel_indices, raster.spatial_shape
    )
    torch.testing.assert_close(
        reconstructed[raster.valid_mask], raster.intensity[raster.valid_mask]
    )
    assert torch.isnan(reconstructed[~raster.valid_mask]).all()


def test_image_dataset_diagnostic_indices_are_complete_or_spatially_distributed():
    raster = _raster(171, (4, 5), time=1.0)
    irregular_mask = torch.tensor(
        (
            (True, False, True, False, True),
            (False, True, False, True, False),
            (True, True, False, False, True),
            (False, False, True, False, True),
        )
    )
    raster = replace(raster, valid_mask=irregular_mask)
    dataset = ImagePixelDataset(raster, image_index=0, channel_index=0)

    torch.testing.assert_close(
        dataset.diagnostic_dataset_indices(20),
        torch.arange(len(dataset), dtype=torch.long),
    )

    expected_ranks = torch.tensor((0, 3, 6, 9), dtype=torch.long)
    first = dataset.diagnostic_dataset_indices(4)
    second = dataset.diagnostic_dataset_indices(4)
    torch.testing.assert_close(first, expected_ranks)
    torch.testing.assert_close(second, expected_ranks)
    assert torch.unique(first).numel() == first.numel()
    torch.testing.assert_close(
        dataset.pixel_indices[first],
        torch.tensor(((0, 0), (1, 1), (2, 1), (3, 4))),
    )


def test_image_dataset_normalizes_intensity_and_uncertainty_without_changing_raster():
    raster = _raster(171, (2, 2), time=1.0)
    original = raster.intensity.clone()
    dataset = ImagePixelDataset(
        raster, image_index=0, channel_index=0, intensity_scale=200.0
    )
    batch = collate_image_samples([dataset[index] for index in range(len(dataset))])
    torch.testing.assert_close(batch["intensity"], original.flatten() / 200.0)
    torch.testing.assert_close(batch["uncertainty"], raster.uncertainty.flatten() / 200.0)
    torch.testing.assert_close(raster.intensity, original)
    assert batch["intensity"].dtype == torch.float32


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_image_dataset_rejects_invalid_intensity_scale(scale):
    with pytest.raises(ValueError, match="intensity_scale"):
        ImagePixelDataset(
            _raster(171, (2, 2), time=1.0), image_index=0, channel_index=0,
            intensity_scale=scale,
        )


def test_image_data_shares_fixed_channel_scales_across_exposures_and_loaders():
    rasters = [
        _raster(channel, (2, 2), time=1000.0 + group_index, group=group)
        for group_index, group in enumerate(("exposure-0", "exposure-1"))
        for channel in (171, 193, 211)
    ]
    scales = {171: 200.0, 193: 300.0, 211: 400.0}
    data = StoredImageDataModule(
        rasters, [str(index) for index in range(len(rasters))],
        _spec(("exposure-0", "exposure-1")),
        validation_exposure_group="exposure-1", intensity_scales=scales,
    )
    data.setup()
    for dataset in [*data.image_datasets, *data.validation_datasets.values()]:
        channel = dataset.raster.channel_angstrom
        assert dataset.intensity_scale == scales[channel]
        torch.testing.assert_close(
            dataset[0]["intensity"], dataset.raster.intensity[0, 0] / scales[channel]
        )
    for channel, loader in data.validation_dataloaders().items():
        batch = next(iter(loader))
        dataset = data.validation_datasets[channel]
        torch.testing.assert_close(
            batch["intensity"], dataset.raster.intensity.flatten() / scales[channel]
        )
    metadata = data.run_metadata()["intensity_normalization"]
    assert metadata["batch_intensity_unit"] == "dimensionless"
    assert metadata["source_intensity_unit"] == INTENSITY_UNIT
    assert metadata["scale_by_channel_angstrom"] == {"171": 200.0, "193": 300.0, "211": 400.0}


def test_stored_image_data_module_keeps_native_grids_and_complete_group():
    rasters = [
        _raster(171, (2, 3), time=1_000.0),
        _raster(193, (3, 2), time=1_001.25),
        _raster(211, (1, 4), time=1_002.5),
    ]
    data = StoredImageDataModule(
        rasters,
        ["aia-171", "aia-193", "aia-211"],
        _spec(),
        validation_exposure_group="exposure-0",
        batch_size=3,
        validation_batch_size=2,
        num_workers=0,
    )
    data.setup("fit")
    assert [raster.spatial_shape for raster in data.rasters] == [(2, 3), (3, 2), (1, 4)]
    assert set(data.validation_dataloaders()) == {171, 193, 211}
    assert data.validation_raster_names == {
        171: "aia-171",
        193: "aia-193",
        211: "aia-211",
    }

    training_batches = list(data.train_dataloader())
    assert len(training_batches) == 6
    assert all(
        torch.bincount(batch["channel_index"], minlength=3).tolist()
        == [1, 1, 1]
        for batch in training_batches
    )
    assert all("pixel_index" not in batch for batch in training_batches)

    recovered_times = {}
    for channel, loader in data.validation_dataloaders().items():
        batches = list(loader)
        recovered_times[channel] = torch.cat(
            [batch["absolute_tai_seconds"] for batch in batches]
        ).unique().item()
        values = torch.cat([batch["intensity"] for batch in batches])
        indices = torch.cat([batch["pixel_index"] for batch in batches])
        raster = data.rasters[data._raster_indices_by_identity[("exposure-0", channel)]]
        reconstructed = reconstruct_image(values, indices, raster.spatial_shape)
        torch.testing.assert_close(reconstructed, raster.intensity)
    assert recovered_times == {171: 1_000.0, 193: 1_001.25, 211: 1_002.5}

    metadata = data.run_metadata()
    assert metadata["validation"]["exposure_group"] == "exposure-0"
    assert metadata["valid_pixel_count"] == 16
    assert metadata["intensity_normalization"]["batch_intensity_unit"] == INTENSITY_UNIT


def test_stored_image_data_module_rejects_incomplete_or_misunit_group():
    rasters = [
        _raster(171, (2, 2), time=1.0),
        _raster(193, (2, 2), time=2.0),
    ]
    with pytest.raises(ValueError, match="must contain every configured channel"):
        StoredImageDataModule(
            rasters,
            ["171", "193"],
            _spec(),
            validation_exposure_group="exposure-0",
        )

    undersized_training_batch = StoredImageDataModule(
        [
            _raster(171, (2, 2), time=1.0),
            _raster(193, (2, 2), time=2.0),
            _raster(211, (2, 2), time=3.0),
        ],
        ["171", "193", "211"],
        _spec(),
        validation_exposure_group="exposure-0",
        batch_size=2,
    )
    with pytest.raises(ValueError, match="at least one pixel from every"):
        undersized_training_batch.train_dataloader()

    wrong_unit = replace(
        _raster(211, (2, 2), time=3.0),
        metadata={
            "intensity_unit": "photons",
            "ray_geometry": {"solar_radius_m": SOLAR_RADIUS_M},
        },
    )
    with pytest.raises(ValueError, match="intensity unit"):
        StoredImageDataModule(
            [*rasters, wrong_unit],
            ["171", "193", "211"],
            _spec(),
            validation_exposure_group="exposure-0",
        )
