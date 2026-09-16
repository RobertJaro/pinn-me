"""Strict, dependency-free tests for offline AIA observation preparation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from prom3theus.instruments.aia_euv.observation import (
    AIA_ASINH_SCALE_ALGORITHM,
    AIA_INTENSITY_UNIT,
    load_prepared_aia_observation,
)
from prom3theus.instruments.aia_euv.preparation import (
    AIA_CHANNELS_ANGSTROM,
    AIA_DEGRADATION_REFERENCE_EPOCH,
    AIA_RAY_DIRECTION_CONVENTION,
    AIA_SURFACE_FRAME,
    AIA_TIME_REPRESENTATION,
    RegisteredAIAChannelRaster,
    prepare_aia_observation_store,
)
from prom3theus.observations import ImageObservationStore
from prom3theus.resources import AIA_RESOURCE_SET_ID, validate_resource_set


SOLAR_RADIUS_M = 6.957e8
SOURCE_DIGEST = "a" * 64


def _convention_id() -> str:
    return validate_resource_set(AIA_RESOURCE_SET_ID)["scientific_contract"][
        "calibration_convention_id"
    ]


def _input_raster(
    channel: int,
    *,
    group: str = "exposure-0",
    time: float | None = None,
    factor: float = 2.0,
    applications: int = 0,
) -> RegisteredAIAChannelRaster:
    ray = torch.zeros(2, 2, 3, dtype=torch.float64)
    ray[..., 2] = -1.0
    surface = torch.zeros_like(ray)
    surface[..., 2] = SOLAR_RADIUS_M
    return RegisteredAIAChannelRaster(
        intensity=torch.tensor(
            [[-4.0, 8.0], [12.0 + channel / 100.0, 16.0]],
            dtype=torch.float64,
        ),
        uncertainty=torch.full((2, 2), 4.0, dtype=torch.float64),
        valid_mask=torch.tensor([[True, True], [False, True]]),
        on_disk_mask=torch.tensor([[True, False], [True, True]]),
        ray_direction=ray,
        surface_position_m=surface,
        absolute_tai_seconds=(
            1_700_000_000.125 + channel if time is None else time
        ),
        exposure_group=group,
        channel_angstrom=channel,
        solar_radius_m=SOLAR_RADIUS_M,
        degradation_factor=factor,
        degradation_correction_applications=applications,
        calibration_convention_id=_convention_id(),
        intensity_unit=AIA_INTENSITY_UNIT,
        time_representation=AIA_TIME_REPRESENTATION,
        surface_frame=AIA_SURFACE_FRAME,
        ray_direction_convention=AIA_RAY_DIRECTION_CONVENTION,
        provenance={"fixture": "registered channel map"},
    )


def test_streamed_statistics_match_global_medians(tmp_path):
    from prom3theus.instruments.aia_euv.observation import aia_objective_statistics

    inputs = [replace(_input_raster(channel, group=f"g{index}"),
                      intensity=_input_raster(channel).intensity * multiplier)
              for index, multiplier in enumerate([1., 3., 20.])
              for channel in AIA_CHANNELS_ANGSTROM]
    output = prepare_aia_observation_store(
        tmp_path / "streamed", rasters=inputs, source_files_sha256=SOURCE_DIGEST,
        preparation_dependencies={},
    )
    rasters, _, metadata = ImageObservationStore.load_sequence(output, mmap=False)
    assert metadata["objective_statistics"] == aia_objective_statistics(rasters, AIA_CHANNELS_ANGSTROM)
    assert not list(output.glob(".statistics*"))


def test_time_window_selects_complete_groups_and_validation_without_rewriting_store(tmp_path):
    inputs = [
        _input_raster(channel, group=f"g{index}", time=time)
        for index, time in enumerate((99., 100., 199., 200.))
        for channel in AIA_CHANNELS_ANGSTROM
    ]
    inputs += [_input_raster(channel, group="straddling", time=200. if channel == 211 else 199.)
               for channel in AIA_CHANNELS_ANGSTROM]
    path = prepare_aia_observation_store(
        tmp_path / "windowed", rasters=inputs, source_files_sha256=SOURCE_DIGEST,
        preparation_dependencies={},
    )
    before = (path / "manifest.json").read_bytes()
    config = {
        "type": "aia_euv", "directory": str(path), "channels_angstrom": [171, 193, 211],
        "selection": {"validation_exposure_group": "g0"},
        "loader": {"batch_size": 3, "validation_batch_size": 4, "workers": 0, "pin_memory": False},
    }
    loaded = load_prepared_aia_observation(
        config, expected_calibration_convention_id=_convention_id(), time_bounds_tai=(100., 200.),
    )
    assert loaded.spec.exposure_groups == ("g1", "g2")
    assert loaded.data_module.validation_exposure_group == "g1"
    assert len(loaded.data_module.rasters) == 6
    assert all(100 <= raster.absolute_tai_seconds < 200 for raster in loaded.data_module.rasters)
    assert (path / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="No complete AIA"):
        load_prepared_aia_observation(
            config, expected_calibration_convention_id=_convention_id(), time_bounds_tai=(201., 300.),
        )


def test_worker_failure_keeps_completed_images_in_final_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("PROM3THEUS_PREP_WORKERS", "1")

    def prepare(channel):
        if channel == 193:
            # The preceding worker already wrote the 171 image.
            assert (tmp_path / "failed" / "i0000_a000.npy").is_file()
            raise ValueError("broken input image")
        return _input_raster(channel)

    with pytest.raises(ValueError, match="broken input"):
        prepare_aia_observation_store(
            tmp_path / "failed", source_products=AIA_CHANNELS_ANGSTROM,
            raster_backend=prepare, source_files_sha256=SOURCE_DIGEST, preparation_dependencies={},
        )
    assert (tmp_path / "failed" / "i0000_a000.npy").is_file()
    assert not (tmp_path / "failed" / "manifest.json").exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "failed"]


def _inputs(**options) -> list[RegisteredAIAChannelRaster]:
    return [_input_raster(channel, **options) for channel in AIA_CHANNELS_ANGSTROM]


def _publish(path: Path, rasters=None) -> Path:
    return prepare_aia_observation_store(
        path,
        source_files_sha256=SOURCE_DIGEST,
        preparation_dependencies={
            "aiapy": "0.12.1",
            "registration": {"algorithm": "synthetic-test"},
        },
        rasters=_inputs() if rasters is None else rasters,
        observation_id="aia-sample",
    )


def _loader_config(path: Path) -> dict:
    return {
        "type": "aia_euv",
        "directory": str(path),
        "channels_angstrom": list(AIA_CHANNELS_ANGSTROM),
        "selection": {"validation_exposure_group": "exposure-0"},
        "loader": {
            "batch_size": 2,
            "validation_batch_size": 2,
            "workers": 0,
            "pin_memory": False,
        },
    }


def test_preparation_publishes_loader_compatible_store_and_applies_factor_once(
    tmp_path,
):
    path = _publish(tmp_path / "aia.image")
    rasters, names, metadata = ImageObservationStore.load_sequence(path, mmap=False)

    assert names == [
        "exposure-0-aia-171",
        "exposure-0-aia-193",
        "exposure-0-aia-211",
    ]
    assert [raster.channel_angstrom for raster in rasters] == [171, 193, 211]
    assert rasters[0].absolute_tai_seconds == 1_700_000_171.125
    torch.testing.assert_close(
        rasters[0].intensity,
        torch.tensor([[-2.0, 4.0], [6.855, 8.0]], dtype=torch.float64),
    )
    assert rasters[0].uncertainty is None
    assert all("uncertainty" not in record["arrays"] for record in ImageObservationStore.manifest(path)["rasters"])
    assert rasters[0].intensity[0, 0] < 0
    assert rasters[0].valid_mask.tolist() == [[True, False], [False, True]]
    calibration = rasters[0].metadata["calibration"]
    assert calibration["input_correction_applications"] == 0
    assert calibration["correction_performed_by_preparation"] is True
    assert calibration["total_correction_applications"] == 1
    assert calibration["reference_epoch"] == AIA_DEGRADATION_REFERENCE_EPOCH

    assert set(metadata) == {
        "adapter",
        "observation",
        "source_files_sha256",
        "preparation",
        "objective_statistics",
    }
    assert metadata["source_files_sha256"] == SOURCE_DIGEST
    assert metadata["objective_statistics"]["asinh_scale_algorithm"] == (
        AIA_ASINH_SCALE_ALGORITHM
    )
    assert set(
        metadata["objective_statistics"][
            "asinh_scale_by_channel_dn_s_pixel"
        ]
    ) == {"171", "193", "211"}
    assert metadata["preparation"]["degradation_correction"][
        "total_applications_per_raster"
    ] == 1

    loaded = load_prepared_aia_observation(
        _loader_config(path),
        expected_calibration_convention_id=_convention_id(),
    )
    assert loaded.spec.channels_angstrom == AIA_CHANNELS_ANGSTROM
    assert set(loaded.data_module.validation_dataloaders()) == {171, 193, 211}


def test_already_corrected_input_is_not_corrected_twice(tmp_path):
    inputs = _inputs(factor=4.0, applications=1)
    original = inputs[0].intensity.clone()
    path = _publish(tmp_path / "already-corrected.image", inputs)
    rasters, _, metadata = ImageObservationStore.load_sequence(path, mmap=False)

    torch.testing.assert_close(rasters[0].intensity, original)
    calibration = rasters[0].metadata["calibration"]
    assert calibration["correction_performed_by_preparation"] is False
    assert calibration["total_correction_applications"] == 1
    record = metadata["preparation"]["degradation_correction"]["records"][0]
    assert record["input_correction_applications"] == 1
    assert record["correction_performed_by_preparation"] is False


def test_dependency_injected_map_and_preparation_backends(tmp_path):
    loaded_sources = []
    prepared_maps = []

    def map_backend(source):
        loaded_sources.append(source)
        return {"source": source, "kind": "fake-map"}

    def preparation_backend(maps):
        prepared_maps.extend(maps)
        return _inputs()

    path = prepare_aia_observation_store(
        tmp_path / "backend.image",
        source_files_sha256=SOURCE_DIGEST,
        preparation_dependencies={"fake_backend": "1"},
        source_products=("aia-171.fits", "aia-193.fits", "aia-211.fits"),
        map_backend=map_backend,
        preparation_backend=preparation_backend,
    )

    assert loaded_sources == ["aia-171.fits", "aia-193.fits", "aia-211.fits"]
    assert [value["source"] for value in prepared_maps] == loaded_sources
    assert ImageObservationStore.manifest(path)["format"] == (
        "prom3theus.image_observation_store"
    )


@pytest.mark.parametrize(
    "rasters, message",
    [
        (_inputs()[:-1], "must contain exactly channels"),
        ([*_inputs(), _input_raster(171)], "one raster per exposure group"),
    ],
)
def test_preparation_rejects_incomplete_or_duplicate_groups(
    tmp_path, rasters, message
):
    path = tmp_path / "invalid.image"
    with pytest.raises(ValueError, match=message):
        _publish(path, rasters)
    assert not (path / "manifest.json").exists()


def test_preparation_rejects_calibration_unit_and_double_correction_errors(
    tmp_path,
):
    with pytest.raises(ValueError, match="intensity_unit"):
        replace(_input_raster(171), intensity_unit="DN pixel^-1")
    with pytest.raises(ValueError, match="double correction"):
        replace(_input_raster(171), degradation_correction_applications=2)

    wrong_convention = replace(
        _input_raster(171), calibration_convention_id="sha256:" + "f" * 64
    )
    path = tmp_path / "wrong-convention.image"
    with pytest.raises(ValueError, match="does not match"):
        _publish(path, [wrong_convention, *_inputs()[1:]])
    assert not (path / "manifest.json").exists()


@pytest.mark.parametrize(
    "change, message",
    [
        ({"time_representation": "unix"}, "unix_tai"),
        ({"surface_frame": "helioprojective"}, "Carrington"),
        ({"ray_direction_convention": "sun_to_observer"}, "observer_to_sun"),
    ],
)
def test_input_contract_rejects_ambiguous_coordinate_metadata(change, message):
    with pytest.raises(ValueError, match=message):
        replace(_input_raster(171), **change)


def test_preparation_rejects_bad_physical_geometry_and_ignores_legacy_uncertainty(tmp_path):
    base = _input_raster(171)
    reversed_ray = replace(base, ray_direction=-base.ray_direction)
    with pytest.raises(ValueError, match="geometry"):
        _publish(tmp_path / "reversed.image", [reversed_ray, *_inputs()[1:]])

    uncertainty = base.uncertainty.clone()
    uncertainty[0, 0] = 0.0
    invalid_uncertainty = replace(base, uncertainty=uncertainty)
    _publish(
        tmp_path / "zero-uncertainty.image",
        [invalid_uncertainty, *_inputs()[1:]],
    )


def test_preparation_requires_exactly_one_input_route(tmp_path):
    arguments = {
        "output_directory": tmp_path / "route.image",
        "source_files_sha256": SOURCE_DIGEST,
        "preparation_dependencies": {},
    }
    with pytest.raises(TypeError, match="exactly one"):
        prepare_aia_observation_store(**arguments)
    with pytest.raises(TypeError, match="exactly one"):
        prepare_aia_observation_store(
            **arguments,
            rasters=_inputs(),
            source_products=("one",),
            preparation_backend=lambda _: _inputs(),
        )
