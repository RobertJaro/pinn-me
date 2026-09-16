import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from prom3theus.instruments.aia_euv.observation import (
    AIA_ASINH_SCALE_ALGORITHM,
    AIA_INTENSITY_UNIT,
    aia_objective_statistics,
    channel_intensity_scales,
    load_prepared_aia_observation,
)
from prom3theus.instruments.aia_euv.preparation import (
    AIA_DEGRADATION_OPERATION,
    AIA_DEGRADATION_REFERENCE_EPOCH,
    AIA_PREPARATION_FORMAT,
    AIA_PREPARATION_VERSION,
    AIA_RAY_DIRECTION_CONVENTION,
    AIA_SURFACE_FRAME,
    AIA_TIME_REPRESENTATION,
)
from prom3theus.observations import (
    ImageObservationRaster,
    ImageObservationSpec,
    ImageObservationStore,
)


RADIUS_M = 6.957e8
CONVENTION_ID = "sha256:" + "1" * 64


def _raster(channel: int, convention_id: str = CONVENTION_ID) -> ImageObservationRaster:
    ray = torch.zeros(2, 2, 3)
    ray[..., 2] = -1.0
    surface = torch.zeros_like(ray)
    surface[..., 2] = RADIUS_M
    return ImageObservationRaster(
        intensity=torch.tensor([[channel / 10, -1.0], [2.0, 3.0]]),
        uncertainty=torch.full((2, 2), 0.5),
        ray_direction=ray,
        surface_position_m=surface,
        valid_mask=torch.ones(2, 2, dtype=torch.bool),
        absolute_tai_seconds=1_700_000_000.0 + channel,
        channel_angstrom=channel,
        exposure_group="group-0",
        metadata={
            "intensity_unit": AIA_INTENSITY_UNIT,
            "observation_time": {
                "representation": AIA_TIME_REPRESENTATION,
                "absolute_tai_seconds": 1_700_000_000.0 + channel,
            },
            "ray_geometry": {
                "solar_radius_m": RADIUS_M,
                "frame": AIA_SURFACE_FRAME,
                "surface_position": "photospheric_surface_intersection",
                "ray_direction": AIA_RAY_DIRECTION_CONVENTION,
            },
            "calibration": {
                "calibration_convention_id": convention_id,
                "degradation_factor": 1.0,
                "input_correction_applications": 1,
                "correction_performed_by_preparation": False,
                "total_correction_applications": 1,
                "operation": AIA_DEGRADATION_OPERATION,
                "reference_epoch": AIA_DEGRADATION_REFERENCE_EPOCH,
            },
            "mask": {
                "definition": "upstream_valid_mask AND on_disk_mask",
                "on_disk_pixel_count": 4,
                "valid_on_disk_pixel_count": 4,
            },
            "provenance": {"fixture": True},
        },
    )


def _store(path: Path, *, convention_id: str = CONVENTION_ID, rasters=None) -> None:
    if rasters is None:
        rasters = [_raster(channel, convention_id) for channel in (171, 193, 211)]
    groups = tuple(dict.fromkeys(raster.exposure_group for raster in rasters))
    spec = ImageObservationSpec(
        observation_id="aia_sample",
        observation_type="aia_euv",
        instrument_type="aia_temperature_response",
        intensity_unit=AIA_INTENSITY_UNIT,
        channels_angstrom=(171, 193, 211),
        exposure_groups=groups,
        calibration_convention={
            "calibration_convention_id": convention_id,
            "degradation_correction": {
                "operation": AIA_DEGRADATION_OPERATION,
                "reference_epoch": AIA_DEGRADATION_REFERENCE_EPOCH,
                "total_applications_per_raster": 1,
            },
            "measurement_semantics": "per_native_pixel",
            "sensitivity_convention": "reference_epoch",
        },
        geometry_convention={
            "ray_direction": AIA_RAY_DIRECTION_CONVENTION,
            "surface_position": "photospheric_surface_intersection",
            "surface_frame": AIA_SURFACE_FRAME,
            "time_representation": AIA_TIME_REPRESENTATION,
        },
        required_resource_sets=("aia_euv_v1",),
    )
    ImageObservationStore.save_sequence(
        path,
        rasters,
        source_signature="a" * 64,
        metadata={
            "adapter": "aia_euv",
            "observation": spec.metadata(),
            "source_files_sha256": "b" * 64,
            "preparation": {
                "format": AIA_PREPARATION_FORMAT,
                "version": AIA_PREPARATION_VERSION,
                "channels_angstrom": [171, 193, 211],
                "exposure_groups": list(groups),
                "output_intensity_unit": AIA_INTENSITY_UNIT,
                "calibration_convention_id": convention_id,
                "time_representation": AIA_TIME_REPRESENTATION,
                "surface_frame": AIA_SURFACE_FRAME,
                "ray_direction": AIA_RAY_DIRECTION_CONVENTION,
                "degradation_correction": {
                    "operation": AIA_DEGRADATION_OPERATION,
                    "reference_epoch": AIA_DEGRADATION_REFERENCE_EPOCH,
                    "total_applications_per_raster": 1,
                    "records": [
                        {
                            "exposure_group": raster.exposure_group,
                            "channel_angstrom": raster.channel_angstrom,
                            "degradation_factor": 1.0,
                            "input_correction_applications": 1,
                            "correction_performed_by_preparation": False,
                            "total_correction_applications": 1,
                        }
                        for raster in rasters
                    ],
                },
                "preparation_dependencies": {"fixture": "1"},
            },
            "objective_statistics": aia_objective_statistics(
                rasters, spec.channels_angstrom
            ),
        },
    )


def _mutate_manifest(path: Path, mutation) -> None:
    manifest_path = path / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(document)
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _config(path: Path) -> dict:
    return {
        "type": "aia_euv",
        "directory": str(path),
        "channels_angstrom": [171, 193, 211],
        "selection": {"validation_exposure_group": "group-0"},
        "loader": {
            "batch_size": 3,
            "validation_batch_size": 4,
            "workers": 0,
            "pin_memory": False,
        },
    }


def test_prepared_aia_adapter_preserves_statistics_and_complete_group(tmp_path):
    path = tmp_path / "aia.image"
    _store(path)
    loaded = load_prepared_aia_observation(
        _config(path), expected_calibration_convention_id=CONVENTION_ID
    )
    assert loaded.spec.channels_angstrom == (171, 193, 211)
    assert all(value > 0 for value in loaded.asinh_scales)
    assert (
        loaded.store_metadata["objective_statistics"]["asinh_scale_algorithm"]
        == AIA_ASINH_SCALE_ALGORITHM
    )
    assert set(loaded.data_module.validation_dataloaders()) == {171, 193, 211}
    assert loaded.intensity_scales == pytest.approx((17.1, 19.3, 21.1))
    for channel, loader in loaded.data_module.validation_dataloaders().items():
        target = next(iter(loader))["intensity"]
        raster = loaded.data_module.validation_datasets[channel].raster
        torch.testing.assert_close(target, raster.intensity.flatten() / loaded.data_module.intensity_scales[channel])
        assert float(target.abs().max()) == pytest.approx(1.0)


def test_channel_maxima_use_all_valid_exposures_and_preserve_negative_values():
    first = replace(
        _raster(171), intensity=torch.tensor([[-20.0, 4.0], [3.0, 1e20]]),
        valid_mask=torch.tensor([[True, True], [True, False]]),
    )
    second = replace(_raster(171), intensity=torch.tensor([[5.0, 2.0], [30.0, 4.0]]))
    third = replace(_raster(193), intensity=torch.zeros(2, 2))
    assert channel_intensity_scales([first, second, third], (171, 193)) == {
        171: 30.0, 193: 1.0,
    }
    assert channel_intensity_scales([first], (171,)) == {171: 20.0}


def test_prepared_aia_channel_maxima_are_estimated_after_time_selection(tmp_path):
    path = tmp_path / "time-series.image"
    early = [_raster(channel) for channel in (171, 193, 211)]
    late = []
    for raster in early:
        time = raster.absolute_tai_seconds + 4000.0
        late.append(replace(
            raster, intensity=raster.intensity * 100.0, exposure_group="group-1",
            absolute_tai_seconds=time,
            metadata={**raster.metadata, "observation_time": {
                **raster.metadata["observation_time"], "absolute_tai_seconds": time,
            }},
        ))
    _store(path, rasters=[*early, *late])
    before = ImageObservationStore.manifest(path)
    full = load_prepared_aia_observation(
        _config(path), expected_calibration_convention_id=CONVENTION_ID
    )
    selected = load_prepared_aia_observation(
        _config(path), expected_calibration_convention_id=CONVENTION_ID,
        time_bounds_tai=(1_700_000_000.0, 1_700_000_300.0),
    )
    assert full.intensity_scales == pytest.approx((1710.0, 1930.0, 2110.0))
    assert selected.intensity_scales == pytest.approx((17.1, 19.3, 21.1))
    assert selected.asinh_scales == full.asinh_scales
    assert selected.spec.exposure_groups == ("group-0",)
    assert ImageObservationStore.manifest(path) == before
    for raster, original in zip(selected.data_module.rasters, early, strict=True):
        torch.testing.assert_close(raster.intensity, original.intensity)


def test_prepared_aia_adapter_rejects_response_calibration_mismatch(tmp_path):
    path = tmp_path / "aia.image"
    _store(path)
    with pytest.raises(ValueError, match="does not match"):
        load_prepared_aia_observation(
            _config(path),
            expected_calibration_convention_id="sha256:" + "2" * 64,
        )


def test_legacy_uncertainty_store_uses_intensity_only_scales_without_rewriting(tmp_path):
    path = tmp_path / "legacy.image"
    _store(path)

    def legacy_statistics(document):
        statistics = document["metadata"]["objective_statistics"]
        statistics["asinh_scale_algorithm"] = "max(median(abs(valid_intensity)),median(valid_sigma))"
        statistics["asinh_scale_by_channel_dn_s_pixel"] = {str(c): 1e8 for c in (171, 193, 211)}
    _mutate_manifest(path, legacy_statistics)
    before = ImageObservationStore.manifest(path)
    loaded = load_prepared_aia_observation(
        _config(path), expected_calibration_convention_id=CONVENTION_ID
    )
    assert loaded.asinh_scales == (2.0, 2.0, 2.0)
    assert all(raster.uncertainty is None for raster in loaded.data_module.rasters)
    assert "uncertainty" not in next(iter(loaded.data_module.validation_dataloaders()[171]))
    assert ImageObservationStore.manifest(path) == before


def test_channel_scale_depends_only_on_observed_intensity():
    from dataclasses import replace
    from prom3theus.instruments.aia_euv.observation import robust_asinh_scales

    raster = replace(_raster(171), uncertainty=torch.full((2, 2), 1e8))
    assert robust_asinh_scales([raster], [171]) == {"171": 2.0}


def test_prepared_aia_adapter_rejects_unversioned_or_double_corrected_metadata(
    tmp_path,
):
    unversioned = tmp_path / "unversioned.image"
    _store(unversioned)
    _mutate_manifest(
        unversioned,
        lambda document: document["metadata"]["preparation"].__setitem__(
            "version", 2
        ),
    )
    with pytest.raises(ValueError, match="format/version"):
        load_prepared_aia_observation(
            _config(unversioned),
            expected_calibration_convention_id=CONVENTION_ID,
        )

    double = tmp_path / "double.image"
    _store(double)
    _mutate_manifest(
        double,
        lambda document: document["metadata"]["preparation"][
            "degradation_correction"
        ].__setitem__("total_applications_per_raster", 2),
    )
    with pytest.raises(ValueError, match="exactly one correction"):
        load_prepared_aia_observation(
            _config(double),
            expected_calibration_convention_id=CONVENTION_ID,
        )


def test_prepared_aia_adapter_rejects_missing_or_inconsistent_records(tmp_path):
    missing = tmp_path / "missing.image"
    _store(missing)
    _mutate_manifest(
        missing,
        lambda document: document["metadata"]["preparation"][
            "degradation_correction"
        ]["records"].pop(),
    )
    with pytest.raises(ValueError, match="cover every configured"):
        load_prepared_aia_observation(
            _config(missing),
            expected_calibration_convention_id=CONVENTION_ID,
        )

    inconsistent = tmp_path / "inconsistent.image"
    _store(inconsistent)
    _mutate_manifest(
        inconsistent,
        lambda document: document["rasters"][0]["metadata"][
            "calibration"
        ].__setitem__("degradation_factor", 2.0),
    )
    with pytest.raises(ValueError, match="factor does not match"):
        load_prepared_aia_observation(
            _config(inconsistent),
            expected_calibration_convention_id=CONVENTION_ID,
        )


def test_prepared_aia_adapter_rejects_time_geometry_unit_and_statistics_drift(
    tmp_path,
):
    wrong_time = tmp_path / "wrong-time.image"
    _store(wrong_time)
    _mutate_manifest(
        wrong_time,
        lambda document: document["metadata"]["preparation"].__setitem__(
            "time_representation", "unix"
        ),
    )
    with pytest.raises(ValueError, match="geometry/time"):
        load_prepared_aia_observation(
            _config(wrong_time),
            expected_calibration_convention_id=CONVENTION_ID,
        )

    wrong_unit = tmp_path / "wrong-unit.image"
    _store(wrong_unit)
    _mutate_manifest(
        wrong_unit,
        lambda document: document["metadata"]["preparation"].__setitem__(
            "output_intensity_unit", "DN pixel^-1"
        ),
    )
    with pytest.raises(ValueError, match="output unit"):
        load_prepared_aia_observation(
            _config(wrong_unit),
            expected_calibration_convention_id=CONVENTION_ID,
        )

    wrong_scale = tmp_path / "wrong-scale.image"
    _store(wrong_scale)
    _mutate_manifest(
        wrong_scale,
        lambda document: document["metadata"]["objective_statistics"][
            "asinh_scale_by_channel_dn_s_pixel"
        ].__setitem__("171", 999.0),
    )
    with pytest.raises(ValueError, match="inconsistent with the stored rasters"):
        load_prepared_aia_observation(
            _config(wrong_scale),
            expected_calibration_convention_id=CONVENTION_ID,
        )
