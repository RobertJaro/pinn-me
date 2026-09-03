"""Scientific invariants for the modular HMI LTE observation adapter."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

from prom3theus.core import sha256_file
from prom3theus.instruments.hmi.acquisition import (
    format_jsoc_time,
    hmi_observation_wcs_header,
    parse_hmi_tai_time,
    read_acquisition_header,
    read_stokes_cube,
    resolve_acquisition_groups,
    resolve_segment_files,
)
from prom3theus.instruments.hmi.calibration import calibrate_stokes
from prom3theus.instruments.hmi.constants import (
    HMI_WAVELENGTH_CENTER,
    HMI_WAVELENGTH_GRID,
)
from prom3theus.instruments.hmi.geometry import (
    HMIDetectorGrid,
    detector_stokes_basis,
    observer_velocity_rwn,
    project_observer_velocity_to_los,
)
from prom3theus.instruments.hmi.operator import HMIFilterProfiles
from prom3theus.instruments.hmi.raster import load_raster
from prom3theus.instruments.hmi.response import (
    HMIResponseArchive,
    MANIFEST_FORMAT,
)
from prom3theus.instruments.hmi.timeline import HMIDataModule
from prom3theus.inversion.runner import _inject_coordinate_contract
from prom3theus.observations import describe_observation_data
from prom3theus.training.lightning import LTEInversionModule


def _write_response(path: Path) -> HMIResponseArchive:
    nominal = HMI_WAVELENGTH_GRID.to_value("Angstrom")
    nodes = np.linspace(-0.01, 0.01, 3)
    offsets = (nodes[None] - nominal[:, None]).astype(np.float32)
    weights = np.zeros((2, 2, 6, 3), dtype=np.float32)
    for filter_index in range(6):
        weights[..., filter_index, 1] = 0.7 + 0.01 * filter_index
    continuum = 1.0 - weights.sum(axis=-1)
    np.savez(
        path,
        offsets=offsets,
        weights=weights,
        continuum_weights=continuum,
        metadata=np.array(
            json.dumps(
                {
                    "format": "prom3theus.hmi_response.v1",
                    "half_width": 0.65,
                    "wavelength_unit": "Angstrom",
                    "ccd_size": 2,
                    "phase_map_shape": [2, 2],
                    "samples": 3,
                    "phase_map_fsn": 1,
                    "HCAMID": 3,
                    "record": "test[1]",
                    "T_START": "2023.10.10_19:25:31_TAI",
                    "T_REC": "2023.10.10_19:26:26_TAI",
                    "T_STOP": "2023.10.10_19:27:22_TAI",
                }
            )
        ),
    )
    return HMIResponseArchive(path)


def _solar_reference() -> dict:
    wavelength = np.linspace(6172.5, 6174.2, 64)
    return {
        "schema_version": 1,
        "source": {"description": "synthetic test atlas"},
        "units": {
            "wavelength": "standard-air angstrom",
            "intensity_radiance": "W m^-3 sr^-1",
            "continuum_radiance": "W m^-3 sr^-1",
        },
        "wavelength_range_air_angstrom": [6172.5, 6174.2],
        "wavelength_air_angstrom": wavelength.tolist(),
        "intensity_radiance_w_m3_sr": np.full(64, 3.0e13).tolist(),
        "continuum_radiance_w_m3_sr": np.full(64, 3.0e13).tolist(),
        "limb_darkening": {"reference": "test"},
    }


def _fits_header(*, date_obs: str, t_rec: str) -> fits.Header:
    return fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CDELT1": 0.5,
            "CDELT2": 0.5,
            "CRPIX1": 2.5,
            "CRPIX2": 2.5,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "DATE-OBS": date_obs,
            "T_OBS": t_rec,
            "T_REC": t_rec,
            "CAMERA": 3,
            "HCAMID": 3,
            "QUALITY": 0,
            "DSUN_OBS": 1.496e11,
            "RSUN_REF": 6.957e8,
            "RSUN_OBS": 959.2,
            "CRLN_OBS": 180.0,
            "CRLT_OBS": -7.0,
            "CROTA2": 0.0,
            "OBS_VR": 1200.0,
            "OBS_VW": -300.0,
            "OBS_VN": 50.0,
            "CCD_X0": 2046,
            "CCD_Y0": 2046,
            "TELESCOP": "SDO/HMI",
            "INSTRUME": "HMI_SIDE1",
        }
    )


def _write_acquisition(directory: Path, timestamp: str, header: fits.Header) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        for component_index, component in enumerate("IQUV"):
            for filter_index in range(6):
                value = 50000.0 if component == "I" else 1.0e-4 * (component_index + 1)
                path = directory / (
                    f"hmi.S_720s.{timestamp}_TAI.3.{component}{filter_index}.fits"
                )
                fits.writeto(path, np.full((4, 4), value, dtype=np.float32), header)


def test_detector_basis_tracks_rotated_ccd_up_and_is_right_handed():
    angle = np.deg2rad(37.0)
    pixel_scale_matrix = 0.5 * np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    los = np.array([[[0.0, 0.0, 1.0]]])
    hpc_x = np.array([[[1.0, 0.0, 0.0]]])
    hpc_y = np.array([[[0.0, 1.0, 0.0]]])
    q_axis, u_axis = detector_stokes_basis(los, hpc_x, hpc_y, pixel_scale_matrix)

    detector_right = np.array([np.cos(angle), np.sin(angle), 0.0])
    detector_up = np.array([-np.sin(angle), np.cos(angle), 0.0])
    np.testing.assert_allclose(q_axis[0, 0], detector_up, atol=1.0e-15)
    np.testing.assert_allclose(u_axis[0, 0], -detector_right, atol=1.0e-15)
    np.testing.assert_allclose(
        np.cross(q_axis[0, 0], u_axis[0, 0]), los[0, 0], atol=1.0e-15
    )


def test_observer_velocity_projection_preserves_component_signs():
    angle = np.deg2rad(12.0)
    longitude = np.deg2rad(37.0)
    latitude = np.deg2rad(-7.0)
    radial = np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    west = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.array(
        [
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        ]
    )
    toward_observer = np.array(
        [
            radial,
            np.cos(angle) * radial - np.sin(angle) * west,
            np.cos(angle) * radial - np.sin(angle) * north,
        ]
    )
    projected = project_observer_velocity_to_los(
        1.5e11 * radial,
        toward_observer,
        np.array([1200.0, 300.0, 80.0]),
    )
    np.testing.assert_allclose(projected[0], 1200.0, atol=1.0e-12)
    np.testing.assert_allclose(
        projected[1],
        1200.0 * np.cos(angle) - 300.0 * np.sin(angle),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        projected[2],
        1200.0 * np.cos(angle) - 80.0 * np.sin(angle),
        atol=1.0e-12,
    )


def test_observer_velocity_requires_all_keywords_and_identical_segments():
    headers = [
        fits.Header({"OBS_VR": 1200.0, "OBS_VW": -300.0, "OBS_VN": 50.0})
        for _ in range(24)
    ]
    np.testing.assert_array_equal(
        observer_velocity_rwn(headers), [1200.0, -300.0, 50.0]
    )
    missing = [header.copy() for header in headers]
    del missing[3]["OBS_VN"]
    with pytest.raises(KeyError, match=r"I3.*OBS_VN"):
        observer_velocity_rwn(missing)
    inconsistent = [header.copy() for header in headers]
    inconsistent[7]["OBS_VW"] = -299.0
    with pytest.raises(ValueError, match=r"identical.*Q1"):
        observer_velocity_rwn(inconsistent)
    with pytest.raises(ValueError, match="exactly 24"):
        observer_velocity_rwn(headers[:-1])


def test_detector_grid_rejects_coordinates_outside_the_physical_ccd():
    with pytest.raises(ValueError, match="inside the CCD"):
        HMIDetectorGrid((4, 4), (4094, 0))
    grid = HMIDetectorGrid((4, 4), (100, 200))
    with pytest.raises(IndexError, match="outside the image"):
        grid.at_indices(np.array([[4, 0]]))


def test_calibration_shifts_atlas_without_shifting_filter_grid():
    class DeltaResponse:
        def __init__(self):
            self.observed_wavelength = None

        def quadrature_wavelength(self, observed_wavelength):
            self.observed_wavelength = np.asarray(observed_wavelength).copy()
            return torch.tensor([6173.0, 6173.5], dtype=torch.float64)

        def sample(self, detector_coordinates):
            count = np.asarray(detector_coordinates).reshape(-1, 2).shape[0]
            weights = torch.zeros((count, 6, 2), dtype=torch.float64)
            weights[..., 0] = 1.0
            return {
                "spectral_weights": weights,
                "continuum_weights": torch.zeros((count, 6), dtype=torch.float64),
            }

    wavelength = np.linspace(6171.0, 6175.0, 4001)
    intensity = 2.0e13 + 1.0e13 * (wavelength - wavelength[0])
    reference = {
        **_solar_reference(),
        "wavelength_range_air_angstrom": [wavelength[0], wavelength[-1]],
        "wavelength_air_angstrom": wavelength.tolist(),
        "intensity_radiance_w_m3_sr": intensity.tolist(),
        "continuum_radiance_w_m3_sr": np.full_like(wavelength, 3.0e13).tolist(),
    }

    def calibrated_gain(observer_velocity):
        sampler = DeltaResponse()
        stokes = np.zeros((1, 1, 4, 6), dtype=np.float64)
        stokes[..., 0, :] = 1.0
        _, metadata = calibrate_stokes(
            stokes,
            np.ones((1, 1)),
            np.ones((1, 1), dtype=bool),
            np.zeros((1, 1, 2), dtype=np.float32),
            sampler,
            reference,
            np.full((1, 1), observer_velocity),
            quiet_sun_max_fractional_polarization=1.0,
            quiet_sun_trim_quantiles=(0.0, 1.0),
            minimum_quiet_sun_pixels=1,
            calibration_sample_limit=1,
        )
        return metadata, sampler.observed_wavelength

    positive, instrument_wavelength = calibrated_gain(30_000.0)
    negative, _ = calibrated_gain(-30_000.0)
    beta = 30_000.0 / 299_792_458.0
    atlas_wavelength = 6173.0 * np.sqrt((1.0 - beta) / (1.0 + beta))
    expected_gain = np.interp(atlas_wavelength, wavelength, intensity)
    assert positive["detector_to_radiance_w_m3_sr_per_raw_unit"] == pytest.approx(
        expected_gain
    )
    assert (
        positive["detector_to_radiance_w_m3_sr_per_raw_unit"]
        < negative["detector_to_radiance_w_m3_sr_per_raw_unit"]
    )
    np.testing.assert_allclose(
        instrument_wavelength,
        (
            HMI_WAVELENGTH_CENTER.to_value("Angstrom")
            + HMI_WAVELENGTH_GRID.to_value("Angstrom")
        )[::-1],
    )


def test_acquisition_grouping_rejects_incomplete_sets(tmp_path):
    for timestamp in ("20240323_234800", "20240324_000000"):
        for component in "IQUV":
            for filter_index in range(6):
                (
                    tmp_path
                    / f"hmi.S_720s.{timestamp}_TAI.3.{component}{filter_index}.fits"
                ).touch()
    groups = resolve_acquisition_groups(directory=tmp_path)
    assert [name for name, _ in groups] == [
        "hmi.S_720s.20240323_234800_TAI.3",
        "hmi.S_720s.20240324_000000_TAI.3",
    ]
    assert [len(paths) for _, paths in groups] == [24, 24]
    groups[0][1][0].unlink()
    with pytest.raises(ValueError, match="missing"):
        resolve_acquisition_groups(directory=tmp_path)


def test_acquisition_grouping_rejects_non_camera_three_filenames(tmp_path):
    for component in "IQUV":
        for filter_index in range(6):
            (
                tmp_path
                / f"hmi.S_720s.20240324_000000_TAI.2.{component}{filter_index}.fits"
            ).touch()

    with pytest.raises(ValueError, match="CAMERA=3 filenames"):
        resolve_acquisition_groups(directory=tmp_path)


def test_physical_time_uses_t_obs_while_record_identity_uses_t_rec(tmp_path):
    header = _fits_header(
        date_obs="2024-03-24T00:58:36.000",
        t_rec="2024.03.24_01:00:00_TAI",
    )
    header["T_OBS"] = "2024.03.24_00:59:58_TAI"
    _write_acquisition(tmp_path, "20240324_010000", header)

    acquisition = read_acquisition_header(
        tmp_path / "hmi.S_720s.20240324_010000_TAI.3.I0.fits"
    )
    assert acquisition["date"].tai.isot == "2024-03-24T00:59:58.000"
    assert acquisition["observation_time"] == "2024.03.24_00:59:58_TAI"
    assert acquisition["record_time"] == "2024.03.24_01:00:00_TAI"
    assert acquisition["acquisition_key"] == ("2024.03.24_01:00:00_TAI|HCAMID=3")
    map_header = hmi_observation_wcs_header(header)
    assert map_header["DATE-OBS"] == "2024.03.24_00:59:58_TAI"
    assert "T_OBS" not in map_header


def test_jsoc_time_conversion_preserves_timezone_aware_instants():
    utc = datetime(2024, 3, 24, 1, 0, tzinfo=timezone.utc)

    assert format_jsoc_time(utc) == "2024.03.24_01:00:37_TAI"
    assert format_jsoc_time("2024-03-24T02:00:00+01:00") == ("2024.03.24_01:00:37_TAI")
    assert parse_hmi_tai_time(utc).utc.isot == "2024-03-24T01:00:00.000"


def test_acquisition_header_requires_explicit_tai_record_time(tmp_path):
    header = _fits_header(
        date_obs="2024-03-24T00:58:36.000",
        t_rec="2024.03.24_01:00:00_TAI",
    )
    header["T_REC"] = "2024-03-24T01:00:00Z"
    _write_acquisition(tmp_path, "20240324_010000", header)

    with pytest.raises(ValueError, match="invalid T_OBS/T_REC"):
        read_acquisition_header(tmp_path / "hmi.S_720s.20240324_010000_TAI.3.I0.fits")


def test_acquisition_header_rejects_non_camera_three(tmp_path):
    header = _fits_header(
        date_obs="2024-03-24T00:58:36.000",
        t_rec="2024.03.24_01:00:00_TAI",
    )
    header["CAMERA"] = 2
    _write_acquisition(tmp_path, "20240324_010000", header)

    with pytest.raises(ValueError, match="CAMERA=3"):
        read_acquisition_header(tmp_path / "hmi.S_720s.20240324_010000_TAI.3.I0.fits")


def test_stokes_cube_rejects_segment_identity_mismatch(tmp_path):
    header = _fits_header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
    )
    _write_acquisition(tmp_path, "20240324_010000", header)
    mismatched = tmp_path / "hmi.S_720s.20240324_010000_TAI.3.Q1.fits"
    fits.setval(mismatched, "T_REC", value="2024.03.24_01:12:00_TAI")
    segments = resolve_segment_files(tmp_path)
    with pytest.raises(ValueError, match=r"Q1.*not aligned"):
        read_stokes_cube(segments, require_quality_zero=True)


def test_filter_operator_uses_packaged_provenance_and_preserves_constant_stokes():
    quadrature = HMI_WAVELENGTH_CENTER.value + np.linspace(-0.4, 0.4, 3)
    operator = HMIFilterProfiles(
        quadrature_wavelength_angstrom=quadrature,
        inner_half_width_angstrom=0.65,
    )
    observed = torch.tensor(
        [6173.1713, 6173.2401, 6173.3089, 6173.3777, 6173.4465, 6173.5153]
    )
    synthesis = operator.synthesis_grid(observed)
    stokes = (
        torch.tensor([2.0, -0.2, 0.1, 0.05])
        .reshape(1, 4, 1)
        .expand(2, 4, synthesis.numel())
    )
    weights = torch.zeros(2, 6, 3)
    weights[..., 1] = 0.75
    continuum = torch.full((2, 6), 0.25)
    output = operator(
        stokes,
        synthesis,
        observed,
        spectral_weights=weights,
        continuum_weights=continuum,
    )
    expected = stokes[..., :1].expand(2, 4, 6).clone()
    expected[:, 1:] *= 0.75
    torch.testing.assert_close(output, expected)
    with pytest.raises(TypeError, match="data_directory"):
        HMIFilterProfiles(
            data_directory="external",
            quadrature_wavelength_angstrom=quadrature,
        )


def test_tiny_sequence_builds_calibrated_rasters_and_response_batches(
    tmp_path, monkeypatch
):
    data_directory = tmp_path / "stokes"
    response_directory = tmp_path / "responses"
    data_directory.mkdir()
    response_directory.mkdir()
    first_header = _fits_header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
    )
    second_header = _fits_header(
        date_obs="2024-03-24T01:11:20.000",
        t_rec="2024.03.24_01:12:00_TAI",
    )
    _write_acquisition(data_directory, "20240324_010000", first_header)
    _write_acquisition(data_directory, "20240324_011200", second_header)
    response_file = response_directory / "tiny.npz"
    sampler = _write_response(response_file)
    first_key = "2024.03.24_01:00:00_TAI|HCAMID=3"
    second_key = "2024.03.24_01:12:00_TAI|HCAMID=3"
    (response_directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "profiles": {
                    "INVPHMAP=1|HCAMID=3": {
                        "file": response_file.name,
                        "phase_map_fsn": 1,
                        "hcamid": 3,
                        "record": "test[1]",
                        "sha256": sha256_file(response_file),
                    }
                },
                "acquisitions": {
                    first_key: {
                        "record_time": "2024.03.24_01:00:00_TAI",
                        "observation_time": "2024.03.24_01:00:00_TAI",
                        "hcamid": 3,
                        "profile": "INVPHMAP=1|HCAMID=3",
                    },
                    second_key: {
                        "record_time": "2024.03.24_01:12:00_TAI",
                        "observation_time": "2024.03.24_01:12:00_TAI",
                        "hcamid": 3,
                        "profile": "INVPHMAP=1|HCAMID=3",
                    },
                },
            }
        )
    )

    raster = load_raster(
        str(data_directory / "*010000*.fits"),
        response_directory,
        quiet_sun_max_fractional_polarization=1.0,
        quiet_sun_trim_quantiles=(0.0, 1.0),
        minimum_quiet_sun_pixels=1,
        calibration_sample_limit=16,
        response_sampler=sampler,
        solar_reference=_solar_reference(),
    )
    assert raster.stokes.shape == (4, 4, 4, 6)
    assert raster.valid_mask.all()
    assert set(raster.auxiliary) == {
        "observer_los_velocity_m_per_s",
        "instrument_response:spectral_weights",
        "instrument_response:continuum_weights",
    }

    monkeypatch.setattr(
        "prom3theus.instruments.hmi.timeline.load_solar_reference",
        lambda *args, **kwargs: _solar_reference(),
    )
    module = HMIDataModule(
        directory=data_directory,
        transmission_profile_directory=response_directory,
        validation_raster=1,
        quiet_sun_max_fractional_polarization=1.0,
        quiet_sun_trim_quantiles=(0.0, 1.0),
        minimum_quiet_sun_pixels=1,
        calibration_sample_limit=16,
        progress=False,
    )
    module.setup()
    assert module.raster_names == [first_key, second_key]
    assert len(module.dataset) == 32
    assert module.raster.metadata["coordinates"]["time_affine"]["range_hours"] == [
        0.0,
        pytest.approx(0.2),
    ]
    batch = next(iter(module.val_dataloader()))
    assert batch["instrument_response"]["spectral_weights"].shape[-2:] == (6, 3)
    assert batch["observer_los_velocity_m_per_s"].ndim == 1
    assert torch.isfinite(batch["observer_los_velocity_m_per_s"]).all()

    observation, _ = describe_observation_data(module, {"type": "hmi_stokes"})
    atmosphere_config = {
        "height_input_scale_m": 1.0e6,
        "time_dependent": True,
        "shell_height_bounds_Mm": [1.5, -0.1],
        "tangent_margin_m": 1_000.0,
        "reference_atmosphere_config": "falc_82",
        "temperature_log10_bounds": [3.4, 4.0],
        "temperature_log_scale": 0.1,
        "velocity_scale_m_per_s": 1_000.0,
        "velocity_max_m_per_s": 100_000.0,
        "magnetic_scale_gauss": 100.0,
        "microturbulence_log10_bounds": [1.0, 4.0],
        "microturbulence_log_scale": 0.2,
        "gas_pressure_log10_bounds": [-1.5, 6.0],
        "gas_pressure_log_scale": 0.5,
        "model_config": {
            "type": "mlp",
            "dim": 8,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    }
    _inject_coordinate_contract(atmosphere_config, module.raster.metadata)
    equations = {
        name: {"enabled": False, "weight": 0.0}
        for name in (
            "hydrostatic_equilibrium",
            "magnetohydrostatic_equilibrium",
            "momentum",
            "magnetic_divergence",
            "induction",
            "continuity",
            "upper_boundary_gas_pressure_prior",
        )
    }
    inversion = LTEInversionModule(
        log_tau500=[-5.0, -2.0, 1.0],
        wavelength_angstrom=observation.wavelength_angstrom,
        atmosphere_config=atmosphere_config,
        synthesizer_config={"line_ids": list(observation.required_line_ids)},
        instrument_config={
            "type": observation.instrument_type,
            **dict(observation.instrument_options),
        },
        normalization_config={"asinh_alphas": {"Q": 1.0e-2, "U": 1.0e-2, "V": 1.0e-2}},
        stokes_loss_config={"type": "mse"},
        weight_config={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0},
        wavelength_weights=None,
        wavelength_exclude_windows_angstrom=[],
        continuum_indices=observation.continuum_indices,
        atlas_continuum_radiance_w_m3_sr=(observation.radiance_scale_w_m3_sr),
        depth_sampling_config={
            "sample_count": 3,
            "coarse_to_fine": {
                "enabled": False,
                "fine_sample_count": 1,
                "uniform_weight_floor": 0.0,
            },
        },
        physics_config={
            "equations": equations,
            "gravity_m_per_s2": None,
            "volume_points_per_step": 0,
            "height_layers_per_step": 0,
            "upper_boundary_points_per_step": 0,
            "validation_height_layers": 0,
            "validation_points_per_height": 0,
            "sampling_domain": None,
            "vector_basis_matches_spatial_coordinates": True,
            "normalization": {"length_m": 1.0e6, "time_s": 3_600.0},
        },
        learning_rate={"start": 1.0e-3, "end": 1.0e-4, "iterations": "auto"},
        run_metadata={},
        observation_id=observation.observation_id,
        velocity_synthesis_mode=observation.velocity_synthesis_mode.value,
        instrument_radial_velocity_correction_m_per_s=0.0,
        vector_regularization_config={
            "enabled": False,
            "magnetic_weight": 0.0,
            "velocity_weight": 0.0,
            "decay_steps": 0,
        },
    ).float()
    result = inversion.synthesize(
        batch["coordinates"],
        ray_direction=batch["ray_direction"],
        stokes_basis=batch["stokes_basis"],
        observer_los_velocity_m_per_s=batch["observer_los_velocity_m_per_s"],
        instrument_response=batch["instrument_response"],
    )
    loss = inversion._stokes_objective(result["stokes"], batch["stokes"])[1]

    assert result["stokes"].shape == batch["stokes"].shape
    assert torch.isfinite(result["stokes"]).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in inversion.atmosphere_model.parameters()
    )
