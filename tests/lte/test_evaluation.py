from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from pme.evaluation.lte import (
    _export_device,
    _normalization_contract,
    _relocated_resource_configs,
    _raster_loader_config,
    evaluate_atmosphere_model,
    evaluate_stokes_model,
)
from pme.lte.atmosphere import StratifiedAtmosphereModel
from pme.lte.hinode import load_hinode_raster
from pme.train.lte_module import LTEModule


class _FastSynthesizer(nn.Module):
    def __init__(self, log_tau500=None, **kwargs):
        super().__init__()

    def forward(
        self, atmosphere, wavelength, mu=1.0, radiance_scale=None, ray_distance_m=None
    ):
        del radiance_scale  # This stand-in already emits normalized intensity.
        del ray_distance_m
        wavelength = torch.as_tensor(wavelength).to(atmosphere.temperature)
        mu = torch.as_tensor(mu).to(atmosphere.temperature)
        if mu.shape[-1:] == (1,):
            mu = mu[..., 0]
        coordinate = (wavelength - wavelength.mean()) / (
            wavelength.max() - wavelength.min()
        )
        intensity = atmosphere.temperature.mean(-1, keepdim=True) / 6000.0
        intensity = intensity * mu[..., None] * (1.0 + 0.01 * coordinate)
        zeros = torch.zeros_like(intensity).expand(*intensity.shape[:-1], wavelength.numel())
        return torch.stack((intensity, zeros, zeros, zeros), dim=-2)


class _DensityLookup:
    @staticmethod
    def reference_mass_density(temperature, gas_pressure):
        return gas_pressure / temperature

    @staticmethod
    def volume_extinction_at_5000(temperature, gas_pressure):
        return gas_pressure / temperature * 1.0e-4


def test_export_device_respects_dtype_and_radiative_transfer_support():
    assert _export_device(
        "cpu", torch.float64, include_stokes=True
    ) == torch.device("cpu")
    with pytest.raises(ValueError, match="float64"):
        _export_device("mps", torch.float64, include_stokes=False)
    with pytest.raises(ValueError, match="matrix_exp"):
        _export_device("mps", torch.float32, include_stokes=True)


def test_full_resolution_config_is_reduced_to_raster_loader_arguments():
    resolved = _raster_loader_config({
        "type": "hinode_lte",
        "files": "scan/*.fits",
        "scan_slice": [0, 1024],
        "batch_size": 128,
        "validation_batch_size": 256,
        "validation_stride": 16,
        "num_workers": 0,
        "pin_memory": True,
    })
    assert resolved == {
        "files": "scan/*.fits",
        "scan_slice": [0, 1024],
    }


def test_normalization_contract_excludes_data_dependent_calibration_results():
    first = {
        "operation": "no raster or per-pixel continuum normalization",
        "type": "edge_mean",
        "edge_samples": 8,
        "indices": [0, 1, 110, 111],
        "continuum_intensity_quantiles": {"50": 12.0},
        "radiometric_calibration": {
            "type": "absolute quiet-Sun atlas calibration",
            "stored_stokes_unit": "I_c,atlas(mu=1)",
            "atlas_disk_center_continuum_radiance_w_m3_sr": 3.06e13,
            "detector_to_radiance_w_m3_sr_per_raw_unit": 1.2e9,
            "quiet_sun": {
                "maximum_mean_fractional_polarization": 0.01,
                "continuum_trim_quantiles": [0.05, 0.95],
                "minimum_pixel_count": 1024,
                "pixel_count": 4096,
                "median_raw_continuum": 10234.0,
            },
            "reference": {"schema_version": 1, "source": {"sha256": "abc"}},
        },
    }
    second = deepcopy(first)
    second["continuum_intensity_quantiles"] = {"50": 27.0}
    second["radiometric_calibration"][
        "detector_to_radiance_w_m3_sr_per_raw_unit"
    ] = 7.8e9
    second["radiometric_calibration"]["quiet_sun"]["pixel_count"] = 2048
    second["radiometric_calibration"]["quiet_sun"]["median_raw_continuum"] = 9987.0
    assert _normalization_contract(first) == _normalization_contract(second)


def test_export_retargets_only_checksum_equivalent_resource_paths():
    checkpoint = {
        "hyper_parameters": {
            "synthesizer_config": {
                "atomic_data_directory": "/old/resources",
                "line_ids": ["FeI_6301.5008", "FeI_6302.4932"],
            },
            "instrument_config": {
                "data_directory": "/old/resources",
                "fwhm_angstrom": 0.025,
            },
        },
        "lte_metadata": {
            "data": {
                "lte_resources": {
                    "instrument": "Hinode/SOT-SP",
                    "resource_sha256": {"stic_continuum_table.json": "abc"},
                }
            }
        },
    }
    resources = {
        "directory": "/new/resources",
        "instrument": "Hinode/SOT-SP",
        "required_production_files": ["stic_continuum_table.json"],
        "resource_sha256": {"stic_continuum_table.json": "abc"},
    }
    synthesis, instrument = _relocated_resource_configs(checkpoint, resources)
    assert synthesis["atomic_data_directory"] == "/new/resources"
    assert synthesis["line_ids"] == ["FeI_6301.5008", "FeI_6302.4932"]
    assert instrument == {
        "data_directory": "/new/resources",
        "fwhm_angstrom": 0.025,
    }

    incompatible = deepcopy(resources)
    incompatible["resource_sha256"] = {"stic_continuum_table.json": "different"}
    with pytest.raises(RuntimeError, match="stic_continuum_table.json"):
        _relocated_resource_configs(checkpoint, incompatible)


def test_dense_lte_evaluation_preserves_raster_and_samples_continuous_depth(
    synthetic_hinode_files,
):
    raster = load_hinode_raster(synthetic_hinode_files)
    ray_geometry = raster.metadata["ray_geometry"]
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 9),
        scene_geometry_config={
            "solar_radius_m": ray_geometry["solar_radius_m"],
            "scene_basis": ray_geometry["scene_basis_rows"],
        },
        model_config={
            "dim": 8,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    )
    model.train()
    dense_grid = torch.linspace(-5.0, 1.0, 31)
    result = evaluate_atmosphere_model(
        model,
        raster,
        log_tau500=dense_grid,
        batch_size=2,
        continuum_opacity=_DensityLookup(),
        storage_dtype="float64",
    )

    assert model.training
    assert result["temperature_k"].shape == (*raster.spatial_shape, 31)
    assert result["v_los_m_per_s"].shape == (*raster.spatial_shape, 31)
    assert result["velocity_field_m_per_s"].shape == (
        *raster.spatial_shape,
        31,
        3,
    )
    assert result["magnetic_field_gauss"].shape == (*raster.spatial_shape, 31, 3)
    assert result["magnetic_field_scene_gauss"].shape == (
        *raster.spatial_shape, 31, 3
    )
    assert result["magnetic_field_spherical_gauss"].shape == (
        *raster.spatial_shape, 31, 3
    )
    assert result["magnetic_field_observer_gauss"].shape == (
        *raster.spatial_shape, 31, 3
    )
    assert result["velocity_field_scene_m_per_s"].shape == (
        *raster.spatial_shape, 31, 3
    )
    assert result["velocity_field_spherical_m_per_s"].shape == (
        *raster.spatial_shape, 31, 3
    )
    assert result["carrington_rotation_velocity_m_per_s"].shape == (
        *raster.spatial_shape, 31, 3
    )
    assert result["velocity_field_inertial_m_per_s"].shape == (
        *raster.spatial_shape, 31, 3
    )
    assert result["velocity_field_inertial_spherical_m_per_s"].shape == (
        *raster.spatial_shape, 31, 3
    )
    assert result["velocity_field_observer_m_per_s"].shape == (
        *raster.spatial_shape, 31, 3
    )
    assert result["carrington_longitude_rad"].shape == (*raster.spatial_shape, 31)
    assert result["carrington_colatitude_rad"].shape == (*raster.spatial_shape, 31)
    assert result["radius_m"].shape == (*raster.spatial_shape, 31)
    assert result["geometric_height_m"].shape == (*raster.spatial_shape, 31)
    assert result["gas_pressure_pa"].shape == (*raster.spatial_shape, 31)
    assert result["mass_density_kg_m3"].shape == (*raster.spatial_shape, 31)
    assert result["temperature_k"].dtype == np.float64
    assert result["magnetic_field_gauss"].dtype == np.float64
    assert result["time_hours"].shape == raster.spatial_shape
    assert result["carrington_chart_x_mm"].shape == raster.spatial_shape
    assert result["carrington_chart_y_mm"].shape == raster.spatial_shape
    np.testing.assert_allclose(result["time_hours"], raster.coords[..., 0])
    np.testing.assert_allclose(result["carrington_chart_x_mm"], raster.coords[..., 1])
    np.testing.assert_allclose(result["carrington_chart_y_mm"], raster.coords[..., 2])
    np.testing.assert_allclose(result["log_tau500"], dense_grid.numpy())
    np.testing.assert_allclose(
        result["velocity_field_inertial_m_per_s"],
        result["velocity_field_m_per_s"]
        + result["carrington_rotation_velocity_m_per_s"],
        rtol=2.0e-6,
        atol=2.0e-4,
    )
    scene_basis = np.asarray(
        raster.metadata["ray_geometry"]["scene_basis_rows"]
    )
    np.testing.assert_allclose(
        result["magnetic_field_scene_gauss"],
        np.einsum("ij,...j->...i", scene_basis, result["magnetic_field_gauss"]),
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    assert np.isfinite(result["temperature_k"][result["valid_mask"]]).all()


def test_stokes_evaluation_restores_raster_shape_and_residuals(
    synthetic_hinode_files, monkeypatch
):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _FastSynthesizer)
    raster = load_hinode_raster(synthetic_hinode_files)
    affine = raster.metadata["coordinates"]["network_affine"]
    ray_geometry = raster.metadata["ray_geometry"]
    module = LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 7),
        wavelength_angstrom=raster.wavelength_angstrom,
        atmosphere_config={
            "spatial_coordinate_center_mm": affine["center_mm"],
            "spatial_coordinate_scale_mm": affine["scale_mm"],
            "scene_geometry_config": {
                "solar_radius_m": ray_geometry["solar_radius_m"],
                "scene_basis": ray_geometry["scene_basis_rows"],
            },
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
    )
    module.train()
    result = evaluate_stokes_model(module, raster, batch_size=2)
    assert module.training
    expected_shape = (*raster.spatial_shape, 4, raster.wavelength_angstrom.numel())
    assert result["predicted_stokes"].shape == expected_shape
    assert result["observed_stokes"].shape == expected_shape
    assert result["predicted_stokes"].dtype == np.float32
    np.testing.assert_allclose(
        result["stokes_residual"],
        result["predicted_stokes"] - result["observed_stokes"],
    )
    valid = raster.valid_mask
    synthesized = module.synthesize(
        raster.coords[valid][:2],
        raster.mu[valid][:2],
        ray_origin_m=raster.ray_origin_m[valid][:2],
        ray_direction=raster.ray_direction[valid][:2],
        stokes_basis=raster.stokes_basis[valid][:2],
    )
    torch.testing.assert_close(
        synthesized["velocity_field_inertial_cartesian"],
        synthesized["atmosphere"].velocity_field
        + synthesized["rotation_velocity_cartesian"],
    )
    assert torch.linalg.vector_norm(
        synthesized["rotation_velocity_cartesian"], dim=-1
    ).mean() > 1_000.0
