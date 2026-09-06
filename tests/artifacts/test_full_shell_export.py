from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prom3theus.artifacts.full_shell import (
    evaluate_full_shell_atmosphere,
    full_shell_height_grid,
    full_shell_metadata,
)
from prom3theus.artifacts.export import export_artifact
from prom3theus.cli.main import _build_parser
from prom3theus.rt import StratifiedAtmosphereModel


class _FakeEOS:
    @staticmethod
    def mass_density(temperature, gas_pressure):
        return gas_pressure / temperature


class _FakeFullShellAtmosphere(torch.nn.Module):
    shell_height_bounds_Mm = (20.0, -0.1)

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("solar_radius_m", torch.tensor(7.0e8))
        self.thermodynamic_eos = _FakeEOS()

    def evaluate_at_height(self, coordinates, geometric_height_m):
        shape = geometric_height_m.shape
        temperature = 10_000.0 + geometric_height_m * 1.0e-4
        gas_pressure = torch.full(shape, 2.0, device=temperature.device)
        magnetic = torch.ones((*shape, 3), device=temperature.device)
        velocity = 2.0 * magnetic
        return {
            "temperature": temperature,
            "gas_pressure": gas_pressure,
            "microturbulence": torch.full(shape, 3.0, device=temperature.device),
            "magnetic_field": magnetic,
            "velocity_field": velocity,
        }

    def position_from_coords_height(self, coordinates, geometric_height_m):
        direction = torch.zeros_like(coordinates)
        direction[..., 0] = 1.0
        return direction * (self.solar_radius_m + geometric_height_m)[..., None]


def _module():
    return SimpleNamespace(atmosphere_model=_FakeFullShellAtmosphere())


def test_full_shell_height_grid_covers_the_configured_physical_domain():
    height = full_shell_height_grid(_module(), 5)

    assert height.shape == (5,)
    assert height[0] == pytest.approx(20.0e6)
    assert height[-1] == pytest.approx(-0.1e6)
    assert torch.all(height[:-1] > height[1:])
    assert full_shell_metadata(height) == {
        "coordinate": "geometric_height_m",
        "coordinate_array": "full_shell_geometric_height_m",
        "height_samples": 5,
        "height_bounds_m": [20.0e6, -0.1e6],
        "height_order": "outer_to_inner",
        "spatial_sampling": "radial_carrington_columns",
        "spatial_coordinates": [
            "carrington_chart_x_mm",
            "carrington_chart_y_mm",
            "time_hours",
        ],
        "valid_mask": "valid_mask",
        "vector_frames": {
            "cartesian": "heliocentric_carrington_cartesian",
            "spherical": "local_carrington_radial_colatitude_longitude",
            "velocity": "carrington_corotating",
        },
        "stokes_synthesis": False,
        "opacity_and_tau500": False,
    }


def test_full_shell_evaluation_uses_radial_columns_and_shared_valid_mask():
    module = _module()
    raster = SimpleNamespace(
        coordinates=torch.tensor(
            [[[0.0, 0.0, 1.0], [2.0, 3.0, 1.0]]], dtype=torch.float32
        ),
        valid_mask=torch.tensor([[True, False]]),
        spatial_shape=(1, 2),
    )
    height = full_shell_height_grid(module, 3)

    arrays = evaluate_full_shell_atmosphere(
        module,
        raster,
        height_grid_m=height,
        batch_size=1,
        storage_dtype="float32",
    )

    np.testing.assert_array_equal(
        arrays["full_shell_geometric_height_m"],
        np.asarray([20.0e6, 9.95e6, -0.1e6], dtype=np.float32),
    )
    assert arrays["full_shell_temperature_k"].shape == (1, 2, 3)
    assert arrays["full_shell_magnetic_field_gauss"].shape == (1, 2, 3, 3)
    assert arrays["full_shell_position_carrington_m"].shape == (1, 2, 3, 3)
    np.testing.assert_allclose(
        arrays["full_shell_radius_m"][0, 0],
        module.atmosphere_model.solar_radius_m.numpy()
        + arrays["full_shell_geometric_height_m"],
    )
    np.testing.assert_allclose(
        arrays["full_shell_mass_density_kg_m3"][0, 0],
        2.0 / arrays["full_shell_temperature_k"][0, 0],
    )
    assert np.isnan(arrays["full_shell_temperature_k"][0, 1]).all()
    assert np.isnan(arrays["full_shell_magnetic_field_gauss"][0, 1]).all()


def test_full_shell_evaluation_matches_real_extrapolation_model_contract():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 5),
        shell_height_bounds_Mm=(20.0, -0.1),
        line_formation_height_bounds_Mm=(1.5, -0.1),
        upper_atmosphere_config={
            "type": "hydrostatic_corona",
            "transition_region_top_megameter": 2.5,
            "coronal_temperature_k": 1.0e6,
            "reference_grid_points": 64,
        },
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "mlp",
            "dim": 4,
            "n_layers": 1,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    )
    module = SimpleNamespace(atmosphere_model=model)
    raster = SimpleNamespace(
        coordinates=torch.tensor([[[0.0, 0.0, 0.0]]]),
        valid_mask=torch.tensor([[True]]),
        spatial_shape=(1, 1),
    )
    height = full_shell_height_grid(module, 4)

    arrays = evaluate_full_shell_atmosphere(
        module,
        raster,
        height_grid_m=height,
        batch_size=1,
        storage_dtype="float32",
    )

    assert arrays["full_shell_temperature_k"].shape == (1, 1, 4)
    assert arrays["full_shell_magnetic_field_gauss"].shape == (1, 1, 4, 3)
    assert np.isfinite(arrays["full_shell_temperature_k"]).all()
    assert np.isfinite(arrays["full_shell_gas_pressure_pa"]).all()
    assert np.isfinite(arrays["full_shell_mass_density_kg_m3"]).all()
    assert (arrays["full_shell_gas_pressure_pa"] > 0).all()
    np.testing.assert_allclose(
        arrays["full_shell_radius_m"][0, 0],
        float(model.solar_radius_m) + arrays["full_shell_geometric_height_m"],
        rtol=2.0e-6,
    )


def test_cli_requires_an_explicit_full_shell_opt_in():
    parser = _build_parser()
    ordinary = parser.parse_args(["export", "artifact", "result.npz"])
    requested = parser.parse_args(
        [
            "export",
            "artifact",
            "result.npz",
            "--include-full-shell",
            "--full-shell-samples",
            "17",
        ]
    )

    assert ordinary.include_full_shell is False
    assert ordinary.full_shell_samples == 101
    assert requested.include_full_shell is True
    assert requested.full_shell_samples == 17


def test_full_shell_sample_count_is_validated_even_without_opt_in(tmp_path):
    with pytest.raises(ValueError, match="full_shell_samples"):
        export_artifact(
            tmp_path,
            tmp_path / "result.npz",
            include_full_shell=False,
            full_shell_samples=1,
        )
