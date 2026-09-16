"""Differentiable physical-ray composition for optically thin images."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from prom3theus.inversion.ray_integral import RayIntegralForwardComposition
from prom3theus.observations import SceneContract
from prom3theus.rt import SI_EMISSION_MEASURE_TO_CGS, integrate_ne2_response


SOLAR_RADIUS_M = 6.957e8
OUTER_HEIGHT_M = 1.0e6


class _AnalyticEOS:
    def __init__(self) -> None:
        self.last_temperature = None
        self.last_pressure = None

    def electron_density(self, temperature, gas_pressure):
        self.last_temperature = temperature
        self.last_pressure = gas_pressure
        return gas_pressure


class _Atmosphere(nn.Module):
    def __init__(self, temperature=2.0, pressure=3.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        self.log_pressure = nn.Parameter(torch.tensor(math.log(pressure)))
        self.register_buffer("solar_radius_m", torch.tensor(SOLAR_RADIUS_M))
        self.solar_radius_value_m = SOLAR_RADIUS_M
        self.shell_height_bounds_Mm = (OUTER_HEIGHT_M / 1.0e6, -0.1)
        self.thermodynamic_eos = _AnalyticEOS()
        self.last_coordinates = None
        self.last_height = None

    def evaluate_chart_height_points(self, coordinates, geometric_height_m):
        self.last_coordinates = coordinates
        self.last_height = geometric_height_m
        shape = geometric_height_m.shape
        return {
            "temperature": self.log_temperature.exp().expand(shape),
            "gas_pressure": self.log_pressure.exp().expand(shape),
        }


class _EmissionOperator(nn.Module):
    channels_angstrom = (171, 193, 211)

    def __init__(self):
        super().__init__()
        self.last_call = None

    def forward(
        self,
        temperature_k,
        electron_density_m3,
        distance_m,
        *,
        sample_dim,
    ):
        self.last_call = (
            temperature_k,
            electron_density_m3,
            distance_m,
            sample_dim,
        )
        factors = temperature_k[..., None] * temperature_k.new_tensor([1.0, 2.0, 3.0])
        return integrate_ne2_response(
            factors,
            electron_density_m3,
            distance_m,
            sample_dim=sample_dim,
        )

    def local_emission_measure_integrand(self, temperature_k, electron_density_m3):
        factors = temperature_k[..., None] * temperature_k.new_tensor([1.0, 2.0, 3.0])
        return factors * electron_density_m3.square().unsqueeze(-1)


def _scene(**changes):
    values = {
        "scene_basis": torch.eye(3, dtype=torch.float64),
        "solar_radius_m": SOLAR_RADIUS_M,
        "reference_time_tai_seconds": 1_700_000_000.0,
        "spatial_coordinate_center_mm": (0.0, 0.0),
        "spatial_coordinate_scale_mm": (100.0, 100.0),
        "time_coordinate_center_hours": 0.0,
        "time_coordinate_scale_hours": 1.0,
        "height_bounds_m": (-1.0e5, OUTER_HEIGHT_M),
    }
    values.update(changes)
    return SceneContract(**values)


def _batch():
    surface = torch.tensor(
        [
            [0.0, 0.0, SOLAR_RADIUS_M],
            [0.6 * SOLAR_RADIUS_M, 0.0, 0.8 * SOLAR_RADIUS_M],
        ],
        dtype=torch.float64,
    )
    ray = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float64)
    return {
        "surface_position_m": surface,
        "ray_direction": ray,
        "absolute_tai_seconds": torch.tensor(
            [1_700_001_800.0, 1_700_001_800.0], dtype=torch.float64
        ),
        "channel_index": torch.tensor([0, 2], dtype=torch.long),
    }


def test_ray_integral_solves_shell_geometry_and_preserves_units():
    atmosphere = _Atmosphere()
    emission = _EmissionOperator()
    composition = RayIntegralForwardComposition(
        atmosphere_model=atmosphere,
        scene=_scene(),
        emission_operator=emission,
        sample_count=5,
        height_power=2.0,
    )
    result = composition.synthesize(_batch())

    assert result.distance_m.shape == (2, 5)
    torch.testing.assert_close(
        result.distance_m[:, 0], torch.zeros(2, dtype=torch.float64)
    )
    assert torch.all(torch.diff(result.distance_m, dim=1) > 0)
    endpoint_radius = torch.linalg.vector_norm(result.position_m[:, -1], dim=-1)
    torch.testing.assert_close(
        endpoint_radius,
        torch.full_like(endpoint_radius, SOLAR_RADIUS_M + OUTER_HEIGHT_M),
    )
    target_height = (
        OUTER_HEIGHT_M * torch.linspace(0.0, 1.0, 5, dtype=torch.float64).square()
    )
    torch.testing.assert_close(
        result.geometric_height_m,
        target_height[None].expand(2, -1),
        rtol=1.0e-10,
        atol=1.0,
    )
    # At disk centre radial height and path distance are both metres.
    torch.testing.assert_close(result.distance_m[0], target_height.to(torch.float64))
    torch.testing.assert_close(
        result.chart_time_coordinates[..., 2],
        torch.full((2, 5), 0.5, dtype=torch.float64),
    )

    temperature, density, distance, sample_dim = emission.last_call
    assert sample_dim == 1
    torch.testing.assert_close(temperature, result.temperature_k)
    torch.testing.assert_close(density, result.electron_density_m3)
    torch.testing.assert_close(distance, result.distance_m)
    expected_first = (
        SI_EMISSION_MEASURE_TO_CGS * 3.0**2 * 2.0 * result.distance_m[0, -1]
    )
    assert result.all_channel_prediction.dtype == temperature.dtype
    torch.testing.assert_close(
        result.all_channel_prediction[0, 0], expected_first.to(temperature)
    )
    torch.testing.assert_close(
        result.raw_prediction,
        torch.stack(
            (result.all_channel_prediction[0, 0], result.all_channel_prediction[1, 2])
        ),
    )
    expected_emission_measure = (
        SI_EMISSION_MEASURE_TO_CGS * 3.0**2 * result.distance_m[:, -1]
    )
    torch.testing.assert_close(
        result.emission_measure_contribution_cm5.sum(dim=1),
        expected_emission_measure,
    )
    assert result.channel_emission_contribution is not None
    torch.testing.assert_close(
        result.channel_emission_contribution.sum(dim=1),
        result.all_channel_prediction.to(result.channel_emission_contribution),
    )

    overridden = composition.synthesize(
        **_batch(), sample_count=3, height_sampling_power=1.0
    )
    assert overridden.distance_m.shape == (2, 3)


def test_ray_integral_gradients_reach_unbounded_atmosphere_temperature_and_pressure():
    atmosphere = _Atmosphere(temperature=4.0e7, pressure=2.0e4)
    composition = RayIntegralForwardComposition(
        atmosphere_model=atmosphere,
        scene=_scene(),
        emission_operator=_EmissionOperator(),
        sample_count=8,
        height_power=1.5,
    )
    result = composition(_batch())
    result.raw_prediction.sum().backward()
    assert atmosphere.log_temperature.grad is not None
    assert atmosphere.log_pressure.grad is not None
    assert torch.isfinite(atmosphere.log_temperature.grad)
    assert torch.isfinite(atmosphere.log_pressure.grad)
    assert atmosphere.log_temperature.grad.abs() > 0
    assert atmosphere.log_pressure.grad.abs() > 0
    # The forward seam passes the decoded values directly to the same EOS.
    torch.testing.assert_close(
        atmosphere.thermodynamic_eos.last_temperature, result.temperature_k
    )
    torch.testing.assert_close(
        atmosphere.thermodynamic_eos.last_pressure, result.gas_pressure
    )
    assert result.temperature_k.min() > 1.0e7


def test_ray_integral_does_not_clip_extreme_positive_primary_fields():
    atmosphere = _Atmosphere(temperature=1.0e12, pressure=1.0e-12)

    class RecordingEmission(_EmissionOperator):
        def forward(
            self, temperature_k, electron_density_m3, distance_m, *, sample_dim
        ):
            self.last_call = (
                temperature_k,
                electron_density_m3,
                distance_m,
                sample_dim,
            )
            mean = temperature_k.mean(dim=sample_dim)
            return mean[:, None] * temperature_k.new_tensor([[1.0, 2.0, 3.0]])

    emission = RecordingEmission()
    result = RayIntegralForwardComposition(
        atmosphere_model=atmosphere,
        scene=_scene(),
        emission_operator=emission,
        sample_count=4,
        height_power=1.0,
    )(_batch())
    assert torch.all(result.temperature_k == atmosphere.log_temperature.exp())
    assert torch.all(result.gas_pressure == atmosphere.log_pressure.exp())
    assert torch.equal(emission.last_call[0], result.temperature_k)
    assert torch.equal(atmosphere.thermodynamic_eos.last_pressure, result.gas_pressure)


@pytest.mark.parametrize(
    ("sample_count", "height_power", "message"),
    [(1, 1.0, "at least two"), (4, 0.0, "strictly positive")],
)
def test_ray_integral_rejects_invalid_sampling(sample_count, height_power, message):
    with pytest.raises(ValueError, match=message):
        RayIntegralForwardComposition(
            atmosphere_model=_Atmosphere(),
            scene=_scene(),
            emission_operator=_EmissionOperator(),
            sample_count=sample_count,
            height_power=height_power,
        )


def test_ray_integral_rejects_bad_ray_channel_and_scene_contract():
    atmosphere = _Atmosphere()
    composition = RayIntegralForwardComposition(
        atmosphere_model=atmosphere,
        scene=_scene(),
        emission_operator=_EmissionOperator(),
        sample_count=4,
    )
    bad_ray = _batch()
    bad_ray["ray_direction"] = bad_ray["ray_direction"] * 2
    with pytest.raises(ValueError, match="unit vectors"):
        composition(bad_ray)

    bad_channel = _batch()
    bad_channel["channel_index"] = torch.tensor([0, 3])
    with pytest.raises(IndexError, match="outside"):
        composition(bad_channel)

    atmosphere.shell_height_bounds_Mm = (2.0, -0.1)
    with pytest.raises(ValueError, match="share the atmosphere outer shell"):
        composition(_batch())


def test_ray_integral_rejects_invalid_emission_shape_and_missing_fields():
    class BadEmission(_EmissionOperator):
        def forward(
            self, temperature_k, electron_density_m3, distance_m, *, sample_dim
        ):
            return temperature_k.mean(dim=sample_dim)

    composition = RayIntegralForwardComposition(
        atmosphere_model=_Atmosphere(),
        scene=_scene(),
        emission_operator=BadEmission(),
        sample_count=4,
    )
    with pytest.raises(ValueError, match=r"return \[ray, channel\]"):
        composition(_batch())
    incomplete = _batch()
    incomplete.pop("absolute_tai_seconds")
    with pytest.raises(KeyError, match="absolute_tai_seconds"):
        composition(incomplete)


def test_shared_adaptive_sampling_jitters_training_and_preserves_constant_integral():
    from prom3theus.inversion.depth_sampling import DepthRefinement

    atmosphere = _Atmosphere()
    composition = RayIntegralForwardComposition(
        atmosphere_model=atmosphere,
        scene=_scene(),
        emission_operator=_EmissionOperator(),
        sample_count=8,
        depth_refinement=DepthRefinement(True, 6, 0.05),
    )
    first = composition.synthesize(_batch(), randomize=True)
    second = composition.synthesize(_batch(), randomize=True)
    assert first.distance_m.shape[-1] == 14
    assert not torch.equal(first.distance_m[:, 1:-1], second.distance_m[:, 1:-1])
    torch.testing.assert_close(
        first.distance_m[:, [0, -1]], second.distance_m[:, [0, -1]]
    )
    assert torch.all(torch.diff(first.distance_m, dim=-1) > 0)
    torch.testing.assert_close(first.raw_prediction, second.raw_prediction)
    rng = torch.random.get_rng_state()
    fixed = composition.synthesize(_batch())
    again = composition.synthesize(_batch())
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(fixed.distance_m, again.distance_m, rtol=0, atol=0)
    fixed.raw_prediction.sum().backward()
    assert atmosphere.log_temperature.grad.abs() > 0
    assert atmosphere.log_pressure.grad.abs() > 0


def test_adaptive_emission_quadrature_converges_to_dense_reference():
    from prom3theus.inversion.depth_sampling import DepthRefinement

    class LocalizedAtmosphere(_Atmosphere):
        def evaluate_chart_height_points(self, coordinates, geometric_height_m):
            fields = super().evaluate_chart_height_points(
                coordinates, geometric_height_m
            )
            fraction = geometric_height_m / OUTER_HEIGHT_M
            fields["gas_pressure"] = fields["gas_pressure"] * (
                0.01 + torch.exp(-0.5 * ((fraction - 0.62) / 0.08).square())
            )
            return fields

    model = LocalizedAtmosphere()
    composition = RayIntegralForwardComposition(
        atmosphere_model=model,
        scene=_scene(),
        emission_operator=_EmissionOperator(),
        sample_count=32,
        depth_refinement=DepthRefinement(True, 64, 0.05),
    )
    reference = composition.synthesize(
        _batch(), sample_count=2048, refine=False
    ).raw_prediction
    low = composition.synthesize(_batch()).raw_prediction
    composition.depth_refinement = DepthRefinement(True, 128, 0.05)
    high = composition.synthesize(_batch(), sample_count=64).raw_prediction
    low_error = ((low - reference) / reference).abs().max()
    high_error = ((high - reference) / reference).abs().max()
    assert high_error < low_error
    assert high_error < 0.002


@pytest.mark.parametrize("mu", [1.0, 0.2, 0.02])
def test_perturbed_refined_line_elements_cover_physical_ray_and_reuse_samples(mu):
    from prom3theus.inversion.depth_sampling import DepthRefinement

    class RecordedAtmosphere(_Atmosphere):
        def __init__(self):
            super().__init__()
            self.calls = []

        def evaluate_chart_height_points(self, coordinates, geometric_height_m):
            fields = super().evaluate_chart_height_points(
                coordinates, geometric_height_m
            )
            for field in fields.values():
                field.retain_grad()
            self.calls.append(fields)
            return fields

    atmosphere = RecordedAtmosphere()
    composition = RayIntegralForwardComposition(
        atmosphere_model=atmosphere,
        scene=_scene(),
        emission_operator=_EmissionOperator(),
        sample_count=8,
        depth_refinement=DepthRefinement(True, 12, 0.05),
    )
    batch = _batch()
    batch["surface_position_m"][:] = torch.tensor(
        [math.sqrt(1 - mu**2) * SOLAR_RADIUS_M, 0.0, mu * SOLAR_RADIUS_M],
        dtype=torch.float64,
    )
    # Tolerated input rounding must not rescale arc length or displace the endpoint.
    batch["ray_direction"] *= 1 + 1e-6
    result = composition.synthesize(batch, randomize=True)
    projection = SOLAR_RADIUS_M * mu
    radial_increment = OUTER_HEIGHT_M * (2 * SOLAR_RADIUS_M + OUTER_HEIGHT_M)
    expected_length = radial_increment / (
        math.sqrt(projection**2 + radial_increment) + projection
    )
    assert result.distance_m.shape == (2, 20)
    assert [call["temperature"].shape[-1] for call in atmosphere.calls] == [8, 12]
    assert torch.all(torch.diff(result.distance_m, dim=-1) > 0)
    assert torch.all(result.line_element_m > 0)
    torch.testing.assert_close(
        result.line_element_m.sum(-1),
        torch.full((2,), expected_length, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.position_m[:, -1], dim=-1),
        torch.full((2,), SOLAR_RADIUS_M + OUTER_HEIGHT_M, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-6,
    )
    physical_intervals = torch.linalg.vector_norm(
        torch.diff(result.position_m, dim=1), dim=-1
    )
    torch.testing.assert_close(
        physical_intervals, torch.diff(result.distance_m, dim=-1), rtol=1e-8, atol=2e-7
    )
    expected = 1e-10 * 3.0**2 * 2.0 * expected_length * torch.tensor([1.0, 3.0])
    torch.testing.assert_close(result.raw_prediction, expected)
    torch.testing.assert_close(
        result.channel_emission_contribution.sum(1),
        result.all_channel_prediction.double(),
    )
    result.raw_prediction.sum().backward()
    for call in atmosphere.calls:
        for field in call.values():
            assert field.grad is not None and torch.isfinite(field.grad).all()
            assert torch.all(field.grad > 0)
