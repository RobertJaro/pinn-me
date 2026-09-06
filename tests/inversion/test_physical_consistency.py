"""Physical unit and reference-frame regression tests for inversion composition."""

from __future__ import annotations

import pytest
import torch

from prom3theus.instruments import MagneticAzimuthConvention
from prom3theus.inversion.forward import (
    DepthRefinement,
    ForwardRuntime,
    LTEForwardComposition,
)
from prom3theus.rt import RayTraceResult, StratifiedAtmosphere


class _CaptureBackend:
    uses_prepared_wavelength = False

    def __init__(self):
        self.atmosphere = None

    @staticmethod
    def prepare_wavelength_grid(wavelength_angstrom):
        del wavelength_angstrom

    @staticmethod
    def reference_extinction(atmosphere):
        return torch.ones_like(atmosphere.temperature)

    def synthesize(self, atmosphere, wavelength_angstrom, *, radiance_scale, path):
        del radiance_scale, path
        self.atmosphere = atmosphere
        return atmosphere.temperature.new_zeros(
            atmosphere.temperature.shape[0], 4, wavelength_angstrom.numel()
        )


class _IdentityInstrument(torch.nn.Module):
    polarization_convention = MagneticAzimuthConvention("identity", 0.0)

    @staticmethod
    def synthesis_grid(observed_wavelength):
        return observed_wavelength

    @staticmethod
    def forward(stokes, synthesis_wavelength, observed_wavelength):
        del synthesis_wavelength, observed_wavelength
        return stokes


class _ConstantAtmosphere(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("log_tau500", torch.tensor([-5.0, 1.0]))
        self.register_buffer("solar_radius_m", torch.tensor(1.0))

    @staticmethod
    def trace_rays(coordinates, ray_direction, depth_grid):
        del ray_direction
        batch = coordinates.shape[0]
        depth = depth_grid.numel()
        scalar = torch.ones(batch, depth, dtype=coordinates.dtype)
        vector = torch.zeros(batch, depth, 3, dtype=coordinates.dtype)
        vector[..., 0] = 100.0
        position = torch.zeros_like(vector)
        position[..., 0] = 1.0
        distance = torch.arange(depth, dtype=coordinates.dtype).expand(batch, -1)
        atmosphere = StratifiedAtmosphere(
            log_tau500=depth_grid,
            temperature=scalar * 6_000.0,
            velocity_field=vector,
            microturbulence=scalar * 1_000.0,
            magnetic_field=torch.zeros_like(vector),
            gas_pressure=scalar * 1_000.0,
            geometric_height_m=distance,
        )
        return atmosphere, RayTraceResult(
            position_m=position,
            distance_m=distance,
            chart_xy_mm=position[..., :2],
            geometric_height_m=distance,
        )


def _registered_composition():
    backend = _CaptureBackend()
    composition = LTEForwardComposition(
        atmosphere_model=_ConstantAtmosphere(),
        backend=backend,
        instrument=_IdentityInstrument(),
        velocity_synthesis_mode="carrington_registered_relative",
        depth_refinement=DepthRefinement(False, 1, 0.0),
    )
    return composition, backend


def _stokes_basis() -> torch.Tensor:
    return torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).unsqueeze(
        0
    )


def test_hinode_registered_velocity_subtracts_the_removed_solar_shift():
    composition, backend = _registered_composition()
    wavelength = torch.tensor([6301.0, 6302.0])
    runtime = ForwardRuntime(
        coarse_depth_grid=torch.tensor([-5.0, 1.0]),
        observed_wavelength_angstrom=wavelength,
        synthesis_wavelength_angstrom=wavelength,
        radiance_scale=1.0,
        carrington_angular_velocity_rad_per_s=0.0,
        instrument_line_of_sight_velocity_correction_m_per_s=0.0,
    )

    composition.synthesize(
        torch.zeros(1, 3),
        runtime=runtime,
        ray_direction=torch.tensor([[-1.0, 0.0, 0.0]]),
        stokes_basis=_stokes_basis(),
        removed_solar_los_velocity_m_per_s=torch.tensor([30.0]),
    )

    assert backend.atmosphere is not None
    expected = torch.tensor([0.0, 0.0, 70.0]).expand(1, 2, 3)
    torch.testing.assert_close(backend.atmosphere.velocity_field, expected)


def test_velocity_offsets_and_instrument_zero_point_must_be_physical():
    composition, _ = _registered_composition()
    wavelength = torch.tensor([6301.0, 6302.0])
    runtime = ForwardRuntime(
        coarse_depth_grid=torch.tensor([-5.0, 1.0]),
        observed_wavelength_angstrom=wavelength,
        synthesis_wavelength_angstrom=wavelength,
        radiance_scale=1.0,
        carrington_angular_velocity_rad_per_s=0.0,
        instrument_line_of_sight_velocity_correction_m_per_s=torch.tensor([0.0, 1.0]),
    )

    with torch.no_grad(), pytest.raises(ValueError, match="must be scalar"):
        composition.synthesize(
            torch.zeros(1, 3),
            runtime=runtime,
            ray_direction=torch.tensor([[-1.0, 0.0, 0.0]]),
            stokes_basis=_stokes_basis(),
            removed_solar_los_velocity_m_per_s=torch.tensor([30.0]),
        )

    physical_runtime = ForwardRuntime(
        coarse_depth_grid=runtime.coarse_depth_grid,
        observed_wavelength_angstrom=wavelength,
        synthesis_wavelength_angstrom=wavelength,
        radiance_scale=1.0,
        carrington_angular_velocity_rad_per_s=0.0,
        instrument_line_of_sight_velocity_correction_m_per_s=0.0,
    )
    with pytest.raises(ValueError, match="finite and subluminal"):
        composition.synthesize(
            torch.zeros(1, 3),
            runtime=physical_runtime,
            ray_direction=torch.tensor([[-1.0, 0.0, 0.0]]),
            stokes_basis=_stokes_basis(),
            removed_solar_los_velocity_m_per_s=torch.tensor([float("nan")]),
        )
