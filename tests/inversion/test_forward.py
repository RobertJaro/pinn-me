import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from prom3theus.instruments import MagneticAzimuthConvention
from prom3theus.inversion.forward import (
    DepthRefinement,
    ForwardRuntime,
    ForwardSynthesisBackend,
    LTEForwardComposition,
    LTESynthesisBackend,
)
from prom3theus.rt import (
    LTESynthesizer,
    RayDistancePath,
    RayTraceResult,
    StratifiedAtmosphere,
    THOMSON_CROSS_SECTION_M2,
)
from prom3theus.training.lightning import LTEInversionModule


class _Backend:
    def __init__(self):
        self.uses_prepared_wavelength = False
        self.prepared_wavelength = None
        self.synthesis_call = None

    def prepare_wavelength_grid(self, wavelength_angstrom):
        self.prepared_wavelength = wavelength_angstrom

    @staticmethod
    def reference_extinction(atmosphere):
        return torch.ones_like(atmosphere.temperature)

    def synthesize(self, atmosphere, wavelength_angstrom, *, radiance_scale, path):
        self.synthesis_call = {
            "atmosphere": atmosphere,
            "wavelength_angstrom": wavelength_angstrom,
            "radiance_scale": radiance_scale,
            "path": path,
        }
        return torch.ones(
            *atmosphere.temperature.shape[:-1],
            4,
            wavelength_angstrom.numel(),
            dtype=atmosphere.temperature.dtype,
        )


class _Instrument(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prepared = None
        self.polarization_convention = MagneticAzimuthConvention("identity", 0.0)

    @staticmethod
    def synthesis_grid(observed_wavelength):
        return observed_wavelength + 0.25

    def prepare(self, synthesis_wavelength, observed_wavelength):
        self.prepared = (synthesis_wavelength, observed_wavelength)

    @staticmethod
    def forward(stokes, synthesis_wavelength, observed_wavelength, *, gain):
        del synthesis_wavelength, observed_wavelength
        return stokes * gain[..., None, None]


class _AtmosphereModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("log_tau500", torch.tensor([-5.0, 1.0]))
        self.register_buffer("solar_radius_m", torch.tensor(1.0))

    def trace_rays(self, coordinates, ray_direction, depth_grid):
        del ray_direction
        batch = coordinates.shape[0]
        depth = depth_grid.numel()
        scalar = torch.ones(batch, depth)
        position = torch.zeros(batch, depth, 3)
        position[..., 0] = 1.0
        distance = torch.arange(depth, dtype=coordinates.dtype).expand(batch, -1)
        atmosphere = StratifiedAtmosphere(
            log_tau500=depth_grid,
            temperature=scalar * 6000.0,
            velocity_field=torch.tensor([1.0, 2.0, 3.0]).expand(batch, depth, 3),
            microturbulence=scalar * 1000.0,
            magnetic_field=torch.tensor([4.0, 5.0, 6.0]).expand(batch, depth, 3),
            gas_pressure=scalar * 1000.0,
            geometric_height_m=distance,
        )
        trace = RayTraceResult(
            position_m=position,
            distance_m=distance,
            chart_xy_mm=position[..., :2],
            geometric_height_m=distance,
        )
        return atmosphere, trace


def _composition(*, mode="carrington_observer_relative", refinement=False):
    backend = _Backend()
    instrument = _Instrument()
    composition = LTEForwardComposition(
        atmosphere_model=_AtmosphereModel(),
        backend=backend,
        instrument=instrument,
        velocity_synthesis_mode=mode,
        depth_refinement=DepthRefinement(refinement, 2, 0.05),
    )
    return composition, backend, instrument


def test_backend_protocol_and_wavelength_preparation_are_solver_neutral():
    composition, backend, instrument = _composition()
    assert isinstance(backend, ForwardSynthesisBackend)
    observed = torch.tensor([1.0, 2.0])

    synthesis = composition.prepare_wavelength_grid(observed)

    torch.testing.assert_close(synthesis, observed + 0.25)
    assert backend.prepared_wavelength is synthesis
    assert instrument.prepared == (synthesis, observed)


def test_lte_backend_calls_current_synthesizer_contract():
    depth = torch.tensor([-4.0, -2.0, 0.0])
    wavelength = torch.linspace(6173.1, 6173.6, 5)
    atmosphere = StratifiedAtmosphere(
        log_tau500=depth,
        temperature=torch.full((1, 3), 5_500.0),
        velocity_field=torch.zeros(1, 3, 3),
        microturbulence=torch.full((1, 3), 1_000.0),
        magnetic_field=torch.zeros(1, 3, 3),
        gas_pressure=torch.logspace(1.0, 4.0, 3).reshape(1, 3),
    )
    backend = LTESynthesisBackend(
        LTESynthesizer(log_tau500=None, line_ids=("FeI_6173.3352",))
    )
    backend.prepare_wavelength_grid(wavelength)

    stokes = backend.synthesize(
        atmosphere,
        wavelength,
        radiance_scale=1.0,
        path=RayDistancePath(torch.tensor([[0.0, 1.0e5, 2.0e5]])),
    )

    assert stokes.shape == (1, 4, 5)
    assert torch.isfinite(stokes).all()


def test_lte_backend_reference_extinction_uses_combined_plasma_state():
    depth = torch.tensor([-4.0, -2.0, 0.0], dtype=torch.float64)
    backend = LTESynthesisBackend(
        LTESynthesizer(log_tau500=None, line_ids=("FeI_6173.3352",)).to(
            dtype=torch.float64
        )
    )
    temperature = torch.tensor([[5_500.0, 2.0e4, 1.0e6]], dtype=torch.float64)
    pressure = torch.tensor([[1.0e-3, 1.0e2, 1.0e7]], dtype=torch.float64)
    base = {
        "log_tau500": depth,
        "velocity_field": torch.zeros(1, 3, 3, dtype=torch.float64),
        "microturbulence": torch.full((1, 3), 1_000.0, dtype=torch.float64),
        "magnetic_field": torch.zeros(1, 3, 3, dtype=torch.float64),
    }
    atmosphere = StratifiedAtmosphere(
        temperature=temperature,
        gas_pressure=pressure,
        **base,
    )
    extinction = backend.reference_extinction(atmosphere)
    assert torch.isfinite(extinction).all()
    assert torch.all(extinction > 0.0)
    electron_density = backend.synthesizer.continuum_opacity.electron_density(
        temperature,
        pressure,
    )
    torch.testing.assert_close(
        extinction[0, -1],
        electron_density[0, -1] * THOMSON_CROSS_SECTION_M2,
    )


def test_forward_import_does_not_load_lightning():
    code = (
        "import sys; import prom3theus.inversion.forward; "
        "assert 'prom3theus.observations.data' not in sys.modules; "
        "assert 'prom3theus.training.lightning' not in sys.modules; "
        "assert 'pytorch_lightning' not in sys.modules"
    )
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_forward_composition_owns_velocity_path_and_instrument_application():
    composition, backend, _ = _composition()
    observed = torch.tensor([1.0, 2.0])
    synthesis = composition.prepare_wavelength_grid(observed)
    runtime = ForwardRuntime(
        coarse_depth_grid=torch.tensor([-5.0, 1.0]),
        observed_wavelength_angstrom=observed,
        synthesis_wavelength_angstrom=synthesis,
        radiance_scale=torch.tensor(3.0),
        carrington_angular_velocity_rad_per_s=torch.tensor(0.0),
        instrument_line_of_sight_velocity_correction_m_per_s=torch.tensor(10.0),
    )

    result = composition.synthesize(
        torch.zeros(1, 3),
        runtime=runtime,
        ray_direction=torch.tensor([[-1.0, 0.0, 0.0]]),
        stokes_basis=torch.eye(3).unsqueeze(0),
        observer_los_velocity_m_per_s=torch.tensor([7.0]),
        instrument_response={"gain": torch.tensor([2.0])},
        return_details=True,
    )

    torch.testing.assert_close(result["stokes"], torch.full((1, 4, 2), 2.0))
    assert isinstance(backend.synthesis_call["path"], RayDistancePath)
    torch.testing.assert_close(
        backend.synthesis_call["path"].distance_m,
        result["ray_trace"].distance_m,
    )
    # +10 m/s is a positive-redshift LOS correction, hence -10 in the
    # toward-observer basis component, followed by the +7 m/s observer offset.
    expected_velocity = torch.tensor([1.0, 2.0, -14.0]).expand(1, 2, 3)
    torch.testing.assert_close(
        backend.synthesis_call["atmosphere"].velocity_field,
        expected_velocity,
    )
    torch.testing.assert_close(
        backend.synthesis_call["atmosphere"].magnetic_field,
        torch.tensor([4.0, 5.0, 6.0]).expand(1, 2, 3),
    )
    torch.testing.assert_close(
        result["magnetic_field_synthesis_observer"],
        result["magnetic_field_observer"],
    )
    torch.testing.assert_close(
        result["atmosphere"].velocity_field,
        torch.tensor([1.0, 2.0, 3.0]).expand(1, 2, 3),
    )


def test_forward_applies_only_the_instrument_magnetic_azimuth_convention():
    composition, backend, instrument = _composition()
    instrument.polarization_convention = MagneticAzimuthConvention(
        "test_plus_90",
        90.0,
    )
    wavelength = torch.tensor([1.0, 2.0])
    runtime = ForwardRuntime(
        coarse_depth_grid=torch.tensor([-5.0, 1.0]),
        observed_wavelength_angstrom=wavelength,
        synthesis_wavelength_angstrom=wavelength,
        radiance_scale=1.0,
        carrington_angular_velocity_rad_per_s=0.0,
        instrument_line_of_sight_velocity_correction_m_per_s=0.0,
    )

    result = composition.synthesize(
        torch.zeros(1, 3),
        runtime=runtime,
        ray_direction=torch.tensor([[-1.0, 0.0, 0.0]]),
        stokes_basis=torch.eye(3).unsqueeze(0),
        observer_los_velocity_m_per_s=torch.zeros(1),
        instrument_response={"gain": torch.ones(1)},
        return_details=True,
    )

    physical = torch.tensor([4.0, 5.0, 6.0]).expand(1, 2, 3)
    synthesis = torch.tensor([-5.0, 4.0, 6.0]).expand(1, 2, 3)
    torch.testing.assert_close(result["magnetic_field_observer"], physical)
    torch.testing.assert_close(result["magnetic_field_synthesis_observer"], synthesis)
    torch.testing.assert_close(
        backend.synthesis_call["atmosphere"].magnetic_field,
        synthesis,
    )
    torch.testing.assert_close(result["atmosphere"].magnetic_field, physical)


def test_line_of_sight_velocity_correction_has_a_spectral_gradient():
    class VelocitySensitiveBackend(_Backend):
        def synthesize(self, atmosphere, wavelength_angstrom, *, radiance_scale, path):
            del radiance_scale, path
            los_velocity = atmosphere.velocity_field[..., 2].mean(dim=-1)
            return los_velocity[:, None, None].expand(
                -1, 4, wavelength_angstrom.numel()
            )

    model = _AtmosphereModel()
    backend = VelocitySensitiveBackend()
    composition = LTEForwardComposition(
        atmosphere_model=model,
        backend=backend,
        instrument=_Instrument(),
        velocity_synthesis_mode="carrington_observer_relative",
        depth_refinement=DepthRefinement(False, 1, 0.0),
    )
    wavelength = torch.tensor([1.0, 2.0])
    correction = torch.tensor(0.0, requires_grad=True)
    runtime = ForwardRuntime(
        coarse_depth_grid=torch.tensor([-5.0, 1.0]),
        observed_wavelength_angstrom=wavelength,
        synthesis_wavelength_angstrom=wavelength,
        radiance_scale=1.0,
        carrington_angular_velocity_rad_per_s=0.0,
        instrument_line_of_sight_velocity_correction_m_per_s=correction,
    )
    stokes_basis = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])

    result = composition.synthesize(
        torch.zeros(1, 3),
        runtime=runtime,
        ray_direction=torch.tensor([[-1.0, 0.0, 0.0]]),
        stokes_basis=stokes_basis,
        observer_los_velocity_m_per_s=torch.zeros(1),
        instrument_response={"gain": torch.ones(1)},
    )
    result["stokes"].sum().backward()

    torch.testing.assert_close(correction.grad, torch.tensor(-8.0))


def test_refinement_retains_coarse_gradients_and_evaluates_only_new_points():
    coarse_grid = torch.tensor([-5.0, -2.0, 1.0])
    coarse_distance = torch.tensor([[0.0, 1.0, 2.0]])
    coarse_signal = torch.tensor(
        [[5100.0, 5600.0, 6200.0]],
        requires_grad=True,
    )
    zeros = torch.zeros((1, 3, 3))
    coarse_atmosphere = StratifiedAtmosphere(
        log_tau500=coarse_grid,
        temperature=coarse_signal,
        velocity_field=zeros,
        microturbulence=torch.full((1, 3), 1000.0),
        magnetic_field=zeros.clone(),
        gas_pressure=torch.full((1, 3), 1000.0),
        geometric_height_m=coarse_distance,
    )
    coarse_position = torch.stack(
        (
            torch.ones_like(coarse_distance),
            torch.zeros_like(coarse_distance),
            coarse_distance,
        ),
        dim=-1,
    )
    coarse_trace = RayTraceResult(
        position_m=coarse_position,
        distance_m=coarse_distance,
        chart_xy_mm=coarse_position[..., :2],
        geometric_height_m=coarse_distance,
    )

    class FineAtmosphere(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("solar_radius_m", torch.tensor(1.0))
            self.fine_temperature = torch.nn.Parameter(torch.tensor(7000.0))
            self.evaluated_points = 0

        @staticmethod
        def _position_chart_height(position_rsun):
            return position_rsun[..., :2], position_rsun[..., 2]

        def evaluate_chart_height_points(self, coordinates, height):
            del coordinates
            self.evaluated_points += int(height.numel())
            scalar = self.fine_temperature.expand_as(height)
            vector = torch.zeros((*height.shape, 3), dtype=height.dtype)
            return {
                "temperature": scalar,
                "velocity_field": vector,
                "microturbulence": torch.full_like(height, 1000.0),
                "magnetic_field": vector.clone(),
                "gas_pressure": torch.full_like(height, 1000.0),
            }

    class CoarseModel(FineAtmosphere):
        def trace_rays(self, coordinates, ray_direction, depth_grid):
            del coordinates, ray_direction, depth_grid
            return coarse_atmosphere, coarse_trace

    model = CoarseModel()
    composition = LTEForwardComposition(
        atmosphere_model=model,
        backend=_Backend(),
        instrument=_Instrument(),
        velocity_synthesis_mode="carrington_registered_relative",
        depth_refinement=DepthRefinement(True, 2, 0.05),
    )
    refined, trace = composition.trace_atmosphere(
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.0, 1.0]]),
        coarse_grid,
    )

    assert model.evaluated_points == 2
    assert refined.depth == 5
    for index, distance in enumerate(coarse_distance[0]):
        merged_index = torch.nonzero(trace.distance_m[0] == distance).item()
        torch.testing.assert_close(
            refined.temperature[0, merged_index],
            coarse_signal[0, index],
        )
    refined.temperature.sum().backward()
    torch.testing.assert_close(coarse_signal.grad, torch.ones_like(coarse_signal))
    torch.testing.assert_close(model.fine_temperature.grad, torch.tensor(2.0))


def test_composition_does_not_change_registered_state_names():
    owner = torch.nn.Module()
    owner.atmosphere_model = torch.nn.Linear(3, 2)
    owner.synthesizer = torch.nn.Linear(2, 2)
    owner.instrument = torch.nn.Linear(2, 2)
    before = tuple(owner.state_dict())
    composition, _, _ = _composition(mode="carrington_registered_relative")

    owner.forward_composition = composition

    assert tuple(owner.state_dict()) == before
    assert not isinstance(composition, torch.nn.Module)


def test_lightning_contains_only_thin_forward_delegates():
    source = inspect.getsourcefile(LTEInversionModule)
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    assert "RayDistancePath" not in text
    assert "project_vectors_to_stokes" not in text
    assert "def _refine_ray_sampling" not in text
    assert "def _apply_observation_operator" not in text


def test_velocity_gauge_rejects_missing_observer_input():
    composition, _, _ = _composition()
    runtime = ForwardRuntime(
        coarse_depth_grid=torch.tensor([-5.0, 1.0]),
        observed_wavelength_angstrom=torch.tensor([1.0, 2.0]),
        synthesis_wavelength_angstrom=torch.tensor([1.0, 2.0]),
        radiance_scale=1.0,
        carrington_angular_velocity_rad_per_s=0.0,
        instrument_line_of_sight_velocity_correction_m_per_s=0.0,
    )

    with pytest.raises(ValueError, match="requires observer_los_velocity"):
        composition.synthesize(
            torch.zeros(1, 3),
            runtime=runtime,
            ray_direction=torch.tensor([[-1.0, 0.0, 0.0]]),
            stokes_basis=torch.eye(3).unsqueeze(0),
            instrument_response={"gain": torch.tensor([1.0])},
        )
