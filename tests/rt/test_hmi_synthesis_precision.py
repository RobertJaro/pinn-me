"""End-to-end Fe I 6173 magnetic response, including float32 training gradients."""

import pytest
import torch

from prom3theus.instruments.hmi.operator import HMIFilterProfiles
from prom3theus.rt import LTESynthesizer, OpticalDepthPath, StratifiedAtmosphere


@pytest.fixture(scope="module")
def problem():
    # Simple fixed thermodynamics isolate polarized synthesis from network/PDEs.
    grid = torch.linspace(-5.0, 1.0, 25)
    solver = LTESynthesizer(log_tau500=grid, line_ids=("FeI_6173.3352",)).float()
    wave = torch.linspace(6172.9352, 6173.7352, 121)
    return solver, grid, wave


def synth(problem, field, *, prepared=False, velocity=0.0):
    solver, grid, wave = problem
    field = torch.as_tensor(field, dtype=grid.dtype)
    if field.ndim == 1:
        field = field[None]
    n = len(field)
    atmosphere = StratifiedAtmosphere(
        depth_coordinate=grid,
        temperature=torch.linspace(4500.0, 7000.0, len(grid), dtype=grid.dtype)[
            None
        ].expand(n, -1),
        gas_pressure=torch.logspace(0.0, 4.5, len(grid), dtype=grid.dtype)[None].expand(
            n, -1
        ),
        magnetic_field=field[:, None].expand(-1, len(grid), -1),
        velocity_field=torch.stack(
            (
                torch.zeros_like(grid),
                torch.zeros_like(grid),
                torch.ones_like(grid) * velocity,
            ),
            -1,
        )[None].expand(n, -1, -1),
        microturbulence=torch.full((n, len(grid)), 1000.0, dtype=grid.dtype),
    )
    return solver(
        atmosphere,
        None if prepared else wave,
        path=OpticalDepthPath(),
        radiance_scale=1e13,
    )


@pytest.mark.parametrize("prepared", [False, True])
def test_float32_small_magnetic_updates_agree_with_autograd(problem, prepared):
    solver, _, wave = problem
    solver.prepare_wavelength_grid(wave)
    field = torch.tensor([0.0, 0.0, 900.0], requires_grad=True)
    profile = synth(problem, field, prepared=prepared)
    gradient = torch.autograd.grad(profile[0, 3, 45], field)[0][2]
    with torch.no_grad():
        plus = synth(problem, [0.0, 0.0, 900.1], prepared=prepared)[0, 3, 45]
        minus = synth(problem, [0.0, 0.0, 899.9], prepared=prepared)[0, 3, 45]
    assert abs(gradient) > 1e-5
    # Previously the finite difference was exactly zero despite nonzero AD.
    torch.testing.assert_close((plus - minus) / 0.2, gradient, atol=2e-7, rtol=0.003)


def test_float32_small_velocity_updates_agree_with_autograd(problem):
    speed = torch.tensor(0.0, requires_grad=True)
    profile = synth(problem, [300.0, 400.0, 900.0], velocity=speed)
    gradient = torch.autograd.grad(profile[0, 0, 45], speed)[0]
    with torch.no_grad():
        finite = (
            synth(problem, [300.0, 400.0, 900.0], velocity=1.0)[0, 0, 45]
            - synth(problem, [300.0, 400.0, 900.0], velocity=-1.0)[0, 0, 45]
        ) / 2
    assert abs(gradient) > 1e-5
    torch.testing.assert_close(finite, gradient, atol=2e-6, rtol=0.005)


def test_full_6173_profiles_preserve_azimuth_and_polarity_symmetries(problem):
    with torch.no_grad():
        spectra = synth(
            problem,
            [
                [600.0, 400.0, 900.0],
                [-600.0, -400.0, 900.0],
                [-400.0, 600.0, 900.0],
                [0.0, 0.0, 900.0],
                [0.0, 0.0, -900.0],
            ],
        )
    torch.testing.assert_close(spectra[0], spectra[1], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        spectra[2], spectra[0] * torch.tensor([1, -1, -1, 1])[:, None]
    )
    torch.testing.assert_close(
        spectra[4], spectra[3] * torch.tensor([1, 1, 1, -1])[:, None]
    )
    assert (spectra[0, 1:].abs().amax(-1) > 0.01).all()


def test_hmi_filter_integration_keeps_small_field_response(problem):
    solver, grid, _ = problem
    observed = torch.tensor(
        [6173.1713, 6173.2401, 6173.3089, 6173.3777, 6173.4465, 6173.5153]
    )
    quadrature = torch.linspace(6172.94, 6173.73, 121)
    instrument = HMIFilterProfiles(quadrature, 0.65, 90.0)
    wave = instrument.synthesis_grid(observed)
    # Synthetic smooth passbands test integration/gradients, not the production calibration.
    weights = torch.exp(-0.5 * ((quadrature[None] - observed[:, None]) / 0.04).square())
    weights = weights / weights.sum(-1, keepdim=True)

    def sample(field):
        rotated = instrument.polarization_convention.to_synthesis_frame(field)
        spectra = synth((solver, grid, wave), rotated)
        return instrument(
            spectra,
            wave,
            observed,
            spectral_weights=weights[None],
            continuum_weights=torch.zeros(1, 6),
        )

    field = torch.tensor([0.0, 0.0, 900.0], requires_grad=True)
    gradient = torch.autograd.grad(sample(field)[0, 3, 1], field)[0][2]
    with torch.no_grad():
        finite = (
            sample(torch.tensor([0.0, 0.0, 900.1]))[0, 3, 1]
            - sample(torch.tensor([0.0, 0.0, 899.9]))[0, 3, 1]
        ) / 0.2
    assert gradient.abs() > 1e-5
    torch.testing.assert_close(finite, gradient, atol=3e-7, rtol=0.005)


def test_prepared_offsets_survive_float32_cast_without_frequency_quantization():
    grid = torch.linspace(-5.0, 1.0, 25, dtype=torch.float64)
    solver = LTESynthesizer(log_tau500=grid, line_ids=("FeI_6173.3352",)).double()
    wave = torch.linspace(6172.9352, 6173.7352, 121).double()
    solver.prepare_wavelength_grid(wave)
    with torch.no_grad():
        reference = synth((solver, grid, wave), [600.0, 400.0, 900.0], prepared=True)
    solver.float()
    field = torch.tensor([0.0, 0.0, 900.0], requires_grad=True)
    actual = synth((solver, grid.float(), wave.float()), field, prepared=True)
    gradient = torch.autograd.grad(actual[0, 3, 45], field)[0][2]
    with torch.no_grad():
        result = synth(
            (solver, grid.float(), wave.float()), [600.0, 400.0, 900.0], prepared=True
        )
    torch.testing.assert_close(result.double(), reference, atol=2e-5, rtol=2e-4)
    assert gradient.abs() > 1e-5


def test_quv_cartesian_magnetic_jacobian_matches_finite_differences(problem):
    field = torch.tensor([600.0, 400.0, 900.0], requires_grad=True)

    def observables(vector):
        return synth(problem, vector)[0, 1:, 45]

    automatic = torch.autograd.functional.jacobian(observables, field)
    with torch.no_grad():
        finite = torch.stack(
            [
                (observables(field + delta) - observables(field - delta)) / 2
                for delta in torch.eye(3)
            ],
            dim=-1,
        )
    assert (automatic.abs().amax(dim=0) > 1e-5).all()
    torch.testing.assert_close(automatic, finite, atol=2e-7, rtol=0.005)


@pytest.mark.parametrize("prepared", [False, True])
def test_float32_synthesis_never_promotes_to_float64(problem, prepared):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class ModelDtypeOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if isinstance(value, torch.Tensor):
                    assert value.dtype not in (torch.float64, torch.complex128), func
            return result

    solver, _, wave = problem
    with ModelDtypeOnly():
        solver.prepare_wavelength_grid(wave)
        field = torch.tensor([600.0, 400.0, 900.0], requires_grad=True)
        result = synth(problem, field, prepared=prepared)
        torch.autograd.grad(result.square().sum(), field)
    assert solver._prepared_line_velocity_offsets_km_s.dtype == torch.float32
    assert solver._prepared_line_velocity_offsets_km_s.abs().max() < 100


def test_scaled_air_offsets_match_independent_vacuum_frequency_reference(problem):
    from prom3theus.core import SPEED_OF_LIGHT
    from prom3theus.rt.wavelength import air_to_vacuum_angstrom

    solver, _, wave = problem
    # Float64 is used only as an independent test oracle, never in production.
    vacuum = air_to_vacuum_angstrom(wave.double())
    centre = air_to_vacuum_angstrom(torch.tensor([6173.3352], dtype=torch.float64))
    expected = (SPEED_OF_LIGHT / 1000) * (vacuum - centre) / vacuum
    actual = solver.line_opacity.velocity_offsets_km_s(wave)[0]
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-6)
