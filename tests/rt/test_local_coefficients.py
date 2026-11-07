"""Scientific contracts for pointwise polarized LTE coefficients."""

from unittest.mock import patch

import pytest
import torch

from prom3theus.core import H_PLANCK, K_BOLTZMANN, SPEED_OF_LIGHT
from prom3theus.rt.local_coefficients import LTELocalCoefficients
from prom3theus.rt.wavelength import air_to_vacuum_angstrom


def _problem(shape=(2,), dtype=torch.float64):
    provider = LTELocalCoefficients(line_ids=["FeI_6302.4932"]).to(dtype=dtype)
    state = {
        "temperature": torch.full(shape, 5800.0, dtype=dtype),
        "gas_pressure": torch.full(shape, 2500.0, dtype=dtype),
        "microturbulence": torch.full(shape, 900.0, dtype=dtype),
        "velocity_field": torch.tensor([100.0, -200.0, 300.0], dtype=dtype).expand(*shape, 3).clone(),
        "magnetic_field": torch.tensor([250.0, -350.0, 900.0], dtype=dtype).expand(*shape, 3).clone(),
    }
    air = torch.tensor([6302.44, 6302.51, 6302.58], dtype=torch.float64)
    vacuum = air_to_vacuum_angstrom(air)
    geometry = {"direction": torch.tensor([0.0, 0.0, 1.0], dtype=dtype), "stokes_basis": torch.eye(3, dtype=dtype), "radiance_scale": 1.0e13}
    return provider, state, air, vacuum, geometry


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_local_coefficients_accept_arbitrary_point_shapes_and_match_line_physics(shape):
    provider, atmosphere, air, vacuum, geometry = _problem(shape)
    result = provider(atmosphere, vacuum, **geometry)
    assert result.propagation_m1.shape == (*shape, 3, 4, 4)
    assert result.source.shape == (*shape, 3, 4)

    # Independent use of the established line-opacity interface, with its air
    # frequency offsets and physical ray-basis inputs, checks the extraction.
    temperature, pressure = atmosphere["temperature"], atmosphere["gas_pressure"]
    plasma = provider.plasma_provider
    state = plasma.prepare_plasma_state(temperature, pressure)
    reservoir = plasma.reference_line_thermodynamics(temperature, pressure, state)
    direct = provider.line_opacity(
        air, temperature, -atmosphere["velocity_field"][..., 2],
        atmosphere["microturbulence"], atmosphere["magnetic_field"],
        plasma(vacuum, temperature, pressure, state), torch.ones_like(temperature),
        lower_level_populations=plasma.reference_lower_level_populations(provider.lines, temperature, pressure, state),
        damping_electron_density=None,
        damping_hydrogen_neutral=reservoir["hydrogen_neutral"], normalize_to_alpha500=False,
    )
    torch.testing.assert_close(result.propagation_m1, direct, rtol=2e-10, atol=1e-15)

    wavelength_m = vacuum * 1e-10
    exact_source = (2 * H_PLANCK * SPEED_OF_LIGHT**2 / wavelength_m**5) / torch.expm1(
        H_PLANCK * SPEED_OF_LIGHT / (K_BOLTZMANN * wavelength_m * temperature[..., None])
    ) / geometry["radiance_scale"]
    torch.testing.assert_close(result.source[..., 0], exact_source, rtol=2e-14, atol=0)
    torch.testing.assert_close(result.source[..., 1:], torch.zeros_like(result.source[..., 1:]))


def test_paired_queries_match_block_diagonal_in_values_and_derivatives():
    provider, state, _, vacuum, geometry = _problem((3,))
    state["temperature"].requires_grad_()
    state["magnetic_field"].requires_grad_()
    vacuum = vacuum.clone().requires_grad_()
    block = provider(state, vacuum, **geometry)
    paired = provider.evaluate_paired(state, vacuum, **geometry)
    index = torch.arange(3)
    torch.testing.assert_close(paired.propagation_m1, block.propagation_m1[index, index])
    torch.testing.assert_close(paired.source, block.source[index, index])
    inputs = (state["temperature"], state["magnetic_field"], vacuum)
    loss_paired = paired.propagation_m1.square().sum() + paired.source.sum()
    loss_block = block.propagation_m1[index, index].square().sum() + block.source[index, index].sum()
    paired_grad = torch.autograd.grad(loss_paired, inputs, retain_graph=True)
    block_grad = torch.autograd.grad(loss_block, inputs)
    for actual, expected in zip(paired_grad, block_grad):
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all()


def test_stokes_basis_rotation_transforms_full_propagation_matrix():
    provider, state, _, vacuum, geometry = _problem(())
    original = provider(state, vacuum, **geometry).propagation_m1
    psi = torch.tensor(0.37, dtype=torch.float64)
    c, s = psi.cos(), psi.sin()
    geometry["stokes_basis"] = torch.stack((torch.stack((c,s,c*0)), torch.stack((-s,c,c*0)), c.new_tensor([0,0,1])))
    rotated = provider(state, vacuum, **geometry).propagation_m1
    rotation = torch.eye(4, dtype=torch.float64)
    rotation[1, 1] = rotation[2, 2] = (2*psi).cos()
    rotation[1, 2] = (2*psi).sin()
    rotation[2, 1] = -(2*psi).sin()
    torch.testing.assert_close(rotated, rotation @ original @ rotation.T, atol=1e-16, rtol=1e-10)


def test_material_motion_toward_observer_blueshifts_opacity():
    provider, state, _, _, geometry = _problem(())
    air = torch.linspace(6302.25, 6302.75, 301, dtype=torch.float64)
    vacuum = air_to_vacuum_angstrom(air)
    state["magnetic_field"].zero_()
    state["velocity_field"] = torch.tensor([0.,0.,3000.], dtype=torch.float64)
    approaching = provider(state, vacuum, **geometry).propagation_m1[..., 0, 0]
    state["velocity_field"] = -state["velocity_field"]
    receding = provider(state, vacuum, **geometry).propagation_m1[..., 0, 0]
    assert vacuum[approaching.argmax()] < vacuum[receding.argmax()]


def test_float32_hot_fade_has_valid_damping_densities_and_finite_gradients():
    provider, state, _, _, geometry = _problem((7,), dtype=torch.float32)
    # The first five values previously yielded negative neutral H despite
    # finite positive T/P, stopping evaluate_paired during a training step.
    state["temperature"] = torch.tensor(
        [31507.31640625, 31508.181640625, 31510.9921875, 31511.857421875,
         31513.8046875, 31623., 48000.], requires_grad=True)
    state["gas_pressure"] = torch.full((7,), 100., requires_grad=True)
    reservoir = provider.plasma_provider.reference_line_thermodynamics(
        state["temperature"], state["gas_pressure"])
    assert torch.all(reservoir["hydrogen_neutral"] >= 0)
    assert torch.all(reservoir["fe_i_population_over_partition"] >= 0)
    wave = torch.full((7,), 6304., dtype=torch.float64)
    result = provider.evaluate_paired(state, wave, **{**geometry, "radiance_scale": 2.5e13})
    assert torch.isfinite(result.propagation_m1).all()
    loss = (result.propagation_m1 * 1e6).square().mean() + result.source.square().mean()
    loss.backward()
    for key in ("temperature", "gas_pressure"):
        assert torch.isfinite(state[key].grad).all()


def test_temperature_and_magnetic_gradients_match_finite_differences():
    provider, state, _, vacuum, geometry = _problem(())
    state["temperature"].requires_grad_()
    state["magnetic_field"].requires_grad_()

    def objective():
        result = provider(state, vacuum, **geometry)
        return result.source.sum() + 1e5 * result.propagation_m1[..., 0, 3].sum()

    analytic_t, analytic_b = torch.autograd.grad(objective(), (state["temperature"], state["magnetic_field"]))
    for key, index, step, analytic in (("temperature", None, 0.01, analytic_t), ("magnetic_field", 2, 0.01, analytic_b[2])):
        base = state[key].detach().clone()
        plus, minus = base.clone(), base.clone()
        if index is None:
            plus += step
            minus -= step
        else:
            plus[index] += step
            minus[index] -= step
        state[key] = plus
        upper = objective()
        state[key] = minus
        lower = objective()
        state[key] = base
        torch.testing.assert_close(analytic, (upper-lower)/(2*step), rtol=1e-5, atol=1e-10)


def test_prepared_float32_grid_preserves_precise_offsets():
    double, state64, _, vacuum, geometry64 = _problem(())
    reference = double(state64, vacuum, **geometry64)
    provider, state32, _, _, geometry32 = _problem((), torch.float32)
    provider.prepare_wavelength_grid(vacuum)
    provider.float()
    result = provider(state32, **geometry32)
    torch.testing.assert_close(result.propagation_m1.double(), reference.propagation_m1, rtol=8e-5, atol=1e-12)
    torch.testing.assert_close(result.source.double(), reference.source, rtol=5e-6, atol=0)


def test_combined_plasma_state_reuses_the_joint_eos_lookup():
    provider, state, _, _, _ = _problem(())
    eos = provider.plasma_provider.eos
    with patch.object(eos, "_states", wraps=eos._states) as evaluate:
        plasma = provider.plasma_provider.prepare_plasma_state(state["temperature"], state["gas_pressure"])
    # One actual state and one pressure-anchor state, rather than separate
    # actual-state composition evaluations for density and electron density.
    assert evaluate.call_count == 2
    torch.testing.assert_close(plasma.mass_density, eos.mass_density(state["temperature"], state["gas_pressure"]))
    torch.testing.assert_close(plasma.electron_density, eos.electron_density(state["temperature"], state["gas_pressure"]))


def test_invalid_basis_is_rejected():
    provider, state, _, vacuum, geometry = _problem(())
    geometry["stokes_basis"][1] *= -1
    with pytest.raises(ValueError, match="right-handed"):
        provider(state, vacuum, **geometry)


def test_reusing_prepared_physics_preserves_spectral_gradients():
    provider, state, _, wave, geometry = _problem((2,))
    state = {name: value.clone().requires_grad_() for name, value in state.items()}
    prepared = provider.prepare_atmosphere(state,
        direction=geometry['direction'], stokes_basis=geometry['stokes_basis'])
    reused = [provider.evaluate_prepared(prepared, block,
        radiance_scale=geometry['radiance_scale']) for block in wave.split(1)]
    independent = [provider(state, block, **geometry) for block in wave.split(1)]
    def objective(blocks):
        return sum(block.propagation_m1.square().sum() + block.source.square().sum()
                   for block in blocks)
    torch.testing.assert_close(objective(reused), objective(independent))
    gradients = [torch.autograd.grad(objective(blocks), tuple(state.values()),
                 retain_graph=True) for blocks in (reused, independent)]
    for actual, expected in zip(*gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_native_per_ray_spectral_grid_preserves_values_and_gradients():
    """A shifted grid broadcasts over depth without replicating plasma tables."""
    provider, state, _, wave, geometry = _problem((2, 3))
    state = {key: value.clone().requires_grad_() for key, value in state.items()}
    shifts = torch.tensor([-.025, .04], dtype=wave.dtype, requires_grad=True)
    grid = wave[None, None, :] + shifts[:, None, None]
    prepared = provider.prepare_atmosphere(state, direction=geometry['direction'],
                                          stokes_basis=geometry['stokes_basis'])
    table_shape = prepared.continuum_logs.shape
    actual = provider.evaluate_spectrum(prepared, provider.prepare_spectrum(grid),
                                       radiance_scale=geometry['radiance_scale'])
    expected = [provider({key: value[ray] for key, value in state.items()},
                        wave + shifts[ray], **geometry) for ray in range(2)]
    assert prepared.continuum_logs.shape == table_shape
    k = torch.stack([value.propagation_m1 for value in expected])
    source = torch.stack([value.source for value in expected])
    torch.testing.assert_close(actual.propagation_m1, k)
    torch.testing.assert_close(actual.source, source)
    inputs = (*state.values(), shifts)
    objectives = (actual.propagation_m1.square().sum() + actual.source.sum(),
                  k.square().sum() + source.sum())
    gradients = [torch.autograd.grad(loss, inputs, create_graph=True) for loss in objectives]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
    second = [torch.autograd.grad(sum(g.square().sum() for g in grads), inputs)
              for grads in gradients]
    for a, b in zip(*second):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)


def test_centered_float32_wavelengths_preserve_subpixel_line_offsets_and_gradients():
    from prom3theus.rt.wavelength import CenteredWavelength
    provider, _, _, vacuum, _ = _problem(dtype=torch.float32)
    # These wavelengths differ by much less than one float32 ULP near 6304 A.
    center = provider._rest_vacuum_angstrom[0]
    exact = torch.tensor([center - 1e-5, center, center + 1e-5], dtype=torch.float64)
    origin = float(int(center))
    offset = (exact - origin).float().requires_grad_()
    spectrum = provider.prepare_spectrum(CenteredWavelength(origin, offset))
    assert spectrum.vacuum.dtype == spectrum.velocity_offsets.dtype == torch.float32
    expected = provider._velocity_offsets(exact)
    torch.testing.assert_close(spectrum.velocity_offsets.double(), expected, rtol=2e-3, atol=2e-6)
    assert spectrum.velocity_offsets[0, 0] < spectrum.velocity_offsets[0, 1] < spectrum.velocity_offsets[0, 2]
    derivative = torch.autograd.grad(spectrum.velocity_offsets.sum(), offset)[0]
    torch.testing.assert_close(derivative, torch.full_like(offset, (SPEED_OF_LIGHT / 1000.) / center), rtol=1e-6, atol=0)
