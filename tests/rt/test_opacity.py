import pytest
import torch

from prom3theus.core import ATOMIC_MASS_UNIT, K_BOLTZMANN
from prom3theus.rt import (
    AtomicDatabase,
    ContinuumOpacity,
    HybridSolarEOS,
    VoigtFaraday,
    damping_rate,
    doppler_velocity,
    doppler_width_frequency,
    integrated_line_opacity,
    planck_lambda,
)


def test_stic_continuum_matches_pinned_reference_and_differentiates():
    opacity = ContinuumOpacity()
    temperature = torch.tensor(5772.0, dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor(1.0e3, dtype=torch.float64, requires_grad=True)
    wavelength = torch.tensor([5000.0, 6300.0], dtype=torch.float64)
    absorption = opacity.true_absorption(wavelength, temperature, pressure)
    scattering = opacity.scattering_extinction(wavelength, temperature, pressure)
    total = opacity(wavelength, temperature, pressure)
    torch.testing.assert_close(total, absorption + scattering)
    torch.testing.assert_close(
        absorption.detach(),
        torch.tensor([1.83567669e-7, 2.23369566e-7], dtype=torch.float64),
        rtol=1e-5,
        atol=0,
    )
    torch.testing.assert_close(
        scattering.detach(),
        torch.tensor([1.49828433e-9, 7.68653651e-10], dtype=torch.float64),
        rtol=1e-5,
        atol=0,
    )
    total.log().sum().backward()
    assert temperature.grad is not None and torch.isfinite(temperature.grad)
    assert pressure.grad is not None and torch.isfinite(pressure.grad)


def test_atomic_chain_closes_in_si_units():
    atomic = AtomicDatabase()
    line = atomic.get_line("FeI_6302.4932")
    temperature = torch.tensor(6000.0, dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor(1000.0, dtype=torch.float64)
    microturbulence = torch.tensor(1000.0, dtype=torch.float64)
    opacity = ContinuumOpacity(atomic)
    lower_population = opacity.reference_lower_level_populations(
        (line,), temperature, pressure
    )[line.id]
    integrated = integrated_line_opacity(line, lower_population, temperature)
    velocity = doppler_velocity(
        temperature, microturbulence, atomic.element("Fe").atomic_mass_u
    )
    reference_velocity = torch.sqrt(
        2.0
        * K_BOLTZMANN
        * temperature
        / (atomic.element("Fe").atomic_mass_u * ATOMIC_MASS_UNIT)
        + microturbulence.square()
    )
    torch.testing.assert_close(velocity, reference_velocity)
    width = doppler_width_frequency(
        line, temperature, microturbulence, atomic.element("Fe").atomic_mass_u
    )
    profile, _ = VoigtFaraday()(
        torch.zeros((), dtype=torch.float64),
        torch.tensor(0.1, dtype=torch.float64),
        doppler_width=width,
    )
    alpha500 = opacity.volume_extinction_at_5000(temperature, pressure)
    assert torch.isfinite(integrated * profile / alpha500)
    assert torch.all(planck_lambda(temperature, torch.tensor([6302.5])) > 0)
    damping = damping_rate(
        line,
        temperature,
        torch.tensor(1.0e19, dtype=torch.float64),
        torch.tensor(1.0e22, dtype=torch.float64),
    )
    (integrated.log() + width.log() + damping.log()).backward()
    assert temperature.grad is not None and torch.isfinite(temperature.grad)


def test_opacity_primitives_reject_nonphysical_thermodynamics():
    atomic = AtomicDatabase()
    line = atomic.get_line("FeI_6302.4932")
    dtype = torch.float64

    with pytest.raises(ValueError, match="finite and strictly positive"):
        ContinuumOpacity(atomic).prepare_stic_lookup(
            torch.tensor(0.0, dtype=dtype),
            torch.tensor(1000.0, dtype=dtype),
        )
    with pytest.raises(ValueError, match="finite and strictly positive"):
        planck_lambda(
            torch.tensor(float("nan"), dtype=dtype),
            torch.tensor([6300.0], dtype=dtype),
        )
    with pytest.raises(ValueError, match="lower_population"):
        integrated_line_opacity(
            line,
            torch.tensor(-1.0, dtype=dtype),
            torch.tensor(6000.0, dtype=dtype),
        )
    with pytest.raises(ValueError, match="microturbulence"):
        doppler_velocity(
            torch.tensor(6000.0, dtype=dtype),
            torch.tensor(-1.0, dtype=dtype),
            atomic.element("Fe").atomic_mass_u,
        )
    with pytest.raises(ValueError, match="number densities"):
        damping_rate(
            line,
            torch.tensor(6000.0, dtype=dtype),
            torch.tensor(-1.0, dtype=dtype),
            torch.tensor(1.0e22, dtype=dtype),
        )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_stic_lookup_edge_saturates_all_thermodynamic_bounds(dtype):
    opacity = ContinuumOpacity()
    log_temperature = torch.tensor([3.0, 3.6, 5.0], dtype=dtype, requires_grad=True)
    log_pressure = torch.tensor([-3.0, 2.0, 8.0], dtype=dtype, requires_grad=True)
    temperature = 10.0**log_temperature
    pressure = 10.0**log_pressure

    state = opacity.prepare_stic_lookup(temperature, pressure)
    t_axis = opacity.stic_log_temperature.to(log_temperature)
    p_axis = opacity.stic_log_pressure.to(log_pressure)
    expected_temperature = log_temperature.clamp(t_axis[0], t_axis[-1])
    expected_pressure = log_pressure.clamp(p_axis[0], p_axis[-1])
    torch.testing.assert_close(torch.log10(state.temperature), expected_temperature)
    torch.testing.assert_close(torch.log10(state.gas_pressure), expected_pressure)

    temperature_residual, pressure_residual = opacity.stic_support_residuals(
        temperature,
        pressure,
    )
    torch.testing.assert_close(
        temperature_residual,
        log_temperature - expected_temperature,
    )
    torch.testing.assert_close(
        pressure_residual,
        log_pressure - expected_pressure,
    )
    (temperature_residual.square() + pressure_residual.square()).sum().backward()
    assert log_temperature.grad[0] < 0.0
    assert log_temperature.grad[1] == 0.0
    assert log_temperature.grad[2] > 0.0
    assert log_pressure.grad[0] < 0.0
    assert log_pressure.grad[1] == 0.0
    assert log_pressure.grad[2] > 0.0

    metadata = opacity.metadata()
    assert "saturated at the nearest STiC" in metadata["thermodynamic_lookup_policy"]
    assert metadata["log10_temperature_bounds_k"] == [3.4, 4.0]
    assert metadata["log10_gas_pressure_bounds_pa"] == [-1.5, 6.0]


def test_stic_table_quantities_are_constant_beyond_each_edge():
    opacity = ContinuumOpacity()
    temperature = torch.tensor([1.0e3, 2.0e4], dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor([1.0e-3, 1.0e7], dtype=torch.float64, requires_grad=True)
    state = opacity.prepare_stic_lookup(temperature, pressure)
    wavelength = torch.tensor([5000.0, 6300.0], dtype=torch.float64)

    actual = opacity(wavelength, temperature, pressure)
    edge = opacity(wavelength, state.temperature, state.gas_pressure)
    torch.testing.assert_close(actual, edge)
    actual_thermodynamics = opacity.reference_line_thermodynamics(
        temperature,
        pressure,
    )
    edge_thermodynamics = opacity.reference_line_thermodynamics(
        state.temperature,
        state.gas_pressure,
    )
    for name in actual_thermodynamics:
        torch.testing.assert_close(
            actual_thermodynamics[name],
            edge_thermodynamics[name],
        )

    actual.sum().backward()
    torch.testing.assert_close(temperature.grad, torch.zeros_like(temperature.grad))
    torch.testing.assert_close(pressure.grad, torch.zeros_like(pressure.grad))


def test_thermodynamic_eos_continues_below_saturated_stic_pressure_edge():
    opacity = ContinuumOpacity(AtomicDatabase())
    eos = HybridSolarEOS()
    temperature = torch.tensor([7000.0, 7000.0], requires_grad=True)
    pressure = torch.tensor([1.0, 1.0e-4], requires_grad=True)

    opacity_density = opacity.reference_mass_density(temperature, pressure)
    edge_pressure = pressure.new_full(
        pressure.shape,
        10.0 ** float(opacity.stic_log_pressure[0]),
    )
    edge_density = opacity.reference_mass_density(temperature, edge_pressure)
    torch.testing.assert_close(opacity_density[1], edge_density[1])
    density = eos.mass_density(temperature, pressure)

    assert torch.isfinite(density).all()
    assert torch.all(density > 0)
    assert density[1] < density[0]
    density.sum().backward()
    assert torch.isfinite(temperature.grad).all()
    assert torch.isfinite(pressure.grad).all()
