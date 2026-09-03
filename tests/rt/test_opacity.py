import pytest
import torch

from prom3theus.core import ATOMIC_MASS_UNIT, K_BOLTZMANN
from prom3theus.rt import (
    AtomicDatabase,
    ContinuumOpacity,
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

    with pytest.raises(ValueError, match="outside the prepared STiC"):
        ContinuumOpacity(atomic).prepare_stic_lookup(
            torch.tensor(2000.0, dtype=dtype),
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
