import torch

from prom3theus.rt import (
    ContinuumOpacity,
    SolarPlasmaTable,
    THOMSON_CROSS_SECTION_M2,
)


def test_combined_table_reproduces_stic_radiative_values_in_native_domain():
    stic = ContinuumOpacity().to(dtype=torch.float64)
    plasma = SolarPlasmaTable().to(dtype=torch.float64)
    temperature = torch.tensor([3_000.0, 5_500.0, 9_000.0], dtype=torch.float64)
    pressure = torch.tensor([1.0e-1, 1.0e2, 1.0e5], dtype=torch.float64)
    wavelength = torch.tensor([5000.0, 6302.5], dtype=torch.float64)
    state = plasma.prepare_plasma_state(temperature, pressure)

    torch.testing.assert_close(
        plasma.true_absorption(wavelength, temperature, pressure, state),
        stic.true_absorption(wavelength, temperature, pressure),
    )
    torch.testing.assert_close(
        plasma.scattering_extinction(wavelength, temperature, pressure, state),
        stic.scattering_extinction(wavelength, temperature, pressure),
    )
    torch.testing.assert_close(
        plasma.reference_neutral_hydrogen_density(temperature, pressure, state),
        stic.reference_neutral_hydrogen_density(temperature, pressure),
    )
    torch.testing.assert_close(
        plasma.reference_fe_i_population_over_partition(temperature, pressure, state),
        stic.reference_fe_i_population_over_partition(temperature, pressure),
    )


def test_photospheric_quantities_fade_to_zero_and_scattering_becomes_thomson():
    plasma = SolarPlasmaTable().to(dtype=torch.float64)
    lower, upper = plasma.eos.transition_temperature_bounds_k
    temperature = torch.tensor(
        [lower, (lower * upper) ** 0.5, upper, 1.0e6], dtype=torch.float64
    )
    pressure = torch.full_like(temperature, 100.0)
    wavelength = torch.tensor([5000.0, 6302.5], dtype=torch.float64)
    state = plasma.prepare_plasma_state(temperature, pressure)

    torch.testing.assert_close(
        state.photospheric_weight,
        torch.tensor([1.0, 0.5, 0.0, 0.0], dtype=torch.float64),
    )
    absorption = plasma.true_absorption(wavelength, temperature, pressure, state)
    scattering = plasma.scattering_extinction(wavelength, temperature, pressure, state)
    neutral_h = plasma.reference_neutral_hydrogen_density(temperature, pressure, state)
    fe_i = plasma.reference_fe_i_population_over_partition(temperature, pressure, state)

    torch.testing.assert_close(absorption[2:], torch.zeros_like(absorption[2:]))
    torch.testing.assert_close(neutral_h[2:], torch.zeros_like(neutral_h[2:]))
    torch.testing.assert_close(fe_i[2:], torch.zeros_like(fe_i[2:]))
    expected_thomson = (
        state.electron_density[2:, None] * THOMSON_CROSS_SECTION_M2
    ).expand_as(scattering[2:])
    torch.testing.assert_close(scattering[2:], expected_thomson)


def test_fade_endpoints_have_zero_weight_derivative():
    plasma = SolarPlasmaTable().to(dtype=torch.float64)
    bounds = torch.tensor(
        plasma.eos.transition_temperature_bounds_k,
        dtype=torch.float64,
        requires_grad=True,
    )
    weight = plasma._photospheric_weight(bounds)
    derivative = torch.autograd.grad(weight.sum(), bounds)[0]

    torch.testing.assert_close(
        derivative, torch.zeros_like(derivative), atol=1.0e-15, rtol=0.0
    )


def test_hot_thermodynamics_reach_exact_fully_ionized_ideal_limit():
    plasma = SolarPlasmaTable().to(dtype=torch.float64)
    lower, upper = plasma.eos.ideal_transition_temperature_bounds_k
    temperature = torch.tensor([lower, upper, 1.0e8], dtype=torch.float64)
    pressure = torch.tensor([1.0, 100.0, 1.0e5], dtype=torch.float64)

    expected_mu = torch.full_like(
        temperature, plasma.eos.mean_molecular_weight_fully_ionized
    )
    torch.testing.assert_close(
        plasma.mean_molecular_weight(temperature[1:], pressure[1:]),
        expected_mu[1:],
    )
    density = plasma.mass_density(temperature[1:], pressure[1:])
    density_ratio = density[1] / density[0]
    torch.testing.assert_close(
        density_ratio,
        (pressure[2] / temperature[2]) / (pressure[1] / temperature[1]),
    )


def test_pressure_continuation_preserves_unbounded_scaling():
    plasma = SolarPlasmaTable().to(dtype=torch.float64)
    temperature = torch.tensor([5_500.0, 5_500.0], dtype=torch.float64)
    pressure = torch.tensor([1.0e7, 2.0e7], dtype=torch.float64)
    wavelength = torch.tensor([5000.0], dtype=torch.float64)
    state = plasma.prepare_plasma_state(temperature, pressure)
    absorption = plasma.true_absorption(wavelength, temperature, pressure, state)
    neutral_h = plasma.reference_neutral_hydrogen_density(temperature, pressure, state)

    torch.testing.assert_close(
        absorption[1] / absorption[0], torch.tensor([4.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        neutral_h[1] / neutral_h[0], torch.tensor(2.0, dtype=torch.float64)
    )
