import math

import pytest
import torch

from prom3theus.core import ATOMIC_MASS_UNIT, K_BOLTZMANN
from prom3theus.rt import ContinuumOpacity, HybridSolarEOS


def test_hybrid_eos_is_exactly_stic_in_the_photospheric_domain():
    eos = HybridSolarEOS()
    opacity = ContinuumOpacity()
    temperature = torch.tensor(
        [10.0**3.4, 3500.0, 5000.0, 9000.0, 1.0e4], dtype=torch.float64
    )
    pressure = torch.tensor([10.0**-1.5, 0.1, 100.0, 1.0e5, 1.0e6], dtype=torch.float64)

    torch.testing.assert_close(
        eos.mass_density(temperature, pressure),
        opacity.reference_mass_density(temperature, pressure),
    )
    torch.testing.assert_close(
        eos.electron_density(temperature, pressure),
        opacity.reference_electron_density(temperature, pressure),
    )


@pytest.mark.parametrize(
    ("dtype", "relative_tolerance"),
    ((torch.float32, 2.0e-5), (torch.float64, 1.0e-12)),
)
def test_hybrid_eos_matches_stic_inside_boundary_cells(dtype, relative_tolerance):
    eos = HybridSolarEOS()
    opacity = ContinuumOpacity()
    log_temperature = torch.tensor([3.4001, 3.405, 3.995, 3.9999], dtype=dtype)[:, None]
    log_pressure = torch.tensor([-1.4999, -1.485, 5.985, 5.9999], dtype=dtype)[None, :]
    temperature = 10.0**log_temperature
    pressure = 10.0**log_pressure

    torch.testing.assert_close(
        eos.mass_density(temperature, pressure),
        opacity.reference_mass_density(temperature, pressure),
        rtol=relative_tolerance,
        atol=0.0,
    )
    torch.testing.assert_close(
        eos.electron_density(temperature, pressure),
        opacity.reference_electron_density(temperature, pressure),
        rtol=relative_tolerance,
        atol=0.0,
    )


def test_hybrid_eos_reaches_the_fully_ionized_coronal_limit_and_differentiates():
    eos = HybridSolarEOS()
    temperature = torch.tensor(1.0e8, requires_grad=True)
    pressure = torch.tensor(3.0e-2, requires_grad=True)

    density = eos.mass_density(temperature, pressure)
    expected = (
        eos.mean_molecular_weight_fully_ionized
        * ATOMIC_MASS_UNIT
        * pressure
        / (K_BOLTZMANN * temperature)
    )
    torch.testing.assert_close(density, expected, rtol=3.0e-6, atol=0.0)
    assert 0.5 < eos.mean_molecular_weight_fully_ionized < 0.7
    electron_density = eos.electron_density(temperature, pressure)
    expected_electron_density = density / (
        eos.mean_molecular_weight_per_electron * ATOMIC_MASS_UNIT
    )
    torch.testing.assert_close(
        electron_density, expected_electron_density, rtol=3.0e-6, atol=0.0
    )
    torch.autograd.backward((density, electron_density))
    assert torch.isfinite(temperature.grad)
    assert torch.isfinite(pressure.grad)


def test_chianti_mapping_remains_exact_through_ten_mk():
    eos = HybridSolarEOS()
    log_temperature = torch.tensor([4.5, 5.0, 6.0, 7.0], dtype=torch.float64)
    temperature = 10.0**log_temperature
    pressure = torch.logspace(-3.0, 3.0, log_temperature.numel(), dtype=torch.float64)

    _, _, log_particles, log_electrons = eos._log_mappings(temperature, pressure)
    expected_log_electrons = eos._chianti_log_electrons(log_temperature)
    expected_log_particles = torch.log10(
        10.0**expected_log_electrons + eos.nuclei_per_h_nucleus
    )

    torch.testing.assert_close(
        log_electrons, expected_log_electrons, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        log_particles, expected_log_particles, rtol=0.0, atol=0.0
    )


def test_upper_transition_blends_from_ten_to_exactly_twenty_mk():
    eos = HybridSolarEOS()
    midpoint_temperature = math.sqrt(1.0e7 * 2.0e7)
    temperature = torch.tensor(
        [1.0e7, midpoint_temperature, 2.0e7, 3.0e7], dtype=torch.float64
    )
    pressure = torch.full_like(temperature, 3.0e-2)

    _, _, log_particles, log_electrons = eos._log_mappings(temperature, pressure)
    chianti_log_electrons = eos._chianti_log_electrons(torch.log10(temperature))
    ideal_log_electrons = math.log10(eos.fully_ionized_electrons_per_h_nucleus)
    expected_midpoint = 0.5 * (chianti_log_electrons[1] + ideal_log_electrons)

    assert eos.ideal_transition_temperature_bounds_k == (1.0e7, 2.0e7)
    torch.testing.assert_close(
        log_electrons[0], chianti_log_electrons[0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(log_electrons[1], expected_midpoint)
    torch.testing.assert_close(
        log_electrons[2:],
        torch.full_like(log_electrons[2:], ideal_log_electrons),
        rtol=0.0,
        atol=0.0,
    )
    expected_log_particles = math.log10(
        eos.nuclei_per_h_nucleus + eos.fully_ionized_electrons_per_h_nucleus
    )
    torch.testing.assert_close(
        log_particles[2:],
        torch.full_like(log_particles[2:], expected_log_particles),
        rtol=0.0,
        atol=0.0,
    )


def test_chianti_mapping_is_shape_preserving_and_exact_at_native_nodes():
    eos = HybridSolarEOS()

    interpolated = eos._chianti_log_electrons(eos.chianti_log_temperature)

    torch.testing.assert_close(
        interpolated, eos.chianti_log_electrons_per_h, rtol=0.0, atol=1.0e-14
    )
    dense = torch.linspace(4.0, 9.0, 2001, dtype=torch.float64)
    values = eos._chianti_log_electrons(dense)
    assert torch.all(values[1:] > values[:-1])


def test_chianti_analytic_interpolation_slope_matches_autograd():
    eos = HybridSolarEOS()
    coordinate = torch.tensor(
        [4.13, 4.5, 5.77, 6.45, 8.92], dtype=torch.float64, requires_grad=True
    )

    values, analytic_slope = eos._chianti_log_electrons_and_slope(coordinate)
    automatic_slope = torch.autograd.grad(values.sum(), coordinate)[0]

    torch.testing.assert_close(analytic_slope, automatic_slope)


def _log_value_and_slope(eos, log_temperature, pressure, quantity):
    coordinate = torch.tensor(log_temperature, dtype=pressure.dtype, requires_grad=True)
    value = torch.log10(quantity(10.0**coordinate, pressure))
    slope = torch.autograd.grad(value, coordinate)[0]
    return float(value.detach()), float(slope.detach())


@pytest.mark.parametrize(
    ("dtype", "epsilon", "value_tolerance", "slope_tolerance"),
    (
        (torch.float32, 2.0e-6, 2.0e-5, 1.0e-2),
        (torch.float64, 1.0e-7, 1.0e-5, 2.0e-3),
    ),
)
def test_temperature_joins_are_value_and_first_derivative_continuous(
    dtype, epsilon, value_tolerance, slope_tolerance
):
    eos = HybridSolarEOS()
    seams = (
        *eos.transition_temperature_bounds_k,
        *eos.ideal_transition_temperature_bounds_k,
    )

    for pressure in (0.09, 100.0, 1.0e6):
        pressure_tensor = torch.tensor(pressure, dtype=dtype)
        for seam in seams:
            coordinate = math.log10(seam)
            for quantity in (eos.mass_density, eos.electron_density):
                left_value, left_slope = _log_value_and_slope(
                    eos, coordinate - epsilon, pressure_tensor, quantity
                )
                right_value, right_slope = _log_value_and_slope(
                    eos, coordinate + epsilon, pressure_tensor, quantity
                )
                assert math.isclose(left_value, right_value, abs_tol=value_tolerance)
                assert math.isclose(left_slope, right_slope, abs_tol=slope_tolerance)


@pytest.mark.parametrize(
    ("dtype", "epsilon", "value_tolerance", "slope_tolerance"),
    (
        (torch.float32, 2.0e-6, 2.0e-5, 1.0e-2),
        (torch.float64, 1.0e-7, 1.0e-5, 2.0e-3),
    ),
)
def test_stic_pressure_edges_and_shoulders_are_value_and_derivative_continuous(
    dtype, epsilon, value_tolerance, slope_tolerance
):
    eos = HybridSolarEOS()
    pressure_seams = (
        float(eos.stic_log_pressure[0])
        - eos.pressure_tangent_decay_width_log10_pressure,
        float(eos.stic_log_pressure[0]),
        float(eos.stic_log_pressure[-1]),
        float(eos.stic_log_pressure[-1])
        + eos.pressure_tangent_decay_width_log10_pressure,
    )

    for temperature in (5000.0, 1.0e4, 2.0e4):
        temperature_tensor = torch.tensor(temperature, dtype=dtype)
        for coordinate in pressure_seams:
            for quantity in (eos.mass_density, eos.electron_density):
                pressure_coordinate = torch.tensor(
                    coordinate - epsilon, dtype=dtype, requires_grad=True
                )
                left = torch.log10(
                    quantity(temperature_tensor, 10.0**pressure_coordinate)
                )
                left_slope = torch.autograd.grad(left, pressure_coordinate)[0]
                pressure_coordinate = torch.tensor(
                    coordinate + epsilon, dtype=dtype, requires_grad=True
                )
                right = torch.log10(
                    quantity(temperature_tensor, 10.0**pressure_coordinate)
                )
                right_slope = torch.autograd.grad(right, pressure_coordinate)[0]

                assert math.isclose(
                    float(left.detach()),
                    float(right.detach()),
                    abs_tol=value_tolerance,
                )
                assert math.isclose(
                    float(left_slope.detach()),
                    float(right_slope.detach()),
                    abs_tol=slope_tolerance,
                )


def test_shared_particle_mapping_closes_pressure_density_and_electron_density():
    eos = HybridSolarEOS()
    temperature = torch.logspace(3.4, 7.0, 73, dtype=torch.float64)[:, None]
    pressure = torch.logspace(-3.0, 7.0, 21, dtype=torch.float64)[None, :]

    density = eos.mass_density(temperature, pressure)
    electron_density = eos.electron_density(temperature, pressure)
    mean_molecular_weight = eos.mean_molecular_weight(temperature, pressure)
    _, _, log_particles, log_electrons = eos._log_mappings(temperature, pressure)
    particles_per_h = 10.0**log_particles
    electrons_per_h = 10.0**log_electrons
    recovered_electrons_per_h = (
        electron_density * eos.mass_u_per_h_nucleus * ATOMIC_MASS_UNIT / density
    )
    recovered_pressure = (
        density * K_BOLTZMANN * temperature / (mean_molecular_weight * ATOMIC_MASS_UNIT)
    )

    torch.testing.assert_close(recovered_pressure, pressure.expand_as(density))
    torch.testing.assert_close(recovered_electrons_per_h, electrons_per_h)
    assert torch.all(density > 0.0)
    assert torch.all(electron_density > 0.0)
    assert torch.all(electrons_per_h > 0.0)
    assert torch.all(electrons_per_h <= eos.fully_ionized_electrons_per_h_nucleus)
    electron_pressure_fraction = electron_density * K_BOLTZMANN * temperature / pressure
    torch.testing.assert_close(
        electron_pressure_fraction,
        electrons_per_h / particles_per_h,
    )

    chianti_regime = temperature >= eos.transition_temperature_bounds_k[1]
    non_electron_particles = particles_per_h - electrons_per_h
    torch.testing.assert_close(
        non_electron_particles.expand_as(density)[chianti_regime.expand_as(density)],
        torch.full_like(
            non_electron_particles.expand_as(density)[
                chianti_regime.expand_as(density)
            ],
            eos.nuclei_per_h_nucleus,
        ),
        rtol=5.0e-14,
        atol=5.0e-14,
    )


def test_stic_pressure_continuation_is_bounded_and_asymptotically_linear():
    eos = HybridSolarEOS()
    temperature = torch.logspace(3.4, 4.0, 31, dtype=torch.float64)[:, None]
    pressure = torch.tensor([1.0e-12, 1.0e-8, 1.0e8, 1.0e12], dtype=torch.float64)[
        None, :
    ]

    density = eos.mass_density(temperature, pressure)
    electron_density = eos.electron_density(temperature, pressure)
    _, _, log_particles, log_electrons = eos._log_mappings(temperature, pressure)
    particles_per_h = 10.0**log_particles
    electrons_per_h = 10.0**log_electrons

    assert torch.isfinite(density).all()
    assert torch.isfinite(electron_density).all()
    assert torch.all(particles_per_h - electrons_per_h > 0.0)
    assert torch.all(electrons_per_h > 0.0)
    assert torch.all(electrons_per_h < eos.fully_ionized_electrons_per_h_nucleus)
    for quantity in (density, electron_density):
        expected_ratio = torch.full_like(quantity[:, 0], 1.0e4)
        torch.testing.assert_close(quantity[:, 1] / quantity[:, 0], expected_ratio)
        torch.testing.assert_close(quantity[:, 3] / quantity[:, 2], expected_ratio)


def test_lower_bridge_is_charge_bounded_monotone_and_composition_safe():
    eos = HybridSolarEOS()
    log_temperature = torch.linspace(4.0, 4.5, 1001, dtype=torch.float64)[:, None]
    pressure = torch.logspace(-3.0, 7.0, 101, dtype=torch.float64)[None, :]
    _, _, log_particles, log_electrons = eos._log_mappings(
        10.0**log_temperature, pressure
    )
    particles_per_h = 10.0**log_particles
    electrons_per_h = 10.0**log_electrons
    non_electron_particles = particles_per_h - electrons_per_h

    assert torch.all(electrons_per_h[1:] >= electrons_per_h[:-1])
    assert float(electrons_per_h.max()) <= eos.fully_ionized_electrons_per_h_nucleus
    assert torch.all(non_electron_particles > 0.0)
    assert float(non_electron_particles.min()) > 0.95 * eos.nuclei_per_h_nucleus
    assert float(non_electron_particles.max()) < 1.05 * eos.nuclei_per_h_nucleus
    torch.testing.assert_close(
        non_electron_particles[-1],
        torch.full_like(non_electron_particles[-1], eos.nuclei_per_h_nucleus),
        rtol=5.0e-14,
        atol=5.0e-14,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hybrid_eos_has_finite_second_temperature_derivatives(dtype):
    eos = HybridSolarEOS()
    coordinate = torch.tensor(
        [4.0, 4.01, 4.24, 4.49, 4.5, 5.31, 6.6, 7.0, 7.15, 7.3, 7.5],
        dtype=dtype,
        requires_grad=True,
    )
    pressure = torch.logspace(-2.0, 6.0, coordinate.numel(), dtype=dtype)
    value = torch.log10(eos.mass_density(10.0**coordinate, pressure))
    first = torch.autograd.grad(value.sum(), coordinate, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), coordinate)[0]

    assert torch.isfinite(value).all()
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hybrid_eos_has_finite_second_pressure_derivatives(dtype):
    eos = HybridSolarEOS()
    coordinate = torch.tensor(
        [-3.0, -1.75, -1.5, -1.49, 5.99, 6.0, 6.25, 8.0],
        dtype=dtype,
        requires_grad=True,
    )
    temperature = torch.logspace(3.4, 4.4, coordinate.numel(), dtype=dtype)
    value = torch.log10(eos.mass_density(temperature, 10.0**coordinate))
    first = torch.autograd.grad(value.sum(), coordinate, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), coordinate)[0]

    assert torch.isfinite(value).all()
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_hybrid_eos_rejects_nonfinite_or_nonpositive_states():
    eos = HybridSolarEOS()

    for temperature, pressure in (
        (float("nan"), 1.0),
        (1.0e4, float("inf")),
        (0.0, 1.0),
        (1.0e4, -1.0),
    ):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            eos.mass_density(torch.tensor(temperature), torch.tensor(pressure))


@pytest.mark.parametrize("pressure", (1.0e-4, 1.0, 1.0e7))
def test_cold_eos_freezes_composition_and_retains_ideal_gas_scaling(pressure):
    eos = HybridSolarEOS()
    minimum_temperature = eos.minimum_temperature_k
    temperature = torch.tensor(
        [0.25 * minimum_temperature, 0.5 * minimum_temperature],
        dtype=torch.float64,
        requires_grad=True,
    )
    gas_pressure = torch.full_like(temperature, pressure, requires_grad=True)
    edge_temperature = torch.full_like(temperature, minimum_temperature)

    _, _, log_particles, log_electrons = eos._log_mappings(
        temperature,
        gas_pressure,
    )
    _, _, edge_log_particles, edge_log_electrons = eos._log_mappings(
        edge_temperature,
        gas_pressure,
    )
    torch.testing.assert_close(log_particles, edge_log_particles)
    torch.testing.assert_close(log_electrons, edge_log_electrons)

    density = eos.mass_density(temperature, gas_pressure)
    electron_density = eos.electron_density(temperature, gas_pressure)
    torch.testing.assert_close(density[0] / density[1], density.new_tensor(2.0))
    torch.testing.assert_close(
        electron_density[0] / electron_density[1],
        electron_density.new_tensor(2.0),
    )
    torch.autograd.backward((density.log().sum(), electron_density.log().sum()))
    assert torch.all(temperature.grad < 0.0)
    assert torch.isfinite(temperature.grad).all()
    assert torch.isfinite(gas_pressure.grad).all()

    metadata = eos.metadata()
    assert "freeze STiC composition" in metadata["cold_temperature_policy"]


def test_coronal_eos_is_linear_in_pressure_and_uses_resource_owned_buffers():
    eos = HybridSolarEOS()
    temperature = torch.tensor([5.0e4, 1.0e6, 4.0e6, 1.0e7], dtype=torch.float64)
    low_pressure = torch.full_like(temperature, 1.0e-20)
    high_pressure = torch.full_like(temperature, 1.0e20)
    ratio = high_pressure / low_pressure

    torch.testing.assert_close(
        eos.mass_density(temperature, high_pressure)
        / eos.mass_density(temperature, low_pressure),
        ratio,
    )
    torch.testing.assert_close(
        eos.electron_density(temperature, high_pressure)
        / eos.electron_density(temperature, low_pressure),
        ratio,
    )
    assert eos.state_dict() == {}
