import math
from pathlib import Path

import pytest
import torch

from pme.lte.atomic import AtomicDatabase
from pme.lte.eos import (
    ATOMIC_MASS_UNIT,
    ELECTRON_VOLT,
    H_PLANCK,
    K_BOLTZMANN,
    M_ELECTRON,
    LTEEOS,
)
from pme.lte.opacity import (
    C_LIGHT,
    ContinuumOpacity,
    damping_rate,
    doppler_velocity,
    doppler_width_frequency,
    integrated_line_opacity,
    planck_lambda,
)
from pme.lte.polarization import ZEEMAN_HZ_PER_GAUSS
from pme.lte.profiles import VoigtFaraday
from pme.lte.synthesis import LTESynthesizer
from pme.lte.wavelength import air_to_vacuum_angstrom


@pytest.fixture(scope="module")
def atomic():
    return AtomicDatabase()


@pytest.fixture(scope="module")
def eos(atomic):
    return LTEEOS(atomic)


def test_raw_barklem_collet_table_and_semilog_interpolation(atomic):
    table = atomic.partition_table
    assert len(table.values) == 284
    assert len(table.temperatures) == 42
    assert table.values["Fe_I"][table.temperatures.index(5000.0)] == 27.794

    temperature = torch.tensor(5500.0, dtype=torch.float64, requires_grad=True)
    partition = atomic.partition_function("Fe_I", temperature)
    expected = math.sqrt(27.794 * 31.7409)
    torch.testing.assert_close(partition, torch.tensor(expected, dtype=torch.float64))
    partition.backward()
    assert temperature.grad is not None
    assert torch.isfinite(temperature.grad)
    assert temperature.grad > 0

    with pytest.raises(KeyError, match="No Barklem--Collet"):
        atomic.partition_function("Unobtainium_I", temperature.detach())

    ionization_path = (
        Path(atomic.partition_table.source).parent
        / "barklem_collet_2016_table4.dat"
    )
    raw_ionization = {}
    with open(ionization_path, encoding="ascii") as handle:
        for line in handle:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            _, symbol, first, second, _ = line.split()
            raw_ionization[symbol] = (float(first), float(second))
    for symbol, element in atomic.elements.items():
        first, second = raw_ionization[symbol]
        assert element.ionization_ev == pytest.approx(first, abs=0.0)
        if symbol == "H":
            assert second == -1.0 and element.second_ionization_ev is None
        else:
            assert element.second_ionization_ev == pytest.approx(second, abs=0.0)


def test_line_metadata_has_explicit_units_and_provenance(atomic):
    line = atomic.get_line("FeI_6302.4932")
    assert line.element == "Fe"
    assert line.ion_stage == 1
    assert line.wavelength_air_angstrom == 6302.4932
    assert line.lower_statistical_weight == 3.0
    assert line.oscillator_strength == pytest.approx(10.0**line.log_gf / 3.0)
    assert set(line.field_provenance["abo_sigma_a0_squared"]) == {
        "SPIN4D_SIR_TABLE3",
        "ABO_CONVENTION",
    }
    assert line.lande_lower == 2.487
    assert line.log_gamma_rad_s == pytest.approx(7.7477331199049475)
    assert line.log_gamma_stark_s_cm3 is None
    assert set(line.field_provenance["log_gamma_rad_s"]) == {
        "SIR2015_CLASSICAL_RADIATIVE"
    }


def test_complete_eos_lookup_contains_runtime_populations_and_is_differentiable(eos):
    assert eos.hydrogen_fraction_names == ("H_I", "H_II", "H_minus")
    assert eos.ion_species == ("Fe_I", "Fe_II", "Fe_III")
    assert eos.lower_level_ids == ("FeI_6301.5008", "FeI_6302.4932")
    temperature = torch.tensor(5772.0, dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor(1.0e3, dtype=torch.float64, requires_grad=True)
    state = eos(temperature, pressure)
    objective = (
        torch.log(state.electron_density)
        + torch.log(state.hydrogen_minus)
        + torch.log(state.lower_level_populations["FeI_6301.5008"])
    )
    gradients = torch.autograd.grad(objective, (temperature, pressure))
    assert all(torch.isfinite(gradient) and gradient != 0 for gradient in gradients)
    assert eos.table_log_pressure[0].item() == -1.5


def test_saha_ratio_matches_direct_reference_calculation(atomic, eos):
    temperature = torch.tensor(5772.0, dtype=torch.float64)
    electron_density = torch.tensor(1.0e19, dtype=torch.float64)
    u_i = atomic.partition_function("Fe_I", temperature)
    u_ii = atomic.partition_function("Fe_II", temperature)
    chi = atomic.element("Fe").ionization_ev * ELECTRON_VOLT
    reference = (
        2.0
        * (2.0 * math.pi * M_ELECTRON * K_BOLTZMANN * temperature / H_PLANCK**2)
        ** 1.5
        * u_ii
        / u_i
        * torch.exp(-chi / (K_BOLTZMANN * temperature))
        / electron_density
    )
    torch.testing.assert_close(
        eos.saha_ratio("Fe", temperature, electron_density),
        reference,
        rtol=2e-13,
        atol=0.0,
    )

    u_iii = atomic.partition_function("Fe_III", temperature)
    chi_second = atomic.element("Fe").second_ionization_ev * ELECTRON_VOLT
    reference_second = (
        2.0
        * (2.0 * math.pi * M_ELECTRON * K_BOLTZMANN * temperature / H_PLANCK**2)
        ** 1.5
        * u_iii
        / u_ii
        * torch.exp(-chi_second / (K_BOLTZMANN * temperature))
        / electron_density
    )
    torch.testing.assert_close(
        eos.saha_ratio("Fe", temperature, electron_density, ionization_stage=2),
        reference_second,
        rtol=2e-13,
        atol=0.0,
    )


def test_eos_conserves_particles_hydrogen_and_charge(atomic, eos):
    temperature = torch.tensor([4200.0, 5772.0, 8000.0], dtype=torch.float64)
    gas_pressure = torch.tensor([30.0, 1.0e3, 1.0e5], dtype=torch.float64)
    state = eos(temperature, gas_pressure)

    nuclei_per_hydrogen = sum(atomic.abundance_ratio(symbol) for symbol in atomic.elements)
    reconstructed_pressure = (
        state.hydrogen_total * nuclei_per_hydrogen + state.electron_density
    ) * K_BOLTZMANN * temperature
    torch.testing.assert_close(reconstructed_pressure, gas_pressure, rtol=2e-12, atol=0.0)
    torch.testing.assert_close(
        state.hydrogen_neutral + state.hydrogen_ionized + state.hydrogen_minus,
        state.hydrogen_total,
        rtol=2e-12,
        atol=0.0,
    )

    # Reconstruct all electron-donor stages independently. Runtime tables only
    # retain ion fractions for elements used by configured lines (Fe here),
    # while the tabulated ne includes every abundance species.
    positive_charge = state.hydrogen_ionized
    for symbol in atomic.elements:
        if symbol == "H":
            continue
        first = eos.saha_ratio(symbol, temperature, state.electron_density)
        second = eos.saha_ratio(
            symbol, temperature, state.electron_density, ionization_stage=2
        )
        denominator = 1.0 + first + first * second
        total = state.hydrogen_total * atomic.abundance_ratio(symbol)
        positive_charge = positive_charge + total * (
            first / denominator + 2.0 * first * second / denominator
        )
    torch.testing.assert_close(
        positive_charge - state.hydrogen_minus,
        state.electron_density,
        # The prepared cubic log(ne) table approximates the offline exact
        # charge-neutrality root to a few parts per million between nodes.
        rtol=5e-6,
        atol=0.0,
    )
    assert eos.table_metadata["runtime_iterations"] == 0
    assert torch.all(state.electron_density > 0)
    assert torch.all(state.mass_density > 0)
    for symbol in ("Fe",):
        total = sum(state.populations[f"{symbol}_{suffix}"] for suffix in ("I", "II", "III"))
        expected = state.hydrogen_total * atomic.abundance_ratio(symbol)
        torch.testing.assert_close(total, expected, rtol=2e-12, atol=0.0)


def test_boltzmann_population_and_hminus_equilibrium(atomic, eos):
    line = atomic.get_line("FeI_6301.5008")
    temperature = torch.tensor([5000.0, 6500.0], dtype=torch.float64)
    gas_pressure = torch.tensor([300.0, 3000.0], dtype=torch.float64)
    state = eos(temperature, gas_pressure)
    lower = eos.lower_level_population(line, temperature, gas_pressure, eos_state=state)
    reference = (
        state.populations["Fe_I"]
        * line.lower_statistical_weight
        / atomic.partition_function("Fe_I", temperature)
        * torch.exp(
            -(line.lower_excitation_ev * ELECTRON_VOLT / K_BOLTZMANN)
            / temperature
        )
    )
    # The complete lookup stores the line-level fraction directly; its
    # interpolation error remains below one part in 10^4 on off-grid states.
    torch.testing.assert_close(lower, reference, rtol=1e-4, atol=0.0)
    torch.testing.assert_close(
        state.hydrogen_minus,
        eos.h_minus_population(
            temperature, state.electron_density, state.hydrogen_neutral
        ),
        rtol=5e-6,
        atol=0.0,
    )
def test_stic_total_continuum_and_hminus_diagnostic_references(atomic, eos):
    opacity = ContinuumOpacity(atomic)
    temperature = torch.tensor(5772.0, dtype=torch.float64, requires_grad=True)
    gas_pressure = torch.tensor(1.0e3, dtype=torch.float64, requires_grad=True)
    state = eos(temperature, gas_pressure)
    wavelength = torch.tensor([5000.0, 6300.0], dtype=torch.float64)
    bound_free = opacity.bound_free_absorption(wavelength, temperature, state)
    free_free = opacity.free_free_absorption(wavelength, temperature, state)
    hminus = opacity.hminus_absorption(wavelength, temperature, state)
    true_absorption = opacity.true_absorption(
        wavelength, temperature, gas_pressure
    )
    scattering = opacity.scattering_extinction(
        wavelength, temperature, gas_pressure
    )
    total = opacity(wavelength, temperature, gas_pressure)
    torch.testing.assert_close(hminus, bound_free + free_free)
    torch.testing.assert_close(total, true_absorption + scattering)
    assert torch.all(bound_free > 0)
    assert torch.all(free_free > 0)
    # Direct results from the pinned STiC Wittmann/cop implementation at the
    # off-grid state T=5772 K, Pgas=1000 Pa.  The tolerance includes only the
    # documented offline cubic interpolation error.
    torch.testing.assert_close(
        true_absorption.detach(),
        torch.tensor([1.83567669e-7, 2.23369566e-7], dtype=torch.float64),
        rtol=1.0e-5,
        atol=0.0,
    )
    torch.testing.assert_close(
        scattering.detach(),
        torch.tensor([1.49828433e-9, 7.68653651e-10], dtype=torch.float64),
        rtol=1.0e-5,
        atol=0.0,
    )

    stimulated = 1.0 - math.exp(
        -H_PLANCK * C_LIGHT / (K_BOLTZMANN * 5772.0 * 5000.0e-10)
    )
    reference_5000 = state.hydrogen_minus.detach() * stimulated * 2.84e-21
    torch.testing.assert_close(bound_free[0].detach(), reference_5000, rtol=2e-13, atol=0.0)
    total.log().sum().backward()
    assert torch.isfinite(temperature.grad)
    assert torch.isfinite(gas_pressure.grad)

    outside_photodetachment_edge = opacity.bound_free_absorption(
        torch.tensor([20_000.0], dtype=torch.float64), temperature.detach(), state
    )
    assert outside_photodetachment_edge.item() == 0.0


def test_stic_falc_contract_and_production_populations_are_coherent(atomic):
    opacity = ContinuumOpacity(atomic)
    assert opacity.reference_solver["commit"] == (
        "18cda77d038a97f007a783dcb61ea9a9a1244bf7"
    )
    assert opacity.reference_top_log_tau500 == -5.0
    assert opacity.reference_top_pressure_pa == pytest.approx(
        0.0903679135719897, rel=0.0, abs=1.0e-15
    )
    assert opacity.reference_top_boundary["temperature_k"] == pytest.approx(
        7456.57046516284
    )
    boundary_state = opacity.reference_top_boundary[
        "stic_state_at_interpolated_boundary"
    ]
    assert boundary_state["total_extinction_m1"] == pytest.approx(
        2.8784599239306297e-11
    )
    assert boundary_state["mass_density_kg_m3"] == pytest.approx(
        1.1239128335585617e-9
    )
    assert boundary_state["required_metric_m_per_log10_tau"] == pytest.approx(
        799936.4777848955
    )
    assert boundary_state["pressure_efold_width_log10_tau"] == pytest.approx(
        0.36494391239092006
    )
    # Dimensional closure at the cgs-generator/SI-model boundary.  These
    # identities detect the common factor-10/100/1000 conversion mistakes
    # independently of the descriptive field names.
    alpha500_m1 = boundary_state["total_extinction_m1"]
    density_kg_m3 = boundary_state["mass_density_kg_m3"]
    metric_m_per_dex = boundary_state["required_metric_m_per_log10_tau"]
    pressure_pa = opacity.reference_top_pressure_pa
    gravity_m_s2 = opacity.reference_gravity_m_per_s2
    assert boundary_state["mass_extinction_m2_kg"] == pytest.approx(
        alpha500_m1 / density_kg_m3
    )
    assert alpha500_m1 * metric_m_per_dex == pytest.approx(
        math.log(10.0) * 10.0**opacity.reference_top_log_tau500
    )
    assert boundary_state["dlog_pressure_dlog10_tau"] == pytest.approx(
        density_kg_m3 * gravity_m_s2 * metric_m_per_dex / pressure_pa
    )
    assert boundary_state["pressure_efold_width_log10_tau"] == pytest.approx(
        1.0 / boundary_state["dlog_pressure_dlog10_tau"]
    )
    assert opacity.metadata()["lookup_units"] == {
        "temperature": "K",
        "gas_pressure": "Pa",
        "wavelength": "vacuum Angstrom",
        "true_absorption": "m^-1",
        "scattering_extinction": "m^-1",
        "mass_density": "kg m^-3",
        "electron_density": "m^-3",
        "neutral_hydrogen_density": "m^-3",
        "fe_i_population_over_partition": "m^-3",
    }
    assert opacity.continuum_contract["physical_height_qualified"] is True
    assert "does not iterate a coherent-scattering source" in (
        opacity.continuum_contract["source_function_limitation"]
    )
    opacity.validate_physical_height_contract(-5.0, 0.0903679135719897)

    temperature = torch.tensor(5772.0, dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
    thermodynamics = opacity.reference_thermodynamics(temperature, pressure)
    neutral_hydrogen = opacity.reference_neutral_hydrogen_density(
        temperature, pressure
    )
    fe_i_over_partition = opacity.reference_fe_i_population_over_partition(
        temperature, pressure
    )
    populations = opacity.reference_lower_level_populations(
        tuple(atomic.lines), temperature, pressure
    )
    assert thermodynamics["mass_density"] > 0
    assert thermodynamics["electron_density"] > 0
    torch.testing.assert_close(
        neutral_hydrogen.detach(),
        torch.tensor(1.14124e22, dtype=torch.float64),
        rtol=2.0e-5,
        atol=0.0,
    )
    torch.testing.assert_close(
        fe_i_over_partition.detach(),
        torch.tensor(1.19250e14, dtype=torch.float64),
        rtol=2.0e-5,
        atol=0.0,
    )
    expected_populations = {
        "FeI_6301.5008": 3.84798e11,
        "FeI_6302.4932": 2.16190e11,
    }
    for line_id, expected in expected_populations.items():
        torch.testing.assert_close(
            populations[line_id].detach(),
            torch.tensor(expected, dtype=torch.float64),
            rtol=2.0e-5,
            atol=0.0,
        )
    total_population = sum(populations.values())
    total_population.log().backward()
    assert torch.isfinite(temperature.grad)
    assert torch.isfinite(pressure.grad)


def test_planck_line_opacity_doppler_and_damping_are_physical(atomic):
    line = atomic.get_line("FeI_6302.4932")
    temperature = torch.tensor(6000.0, dtype=torch.float64, requires_grad=True)
    lower_population = torch.tensor(2.0e15, dtype=torch.float64)
    wavelength = torch.tensor([6302.5], dtype=torch.float64)

    source = planck_lambda(temperature, wavelength)
    integrated = integrated_line_opacity(line, lower_population, temperature)
    microturbulence = torch.tensor(1000.0, dtype=torch.float64)
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
    assert width > 0
    assert source.item() > 0
    assert integrated.item() > 0

    zero_density_rate = damping_rate(
        line,
        temperature,
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    dense_rate = damping_rate(
        line,
        temperature,
        torch.tensor(1.0e19, dtype=torch.float64),
        torch.tensor(1.0e22, dtype=torch.float64),
    )
    assert zero_density_rate.item() == pytest.approx(10.0 ** line.log_gamma_rad_s)
    assert dense_rate > zero_density_rate
    (source.sum().log() + integrated.log() + width.log() + dense_rate.log()).backward()
    assert torch.isfinite(temperature.grad)

    temperature32 = torch.tensor(6000.0, dtype=torch.float32, requires_grad=True)
    wavelength32 = torch.tensor([6302.5], dtype=torch.float32, requires_grad=True)
    planck_lambda(temperature32, wavelength32).log().sum().backward()
    assert torch.isfinite(temperature32.grad)
    assert wavelength32.grad is not None
    assert torch.isfinite(wavelength32.grad).all()

    normalized_temperature = torch.tensor(
        6000.0, dtype=torch.float32, requires_grad=True
    )
    radiance_scale = torch.tensor(2.4e13, dtype=torch.float32)
    physical = planck_lambda(normalized_temperature.detach(), wavelength32.detach())
    normalized = planck_lambda(
        normalized_temperature,
        wavelength32.detach(),
        radiance_scale=radiance_scale,
    )
    torch.testing.assert_close(normalized, physical / radiance_scale)
    normalized.sum().backward()
    assert torch.isfinite(normalized_temperature.grad)

    for invalid_scale in (torch.tensor(float("nan")), torch.tensor(-1.0)):
        with pytest.raises(ValueError, match="finite positive scalar"):
            planck_lambda(
                torch.tensor(6000.0),
                torch.tensor([6302.5]),
                radiance_scale=invalid_scale,
            )


def test_atomic_quantities_close_in_model_units(atomic):
    """Close the atomic chain from reviewed metadata to dimensionless K_lambda."""

    line = atomic.get_line("FeI_6302.4932")
    temperature = torch.tensor(6000.0, dtype=torch.float64)
    pressure = torch.tensor(1000.0, dtype=torch.float64)
    microturbulence = torch.tensor(1000.0, dtype=torch.float64)
    opacity = ContinuumOpacity(atomic)
    lower_population = opacity.reference_lower_level_populations(
        (line,), temperature, pressure
    )[line.id]
    integrated_m1_hz = integrated_line_opacity(
        line, lower_population, temperature
    )
    doppler_width_hz = doppler_width_frequency(
        line,
        temperature,
        microturbulence,
        atomic.element("Fe").atomic_mass_u,
    )
    profile_hz1, _ = VoigtFaraday()(
        torch.zeros((), dtype=torch.float64),
        torch.tensor(0.1, dtype=torch.float64),
        doppler_width=doppler_width_hz,
    )
    line_extinction_m1 = integrated_m1_hz * profile_hz1
    alpha500_m1 = opacity.volume_extinction_at_5000(temperature, pressure)
    propagation_ratio = line_extinction_m1 / alpha500_m1

    assert lower_population > 0
    assert integrated_m1_hz > 0
    assert doppler_width_hz > 0
    assert profile_hz1 > 0
    assert line_extinction_m1 > 0
    assert torch.isfinite(propagation_ratio)

    # The frequency and traditional wavelength Zeeman constants must describe
    # the same shift when B is supplied in gauss and wavelength in Angstrom.
    wavelength_vacuum_angstrom = air_to_vacuum_angstrom(
        temperature.new_tensor(line.wavelength_air_angstrom)
    )
    wavelength_constant_angstrom_per_gauss = (
        wavelength_vacuum_angstrom.square()
        * 1.0e-10
        * ZEEMAN_HZ_PER_GAUSS
        / C_LIGHT
    )
    assert wavelength_constant_angstrom_per_gauss.item() == pytest.approx(
        4.668645e-13 * wavelength_vacuum_angstrom.item() ** 2,
        rel=2.0e-7,
    )

    units = LTESynthesizer(atomic_database=atomic).metadata()["atomic_model_units"]
    assert units["lower_level_population"] == "m^-3"
    assert units["integrated_line_extinction"] == "m^-1 Hz"
    assert units["frequency_profile"] == "Hz^-1"
    assert units["propagation_matrix"] == (
        "constructed dimensionless per unit vertical tau500; multiplied by "
        "alpha500 to m^-1 when geometric heights are present"
    )


def test_air_wavelengths_are_converted_before_vacuum_physics(atomic):
    dtype = torch.float64
    wavelength_air = torch.tensor(
        [5000.0, 6301.5008, 6302.4932], dtype=dtype, requires_grad=True
    )
    wavelength_vacuum = air_to_vacuum_angstrom(wavelength_air)
    expected = torch.tensor(
        [5001.39484863807, 6303.2435744457125, 6304.236240909655],
        dtype=dtype,
    )
    torch.testing.assert_close(wavelength_vacuum, expected, rtol=0.0, atol=2e-12)
    assert torch.all(wavelength_vacuum > wavelength_air)

    temperature = torch.tensor(6000.0, dtype=dtype)
    line = atomic.get_line("FeI_6302.4932")
    width = doppler_width_frequency(
        line,
        temperature,
        torch.tensor(1000.0, dtype=dtype),
        atomic.element("Fe").atomic_mass_u,
    )
    rest_frequency = C_LIGHT / (wavelength_vacuum[-1] * 1.0e-10)
    thermal_speed = torch.sqrt(
        2.0 * K_BOLTZMANN * temperature
        / (atomic.element("Fe").atomic_mass_u * ATOMIC_MASS_UNIT)
        + temperature.new_tensor(1000.0).square()
    )
    torch.testing.assert_close(width, rest_frequency * thermal_speed / C_LIGHT)

    # B_lambda is per metre of vacuum wavelength, not per Angstrom and not at
    # the standard-air label.
    source = planck_lambda(temperature, wavelength_vacuum[-1:])
    wavelength_m = wavelength_vacuum[-1] * 1.0e-10
    expected_source = (
        2.0 * H_PLANCK * C_LIGHT**2 / wavelength_m**5
        / torch.expm1(H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZMANN * temperature))
    )
    torch.testing.assert_close(source[0], expected_source, rtol=2e-13, atol=0.0)
    source.sum().backward()
    assert torch.isfinite(wavelength_air.grad).all()
