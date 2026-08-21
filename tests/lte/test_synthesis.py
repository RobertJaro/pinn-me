from dataclasses import replace

import pytest
import torch

from pme.lte.atmosphere import StratifiedAtmosphere
from pme.lte.examples import synthetic_temperature_profile
from pme.lte.synthesis import LTESynthesizer
from pme.lte.wavelength import air_to_vacuum_angstrom


def _atmosphere(
    *,
    depth=15,
    batch=1,
    magnetic_vector=(0.0, 0.0, 0.0),
    velocity_los=0.0,
    dtype=torch.float64,
    requires_grad=False,
):
    log_tau500 = torch.linspace(-4.0, 1.0, depth, dtype=dtype)
    temperature = synthetic_temperature_profile(log_tau500).expand(batch, -1).clone()
    velocity_los_tensor = torch.full((batch, depth), velocity_los, dtype=dtype)
    velocity = torch.stack(
        (
            torch.zeros_like(velocity_los_tensor),
            torch.zeros_like(velocity_los_tensor),
            -velocity_los_tensor,
        ),
        dim=-1,
    )
    microturbulence = torch.full((batch, depth), 1000.0, dtype=dtype)
    magnetic = torch.tensor(magnetic_vector, dtype=dtype).expand(batch, depth, 3).clone()
    pressure = torch.logspace(-0.5, 5.0, depth, dtype=dtype).expand(batch, -1).clone()
    if requires_grad:
        temperature.requires_grad_()
        velocity.requires_grad_()
        microturbulence.requires_grad_()
        magnetic.requires_grad_()
        pressure.requires_grad_()
    return StratifiedAtmosphere(
        log_tau500=log_tau500,
        temperature=temperature,
        velocity_field=velocity,
        microturbulence=microturbulence,
        magnetic_field=magnetic,
        gas_pressure=pressure,
    )


def test_zero_field_synthesis_is_unpolarized_and_contains_both_lines():
    atmosphere = _atmosphere(depth=15)
    wavelength = torch.linspace(6300.8, 6303.2, 121, dtype=torch.float64)
    synthesizer = LTESynthesizer(log_tau500=atmosphere.log_tau500)
    stokes, diagnostics = synthesizer(
        atmosphere,
        wavelength,
        return_diagnostics=True,
    )

    assert stokes.shape == (1, 4, 121)
    assert torch.isfinite(stokes).all()
    assert torch.all(stokes[:, 0] > 0)
    torch.testing.assert_close(stokes[:, 1:], torch.zeros_like(stokes[:, 1:]), atol=1e-10, rtol=0)
    assert stokes[:, 0].amax() - stokes[:, 0].amin() > 1e-4
    assert set(diagnostics.propagation.line_absorption) >= {"FeI_6301.5008", "FeI_6302.4932"}
    assert synthesizer.metadata()["lines"] == [
        "FeI_6301.5008",
        "FeI_6302.4932",
    ]
    assert diagnostics.continuum_extinction.shape == (1, 15, 121)
    assert diagnostics.propagation.damping_electron_density is None
    torch.testing.assert_close(
        diagnostics.propagation.damping_hydrogen_neutral,
        diagnostics.reference_thermodynamics["hydrogen_neutral"],
    )
    assert set(diagnostics.propagation.lower_level_populations) == {
        "FeI_6301.5008",
        "FeI_6302.4932",
    }
    assert synthesizer.atomic._partition_table is None
    assert synthesizer.continuum_opacity._hminus_reference_loaded is False


def test_mu_changes_continuum_and_line_formation_without_local_renormalization():
    single = _atmosphere(depth=25)
    atmosphere = replace(
        single,
        temperature=single.temperature.expand(2, -1),
        velocity_field=single.velocity_field.expand(2, -1, -1),
        microturbulence=single.microturbulence.expand(2, -1),
        magnetic_field=single.magnetic_field.expand(2, -1, -1),
        gas_pressure=single.gas_pressure.expand(2, -1),
    )
    wavelength = torch.linspace(6301.1, 6302.9, 91, dtype=torch.float64)
    synthesizer = LTESynthesizer(log_tau500=atmosphere.log_tau500)
    stokes = synthesizer(
        atmosphere,
        wavelength,
        mu=torch.tensor([[1.0], [0.5]], dtype=torch.float64),
        radiance_scale=torch.tensor(3.0604866e13, dtype=torch.float64),
    )[:, 0]

    # The inclined ray samples higher, cooler layers. Its continuum remains in
    # the same disk-center atlas unit rather than being normalized back to one.
    continuum = stokes[:, (0, -1)].mean(dim=-1)
    assert continuum[1] < continuum[0]
    assert 0.0 < continuum[1] < 1.0
    # Dividing out each profile's continuum still leaves different line shapes,
    # demonstrating that mu enters formation depth rather than a final scalar.
    disk_center_shape = stokes[0] / continuum[0]
    limb_shape = stokes[1] / continuum[1]
    assert torch.max(torch.abs(disk_center_shape - limb_shape)) > 1.0e-3


def test_synthesis_rejects_nonfinite_or_out_of_range_mu():
    atmosphere = _atmosphere(depth=7)
    wavelength = torch.linspace(6301.3, 6302.7, 11, dtype=torch.float64)
    synthesizer = LTESynthesizer(log_tau500=atmosphere.log_tau500)
    for invalid_mu in (0.0, -0.1, 1.01, float("nan")):
        with pytest.raises(ValueError, match="0 < mu <= 1"):
            synthesizer(atmosphere, wavelength, mu=invalid_mu)


def test_synthesis_only_requests_line_diagnostics_when_requested(monkeypatch):
    atmosphere = _atmosphere(depth=7)
    wavelength = torch.linspace(6301.3, 6302.7, 21, dtype=torch.float64)
    synthesizer = LTESynthesizer(log_tau500=atmosphere.log_tau500)
    observed_flags = []
    original_forward = synthesizer.line_opacity.forward

    def record_flag(*args, **kwargs):
        observed_flags.append(kwargs.get("return_diagnostics"))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(synthesizer.line_opacity, "forward", record_flag)
    plain = synthesizer(atmosphere, wavelength, return_diagnostics=False)
    diagnosed, diagnostics = synthesizer(
        atmosphere, wavelength, return_diagnostics=True
    )

    assert observed_flags == [False, True]
    torch.testing.assert_close(plain, diagnosed)
    assert diagnostics.propagation.line_absorption


def test_radiance_normalization_is_applied_before_polarized_transfer(monkeypatch):
    atmosphere = _atmosphere(depth=7, dtype=torch.float32, requires_grad=True)
    wavelength = torch.linspace(6301.3, 6302.7, 21, dtype=torch.float32)
    synthesizer = LTESynthesizer(log_tau500=atmosphere.log_tau500)
    radiance_scale = torch.tensor(2.4e13, dtype=torch.float32)
    captured = {}
    original_forward = synthesizer.formal_solver.forward

    def inspect_source(propagation, source, *args, **kwargs):
        captured["source_abs_max"] = float(source.detach().abs().amax())
        return original_forward(propagation, source, *args, **kwargs)

    monkeypatch.setattr(synthesizer.formal_solver, "forward", inspect_source)
    normalized = synthesizer(
        atmosphere,
        wavelength,
        radiance_scale=radiance_scale,
    )
    normalized.square().mean().backward()

    assert captured["source_abs_max"] < 10.0
    assert torch.isfinite(normalized).all()
    for field in (
        atmosphere.temperature,
        atmosphere.velocity_field,
        atmosphere.microturbulence,
        atmosphere.magnetic_field,
        atmosphere.gas_pressure,
    ):
        assert field.grad is not None
        assert torch.isfinite(field.grad).all()


def test_synthesis_metadata_records_split_damping_eos_roles():
    roles = LTESynthesizer().metadata()["thermodynamic_roles"]

    assert "electron density for Stark broadening" in roles["stic_wittmann"]
    assert "physical neutral atomic-H density for ABO broadening" in roles["stic_wittmann"]
    assert "not used for production" in roles["barklem_saha"]


def test_reviewed_lines_skip_unneeded_stic_electron_lookup(monkeypatch):
    atmosphere = _atmosphere(depth=7)
    synthesizer = LTESynthesizer(log_tau500=atmosphere.log_tau500)
    monkeypatch.setattr(
        synthesizer.continuum_opacity,
        "reference_electron_density",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Stark-null lines requested electron density")
        ),
    )
    stokes = synthesizer(
        atmosphere,
        torch.linspace(6301.3, 6302.7, 21, dtype=torch.float64),
        return_diagnostics=False,
    )
    assert torch.isfinite(stokes).all()


def test_synthetic_stark_line_uses_stic_electron_density():
    atmosphere = _atmosphere(depth=7)
    base = LTESynthesizer()
    stark_line = replace(
        base.lines[0],
        log_gamma_stark_s_cm3=-8.0,
        stark_temperature_exponent=0.0,
    )
    synthesizer = LTESynthesizer(
        log_tau500=atmosphere.log_tau500,
        atomic_database=base.atomic,
        lines=(stark_line,),
        continuum_opacity=base.continuum_opacity,
    )
    _, diagnostics = synthesizer(
        atmosphere,
        torch.linspace(6301.3, 6301.7, 17, dtype=torch.float64),
        return_diagnostics=True,
    )
    expected = synthesizer.continuum_opacity.reference_electron_density(
        atmosphere.temperature, atmosphere.gas_pressure
    )
    torch.testing.assert_close(
        diagnostics.propagation.damping_electron_density, expected
    )


def test_stic_population_lookups_match_pinned_witt_reference_and_differentiate():
    synthesizer = LTESynthesizer()
    opacity = synthesizer.continuum_opacity
    temperature = torch.tensor(5772.0, dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor(1.0e3, dtype=torch.float64, requires_grad=True)
    hydrogen = opacity.reference_neutral_hydrogen_density(temperature, pressure)
    fe_over_u = opacity.reference_fe_i_population_over_partition(
        temperature, pressure
    )
    lower = opacity.reference_lower_level_populations(
        synthesizer.lines, temperature, pressure
    )

    torch.testing.assert_close(
        hydrogen,
        hydrogen.new_tensor(1.1412385943720008e22),
        rtol=2.0e-5,
        atol=0.0,
    )
    torch.testing.assert_close(
        lower["FeI_6301.5008"],
        hydrogen.new_tensor(3.8479798214992334e11),
        rtol=3.0e-5,
        atol=0.0,
    )
    assert fe_over_u > 0.0
    gradients = torch.autograd.grad(
        torch.log(hydrogen) + torch.log(fe_over_u) + torch.log(sum(lower.values())),
        (temperature, pressure),
    )
    assert all(torch.isfinite(gradient) and gradient != 0.0 for gradient in gradients)
    assert opacity.stic_table_schema_version == 2
    assert opacity.reference_solver["runtime_iterations"] == 0
    assert opacity.reference_solver["abundance_input_contract"][
        "iron_log10_n_over_n_h_plus_12"
    ] == 7.44


def test_faraday_profile_uses_red_positive_wavelength_coordinate():
    """Protect the ME/SIR dispersion convention that controls Stokes-U signs."""

    synthesizer = LTESynthesizer()
    line_opacity = synthesizer.line_opacity
    line = synthesizer.lines[0]
    pattern = line_opacity.patterns[0]
    dtype = torch.float64
    wavelength = torch.tensor(
        [line.wavelength_air_angstrom - 0.03, line.wavelength_air_angstrom + 0.03],
        dtype=dtype,
    )
    frequency = wavelength.new_tensor(299_792_458.0) / (
        air_to_vacuum_angstrom(wavelength) * 1.0e-10
    )
    central_frequency = (
        wavelength.new_tensor(299_792_458.0)
        / (
            air_to_vacuum_angstrom(
                wavelength.new_tensor(line.wavelength_air_angstrom)
            )
            * 1.0e-10
        )
    ).reshape(1)
    profiles = line_opacity._group_profiles(
        pattern,
        frequency,
        central_frequency,
        torch.tensor([3.0e9], dtype=dtype),
        torch.tensor([0.05], dtype=dtype),
        torch.tensor([0.0], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
    )
    pi_dispersion = profiles["pi"][1][0]
    assert pi_dispersion[0] < 0  # blue wavelength offset
    assert pi_dispersion[1] > 0  # red wavelength offset


def test_longitudinal_and_transverse_fields_generate_expected_polarization():
    wavelength = torch.linspace(6301.25, 6302.75, 101, dtype=torch.float64)
    longitudinal = _atmosphere(magnetic_vector=(0.0, 0.0, 1200.0))
    transverse = _atmosphere(magnetic_vector=(1200.0, 0.0, 0.0))
    synthesizer = LTESynthesizer(log_tau500=longitudinal.log_tau500)

    longitudinal_stokes = synthesizer(longitudinal, wavelength)
    transverse_stokes = synthesizer(transverse, wavelength)
    assert longitudinal_stokes[:, 3].abs().amax() > 1e-7
    assert transverse_stokes[:, 1:3].abs().amax() > 1e-7
    torch.testing.assert_close(
        longitudinal_stokes[:, 1:3],
        torch.zeros_like(longitudinal_stokes[:, 1:3]),
        atol=2e-8,
        rtol=0,
    )
    torch.testing.assert_close(
        transverse_stokes[:, 3],
        torch.zeros_like(transverse_stokes[:, 3]),
        atol=2e-8,
        rtol=0,
    )


def test_complete_synthesis_has_gradients_for_all_inferred_atmospheric_fields():
    atmosphere = _atmosphere(
        depth=11,
        batch=2,
        magnetic_vector=(350.0, -240.0, 700.0),
        velocity_los=850.0,
        requires_grad=True,
    )
    wavelength = torch.linspace(6302.15, 6302.82, 45, dtype=torch.float64)
    synthesizer = LTESynthesizer(log_tau500=atmosphere.log_tau500)
    stokes = synthesizer(atmosphere, wavelength, mu=torch.tensor([[1.0], [0.72]], dtype=torch.float64))
    weights = stokes.new_tensor([1.0, 20.0, 20.0, 10.0]).reshape(1, 4, 1)
    objective = (weights * stokes.square()).mean()
    objective.backward()

    for field in (
        atmosphere.temperature,
        atmosphere.velocity_field,
        atmosphere.microturbulence,
        atmosphere.magnetic_field,
        atmosphere.gas_pressure,
    ):
        assert field.grad is not None
        assert torch.isfinite(field.grad).all()
        assert torch.count_nonzero(field.grad) > 0


def test_noiseless_spectrum_recovers_uniform_los_velocity():
    dtype = torch.float32
    target_atmosphere = _atmosphere(
        depth=9,
        velocity_los=1200.0,
        dtype=dtype,
    )
    wavelength = torch.linspace(6302.20, 6302.78, 33, dtype=dtype)
    synthesizer = LTESynthesizer(log_tau500=target_atmosphere.log_tau500)
    with torch.no_grad():
        target = synthesizer(target_atmosphere, wavelength)[:, 0]
        radiance_scale = target.abs().amax()

    velocity_km_s = torch.nn.Parameter(torch.tensor(-0.4, dtype=dtype))
    optimizer = torch.optim.Adam([velocity_km_s], lr=0.15)
    losses = []
    for _ in range(35):
        optimizer.zero_grad()
        candidate = StratifiedAtmosphere(
            log_tau500=target_atmosphere.log_tau500,
            temperature=target_atmosphere.temperature,
            velocity_field=torch.stack(
                (
                    torch.zeros_like(target_atmosphere.v_los),
                    torch.zeros_like(target_atmosphere.v_los),
                    -(1000.0 * velocity_km_s).expand_as(target_atmosphere.v_los),
                ),
                dim=-1,
            ),
            microturbulence=target_atmosphere.microturbulence,
            magnetic_field=target_atmosphere.magnetic_field,
            gas_pressure=target_atmosphere.gas_pressure,
        )
        prediction = synthesizer(candidate, wavelength)[:, 0]
        loss = ((prediction - target) / radiance_scale).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < 0.05 * losses[0]
    assert abs(velocity_km_s.item() - 1.2) < 0.25


def test_realistic_stratification_converges_with_depth_quadrature():
    """Guard convergence of the complete stratified polarized formal solution.

    Constant-layer partition tests cannot expose errors caused by a varying
    source function, opacity, and magnetic atmosphere.  A 101-point result is
    used only as an internal refinement reference; the assertion checks the
    expected second-order convergence trend rather than claiming it is an
    external scientific truth.
    """

    dtype = torch.float64
    wavelength = torch.linspace(6301.20, 6302.80, 49, dtype=dtype)

    def solve(depth: int) -> torch.Tensor:
        grid = torch.linspace(-5.0, 1.0, depth, dtype=dtype)
        temperature = synthetic_temperature_profile(grid).unsqueeze(0)
        atmosphere = StratifiedAtmosphere(
            log_tau500=grid,
            temperature=temperature,
            velocity_field=torch.stack(
                (
                    torch.zeros_like(temperature),
                    torch.zeros_like(temperature),
                    torch.full_like(temperature, -750.0),
                ),
                dim=-1,
            ),
            microturbulence=torch.full_like(temperature, 1_000.0),
            magnetic_field=torch.tensor(
                [350.0, -250.0, 900.0], dtype=dtype
            ).expand(1, depth, 3).clone(),
            gas_pressure=torch.logspace(-0.5, 5.0, depth, dtype=dtype).unsqueeze(0),
        )
        with torch.no_grad():
            return LTESynthesizer()(atmosphere, wavelength)[0]

    reference = solve(101)
    continuum_scale = reference[0, [*range(8), *range(41, 49)]].mean()
    error_25 = (solve(25) - reference).square().mean(dim=-1).sqrt() / continuum_scale
    error_51 = (solve(51) - reference).square().mean(dim=-1).sqrt() / continuum_scale

    assert torch.all(error_51 < 0.4 * error_25)
    assert error_25[0] < 6.0e-3
    assert error_25[3] < 1.1e-3
