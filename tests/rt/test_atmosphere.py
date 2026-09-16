import json

import pytest
import torch

from prom3theus.core import ATOMIC_MASS_UNIT, K_BOLTZMANN
from prom3theus.resources import resource_path
from prom3theus.rt import (
    AtomicDatabase,
    RadialReferenceAtmosphere,
    SolarPlasmaTable,
    StratifiedAtmosphere,
    StratifiedAtmosphereModel,
)


def _model() -> StratifiedAtmosphereModel:
    return StratifiedAtmosphereModel(
        shell_height_bounds_Mm=(1.5, -0.1),
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "mlp",
            "dim": 10,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    )


def _siren_model() -> StratifiedAtmosphereModel:
    return StratifiedAtmosphereModel(
        shell_height_bounds_Mm=(20.0, -0.1),
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "siren",
            "dim": 10,
            "n_layers": 2,
            "first_omega_0": 30.0,
            "hidden_omega_0": 1.0,
            "radial_weighting_config": {
                "near_sun_weight": 1.0,
                "outer_weight": 0.1,
                "exponent": 1.0,
            },
        },
    )


def _extended_model(*, maximum_height_megameter: float = 20.0, points: int = 512):
    return StratifiedAtmosphereModel(
        shell_height_bounds_Mm=(maximum_height_megameter, -0.1),
        line_formation_height_bounds_Mm=(1.5, -0.1),
        upper_atmosphere_config={
            "type": "hydrostatic_corona",
            "transition_region_top_megameter": 2.5,
            "coronal_temperature_k": 1.0e6,
            "reference_grid_points": points,
        },
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "mlp",
            "dim": 10,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    )


def test_complete_falc_reference_restores_only_the_unused_native_upper_nodes():
    document = json.loads(
        resource_path("common/falc_reference_atmosphere.json").read_text(
            encoding="utf-8"
        )
    )
    reference = RadialReferenceAtmosphere("falc_82")
    line = document["line_formation_reference"]
    coordinates = document["coordinates"]

    torch.testing.assert_close(
        reference.log_tau500,
        torch.tensor(line["log10_tau500"], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        reference.height_descending_m,
        torch.tensor(line["height_m"], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert reference.height_descending_m[20] == 0.0

    native_log_tau = torch.tensor(coordinates["log10_tau500"], dtype=torch.float64)
    restored = native_log_tau < -5.0
    assert restored.sum().item() == 37
    expected_height = torch.cat(
        (
            torch.tensor(line["height_m"], dtype=torch.float64).flip(0),
            torch.tensor(coordinates["height_m"], dtype=torch.float64)[restored].flip(
                0
            ),
        )
    )
    expected_logs = torch.stack(
        tuple(
            torch.log(
                torch.cat(
                    (
                        torch.tensor(line[line_name], dtype=torch.float64).flip(0),
                        torch.tensor(document[native_name], dtype=torch.float64)[
                            restored
                        ].flip(0),
                    )
                )
            )
            for line_name, native_name in (
                ("temperature_k", "temperature_k"),
                ("gas_pressure_pa", "gas_pressure_pa"),
                ("microturbulence_m_per_s", "microturbulence_m_per_s"),
            )
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        reference.thermodynamic_height_ascending_m,
        expected_height,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        reference.thermodynamic_logs_ascending,
        expected_logs,
        rtol=0.0,
        atol=0.0,
    )
    assert reference.native_depth_count == 82
    assert reference.thermodynamic_height_ascending_m.numel() == 25 + 37
    assert reference.native_atmosphere_top_m == 2073502.4593743724
    assert reference.native_atmosphere_top_temperature_k == 100000.0


def test_atmosphere_model_starts_at_radial_reference():
    model = _model()
    height = torch.linspace(1.5e6, -1.0e5, 13)
    atmosphere = model(torch.tensor([[0.0, 0.0, 0.0]]), height)
    reference = model.reference_atmosphere.logs_at_height(height)
    torch.testing.assert_close(atmosphere.temperature[0], reference[0].exp())
    torch.testing.assert_close(atmosphere.gas_pressure[0], reference[1].exp())
    torch.testing.assert_close(atmosphere.microturbulence[0], reference[2].exp())
    assert atmosphere.velocity_field.shape == (1, 13, 3)
    assert atmosphere.magnetic_field.shape == (1, 13, 3)


def test_siren_atmosphere_starts_at_radial_reference_and_weights_radius():
    model = _siren_model()
    height = torch.linspace(1.5e6, -1.0e5, 13)
    atmosphere = model(torch.tensor([[0.0, 0.0, 0.0]]), height)
    reference = model.reference_atmosphere.logs_at_height(height)
    torch.testing.assert_close(atmosphere.temperature[0], reference[0].exp())
    torch.testing.assert_close(atmosphere.gas_pressure[0], reference[1].exp())
    torch.testing.assert_close(atmosphere.microturbulence[0], reference[2].exp())

    weighting = model.network.coordinate_weighting
    normalized_radius = torch.tensor(
        [[0.0, 0.0, -0.1], [0.0, 0.0, 2.0]], dtype=model.solar_radius_m.dtype
    )
    envelope = weighting.envelope(normalized_radius)
    assert envelope[0] == pytest.approx(1.0)
    assert 0.1 < envelope[1] < envelope[0]


def test_atmosphere_is_a_differentiable_continuous_field():
    model = _model()
    coordinates = torch.tensor([[-0.2, 0.1, 0.0], [0.3, -0.1, 0.0]])
    height = torch.tensor([[-3.0e4], [4.0e5]], requires_grad=True)
    fields = model.evaluate_at_height(coordinates, height)
    (fields["temperature"].mean() + fields["magnetic_field"].square().mean()).backward()
    assert height.grad is not None and torch.isfinite(height.grad).all()


@pytest.mark.parametrize("activation", ["relu", "swish"])
def test_atmosphere_rejects_noncanonical_or_nonsmooth_activations(activation):
    with pytest.raises(ValueError, match="smooth activation"):
        StratifiedAtmosphereModel(
            shell_height_bounds_Mm=(1.5, -0.1),
            scene_geometry_config={
                "solar_radius_m": 695_700_000.0,
                "scene_basis": torch.eye(3),
            },
            model_config={
                "type": "mlp",
                "dim": 10,
                "n_layers": 2,
                "activation": activation,
                "encoding_config": {"type": "identity"},
            },
        )


def test_stratified_atmosphere_enforces_depth_and_dtype_contracts():
    fields = {
        "temperature": torch.ones(1, 2),
        "velocity_field": torch.zeros(1, 2, 3),
        "microturbulence": torch.ones(1, 2),
        "magnetic_field": torch.zeros(1, 2, 3),
        "gas_pressure": torch.ones(1, 2),
    }
    with pytest.raises(ValueError, match="strictly increase"):
        StratifiedAtmosphere(depth_coordinate=torch.tensor([0.0, -1.0]), **fields)
    with pytest.raises(TypeError, match="same dtype"):
        StratifiedAtmosphere(
            depth_coordinate=torch.tensor([-1.0, 0.0]),
            **{**fields, "gas_pressure": torch.ones(1, 2, dtype=torch.float64)},
        )


def test_atmosphere_model_rejects_invalid_evaluation_grids_and_rays():
    model = _model()
    coordinates = torch.tensor([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="strictly decrease"):
        model(coordinates, torch.tensor([-4.0, -3.0, -3.5]))
    with pytest.raises(ValueError, match="spherical shell"):
        model(coordinates, torch.tensor([1.6e6, 0.0]))
    with pytest.raises(ValueError, match="finite non-zero"):
        model.trace_rays(
            coordinates,
            torch.zeros(1, 3),
            torch.tensor([-4.0, 0.0]),
        )


def test_extended_model_keeps_ray_sampling_in_the_line_formation_domain():
    model = _extended_model(maximum_height_megameter=3.0, points=128)

    ray_heights = torch.tensor(model.line_formation_height_bounds_Mm) * 1e6
    sampled, trace = model.trace_rays(
        torch.zeros(1, 3), torch.tensor([[0.0, 0.0, -1.0]]), ray_heights
    )
    assert sampled.depth == 2
    torch.testing.assert_close(ray_heights[[0, -1]], torch.tensor([1.5e6, -1.0e5]))
    assert model.extrapolation_enabled
    top_pressure = model.top_boundary_reference_log_pressure.exp()
    line_top_pressure = model.reference_atmosphere.logs_at_height(torch.tensor(1.5e6))[
        1
    ].exp()
    assert 0.0 < top_pressure < line_top_pressure

    coords = torch.zeros(1, 3)
    fields = model.evaluate_at_height(coords, torch.tensor([[3.0e6]]))
    torch.testing.assert_close(
        fields["gas_pressure"].squeeze(), top_pressure, rtol=2e-5, atol=0.0
    )


def test_twenty_megameter_reference_is_coronal_and_hydrostatic():
    model = _extended_model()
    heights = torch.linspace(2.6e6, 20.0e6, 128)
    log_temperature, log_pressure, _ = model.reference_atmosphere.logs_at_height(
        heights
    )
    pressure = torch.exp(log_pressure)

    torch.testing.assert_close(
        torch.exp(log_temperature),
        torch.full_like(log_temperature, 1.0e6),
        rtol=2.0e-4,
        atol=0.0,
    )
    assert torch.all(pressure[1:] < pressure[:-1])
    assert torch.isfinite(pressure).all() and torch.all(pressure > 0)
    assert 1.0e-3 < pressure[-1] < 1.0e-1


def test_coronal_continuation_uses_the_native_top_and_satisfies_hydrostatics():
    point_count = 512
    model = _extended_model(points=point_count)
    reference = model.reference_atmosphere
    heights = reference.thermodynamic_height_ascending_m
    logs = reference.thermodynamic_logs_ascending
    upper_heights = heights[-point_count:]
    upper_logs = logs[-point_count:]
    base_height = heights[-point_count - 1]
    base_logs = logs[-point_count - 1]

    # 25 line-formation nodes, 37 restored native FALC nodes, the coronal
    # continuation, and the adiabatic interior nodes carrying the -0.1 Mm
    # shell floor below the FALC table bottom.
    interior_nodes = reference.metadata()["lower_atmosphere"]["interval_count"]
    assert interior_nodes == 16
    assert heights.numel() == interior_nodes + 25 + 37 + point_count
    assert float(base_height) == 2073502.4593743724
    torch.testing.assert_close(
        torch.exp(base_logs),
        torch.tensor(
            [100000.0, 0.031934421263776124, 10680.960000000001],
            dtype=torch.float64,
        ),
        rtol=2.0e-15,
        atol=0.0,
    )
    assert torch.any(upper_heights == 2.5e6)

    transition_fraction = ((upper_heights - base_height) / (2.5e6 - base_height)).clamp(
        0.0, 1.0
    )
    smootherstep = transition_fraction.pow(3) * (
        transition_fraction * (transition_fraction * 6.0 - 15.0) + 10.0
    )
    expected_log_temperature = torch.lerp(
        base_logs[0],
        base_logs.new_tensor(1.0e6).log(),
        smootherstep,
    )
    torch.testing.assert_close(
        upper_logs[:, 0], expected_log_temperature, rtol=2.0e-15, atol=2.0e-15
    )
    coronal = upper_heights >= 2.5e6
    torch.testing.assert_close(
        torch.exp(upper_logs[coronal, 0]),
        torch.full_like(upper_logs[coronal, 0], 1.0e6),
        rtol=2.0e-15,
        atol=0.0,
    )

    hydrostatic_heights = heights[-point_count - 1 :]
    hydrostatic_logs = logs[-point_count - 1 :]
    midpoint_height = 0.5 * (hydrostatic_heights[:-1] + hydrostatic_heights[1:])
    midpoint_temperature = torch.sqrt(
        torch.exp(hydrostatic_logs[:-1, 0] + hydrostatic_logs[1:, 0])
    )
    midpoint_pressure = torch.exp(
        0.5 * (hydrostatic_logs[:-1, 1] + hydrostatic_logs[1:, 1])
    )
    mean_particle_mass_u = model.thermodynamic_eos.mean_molecular_weight(
        midpoint_temperature, midpoint_pressure
    )
    gravity = (
        model.thermodynamic_eos.reference_gravity_m_per_s2
        * (
            model.solar_radius_value_m / (model.solar_radius_value_m + midpoint_height)
        ).square()
    )
    expected_delta_log_pressure = -(
        mean_particle_mass_u
        * ATOMIC_MASS_UNIT
        * gravity
        * torch.diff(hydrostatic_heights)
        / (K_BOLTZMANN * midpoint_temperature)
    )
    torch.testing.assert_close(
        torch.diff(hydrostatic_logs[:, 1]),
        expected_delta_log_pressure,
        rtol=5.0e-12,
        atol=5.0e-16,
    )
    assert torch.all(torch.diff(hydrostatic_logs[:, 1]) < 0.0)


def test_normalized_atmosphere_eos_adapter_round_trips_actual_state():
    model = _model()
    position = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.001]])
    normalized = model.evaluate_position_rsun_normalized(position)
    expected = model.thermodynamic_eos.mass_density(
        normalized["temperature"] * model.temperature_scale_k,
        normalized["gas_pressure"] * model.gas_pressure_scale_pa,
    ) / model.density_scale_kg_m3
    torch.testing.assert_close(
        model.mass_density_normalized(
            normalized["temperature"], normalized["gas_pressure"]
        ),
        expected,
    )


def test_zero_thermodynamic_outputs_follow_the_full_reference_through_the_corona():
    model = _extended_model()
    native_top = model.reference_atmosphere.native_atmosphere_top_m
    heights = torch.tensor(
        [[1.5e6, 2.0e6, native_top, 2.5e6, 20.0e6]], dtype=torch.float32
    )
    reference_logs = model.reference_atmosphere.logs_at_height(heights)
    fields = model.evaluate_at_height(torch.zeros((1, 3)), heights)

    thermodynamic_names = ("temperature", "gas_pressure", "microturbulence")
    for name, reference_log in zip(
        thermodynamic_names,
        reference_logs,
        strict=True,
    ):
        torch.testing.assert_close(fields[name], torch.exp(reference_log))

    probes = torch.tensor(
        [[native_top - 1.0, native_top + 1.0, 2.5e6 - 1.0, 2.5e6 + 1.0]],
        requires_grad=True,
    )
    probe_fields = model.evaluate_at_height(torch.zeros((1, 3)), probes)
    sum(torch.log(probe_fields[name]).sum() for name in thermodynamic_names).backward()
    assert probes.grad is not None and torch.isfinite(probes.grad).all()


@pytest.mark.parametrize(
    ("name", "channel", "reference_index", "scale_attribute"),
    (
        ("temperature", 0, 0, "temperature_log_scale"),
        ("microturbulence", 7, 2, "microturbulence_log_scale"),
        ("gas_pressure", 8, 1, "gas_pressure_log_scale"),
    ),
)
def test_primary_thermodynamic_decoder_is_an_unbounded_linear_log_residual(
    name,
    channel,
    reference_index,
    scale_attribute,
):
    model = _extended_model(points=64)
    values = torch.tensor([-20.0, 0.0, 20.0], dtype=torch.float64).repeat(2)
    heights = torch.tensor(
        [0.0, 0.0, 0.0, 3.0e6, 3.0e6, 3.0e6],
        dtype=torch.float64,
    )
    raw = torch.zeros((values.numel(), len(model.output_names)), dtype=torch.float64)
    raw[:, channel] = values
    raw.requires_grad_()

    fields = model._decode_raw(raw, geometric_height_m=heights)
    reference_log = model.reference_atmosphere.logs_at_height(heights)[reference_index]
    log_residual = torch.log(fields[name]) - reference_log
    scale = getattr(model, scale_attribute)

    torch.testing.assert_close(log_residual, scale * values)
    assert torch.all(log_residual[values.abs() == 20.0].abs() > 10.0 * scale)
    gradient = torch.autograd.grad(log_residual.sum(), raw)[0][:, channel]
    torch.testing.assert_close(gradient, torch.full_like(values, scale))
    assert torch.isfinite(fields[name]).all() and torch.all(fields[name] > 0.0)


def test_coronal_continuation_must_start_above_the_complete_falc_top():
    model = _model()
    with pytest.raises(ValueError, match="above the FALC top"):
        RadialReferenceAtmosphere(
            "falc_82",
            upper_atmosphere_config={
                "type": "hydrostatic_corona",
                "transition_region_top_megameter": 2.0,
                "coronal_temperature_k": 1.0e6,
                "reference_grid_points": 64,
            },
            maximum_height_m=3.0e6,
            solar_radius_m=695_700_000.0,
            thermodynamic_eos=model.thermodynamic_eos,
        )


def test_model_has_no_observation_grid_and_accepts_physical_sampling():
    model = _model()
    assert not hasattr(model, "log_tau500")
    assert not hasattr(model, "depth_to_height")
    assert "log_tau500" not in model.state_dict()
    coordinates = torch.zeros(1, 3)
    for height in (torch.tensor([1.5e6, -1e5]), torch.tensor([1e6, 5e5, 0.0, -1e5])):
        atmosphere = model(coordinates, height)
        torch.testing.assert_close(atmosphere.geometric_height_m[0], height)
        torch.testing.assert_close(
            atmosphere.depth_coordinate, torch.arange(height.numel()).float()
        )


def test_adiabatic_interior_replaces_the_runaway_table_extrapolation():
    """A sub-photospheric shell floor must follow an adiabat, not the table.

    The FALC table stops about 69 km below the surface.  Extrapolating its
    steep bottom gradient reaches 48 kK by -0.5 Mm, which is roughly four
    times the real solar value and drives the forward model past the hot end
    of the photospheric fade, where the Fe I reservoir vanishes entirely.
    """

    atomic = AtomicDatabase()
    eos = SolarPlasmaTable(atomic)
    solar_radius_m = 695_700_000.0
    heights = torch.tensor([-0.1e6, -0.2e6, -0.3e6, -0.5e6], dtype=torch.float64)

    tabulated = RadialReferenceAtmosphere("falc_82", atomic_database=atomic)
    interior = RadialReferenceAtmosphere(
        "falc_82",
        minimum_height_m=-0.5e6,
        solar_radius_m=solar_radius_m,
        thermodynamic_eos=eos,
        atomic_database=atomic,
    )

    extrapolated_temperature = tabulated.logs_at_height(heights)[0].exp()
    temperature, pressure, microturbulence = interior.logs_at_height(heights)
    temperature, pressure = temperature.exp(), pressure.exp()

    assert float(extrapolated_temperature[-1]) > 45_000.0
    # Standard solar envelope values at these depths, to within ten percent.
    expected = torch.tensor([8932.0, 9695.0, 10357.0, 11558.0], dtype=torch.float64)
    torch.testing.assert_close(temperature, expected, rtol=0.1, atol=0.0)
    assert torch.all(temperature[1:] > temperature[:-1])
    assert torch.all(pressure[1:] > pressure[:-1])
    # Microturbulence is held at the native bottom value.
    torch.testing.assert_close(
        microturbulence, torch.full_like(microturbulence, float(microturbulence[0]))
    )
    # The continuation stays far below the hot end of the photospheric fade,
    # so the Fe I reservoir survives everywhere inside the shell.  The old
    # extrapolation ran past it and zeroed the line opacity outright.
    fade_top = eos.eos.transition_temperature_bounds_k[1]
    assert float(temperature[-1]) < 0.5 * fade_top
    assert float(extrapolated_temperature[-1]) > fade_top
    weight = eos.prepare_plasma_state(
        temperature, torch.full_like(temperature, 1.0e5)
    ).photospheric_weight
    assert torch.all(weight > 0.95)


def test_interior_continuation_joins_the_falc_table_continuously():
    atomic = AtomicDatabase()
    eos = SolarPlasmaTable(atomic)
    plain = RadialReferenceAtmosphere("falc_82", atomic_database=atomic)
    interior = RadialReferenceAtmosphere(
        "falc_82",
        minimum_height_m=-0.5e6,
        solar_radius_m=695_700_000.0,
        thermodynamic_eos=eos,
        atomic_database=atomic,
    )

    # Nothing at or above the FALC table bottom may move.
    above = torch.tensor([0.0, 0.5e6, -0.05e6, -0.0688e6], dtype=torch.float64)
    for extended, original in zip(
        interior.logs_at_height(above), plain.logs_at_height(above), strict=True
    ):
        torch.testing.assert_close(extended, original)

    # The stored table stays strictly ascending in height through the join.
    heights = interior.thermodynamic_height_ascending_m
    assert torch.all(heights[1:] > heights[:-1])
    assert float(heights[0]) == -0.5e6

    # A floor inside the table adds no continuation at all.
    shallow = RadialReferenceAtmosphere(
        "falc_82",
        minimum_height_m=-0.05e6,
        solar_radius_m=695_700_000.0,
        thermodynamic_eos=eos,
        atomic_database=atomic,
    )
    assert shallow.metadata()["lower_atmosphere"] is None


def test_interior_ionization_matches_the_stic_table_where_both_apply():
    """Saha must agree with the tabulated composition inside the STiC table.

    The interior gradient needs an ionization degree above the table's 10 kK
    ceiling, where the runtime EoS blends toward CHIANTI coronal equilibrium
    and turns several-fold too neutral for a dense interior.  Saha continues
    the same LTE physics instead -- but only legitimately so if the two agree
    wherever the table itself is valid.
    """

    from prom3theus.rt.atmosphere import _hydrogen_ionization_fraction

    eos = SolarPlasmaTable(AtomicDatabase())
    temperature = torch.tensor([6000.0, 8000.0, 9000.0, 9900.0], dtype=torch.float64)
    gas_pressure = torch.tensor(
        [1.2e4, 1.8e4, 2.2e4, 3.0e4], dtype=torch.float64
    )
    assert torch.all(temperature < eos.eos.transition_temperature_bounds_k[0])

    saha = _hydrogen_ionization_fraction(eos, temperature, gas_pressure)
    tabulated = eos.electron_density(temperature, gas_pressure) / (
        eos.mass_density(temperature, gas_pressure)
        / (eos.eos.mass_u_per_h_nucleus * ATOMIC_MASS_UNIT)
    )

    torch.testing.assert_close(saha, tabulated, rtol=0.05, atol=1.0e-4)
    # And it keeps rising with depth past the ceiling, which the blended
    # coronal-equilibrium electron density does not.
    hot = torch.tensor([10400.0, 11000.0, 11600.0, 12100.0], dtype=torch.float64)
    hot_pressure = torch.tensor([5.0e4, 7.4e4, 1.08e5, 1.55e5], dtype=torch.float64)
    hot_saha = _hydrogen_ionization_fraction(eos, hot, hot_pressure)
    assert torch.all(hot_saha[1:] > hot_saha[:-1])
    assert float(hot_saha[-1]) > 2.0 * float(saha[-1])
