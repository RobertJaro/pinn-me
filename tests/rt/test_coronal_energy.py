import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prom3theus.config.schema import CoronalEnergyConfig
from prom3theus.core import ATOMIC_MASS_UNIT
from prom3theus.inversion.constraints.magnetofluid import MagnetofluidConstraints
from prom3theus.resources import CORONAL_COOLING_RESOURCE, resolve_resource_reference
from prom3theus.rt.coronal_energy import (
    CoronalCoolingTable,
    CoronalEnergy,
    CoronalEnergyOptions,
    _mass_u_per_h_nucleus,
    spitzer_heat_flux,
)


@pytest.fixture
def cooling_path(tmp_path):
    path = tmp_path / "cooling.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "unit": "W m^3",
                "density_convention": "n_e * n_H",
                "log10_temperature_k": [4.0, 6.0, 8.0],
                "log10_lambda_w_m3": [-35.0, -35.0, -35.0],
                "provenance": {"source": "synthetic test fixture, not CHIANTI"},
            }
        )
    )
    return path


class Atmosphere(nn.Module):
    solar_radius_m = 1.0e8
    time_dependent = True

    def __init__(self, slope=0.0, pressure_rate=0.0, velocity_rate=0.0):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope))
        self.pressure_rate = pressure_rate
        self.velocity_rate = velocity_rate
        self.field = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))
        self.thermodynamic_eos = SimpleNamespace(
            mass_density=lambda t, p: (
                torch.ones_like(t) * (1.4 * ATOMIC_MASS_UNIT * 1.0e15)
            ),
            electron_density=lambda t, p: torch.ones_like(t) * 1.0e15,
            mass_u_per_h_nucleus=1.4,
        )

    def evaluate_position_rsun(self, coordinates, time_hours):
        x = (coordinates[:, 0] - 1.04) * self.solar_radius_m
        return {
            "temperature": 1.0e6 + self.slope * x,
            "gas_pressure": torch.exp(self.pressure_rate * time_hours[:, 0] * 3600.0),
            "velocity_field": self.velocity_rate * coordinates * self.solar_radius_m,
            "magnetic_field": self.field.expand_as(coordinates),
        }


def options(cooling_resource, **kwargs):
    """Build rt-level options for direct ``CoronalEnergy`` construction.

    Accepts either a raw ``Path`` to a synthetic cooling table (as tests use)
    or a registry reference string; both are valid ``CoronalEnergyOptions``
    inputs, unlike the schema-level ``CoronalEnergyConfig.cooling_resource``,
    which is a fixed registered-resource literal.
    """

    return CoronalEnergyOptions(cooling_resource=cooling_resource, **kwargs)


def constraint_config(cooling_resource, **kwargs):
    """Build a plain equation-config dict for ``MagnetofluidConstraints``."""

    return {"enabled": True, "weight": 1.0, **asdict(options(cooling_resource, **kwargs))}


def evaluate(model, energy, *, create_graph=True):
    return energy.residual(
        model,
        torch.tensor([[1.04e8, 0.0, 0.0]]),
        torch.zeros(1, 1),
        5.0 / 3.0,
        1.0,
        create_graph=create_graph,
    )


def test_radiation_and_heating_signs_and_si(cooling_path):
    model = Atmosphere()
    opts = options(cooling_path, heating_w_m3=0.0)
    cooling = evaluate(model, CoronalEnergy(opts))
    # n_e*n_H*Lambda = 1e30 * 1e-35 = 1e-5 W/m^3.
    torch.testing.assert_close(
        cooling, torch.tensor([2.0 / 3.0 * 1.0e-5]), rtol=1e-5, atol=1e-10
    )
    heated = evaluate(model, CoronalEnergy(replace(opts, heating_w_m3=1.0e-4)))
    expected = 2.0 / 3.0 * (1.0e-5 - 1.0e-4 * torch.exp(torch.tensor(-1.0 / 30.0)))
    torch.testing.assert_close(heated[0], expected, rtol=1e-5, atol=1e-10)
    assert cooling.dtype == torch.float32


def test_heating_network_reproduces_reference_law_at_construction(cooling_path):
    """A freshly constructed network must equal the fixed reference law exactly."""

    opts = options(
        cooling_path, heating_w_m3=1.0e-4, heating_scale_height_megameter=20.0
    )
    energy = CoronalEnergy(opts)
    magnetic = torch.tensor([[1.0, 2.0, -3.0], [500.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    density = torch.tensor([1.0e-12, 5.0e-13, 2.0e-12])
    amplitude, scale_height_m = energy.heating(magnetic, density)
    torch.testing.assert_close(amplitude, torch.full((3,), 1.0e-4))
    torch.testing.assert_close(scale_height_m, torch.full((3,), 20.0e6))


def test_heating_network_stays_positive_after_perturbation(cooling_path):
    energy = CoronalEnergy(options(cooling_path))
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in energy.heating.network.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    magnetic = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [-1.0e4, 0.0, 0.0]]
    )
    density = torch.tensor([1.0e-15, 1.0e-13, 1.0e-12, 1.0e-9])
    amplitude, scale_height_m = energy.heating(magnetic, density)
    assert torch.isfinite(amplitude).all() and (amplitude > 0).all()
    assert torch.isfinite(scale_height_m).all() and (scale_height_m > 0).all()


def test_heating_network_clamps_extreme_raw_output_without_inf_or_nan_gradients(
    cooling_path,
):
    """Guards the overflow this network is exposed to once coronal_energy is
    evaluated over the shared volume-equation batch: points below
    minimum_height_megameter are no longer pre-filtered out, so this network
    can see wildly out-of-distribution (|B|, rho) once training moves its
    weights. Without a clamp, exp() of a large raw output overflows to inf,
    and an excluded point's inf can still poison a shared parameter's
    gradient (0 * inf = nan) even though its own loss contribution is zero.
    """

    energy = CoronalEnergy(options(cooling_path))
    with torch.no_grad():
        # Forces a huge raw output regardless of input -- well past where a
        # bare exp() would overflow float32.
        energy.heating.network.out_layer.bias.fill_(1000.0)
    magnetic = torch.tensor([[1.0, 0.0, 0.0]])
    density = torch.tensor([1.0e-12])
    amplitude, scale_height_m = energy.heating(magnetic, density)
    assert torch.isfinite(amplitude).all() and torch.isfinite(scale_height_m).all()
    (amplitude + scale_height_m).sum().backward()
    for parameter in energy.heating.network.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_heating_network_parameters_receive_gradients(cooling_path):
    model = Atmosphere(slope=0.01)
    energy = CoronalEnergy(options(cooling_path, heating_w_m3=1.0e-4))
    value = evaluate(model, energy)
    value.sum().backward()
    gradients = [parameter.grad for parameter in energy.heating.network.parameters()]
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    # The zero-initialized output layer receives the network's only guaranteed
    # nonzero gradient at construction; earlier layers only start moving once
    # it does.
    out_layer = energy.heating.network.out_layer
    assert out_layer.weight.grad.abs().max() > 0
    assert out_layer.bias.grad.abs().max() > 0


def test_residual_components_sum_to_the_scalar_residual():
    """``return_components=True`` must reconstruct the exact same residual.

    Guards the sign convention of the diagnostic breakdown: a flipped sign on
    any one term would still look plausible in isolation but silently mislead
    the wandb plot's per-term panel.
    """

    energy = CoronalEnergy(options(CORONAL_COOLING_RESOURCE, heating_w_m3=1.0e-4))
    model = coronal_model(slope=0.01, pressure_rate=0.002, velocity_rate=0.003)
    arguments = (
        model,
        torch.tensor([[1.04e8, 2.0e6, -1.0e6]]),
        torch.full((1, 1), 0.3),
        5.0 / 3.0,
        782.6,
    )
    keywords = dict(create_graph=False, model_time_scale_s=3600.0)
    residual = energy._residual_normalized(*arguments, **keywords)
    residual_again, components, pressure_scale = energy._residual_normalized(
        *arguments, **keywords, return_components=True
    )
    torch.testing.assert_close(residual, residual_again)
    assert set(components) == {
        "D_t p + γp∇·v",
        "-(γ-1)Q_heat",
        "+(γ-1)n_e n_H Λ",
        "+(γ-1)∇·F_c",
    }
    reconstructed = sum(components.values()) / pressure_scale
    torch.testing.assert_close(residual, reconstructed)


def test_coronal_energy_config_validates_heating_network_options():
    CoronalEnergyConfig(
        enabled=True,
        weight=1.0,
        heating_network_hidden_dim=32,
        heating_network_hidden_layers=2,
        heating_network_activation="tanh",
    )
    with pytest.raises(ValueError, match="heating_network_hidden_dim"):
        CoronalEnergyConfig(
            enabled=True,
            weight=1.0,
            heating_network_hidden_dim=0,
        )
    with pytest.raises(ValueError, match="heating_network_activation"):
        CoronalEnergyConfig(
            enabled=True,
            weight=1.0,
            heating_network_activation="relu",
        )


def test_dynamic_pressure_and_compression_use_seconds(cooling_path):
    energy = CoronalEnergy(options(cooling_path, heating_w_m3=0.0))
    baseline = evaluate(Atmosphere(), energy)
    dynamic = evaluate(Atmosphere(pressure_rate=0.002, velocity_rate=0.003), energy)
    torch.testing.assert_close(
        dynamic - baseline, torch.tensor([0.002 + (5.0 / 3.0) * 3 * 0.003])
    )


def test_conduction_second_derivatives_and_magnetic_gradients(cooling_path):
    model = Atmosphere(slope=0.01)
    opts = options(cooling_path, heating_w_m3=0.0)
    value = evaluate(model, CoronalEnergy(opts))
    # Linear T has div(q) = -kappa0 * 2.5*T^1.5*(dT/dx)^2*b_x^2.
    div_q = (
        -opts.conductivity_w_m_k72
        * 2.5
        * 1.0e9
        * 0.01**2
        / (2.0 + opts.magnetic_floor_gauss**2)
    )
    expected = torch.tensor([2.0 / 3.0 * (1.0e-5 + div_q)])
    torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-10)
    value.sum().backward()
    assert torch.isfinite(model.slope.grad) and model.slope.grad.abs() > 0
    assert torch.isfinite(model.field.grad).all() and model.field.grad.abs().max() > 0
    validation = evaluate(model, CoronalEnergy(opts), create_graph=False)
    torch.testing.assert_close(validation, value.detach())


def test_conduction_direction_and_zero_field():
    temperature = torch.tensor([1.0e6])
    gradient = torch.tensor([[1.0, 0.0, 0.0]])
    aligned = spitzer_heat_flux(
        temperature, gradient, torch.tensor([[1.0, 0.0, 0.0]]), 1e-11, 0.1
    )
    assert aligned[0, 0] < 0
    for field in (torch.zeros(1, 3), torch.tensor([[0.0, 1.0, 0.0]])):
        assert torch.equal(
            spitzer_heat_flux(temperature, gradient, field, 1e-11, 0.1),
            torch.zeros(1, 3),
        )


def test_mass_u_per_h_nucleus_unwraps_solar_plasma_table_shape():
    """``SolarPlasmaTable`` only exposes this attribute on its wrapped ``.eos``.

    Every atmosphere in the coronal_energy test suite uses either a
    ``SimpleNamespace`` mock or a bare ``HybridSolarEOS``, both of which
    expose ``mass_u_per_h_nucleus`` directly -- so a regression that only
    works for the flat shape and breaks the wrapped one (production's
    ``StratifiedAtmosphereModel.thermodynamic_eos``, a real
    ``SolarPlasmaTable``) would otherwise go unnoticed.
    """

    flat = SimpleNamespace(mass_u_per_h_nucleus=1.4)
    assert _mass_u_per_h_nucleus(flat) == 1.4
    wrapped = SimpleNamespace(eos=SimpleNamespace(mass_u_per_h_nucleus=1.3))
    assert _mass_u_per_h_nucleus(wrapped) == 1.3


def test_energy_residual_runs_against_the_real_solar_plasma_table(cooling_path):
    """Exercise the actual production EOS wrapper, not just its test double.

    ``HybridSolarEOS`` alone (used by the other tests here) exposes
    ``mass_u_per_h_nucleus`` at its own top level; the real atmosphere uses
    ``SolarPlasmaTable``, which only exposes it on the wrapped ``.eos``. This
    caught a production-only ``AttributeError`` that no other test surfaced.
    """

    from prom3theus.rt.plasma import SolarPlasmaTable

    model = Atmosphere(slope=0.01)
    model.thermodynamic_eos = SolarPlasmaTable()
    residual = evaluate(model, CoronalEnergy(options(cooling_path)))
    residual.square().mean().backward()
    assert torch.isfinite(residual).all()
    assert torch.isfinite(model.slope.grad)
    assert torch.isfinite(model.field.grad).all()


def test_energy_with_shared_hybrid_eos_has_finite_float32_gradients(cooling_path):
    from prom3theus.rt.eos import HybridSolarEOS

    model = Atmosphere(slope=0.01)
    model.thermodynamic_eos = HybridSolarEOS()
    value = evaluate(model, CoronalEnergy(options(cooling_path)))
    value.square().mean().backward()
    assert value.dtype == torch.float32
    assert torch.isfinite(value).all()
    assert torch.isfinite(model.slope.grad)
    assert torch.isfinite(model.field.grad).all()


def test_generated_chianti_table_components_and_energy_gradients():
    from prom3theus.rt.eos import HybridSolarEOS

    document = json.loads(
        resolve_resource_reference(CORONAL_COOLING_RESOURCE).read_text()
    )
    assert document["provenance"]["database_version"] == "11.0.2"
    assert document["provenance"]["fiasco_version"] == "0.8.2"
    components = document["components_w_m3"]
    assert set(components) == {"lines", "free_free", "free_bound", "two_photon"}
    values = torch.tensor(list(components.values()))
    assert torch.isfinite(values).all() and (values >= 0).all()
    assert (values.sum(-1) > 0).all()
    table = CoronalCoolingTable(CORONAL_COOLING_RESOURCE)
    temperature = torch.pow(10.0, table.log_temperature).requires_grad_(True)
    actual = table(temperature)
    torch.testing.assert_close(actual, values.sum(0).log10(), rtol=0, atol=2e-5)
    actual.sum().backward()
    assert torch.isfinite(temperature.grad).all()
    assert 1e-37 < 10.0 ** actual[80].item() < 1e-33
    model = Atmosphere(slope=0.01)
    model.thermodynamic_eos = HybridSolarEOS()
    residual = evaluate(model, CoronalEnergy(options(CORONAL_COOLING_RESOURCE)))
    residual.square().mean().backward()
    assert residual.dtype == torch.float32 and torch.isfinite(residual).all()
    assert torch.isfinite(model.slope.grad)
    assert torch.isfinite(model.field.grad).all()


def test_cooling_tails_do_not_clip_temperature_and_checkpoint_checks_resource(
    cooling_path,
):
    table = CoronalCoolingTable(cooling_path)
    temperature = torch.tensor([1e3, 1e5, 1e9], requires_grad=True)
    values = table(temperature)
    torch.testing.assert_close(values, torch.tensor([-37.0, -35.0, -34.5]))
    values.sum().backward()
    assert (temperature.grad[[0, 2]] > 0).all()
    from prom3theus.application.runtime import state_dict_sha256

    assert len(state_dict_sha256(table)) == 64
    state = {name: value.clone() for name, value in table.state_dict().items()}
    state["source_digest"][0] ^= 1
    with pytest.raises(RuntimeError, match="Checkpoint cooling table"):
        table.load_state_dict(state)


def test_disabled_equation_does_not_load_missing_table(tmp_path):
    constraints = MagnetofluidConstraints(
        {
            "coronal_energy": {
                "enabled": False,
                "weight": 0.0,
                "cooling_resource": tmp_path / "missing.json",
            }
        }
    )
    assert constraints.coronal_energy is None
    assert not constraints.upper_volume_active
    assert not constraints.state_dict()


def test_energy_excludes_low_atmosphere_even_under_no_grad(cooling_path):
    model = Atmosphere()
    constraints = MagnetofluidConstraints(
        {"coronal_energy": constraint_config(cooling_path, heating_w_m3=0.0)},
        vector_basis_matches_spatial_coordinates=True,
    )
    positions = torch.tensor([[1.01e8, 0.0, 0.0], [1.04e8, 0.0, 0.0]])
    with torch.no_grad():
        mixed = constraints.upper_domain(
            model,
            positions,
            torch.zeros(2, 1),
            height_group_shape=(2, 1),
            create_graph=False,
        )
        high = constraints.upper_domain(
            model,
            positions[1:],
            torch.zeros(1, 1),
            height_group_shape=(1, 1),
            create_graph=False,
        )
        low = constraints.upper_domain(
            model,
            positions[:1],
            torch.zeros(1, 1),
            height_group_shape=(1, 1),
            create_graph=False,
        )
    torch.testing.assert_close(
        mixed.losses["coronal_energy"], high.losses["coronal_energy"]
    )
    assert low.losses["coronal_energy"] == 0


def test_upper_domain_diagnostics_match_the_scalar_loss(cooling_path):
    model = coronal_model(slope=0.01)
    constraints = MagnetofluidConstraints(
        {"coronal_energy": constraint_config(cooling_path, heating_w_m3=1.0e-4)},
        vector_basis_matches_spatial_coordinates=True,
    )
    positions = torch.tensor([[1.04e8, 0.0, 0.0], [1.05e8, 0.0, 0.0]])
    result = constraints.upper_domain(
        model,
        positions,
        torch.zeros(2, 1),
        height_group_shape=(2, 1),
        return_diagnostics=True,
    )
    data = result.equation_diagnostics["coronal_energy"]
    assert set(data["terms"]) == set(data["normalized_terms"])
    reconstructed_residual = sum(data["normalized_terms"].values())
    torch.testing.assert_close(
        reconstructed_residual.square().mean(), result.losses["coronal_energy"]
    )
    assert data["weight"] == result.weights["coronal_energy"]


def test_upper_domain_diagnostics_reject_a_partial_height_mask(cooling_path):
    model = Atmosphere(slope=0.01)
    constraints = MagnetofluidConstraints(
        {"coronal_energy": constraint_config(cooling_path)},
        vector_basis_matches_spatial_coordinates=True,
    )
    positions = torch.tensor([[1.01e8, 0.0, 0.0], [1.04e8, 0.0, 0.0]])
    with pytest.raises(ValueError, match="minimum_height_megameter"):
        constraints.upper_domain(
            model,
            positions,
            torch.zeros(2, 1),
            height_group_shape=(2, 1),
            return_diagnostics=True,
        )


def test_upper_domain_computes_the_full_batch_and_masks_only_the_reduction(
    cooling_path,
):
    """Training-path coronal_energy no longer pre-filters below the minimum.

    Both points are evaluated together (sharing the volume equations' batch,
    not a second sampling pass); the excluded one must not raise, and must
    not poison the shared heating-network/atmosphere parameters' gradients.
    """

    model = coronal_model(slope=0.01)
    constraints = MagnetofluidConstraints(
        {"coronal_energy": constraint_config(cooling_path, heating_w_m3=1.0e-4)},
        vector_basis_matches_spatial_coordinates=True,
    )
    # [0]: 1 megameter, below the 3 megameter default minimum -- excluded from the mean.
    # [1]: 4 megameter, above it -- the only point that should shape the loss.
    positions = torch.tensor([[1.01e8, 0.0, 0.0], [1.04e8, 0.0, 0.0]])
    both = constraints.upper_domain(
        model, positions, torch.zeros(2, 1), height_group_shape=(2, 1)
    )
    high_only = constraints.upper_domain(
        model, positions[1:], torch.zeros(1, 1), height_group_shape=(1, 1)
    )
    assert torch.isfinite(both.losses["coronal_energy"])
    torch.testing.assert_close(
        both.losses["coronal_energy"], high_only.losses["coronal_energy"]
    )
    both.losses["coronal_energy"].backward()
    for parameter in (
        *constraints.coronal_energy.heating.network.parameters(),
        *model.parameters(),
    ):
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_energy_defaults_to_registered_resource_and_rejects_adiabatic_overlap(
    cooling_path,
):
    assert (
        CoronalEnergyConfig(enabled=True, weight=1.0).cooling_resource
        == CORONAL_COOLING_RESOURCE
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        MagnetofluidConstraints(
            {
                "coronal_energy": constraint_config(cooling_path),
                "adiabatic_pressure": {"enabled": True, "weight": 1.0},
            },
            vector_basis_matches_spatial_coordinates=True,
        )


class NormalizedAtmosphere(Atmosphere):
    """Atmosphere exposing the dimensionless training contract.

    ``CoronalEnergy.residual`` dispatches on these attributes, so this is the
    path every real inversion takes.  The scales are the FALC_82 values the
    shipped configurations derive, which is what makes the SI density product
    unrepresentable in float32 unless it stays inside the logarithm.
    """

    temperature_scale_k = 6558.0
    gas_pressure_scale_pa = 1.231e4
    density_scale_kg_m3 = 2.911e-4
    electron_density_scale_m3 = 8.256e19
    velocity_scale_m_per_s = 1.0e3
    magnetic_scale_gauss = 100.0
    height_input_scale_m = 1.0e7

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(
            "solar_radius_model",
            torch.tensor(self.solar_radius_m / self.height_input_scale_m),
        )
        from prom3theus.rt.eos import HybridSolarEOS

        self.thermodynamic_eos = HybridSolarEOS()

    # 1 MK at a coronal gas pressure: the state the residual is written for.
    coronal_pressure_pa = 3.041e-2

    def evaluate_position_rsun(self, coordinates, time_hours):
        fields = super().evaluate_position_rsun(coordinates, time_hours)
        fields["gas_pressure"] = fields["gas_pressure"] * self.coronal_pressure_pa
        return fields

    def evaluate_position_rsun_normalized(self, coordinates, time_hours):
        fields = self.evaluate_position_rsun(coordinates, time_hours)
        return {
            "temperature": fields["temperature"] / self.temperature_scale_k,
            "gas_pressure": fields["gas_pressure"] / self.gas_pressure_scale_pa,
            "velocity_field": fields["velocity_field"] / self.velocity_scale_m_per_s,
            "magnetic_field": fields["magnetic_field"] / self.magnetic_scale_gauss,
        }

    def mass_density_normalized(self, temperature, gas_pressure):
        return (
            self.thermodynamic_eos.mass_density(
                temperature * self.temperature_scale_k,
                gas_pressure * self.gas_pressure_scale_pa,
            )
            / self.density_scale_kg_m3
        )

    def electron_density_normalized(self, temperature, gas_pressure):
        return (
            self.thermodynamic_eos.electron_density(
                temperature * self.temperature_scale_k,
                gas_pressure * self.gas_pressure_scale_pa,
            )
            / self.electron_density_scale_m3
        )


def evaluate_both_paths(model, energy, dtype):
    """Return the normalized (production) and physical residuals in ``dtype``."""
    model = model.to(dtype)
    energy = energy.to(dtype)
    position = torch.tensor([[1.04e8, 2.0e6, -1.0e6]], dtype=dtype)
    time = torch.full((1, 1), 0.3, dtype=dtype)
    arguments = (model, position, time, 5.0 / 3.0, 782.6)
    keywords = dict(create_graph=False, model_time_scale_s=3600.0)
    return (
        energy._residual_normalized(*arguments, **keywords),
        energy._residual_physical(*arguments, **keywords),
    )


def coronal_model(**kwargs):
    """A 1 MK, coronal-density state where the SI loss underflows float32."""
    return NormalizedAtmosphere(**kwargs)


def test_normalized_path_is_selected_for_the_training_contract(cooling_path):
    energy = CoronalEnergy(options(cooling_path))
    model = coronal_model()
    dispatched = energy.residual(
        model,
        torch.tensor([[1.04e8, 2.0e6, -1.0e6]]),
        torch.full((1, 1), 0.3),
        5.0 / 3.0,
        782.6,
        create_graph=False,
        model_time_scale_s=3600.0,
    )
    normalized, _ = evaluate_both_paths(model, energy, torch.float32)
    torch.testing.assert_close(dispatched, normalized)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"slope": 0.01},
        {"pressure_rate": 0.002, "velocity_rate": 0.003},
        {"slope": 0.01, "pressure_rate": 0.002, "velocity_rate": 0.003},
    ],
)
def test_normalized_and_physical_residuals_agree(kwargs):
    """The model-unit rewrite must not change the physical residual."""
    energy = CoronalEnergy(options(CORONAL_COOLING_RESOURCE))
    reference, _ = evaluate_both_paths(
        coronal_model(**kwargs), energy, torch.float64
    )
    for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 2e-5)):
        normalized, physical = evaluate_both_paths(
            coronal_model(**kwargs), energy, dtype
        )
        assert torch.isfinite(normalized).all(), f"{dtype} normalized residual"
        torch.testing.assert_close(
            normalized.double(), physical.double(), rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            normalized.double(), reference, rtol=tolerance, atol=tolerance
        )


def test_normalized_radiation_survives_float32_range():
    """n_e * n_H * Lambda is ~1e-48 normalized and ~1e43 in its coefficient.

    Only the summed logarithm is representable, so a regression that moves the
    SI density scale out of the exponent produces inf * 0 = NaN here.
    """
    energy = CoronalEnergy(options(CORONAL_COOLING_RESOURCE, heating_w_m3=0.0))
    model = coronal_model()
    normalized, physical = evaluate_both_paths(model, energy, torch.float32)
    assert torch.isfinite(normalized).all()
    # Radiation alone: a positive residual of order unity, not zero.
    assert 0.1 < float(normalized.detach()) < 10.0
    torch.testing.assert_close(
        normalized.double(), physical.double(), rtol=2e-5, atol=2e-5
    )


def test_normalized_residual_has_finite_float32_gradients():
    model = coronal_model(slope=0.01).to(torch.float32)
    energy = CoronalEnergy(options(CORONAL_COOLING_RESOURCE)).to(torch.float32)
    residual = energy._residual_normalized(
        model,
        torch.tensor([[1.04e8, 2.0e6, -1.0e6]]),
        torch.full((1, 1), 0.3),
        5.0 / 3.0,
        782.6,
        create_graph=True,
        model_time_scale_s=3600.0,
    )
    residual.square().mean().backward()
    assert residual.dtype == torch.float32
    assert torch.isfinite(model.slope.grad) and model.slope.grad.abs() > 0
    assert torch.isfinite(model.field.grad).all() and model.field.grad.abs().max() > 0
