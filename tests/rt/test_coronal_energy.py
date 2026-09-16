import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prom3theus.config.schema import CoronalEnergyConfig
from prom3theus.core import ATOMIC_MASS_UNIT
from prom3theus.inversion.constraints.magnetofluid import MagnetofluidConstraints
from prom3theus.rt.coronal_energy import (
    CoronalCoolingTable,
    CoronalEnergy,
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


def options(path, **kwargs):
    return CoronalEnergyConfig(enabled=True, weight=1.0, cooling_table=path, **kwargs)


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

    path = (
        Path(__file__).resolve().parents[2]
        / "src/prom3theus/resources/sets/plasma/coronal_cooling_v1.json"
    )
    document = json.loads(path.read_text())
    assert document["provenance"]["database_version"] == "11.0.2"
    assert document["provenance"]["fiasco_version"] == "0.8.2"
    components = document["components_w_m3"]
    assert set(components) == {"lines", "free_free", "free_bound", "two_photon"}
    values = torch.tensor(list(components.values()))
    assert torch.isfinite(values).all() and (values >= 0).all()
    assert (values.sum(-1) > 0).all()
    table = CoronalCoolingTable(path)
    temperature = torch.pow(10.0, table.log_temperature).requires_grad_(True)
    actual = table(temperature)
    torch.testing.assert_close(actual, values.sum(0).log10(), rtol=0, atol=2e-5)
    actual.sum().backward()
    assert torch.isfinite(temperature.grad).all()
    assert 1e-37 < 10.0 ** actual[80].item() < 1e-33
    model = Atmosphere(slope=0.01)
    model.thermodynamic_eos = HybridSolarEOS()
    residual = evaluate(model, CoronalEnergy(options(path)))
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
                "cooling_table": tmp_path / "missing.json",
            }
        }
    )
    assert constraints.coronal_energy is None
    assert not constraints.upper_volume_active
    assert not constraints.state_dict()


def test_energy_excludes_low_atmosphere_even_under_no_grad(cooling_path):
    model = Atmosphere()
    constraints = MagnetofluidConstraints(
        {"coronal_energy": options(cooling_path, heating_w_m3=0.0).to_dict()},
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


def test_energy_requires_resource_and_rejects_adiabatic_overlap(cooling_path):
    with pytest.raises(ValueError, match="cooling_table"):
        CoronalEnergyConfig(enabled=True, weight=1.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        MagnetofluidConstraints(
            {
                "coronal_energy": options(cooling_path).to_dict(),
                "adiabatic_pressure": {"enabled": True, "weight": 1.0},
            },
            vector_basis_matches_spatial_coordinates=True,
        )
