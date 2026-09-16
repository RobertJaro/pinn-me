"""A minimal height-independent magnetic field keeps LTE stratification intact."""

from pathlib import Path

import pytest
import torch

from prom3theus.application.configuration import atmosphere_options
from prom3theus.config import ConfigError, load_config, parse_config
from prom3theus.config.schema import MagneticParameterConfig
from prom3theus.rt import StratifiedAtmosphereModel


ROOT = Path(__file__).resolve().parents[2]
_OMITTED = object()


def _model(reference=_OMITTED, *, dynamic=False, warp=False):
    options = {} if reference is _OMITTED else {"magnetic_reference_height_megameter": reference}
    torch.manual_seed(73)
    return StratifiedAtmosphereModel(
        shell_height_bounds_Mm=(1.5, -0.1),
        time_dependent=dynamic,
        scene_geometry_config={"solar_radius_m": 695_700_000.0, "scene_basis": torch.eye(3)},
        model_config={"type": "siren", "dim": 12, "n_layers": 2,
                      "first_omega_0": 30.0, "hidden_omega_0": 1.0,
                      "radial_weighting_config": {"near_sun_weight": 1.0, "outer_weight": .1,
                                                  "exponent": 1.0} if warp else None},
        **options,
    ).double()


@pytest.mark.parametrize("dynamic,warp", [(False, False), (True, False), (True, True)])
def test_magnetic_reference_freezes_only_magnetic_height_dependence(dynamic, warp):
    legacy = _model(dynamic=dynamic, warp=warp)
    fixed = _model(.15, dynamic=dynamic, warp=warp)
    fixed.load_state_dict(legacy.state_dict(), strict=True)
    coords = torch.tensor([[.3, -.2, .1], [-.6, .4, .7]], dtype=torch.float64)
    heights = torch.tensor([[0., .1e6, .3e6, .5e6]], dtype=torch.float64).expand(2, -1)
    old = legacy.evaluate_at_height(coords, heights)
    new = fixed.evaluate_at_height(coords, heights)
    reference = legacy.evaluate_chart_height_points(coords, torch.full((2,), .15e6, dtype=coords.dtype))
    torch.testing.assert_close(new["magnetic_field"], reference["magnetic_field"][:, None].expand(-1, 4, -1), atol=1e-11, rtol=1e-12)
    assert not torch.allclose(old["magnetic_field"][:, 0], old["magnetic_field"][:, -1])
    for name in ("temperature", "gas_pressure", "microturbulence", "velocity_field"):
        torch.testing.assert_close(new[name], old[name], atol=0, rtol=0)
    assert not torch.allclose(new["temperature"][:, 0], new["temperature"][:, -1])
    assert not torch.allclose(new["gas_pressure"][:, 0], new["gas_pressure"][:, -1])
    assert not torch.allclose(new["velocity_field"][:, 0], new["velocity_field"][:, -1])


def test_all_query_paths_use_the_same_magnetic_reference():
    model = _model(.15, dynamic=True)
    coords = torch.tensor([[.3, -.2, .1], [-.6, .4, .7]], dtype=torch.float64)
    heights = torch.tensor([.5e6, .3e6, .1e6, 0.], dtype=torch.float64)
    cube = model(coords, heights)
    repeated = coords[:, None].expand(-1, 4, -1).reshape(-1, 3)
    paired_height = heights[None].expand(2, -1).reshape(-1)
    paired = model.evaluate_chart_height_points(repeated, paired_height)
    positions = model.position_from_coords_height(repeated, paired_height)
    cartesian = model.evaluate_position_points(positions, repeated[:, 2])
    rsun = model.evaluate_position_rsun(positions / model.solar_radius_m, repeated[:, 2])
    for fields in (paired, cartesian, rsun):
        torch.testing.assert_close(fields["magnetic_field"].reshape(2, 4, 3), cube.magnetic_field,
                                   atol=1e-10, rtol=1e-10)
    surface = model.position_from_coords_height(coords, torch.zeros(2, dtype=coords.dtype))
    radial_ray = -surface / surface.norm(dim=-1, keepdim=True)
    traced, geometry = model.trace_rays(coords, radial_ray, heights)
    torch.testing.assert_close(traced.magnetic_field, cube.magnetic_field, atol=1e-10, rtol=1e-10)
    # An oblique ray may traverse angular magnetic structure. Its values must
    # still equal reference-height queries at its actual angular coordinates.
    oblique_ray = radial_ray + torch.tensor([.08, -.02, 0.], dtype=coords.dtype)
    oblique, geometry = model.trace_rays(coords, oblique_ray, heights)
    query = torch.cat((geometry.chart_xy_mm, coords[:, None, 2:3].expand(-1, 4, -1)), -1)
    expected = model.evaluate_chart_height_points(query, torch.full((2, 4), .15e6, dtype=coords.dtype))
    torch.testing.assert_close(oblique.magnetic_field, expected["magnetic_field"], atol=1e-10, rtol=1e-10)


def test_magnetic_gradients_reach_network_and_angular_time_inputs_but_not_height():
    model = _model(.15, dynamic=True)
    coords = torch.tensor([[.3, -.2, .1], [-.6, .4, .7]], dtype=torch.float64, requires_grad=True)
    heights = torch.tensor([0., .3e6], dtype=torch.float64, requires_grad=True)
    fields = model.evaluate_chart_height_points(coords, heights)
    magnetic_loss = fields["magnetic_field"].square().mean()
    angular_gradient, height_gradient = torch.autograd.grad(magnetic_loss, (coords, heights), retain_graph=True)
    torch.testing.assert_close(height_gradient, torch.zeros_like(heights), atol=0, rtol=0)
    assert torch.isfinite(angular_gradient).all()
    assert (angular_gradient.abs().sum(0) > 0).all()  # x, y and time remain learned.
    temperature_gradient = torch.autograd.grad(fields["temperature"].sum(), heights, retain_graph=True)[0]
    assert torch.isfinite(temperature_gradient).all() and (temperature_gradient != 0).all()
    magnetic_loss.backward()
    assert model.network.out_layer.weight.grad[4:7].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.network.parameters() if p.grad is not None)


def test_omitted_and_null_option_preserve_exact_legacy_predictions_and_state_keys():
    omitted, explicit_null = _model(), _model(None)
    coords = torch.tensor([[.3, -.2, .1]], dtype=torch.float64)
    heights = torch.tensor([[0., .15e6, .3e6]], dtype=torch.float64)
    a, b = omitted.evaluate_at_height(coords, heights), explicit_null.evaluate_at_height(coords, heights)
    assert set(omitted.state_dict()) == set(explicit_null.state_dict()) == set(_model(.15).state_dict())
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
    inputs = omitted._network_inputs_at_height(coords, heights)
    old_raw = omitted.network(inputs.reshape(-1, inputs.shape[-1])).reshape(1, 3, -1)
    old = omitted._decode_raw(old_raw, geometric_height_m=heights)
    for name in a:
        torch.testing.assert_close(a[name], old[name], atol=0, rtol=0)
    assert "magnetic_reference_height_megameter" not in omitted.reference_metadata()


@pytest.mark.parametrize("value,error", [(True, TypeError), (False, TypeError), ("0.15", TypeError),
                                         (float("nan"), ValueError), (float("inf"), ValueError),
                                         (-.2, ValueError), (1.6, ValueError)])
def test_constructor_rejects_invalid_magnetic_reference(value, error):
    with pytest.raises(error, match="magnetic_reference_height_megameter"):
        _model(value)


def test_config_wires_reference_and_keeps_legacy_constructor_options_unchanged():
    config = load_config(ROOT / "configs/hmi_local_minimal.yaml")
    document = config.to_dict()
    magnetic = document["atmosphere"]["parameters"]["magnetic_field"]
    magnetic.pop("reference_height_megameter", None)
    absent = atmosphere_options(parse_config(document, base_directory=ROOT).atmosphere)
    magnetic["reference_height_megameter"] = None
    explicit = atmosphere_options(parse_config(document, base_directory=ROOT).atmosphere)
    assert absent == explicit
    assert "magnetic_reference_height_megameter" not in absent
    magnetic["reference_height_megameter"] = .15
    fixed = parse_config(document, base_directory=ROOT)
    assert atmosphere_options(fixed.atmosphere)["magnetic_reference_height_megameter"] == .15
    assert fixed.to_dict()["atmosphere"]["parameters"]["magnetic_field"]["reference_height_megameter"] == .15
    magnetic["reference_height_megameter"] = 1.6
    with pytest.raises(ConfigError, match="reference_height_megameter.*shell"):
        parse_config(document, base_directory=ROOT)


@pytest.mark.parametrize("value", [True, "0.15", float("nan"), float("inf")])
def test_magnetic_parameter_contract_rejects_nonphysical_reference(value):
    with pytest.raises((TypeError, ValueError), match="reference_height_megameter"):
        MagneticParameterConfig(scale_gauss=1000., reference_height_megameter=value)
