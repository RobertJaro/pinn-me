"""Independent polarity and Lorentz-force controls for MURaM Hinode MHS."""

from pathlib import Path

import pytest
import torch
from torch import nn

from prom3theus.application.configuration import atmosphere_options
from prom3theus.config import load_config
from prom3theus.inversion.constraints.magnetofluid import MagnetofluidConstraints
from prom3theus.muram import config as _register_muram_schema  # noqa: F401
from prom3theus.rt import StratifiedAtmosphereModel


class BipolarPotential(nn.Module):
    """A_y = B0 L log(cosh(x/L)) - Bx z, expressed in global Cartesian axes."""

    def __init__(self, model):
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(1500.0, dtype=torch.float64))
        self.radius = model.solar_radius_value_m
        self.length = model.height_input_scale_m
        self.a_scale = (
            model.vector_potential_scale_gauss_m * model.scale_vector_potential
        )
        self.register_buffer("basis", model.scene_basis.detach().clone())

    def forward(self, inputs):
        local = torch.cat(
            (inputs[:, :2] * self.length / self.radius, torch.ones_like(inputs[:, :1])),
            dim=-1,
        )
        local = local / local.norm(dim=-1, keepdim=True)
        local = local * (self.radius + inputs[:, 2:3] * self.length)
        x, _, z = local.unbind(-1)
        ay = self.amplitude * 1.0e7 * torch.log(torch.cosh(x / 1.0e7)) - 300.0 * (
            z - self.radius
        )
        zero = x * 0.0
        potential = torch.stack((zero, ay, zero), dim=-1) @ self.basis
        return torch.cat(
            (
                zero[:, None].expand(-1, 4),
                potential / self.a_scale,
                zero[:, None].expand(-1, 2),
            ),
            dim=-1,
        )


def _model(rotated):
    config = load_config(
        Path(__file__).resolve().parents[2] / "configs/muram/hinode/mhs.yaml"
    )
    options = atmosphere_options(config.atmosphere)
    options.update(magnetic_representation="vector_potential", scale_vector_potential=.01)
    options["model_config"].update(dim=8, n_layers=2)
    # An exact right-handed rotation also places Br along a different global axis.
    basis = (
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        if rotated
        else torch.eye(3)
    )
    model = StratifiedAtmosphereModel(
        **options,
        scene_geometry_config={
            "solar_radius_m": 695700000.0,
            "scene_basis": basis,
        },
    ).double()
    model.network = BipolarPotential(model)
    return model


@pytest.mark.parametrize("rotated", [False, True])
def test_bipolar_photosphere_curl_and_mhs_lorentz_force(rotated):
    model = _model(rotated)
    coords = torch.tensor(
        [[-20.0, -5.0, 0.0], [-8.0, 6.0, 0.0], [8.0, -6.0, 0.0], [20.0, 5.0, 0.0]],
        dtype=torch.float64,
    )
    height = torch.tensor([0.0, 1.0e5, 0.0, 1.0e5], dtype=torch.float64)
    position = model.position_from_coords_height(coords, height)
    local = position @ model.scene_basis.T
    x = local[:, 0]
    bz = model.network.amplitude * torch.tanh(x / 1.0e7)
    expected_b = (
        torch.stack((torch.full_like(x, 300.0), x * 0.0, bz), dim=-1)
        @ model.scene_basis
    )
    curl_y = -model.network.amplitude / 1.0e7 / torch.cosh(x / 1.0e7).square()
    expected_curl = torch.stack((x * 0.0, curl_y, x * 0.0), dim=-1) @ model.scene_basis

    # Radiative transfer uses paired chart/height inputs; physics uses global positions.
    fields = model.evaluate_chart_height_points(coords, height)
    torch.testing.assert_close(
        fields["magnetic_field"], expected_b, rtol=1.0e-7, atol=1.0e-8
    )
    radial_field = (
        fields["magnetic_field"] * position / position.norm(dim=-1, keepdim=True)
    ).sum(-1)
    assert torch.equal(
        radial_field.sign(), torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64)
    )

    constraints = MagnetofluidConstraints(
        {"magnetohydrostatic_equilibrium": {"enabled": True}},
        gravity_m_per_s2=274.0,
        vector_basis_matches_spatial_coordinates=True,
        normalization={"length_m": 1.0e6, "force_balance_pressure_scale_pa": 1.0e4},
    )
    kwargs = {
        "height_group_shape": (4, 1),
        "return_state": True,
        "return_diagnostics": True,
    }
    result = constraints.volume(model, position, coords[:, 2], **kwargs)
    state = result.state
    torch.testing.assert_close(
        state.magnetic_field_gauss, expected_b, rtol=1.0e-7, atol=1.0e-8
    )
    jacobian = state.derivative("magnetic")
    actual_curl = torch.stack(
        (
            jacobian[:, 2, 1] - jacobian[:, 1, 2],
            jacobian[:, 0, 2] - jacobian[:, 2, 0],
            jacobian[:, 1, 0] - jacobian[:, 0, 1],
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual_curl, expected_curl, rtol=1.0e-7, atol=1.0e-14)
    torch.testing.assert_close(
        jacobian.diagonal(dim1=-2, dim2=-1).sum(-1), x * 0.0, atol=1.0e-15, rtol=0.0
    )

    physical = constraints._volume_physical(model, position, coords[:, 2], **kwargs)
    key = "magnetohydrostatic_equilibrium"
    expected_lorentz = torch.linalg.cross(
        expected_curl * 1.0e-4, expected_b * 1.0e-4, dim=-1
    ) / (4.0e-7 * torch.pi)
    torch.testing.assert_close(
        physical.equation_diagnostics[key]["terms"]["−J×B"], -expected_lorentz
    )
    torch.testing.assert_close(
        result.losses[key], physical.losses[key], rtol=1.0e-7, atol=1.0e-10
    )
    (gradient,) = torch.autograd.grad(result.losses[key], model.network.amplitude)
    original = model.network.amplitude.detach().clone()
    values = []
    try:
        for sign in (1, -1):
            with torch.no_grad():
                model.network.amplitude.copy_(original + sign * 0.01)
            values.append(
                constraints.volume(model, position, coords[:, 2], **kwargs)
                .losses[key]
                .detach()
            )
    finally:
        with torch.no_grad():
            model.network.amplitude.copy_(original)
    torch.testing.assert_close(
        gradient, (values[0] - values[1]) / 0.02, rtol=1.0e-7, atol=1.0e-12
    )


def test_hinode_preset_traces_coarse_and_fine_points_in_formation_shell():
    from prom3theus.inversion.lte import LTEWindow

    config = load_config(Path(__file__).resolve().parents[2] / "configs/muram/hinode/mhs.yaml")
    model = _model(rotated=False)
    window_settings = config.synthesis.windows[0].to_dict()
    window_settings.update(coefficient_backend="exact_lte", mapper_checkpoint=None)
    window = LTEWindow(model, configuration=window_settings)
    assert window.depth_refinement.enabled and window.depth_refinement.sample_count == 16
    assert window.jitter_fraction == 1.0
    outer, inner = model.line_formation_height_bounds_megameter
    torch.testing.assert_close(window.height_bounds_m, model.solar_radius_m.new_tensor([inner, outer]) * 1e6)
    geometry = window.boundary_queries(torch.zeros(1, 3), torch.tensor([[0.3, 0., -(1 - 0.3**2)**.5]]))
    samples = window.sample_rays(geometry)
    assert samples.distance_m.shape == (1, 16)
    torch.testing.assert_close(samples.intervals_m, samples.distance_m.diff(dim=-1))
    torch.testing.assert_close(samples.intervals_m.sum(-1), geometry["path_length_m"])


def test_raw_force_and_boundary_curl():
    model = _model(rotated=False)
    constraints = MagnetofluidConstraints(
        {
            name: {"enabled": True}
            for name in (
                "magnetohydrostatic_equilibrium",
                "upper_boundary_current_free",
                "side_boundary_current_free",
            )
        },
        gravity_m_per_s2=274.0,
        vector_basis_matches_spatial_coordinates=True,
        normalization={"length_m": 1e6, "time_s": 3600., "normalize_force_balance": False,
                       "normalize_boundary_current": False},
    )
    coords = torch.tensor([[-8.0, 6.0, 0.0], [20.0, 5.0, 0.0]], dtype=torch.float64)
    time = coords[:, 2]
    position = model.position_from_coords_height(coords, torch.zeros_like(time))
    volume = constraints.volume(
        model, position, time, height_group_shape=(2, 1), return_diagnostics=True
    )
    name = "magnetohydrostatic_equilibrium"
    raw_force = volume.equation_diagnostics[name]["residual"]
    torch.testing.assert_close(volume.losses[name], raw_force.square().mean())

    # For this potential only curl_y is nonzero. Convert G/m to model units.
    curl_y = -model.network.amplitude / 1.0e7 / torch.cosh(position[:, 0] / 1.0e7).square()
    model_curl_y = curl_y * model.height_input_scale_m / model.magnetic_scale_gauss
    expected = model_curl_y.square().mean() / 3
    upper = constraints.upper_boundary(model, position, time)
    normal = torch.zeros_like(position)
    normal[:, 0] = 1
    side = constraints.side_boundary(model, position, time, normal)
    for loss in (upper.losses["upper_boundary_current_free"], side.losses["side_boundary_current_free"]):
        torch.testing.assert_close(loss, expected)
        (gradient,) = torch.autograd.grad(loss, model.network.amplitude)
        torch.testing.assert_close(gradient, 2 * expected / model.network.amplitude)


def test_hinode_volume_collocation_redraws_training_points_and_fixes_validation():
    from prom3theus.application.joint_assembly import _physics_options
    from prom3theus.inversion.assembly import build_physics_assembly
    from prom3theus.inversion.data_terms import PhysicsConstraintTerm
    from prom3theus.inversion.sampling import SphericalShellDomain

    config = load_config(Path(__file__).resolve().parents[2] / "configs/muram/hinode/mhs.yaml")
    assert config.atmosphere.parameters.magnetic_field.representation == "direct"
    assert {name for name, eq in config.physics.equations.to_dict().items() if eq["enabled"]} == {
        "magnetic_divergence", "magnetohydrostatic_equilibrium"}
    geometry = config.atmosphere.geometry
    bounds = (geometry.inner_height_megameter, geometry.outer_height_megameter)
    assert bounds == config.synthesis.windows[0].shell_height_bounds_megameter == (-.5, 1.5)
    domain = SphericalShellDomain(0., (-.1, .1), (-.1, .1), (0., 0.), bounds, 695700000.)
    options = _physics_options(config, domain, 274.)
    assembly = build_physics_assembly(options, reference_gravity_m_per_s2=274.)
    term = PhysicsConstraintTerm(assembly)
    reference = torch.tensor(0.)
    term.set_step(1000)
    first = term._samples(reference)["volume_position_m"]
    second = term._samples(reference)["volume_position_m"]
    assert first.shape == second.shape == (2048, 3)
    assert torch.all(torch.any(first != second, dim=-1))
    assert not term.hard_example_enabled
    term.eval()
    fixed = term._samples(reference)["volume_position_m"]
    assert fixed.shape == (384, 3)
    for points in (first, second, fixed):
        heights = (points.norm(dim=-1) - domain.solar_radius_m) / 1e6
        assert torch.all((bounds[0] <= heights) & (heights <= bounds[1]))
    assert fixed.norm(dim=-1).unique().numel() == 384
    restored = build_physics_assembly(assembly.model_configuration(), reference_gravity_m_per_s2=274.)
    torch.testing.assert_close(fixed, restored.validation_position_m)

    model = _model(rotated=True)
    loss = term.evaluate(model).loss
    (gradient,) = torch.autograd.grad(loss, model.network.amplitude)
    assert torch.isfinite(gradient)
