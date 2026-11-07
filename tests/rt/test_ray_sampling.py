"""Physical-region invariants of the production atmospheric ray tracer."""

import pytest
import torch

from prom3theus.rt import StratifiedAtmosphereModel


def _model(dtype):
    return StratifiedAtmosphereModel(
        shell_height_bounds_megameter=(1.5, -1.0),
        scene_geometry_config={
            "solar_radius_m": 695700000.,
            "scene_basis": torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]),
        },
        model_config={
            "type": "mlp", "dim": 8, "n_layers": 2, "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    ).to(dtype=dtype)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-6), (torch.float32, 2.)])
def test_oblique_rays_visit_requested_heights_and_remain_on_pixel_ray(dtype, tolerance):
    model = _model(dtype)
    mu = torch.tensor([1., .8, .5, .3, .1], dtype=dtype)
    coords = torch.tensor([[21., -12., 7.]], dtype=dtype).expand(5, -1)
    radial = model._chart_direction(coords[:, :2])
    tangent = model.scene_basis[0].expand_as(radial)
    tangent = tangent - (tangent * radial).sum(-1, keepdim=True) * radial
    tangent = torch.nn.functional.normalize(tangent, dim=-1)
    ray = -mu[:, None] * radial + (1 - mu.square()).sqrt()[:, None] * tangent
    heights = torch.tensor([1.5e6, 500e3, 0., -300e3, -600e3, -1e6], dtype=dtype)
    atmosphere, trace = model.trace_rays(coords, ray, heights)
    torch.testing.assert_close(
        trace.geometric_height_m, heights.expand(5, -1), atol=tolerance, rtol=0,
    )
    assert torch.all(trace.distance_m[:, 1:] > trace.distance_m[:, :-1])
    assert torch.all(trace.distance_m[:, 0] == 0)
    # Independent Cartesian check; float32 global positions lose ~one ULP at R_sun.
    zero = trace.position_m[:, 2]
    torch.testing.assert_close(zero, radial * model.solar_radius_m, atol=80 if dtype == torch.float32 else 1e-6, rtol=0)
    displacement = trace.position_m - trace.position_m[:, :1]
    torch.testing.assert_close(
        displacement, trace.distance_m[..., None] * ray[:, None],
        atol=150 if dtype == torch.float32 else 1e-6, rtol=0,
    )
    torch.testing.assert_close(atmosphere.geometric_height_m, trace.geometric_height_m)


def test_limb_sampling_stays_above_tangent_without_repeated_or_reversed_points():
    model = _model(torch.float64)
    coords = torch.zeros(2, 3, dtype=torch.float64)
    mu = torch.tensor([.05, .02], dtype=torch.float64)
    rays = torch.stack((-mu, (1 - mu.square()).sqrt(), torch.zeros_like(mu)), -1)
    requested = torch.tensor([1.5e6, 500e3, 0., -500e3, -1e6], dtype=torch.float64)
    _, trace = model.trace_rays(coords, rays, requested)
    tangent_height = model.solar_radius_m * ((1 - mu.square()).sqrt() - 1)
    bottom = tangent_height + model.tangent_margin_m
    fraction = (requested[0] - requested) / (requested[0] - requested[-1])
    expected = requested + fraction * (bottom[:, None] - requested[-1])
    torch.testing.assert_close(trace.geometric_height_m, expected, atol=1e-6, rtol=0)
    assert torch.all(trace.geometric_height_m[:, 1:] < trace.geometric_height_m[:, :-1])
    assert torch.all(trace.distance_m[:, 1:] > trace.distance_m[:, :-1])


def test_height_intersection_derivative_matches_requested_height():
    model = _model(torch.float64)
    height = torch.tensor([1.5e6, 0., -1e6], dtype=torch.float64, requires_grad=True)
    _, trace = model.trace_rays(
        torch.zeros(1, 3, dtype=torch.float64),
        torch.tensor([[-.1, (1 - .1**2)**.5, 0.]], dtype=torch.float64), height,
    )
    gradient, = torch.autograd.grad(trace.geometric_height_m.sum(), height)
    torch.testing.assert_close(gradient, torch.ones_like(height), atol=1e-10, rtol=0)
