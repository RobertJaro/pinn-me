import torch

from prom3theus.inversion.sampling import SphericalShellDomain


def test_random_grouped_uses_quadratic_radial_sampling_toward_inner_radius():
    torch.manual_seed(4)
    domain = SphericalShellDomain(
        longitude_center_rad=0.0,
        longitude_offset_bounds_rad=(-0.1, 0.1),
        latitude_bounds_rad=(-0.1, 0.1),
        time_bounds_hours=(0.0, 1.0),
        height_bounds_Mm=(-0.1, 20.0),
        solar_radius_m=695_700_000.0,
    )
    samples = domain.random_grouped(16, 2)
    radius = torch.linalg.vector_norm(samples["position_m"][:, 0], dim=-1)
    height = (radius - domain.solar_radius_m) / 1.0e6
    height_fraction = (height - (-0.1)) / (20.0 - (-0.1))
    unwarped_fraction = height_fraction.sqrt()
    lower = torch.arange(16, dtype=height.dtype) / 16

    assert torch.all(unwarped_fraction >= lower)
    assert torch.all(unwarped_fraction < lower + 1 / 16)
    assert height_fraction.mean() < unwarped_fraction.mean()


def test_deterministic_grouped_keeps_uniform_radial_sampling():
    domain = SphericalShellDomain(
        longitude_center_rad=0.0,
        longitude_offset_bounds_rad=(-0.1, 0.1),
        latitude_bounds_rad=(-0.1, 0.1),
        time_bounds_hours=(0.0, 1.0),
        height_bounds_Mm=(0.0, 20.0),
        solar_radius_m=695_700_000.0,
    )

    samples = domain.deterministic_grouped(5, 1)
    radius = torch.linalg.vector_norm(samples["position_m"][:, 0], dim=-1)
    height = (radius - domain.solar_radius_m) / 1.0e6

    torch.testing.assert_close(height, torch.linspace(0.0, 20.0, 5, dtype=height.dtype))


def test_random_sides_balances_four_angular_faces_and_returns_outward_normals():
    torch.manual_seed(8)
    domain = SphericalShellDomain(
        longitude_center_rad=0.2,
        longitude_offset_bounds_rad=(-0.1, 0.15),
        latitude_bounds_rad=(-0.2, 0.1),
        time_bounds_hours=(0.0, 1.0),
        height_bounds_Mm=(0.0, 20.0),
        solar_radius_m=695_700_000.0,
    )

    samples = domain.random_sides(40)
    position = samples["position_m"]
    normal = samples["normal"]
    radial = position / torch.linalg.vector_norm(position, dim=-1, keepdim=True)
    longitude = torch.atan2(position[..., 1], position[..., 0])
    latitude = torch.asin(radial[..., 2])
    longitude_tangent = torch.stack(
        (-torch.sin(longitude), torch.cos(longitude), torch.zeros_like(longitude)),
        dim=-1,
    )
    latitude_tangent = torch.linalg.cross(radial, longitude_tangent, dim=-1)

    assert position.shape == normal.shape == (40, 3)
    assert samples["time_hours"].shape == (40, 1)
    torch.testing.assert_close(
        torch.linalg.vector_norm(normal, dim=-1),
        torch.ones_like(normal[..., 0]),
    )
    torch.testing.assert_close(
        (normal * radial).sum(dim=-1), torch.zeros_like(normal[..., 0])
    )
    torch.testing.assert_close(
        longitude[:10],
        torch.full_like(longitude[:10], 0.1),
    )
    torch.testing.assert_close(
        longitude[10:20],
        torch.full_like(longitude[10:20], 0.35),
    )
    torch.testing.assert_close(
        latitude[20:30],
        torch.full_like(latitude[20:30], -0.2),
    )
    torch.testing.assert_close(
        latitude[30:],
        torch.full_like(latitude[30:], 0.1),
    )
    assert torch.all((normal[:10] * longitude_tangent[:10]).sum(dim=-1) < 0)
    assert torch.all((normal[10:20] * longitude_tangent[10:20]).sum(dim=-1) > 0)
    assert torch.all((normal[20:30] * latitude_tangent[20:30]).sum(dim=-1) < 0)
    assert torch.all((normal[30:] * latitude_tangent[30:]).sum(dim=-1) > 0)


def test_deterministic_sides_are_repeatable_flat_and_cover_radial_domain():
    domain = SphericalShellDomain(
        longitude_center_rad=0.0,
        longitude_offset_bounds_rad=(-0.1, 0.1),
        latitude_bounds_rad=(-0.1, 0.1),
        time_bounds_hours=(0.0, 1.0),
        height_bounds_Mm=(0.0, 20.0),
        solar_radius_m=695_700_000.0,
    )

    first = domain.deterministic_sides(24)
    second = domain.deterministic_sides(24)
    for name in first:
        torch.testing.assert_close(first[name], second[name])
    assert first["position_m"].shape == (24, 3)
    radius = torch.linalg.vector_norm(first["position_m"], dim=-1)
    height = (radius - domain.solar_radius_m) / 1.0e6
    assert torch.all((height >= 0.0) & (height <= 20.0))
    assert height.amin() < 1.0
    assert height.amax() > 15.0


def test_random_samplers_allocate_in_bulk_without_point_or_face_loops(monkeypatch):
    import ast
    import inspect
    import textwrap

    domain = SphericalShellDomain(
        0.0, (-0.1, 0.1), (-0.1, 0.1), (0.0, 1.0), (0.0, 20.0), 695700000.0
    )
    original = torch.rand
    calls = []

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result.numel())
        return result

    monkeypatch.setattr(torch, "rand", counted)
    for name, args in (
        ("random_grouped", (16, 32)),
        ("random_top", (512,)),
        ("random_sides", (512,)),
    ):
        method = getattr(domain, name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        assert not any(
            isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp))
            for node in ast.walk(tree)
        )
        for dtype, expected_calls in ((None, 2), (torch.float32, 1)):
            calls.clear()
            method(*args, dtype=dtype)
            assert len(calls) == expected_calls
            assert min(calls) >= 512
