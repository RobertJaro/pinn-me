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

    torch.testing.assert_close(
        height, torch.linspace(0.0, 20.0, 5, dtype=height.dtype)
    )


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

    samples = domain.random_sides(5, 8)
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

    assert position.shape == normal.shape == (5, 8, 3)
    assert samples["time_hours"].shape == (5, 8, 1)
    torch.testing.assert_close(
        torch.linalg.vector_norm(normal, dim=-1),
        torch.ones_like(normal[..., 0]),
    )
    torch.testing.assert_close(
        (normal * radial).sum(dim=-1), torch.zeros_like(normal[..., 0])
    )
    torch.testing.assert_close(
        longitude[:, :2],
        torch.full_like(longitude[:, :2], 0.1),
    )
    torch.testing.assert_close(
        longitude[:, 2:4],
        torch.full_like(longitude[:, 2:4], 0.35),
    )
    torch.testing.assert_close(
        latitude[:, 4:6],
        torch.full_like(latitude[:, 4:6], -0.2),
    )
    torch.testing.assert_close(
        latitude[:, 6:],
        torch.full_like(latitude[:, 6:], 0.1),
    )
    assert torch.all((normal[:, :2] * longitude_tangent[:, :2]).sum(dim=-1) < 0)
    assert torch.all((normal[:, 2:4] * longitude_tangent[:, 2:4]).sum(dim=-1) > 0)
    assert torch.all((normal[:, 4:6] * latitude_tangent[:, 4:6]).sum(dim=-1) < 0)
    assert torch.all((normal[:, 6:] * latitude_tangent[:, 6:]).sum(dim=-1) > 0)


def test_deterministic_sides_are_repeatable_and_cover_radial_endpoints():
    domain = SphericalShellDomain(
        longitude_center_rad=0.0,
        longitude_offset_bounds_rad=(-0.1, 0.1),
        latitude_bounds_rad=(-0.1, 0.1),
        time_bounds_hours=(0.0, 1.0),
        height_bounds_Mm=(0.0, 20.0),
        solar_radius_m=695_700_000.0,
    )

    first = domain.deterministic_sides(3, 8)
    second = domain.deterministic_sides(3, 8)
    for name in first:
        torch.testing.assert_close(first[name], second[name])
    radius = torch.linalg.vector_norm(first["position_m"][:, 0], dim=-1)
    height = (radius - domain.solar_radius_m) / 1.0e6
    torch.testing.assert_close(
        height, torch.tensor([0.0, 10.0, 20.0], dtype=height.dtype)
    )
