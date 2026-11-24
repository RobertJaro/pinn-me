"""Physical Cartesian box placement and bounded spherical ray sampling."""

import json

import pytest
import torch

from prom3theus.muram.ray_geometry import BoxPlacement, build_ray_image, sample_ray_batch


def placement(**kwargs):
    return BoxPlacement.from_bounds([[-2e5, -1e5, 0.], [2e5, 1e5, 3e6]], bottom_height_megameters=0., **kwargs)


def test_rigid_placement_preserves_lengths_and_vectors():
    box = placement(longitude_deg=35., latitude_deg=-20.)
    points = torch.tensor([[1e5, -1e5, 0.], [-2e5, 0., 3e6]], dtype=torch.float64)
    global_points = box.local_to_global(points)
    torch.testing.assert_close(box.global_to_local(global_points), points, atol=1e-7, rtol=0)
    torch.testing.assert_close(torch.linalg.vector_norm(global_points.diff(dim=0), dim=-1),
                               torch.linalg.vector_norm(points.diff(dim=0), dim=-1))
    torch.testing.assert_close(box.basis @ box.basis.T, torch.eye(3, dtype=torch.float64))
    assert torch.linalg.det(box.basis).item() == pytest.approx(1.)
    torch.testing.assert_close(box.vectors_to_global(torch.tensor([0., 0., 1.])), box.basis[2])


def test_disk_center_lattice_matches_native_centers_and_exact_faces():
    box = placement()
    image = build_ray_image(box, pixel_size_m=(1e5, 1e5))
    assert image.spatial_shape == (3, 5)
    samples = sample_ray_batch(box, image.origins_local_m, image.directions_local)
    assert samples.local_position_m.shape == (15, 128, 3)
    assert samples.valid_mask.all()
    assert samples.bottom_boundary_mask.all()
    assert (samples.boundary_type == 1).all()
    torch.testing.assert_close(samples.local_position_m[:, 0, 2], torch.full((15,), 3e6, dtype=torch.float64))
    torch.testing.assert_close(samples.local_position_m[:, -1, 2], torch.zeros(15, dtype=torch.float64))
    xx, yy = torch.meshgrid(image.image_x_m, image.image_y_m, indexing="xy")
    torch.testing.assert_close(samples.local_position_m[..., 0], xx.reshape(-1, 1).expand(-1, 128))
    torch.testing.assert_close(samples.local_position_m[..., 1], yy.reshape(-1, 1).expand(-1, 128))
    assert (samples.distance_m.diff(dim=-1) >= 0).all()
    selected = samples.select(torch.tensor([1, 4]))
    assert selected.local_position_m.shape == (2, 128, 3)
    torch.testing.assert_close(selected.distance_m, samples.distance_m[[1, 4]])
    assert json.loads(json.dumps(box.metadata()))["height_origin"] == "bottom native cell center"
    assert json.loads(json.dumps(image.metadata()))["spatial_shape"] == [3, 5]


def test_inclined_rays_change_horizontal_position_and_flag_side_inflow():
    box = placement(longitude_deg=35.)
    image = build_ray_image(box, pixel_size_m=(1e5, 1e5))
    samples = sample_ray_batch(box, image.origins_local_m, image.directions_local, depth_samples=64)
    assert samples.valid_mask.any()
    assert (samples.boundary_type == 2).any()
    valid = samples.local_position_m[samples.valid_mask]
    assert ((valid[:, -1, 0] - valid[:, 0, 0]).abs() > 1e3).all()
    assert (valid >= box.bounds_m[0]).all() and (valid <= box.bounds_m[1]).all()
    assert torch.all(samples.distance_m[samples.valid_mask, -1] > 0)


def test_nondivisible_pixel_spacing_keeps_ray_centers_inside_box():
    box = BoxPlacement.from_bounds([[-1.5e5, -2.5e5, 0.], [1.5e5, 2.5e5, 3e6]],
                                   bottom_height_megameters=0.)
    image = build_ray_image(box, pixel_size_m=(2e5, 2e5))
    assert image.spatial_shape == (3, 2)
    torch.testing.assert_close(image.image_x_m, torch.tensor([-1e5, 1e5], dtype=torch.float64))
    torch.testing.assert_close(image.image_y_m, torch.tensor([-2e5, 0., 2e5], dtype=torch.float64))
    samples = sample_ray_batch(box, image.origins_local_m, image.directions_local)
    assert samples.valid_mask.all()
    assert samples.bottom_boundary_mask.all()


def test_clipped_shell_nodes_integrate_only_box_and_retain_endpoints():
    box = placement()
    image = build_ray_image(box, pixel_size_m=(1e5, 1e5))
    # A huge shell and only two nominal nodes still resolve the thin box exactly.
    samples = sample_ray_batch(box, image.origins_local_m, image.directions_local,
                               depth_samples=2, shell_height_bounds_megameters=(-5., 100.))
    torch.testing.assert_close(samples.distance_m[:, -1], torch.full((15,), 3e6, dtype=torch.float64))
    assert samples.interval_mask.all()
    assert samples.bottom_boundary_mask.all()


def test_shell_cutoff_inside_box_has_vacuum_boundary():
    box = placement()
    image = build_ray_image(box, pixel_size_m=(1e5, 1e5))
    samples = sample_ray_batch(box, image.origins_local_m, image.directions_local,
                               shell_height_bounds_megameters=(1., 2.))
    assert samples.valid_mask.all()
    assert not samples.bottom_boundary_mask.any()
    assert (samples.boundary_type == 3).all()
    radii = torch.linalg.vector_norm(samples.global_position_m, dim=-1)
    torch.testing.assert_close(radii[:, 0], torch.full((15,), box.solar_radius_m + 2e6, dtype=torch.float64))
    torch.testing.assert_close(radii[:, -1], torch.full((15,), box.solar_radius_m + 1e6, dtype=torch.float64))


def test_missed_box_has_zero_intervals_and_no_thermal_boundary():
    box = placement()
    origins = torch.tensor([[1e7, 0., 1e7]], dtype=torch.float64)
    samples = sample_ray_batch(box, origins, torch.tensor([[0., 0., -1.]]))
    assert not samples.valid_mask.any()
    assert not samples.interval_mask.any()
    assert not samples.bottom_boundary_mask.any()
    assert (samples.distance_m == 0).all()
    assert torch.isfinite(samples.local_position_m).all()


def test_off_limb_inner_sphere_miss_uses_near_side_box():
    box = BoxPlacement.from_bounds([[-1e5, -1e5, 0.], [1e5, 1e5, 3e6]],
                                   bottom_height_megameters=10., longitude_deg=85.)
    image = build_ray_image(box, pixel_size_m=(1e5, 1e5))
    samples = sample_ray_batch(box, image.origins_local_m, image.directions_local,
                               shell_height_bounds_megameters=(0., 20.))
    assert samples.valid_mask.any()
    positions = samples.global_position_m[samples.valid_mask]
    assert (positions[..., 0] > 0).all()
    assert (torch.linalg.vector_norm(positions, dim=-1) > box.solar_radius_m).all()


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError, match="observer-facing"):
        build_ray_image(placement(longitude_deg=120.), pixel_size_m=(1e5, 1e5))
    with pytest.raises(ValueError, match="ordered"):
        BoxPlacement.from_bounds([[0., 0., 0.], [1., 0., 1.]], bottom_height_megameters=0.)
    image = build_ray_image(placement(), pixel_size_m=(1e5, 1e5))
    with pytest.raises(ValueError, match="depth_samples"):
        sample_ray_batch(placement(), image.origins_local_m, image.directions_local, depth_samples=1)
