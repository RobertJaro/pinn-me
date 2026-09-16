import pytest
import torch
from prom3theus.rt.potential import fft_potential_at_points


def test_fourier_mode_and_uniform_flux():
    n = 16
    dx, dy = .4, .7
    k = 2 * torch.pi / (n * dx)
    x = torch.arange(n, dtype=torch.float64) * dx
    source = (3 + 2 * torch.cos(k * x))[:, None].expand(n, 12)
    points = torch.tensor([[.2, 1., .3], [1.7, -.2, 2.]], dtype=torch.float64, requires_grad=True)
    field = fft_potential_at_points(source, points, spacing=(dx, dy), origin=(0., 0.), plane_z=0.)
    expected = torch.stack((2 * torch.sin(k * points[:, 0]) * torch.exp(-k * points[:, 2]),
                            points[:, 0] * 0,
                            3 + 2 * torch.cos(k * points[:, 0]) * torch.exp(-k * points[:, 2])), -1)
    torch.testing.assert_close(field, expected)
    jac = torch.stack([torch.autograd.grad(field[:, i].sum(), points, retain_graph=True)[0] for i in range(3)], 1)
    torch.testing.assert_close(jac.diagonal(dim1=1, dim2=2).sum(-1), torch.zeros(2, dtype=points.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(jac, jac.transpose(1, 2), atol=1e-12, rtol=0)
    shifted = fft_potential_at_points(source, points + points.new_tensor([4., 5., 6.]), spacing=(dx, dy), origin=(4., 5.), plane_z=6.)
    torch.testing.assert_close(field, shifted)


def test_rejects_downward_extrapolation():
    with pytest.raises(ValueError, match='above'):
        fft_potential_at_points(torch.ones(4, 4), torch.tensor([[0., 0., -1.]]), spacing=(1., 1.), origin=(0., 0.), plane_z=0.)


def test_cube_matches_direct_fourier_solution_at_grid_nodes():
    from prom3theus.rt.potential import fft_potential_cube, sample_potential_cube
    nx, ny, nz = 12, 10, 7
    dx, dy, dz = .4, .7, .3
    x, y = torch.meshgrid(torch.arange(nx, dtype=torch.float64) * dx,
                          torch.arange(ny, dtype=torch.float64) * dy, indexing='ij')
    source = 3 + torch.cos(2 * torch.pi * x / (nx * dx)) + .7 * torch.sin(4 * torch.pi * y / (ny * dy))
    heights = torch.arange(nz, dtype=torch.float64) * dz
    cube = fft_potential_cube(source, spacing=(dx, dy), heights=heights, chunk_size=2)
    z, xx, yy = torch.meshgrid(heights, x[:, 0], y[0], indexing='ij')
    origin = torch.tensor([2., -3., 4.], dtype=torch.float64)
    queries = torch.stack((xx, yy, z), -1).reshape(-1, 3) + origin
    expected = fft_potential_at_points(source, queries, spacing=(dx, dy), origin=origin[:2], plane_z=origin[2])
    torch.testing.assert_close(cube.reshape(-1, 3), expected)
    extracted = sample_potential_cube(cube, queries, origin=origin, spacing=(dx, dy, dz))
    torch.testing.assert_close(extracted, expected)
    # The bottom normal component reproduces the source, including its mean.
    torch.testing.assert_close(cube[0, :, :, 2], source)


def test_cube_trilinear_sampling_uses_physical_axis_order():
    from prom3theus.rt.potential import sample_potential_cube
    origin = torch.tensor([10., -5., 20.], dtype=torch.float64)
    spacing = torch.tensor([2., 3., 4.], dtype=torch.float64)
    z, x, y = torch.meshgrid(torch.arange(4), torch.arange(5), torch.arange(6), indexing='ij')
    positions = torch.stack((x, y, z), -1) * spacing + origin
    def field(p):
        return torch.stack((2 * p[..., 0] + p[..., 2], 3 * p[..., 1], p[..., 0] - p[..., 1] + p[..., 2]), -1)
    queries = torch.tensor([[11., -3., 22.], [16.1, 7.2, 28.5]], dtype=torch.float64)
    torch.testing.assert_close(sample_potential_cube(field(positions), queries, origin=origin, spacing=spacing), field(queries))
    with pytest.raises(ValueError, match='outside'):
        sample_potential_cube(field(positions), queries + 100, origin=origin, spacing=spacing)
