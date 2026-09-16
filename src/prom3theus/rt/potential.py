"""Cartesian half-space potential field, using physical FFT wave numbers.

Equivalent to the alpha=0 Fourier solution in NF2's get_fft_potential_field.
These standalone Cartesian utilities include cube and direct point evaluators.
The inversion boundary objective uses spherical_potential instead.
The source is periodic; zero padding separates its periodic images. Its mean
normal field is retained, with zero mean horizontal field.
"""

import torch


def fft_potential_at_points(normal_field, points, *, spacing, origin, plane_z, chunk_size=128):
    """Return [point,3] fields in source-field units; all lengths use one unit.

    Array axes are x then y. ``origin`` is the center of cell [0,0]. The scalar
    potential is harmonic in the Cartesian half-space above ``plane_z``.
    """
    if normal_field.ndim != 2 or min(normal_field.shape) < 2:
        raise ValueError("Potential source must be a two-dimensional grid")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Potential query points must have shape [N,3]")
    if chunk_size < 1 or any(float(d) <= 0 for d in spacing):
        raise ValueError("Potential spacing and chunk size must be positive")
    if not torch.isfinite(normal_field).all() or not torch.isfinite(points).all():
        raise ValueError("Potential sources and coordinates must be finite")
    if torch.any(points[:, 2] < plane_z):
        raise ValueError("FFT queries must lie above the source plane")
    nx, ny = normal_field.shape
    kx, ky = torch.meshgrid(
        2 * torch.pi * torch.fft.fftfreq(nx, d=float(spacing[0]), device=points.device, dtype=points.dtype),
        2 * torch.pi * torch.fft.fftfreq(ny, d=float(spacing[1]), device=points.device, dtype=points.dtype),
        indexing="ij",
    )
    kx, ky = kx.flatten(), ky.flatten()
    k = (kx.square() + ky.square()).sqrt()
    safe_k = k.clamp_min(torch.finfo(k.dtype).tiny)
    coefficient = torch.fft.fft2(normal_field).flatten() / (nx * ny)
    factors = torch.stack((-1j * kx / safe_k, -1j * ky / safe_k, torch.ones_like(k)), -1)
    result = []
    for query in points.split(chunk_size):
        phase = ((query[:, :1] - origin[0]) * kx + (query[:, 1:2] - origin[1]) * ky)
        decay = -(query[:, 2:3] - plane_z) * k
        modes = torch.exp(decay + 1j * phase) * coefficient
        result.append((modes @ factors).real)
    return torch.cat(result) if result else points.new_empty((0, 3))


def fft_potential_cube(normal_field, *, spacing, heights, chunk_size=8):
    """Compute [z,x,y,component] from one source FFT and inverse FFTs per height.

    Heights are distances above the source plane. Horizontal spacings and heights
    use the same physical units; component units match the source normal field.
    """
    if normal_field.ndim != 2 or min(normal_field.shape) < 2:
        raise ValueError("Potential source must be a two-dimensional grid")
    if heights.ndim != 1 or len(heights) < 2 or chunk_size < 1:
        raise ValueError("Potential cube requires at least two heights and a positive chunk size")
    if not torch.isfinite(normal_field).all() or not torch.isfinite(heights).all() or (heights < 0).any():
        raise ValueError("Potential sources and non-negative heights must be finite")
    if any(float(d) <= 0 for d in spacing):
        raise ValueError("Potential spacing must be positive")
    nx, ny = normal_field.shape
    kx, ky = torch.meshgrid(
        2 * torch.pi * torch.fft.fftfreq(nx, d=float(spacing[0]), device=normal_field.device, dtype=normal_field.dtype),
        2 * torch.pi * torch.fft.fftfreq(ny, d=float(spacing[1]), device=normal_field.device, dtype=normal_field.dtype),
        indexing="ij",
    )
    k = (kx.square() + ky.square()).sqrt()
    safe_k = k.clamp_min(torch.finfo(k.dtype).tiny)
    factors = torch.stack((-1j * kx / safe_k, -1j * ky / safe_k, torch.ones_like(k)))
    coefficients = torch.fft.fft2(normal_field)[None] * factors
    layers = []
    for height in heights.split(chunk_size):
        modes = coefficients[None] * torch.exp(-height[:, None, None, None] * k)
        layers.append(torch.fft.ifft2(modes, dim=(-2, -1)).real.permute(0, 2, 3, 1))
    return torch.cat(layers)


def sample_potential_cube(cube, points, *, origin, spacing):
    """Trilinearly sample a regular [z,x,y,3] cube at physical [x,y,z] points."""
    from torch.nn import functional as F

    if cube.ndim != 4 or cube.shape[-1] != 3 or min(cube.shape[:3]) < 2:
        raise ValueError("Potential cube must have shape [z,x,y,3], with each axis >= 2")
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("Potential query points must have shape [N,3]")
    if not len(points):
        return points.new_empty((0, 3))
    origin = torch.as_tensor(origin, device=points.device, dtype=points.dtype)
    spacing = torch.as_tensor(spacing, device=points.device, dtype=points.dtype)
    extent = spacing * points.new_tensor([cube.shape[1]-1, cube.shape[2]-1, cube.shape[0]-1])
    coordinates = 2 * (points - origin) / extent - 1
    if not torch.isfinite(coordinates).all() or (coordinates.abs() > 1 + 1e-5).any():
        raise ValueError("Potential query lies outside the cube")
    # grid_sample's final coordinate order is width(y), height(x), depth(z).
    grid = coordinates[:, [1, 0, 2]].clamp(-1, 1).reshape(1, 1, 1, -1, 3)
    result = F.grid_sample(cube.permute(3, 0, 1, 2)[None], grid, mode="bilinear", align_corners=True)
    return result[0, :, 0, 0].T
