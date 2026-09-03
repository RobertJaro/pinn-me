"""Differentiable spherical scene geometry for LTE ray synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _unit(value: torch.Tensor, *, name: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if not torch.isfinite(value).all() or torch.any(norm <= 0):
        raise ValueError(f"{name} must contain finite non-zero vectors.")
    return value / norm


def validate_scene_basis(value: torch.Tensor) -> torch.Tensor:
    """Validate a right-handed orthonormal row basis in solar Cartesian space."""

    basis = torch.as_tensor(value)
    if not basis.is_floating_point():
        basis = basis.to(torch.get_default_dtype())
    if basis.shape != (3, 3) or not torch.isfinite(basis).all():
        raise ValueError("scene_basis must be a finite [3, 3] matrix.")
    if basis.dtype not in (torch.float32, torch.float64):
        raise TypeError("scene_basis must use float32 or float64.")
    # Do not form this tiny Gram matrix with ``basis @ basis.T``. CUDA may use
    # TF32 for that float32 matmul when the application requests high matmul
    # performance, making a valid basis fail this geometry invariant on one GPU
    # while passing on CPU. Elementwise products and reductions retain normal
    # float32 precision on every backend supported by the inversion.
    gram = (basis[:, None, :] * basis[None, :, :]).sum(dim=-1)
    identity = torch.eye(3, dtype=basis.dtype, device=basis.device)
    if not torch.allclose(gram, identity, rtol=0.0, atol=2.0e-5):
        maximum_error = (gram - identity).abs().amax().detach().cpu().item()
        raise ValueError(
            "scene_basis rows must be orthonormal; "
            f"maximum |B B^T - I| is {maximum_error:.3e}."
        )
    orientation = (torch.linalg.cross(basis[0], basis[1], dim=0) * basis[2]).sum()
    if orientation <= 0:
        raise ValueError("scene_basis must be right handed.")
    return basis


def direction_to_chart_mm(
    direction: torch.Tensor,
    scene_basis: torch.Tensor,
    solar_radius_m: torch.Tensor | float,
) -> torch.Tensor:
    """Map solar directions to an observer-independent local gnomonic chart."""

    unit = _unit(direction, name="direction")
    basis = validate_scene_basis(scene_basis).to(unit)
    local = torch.einsum("ij,...j->...i", basis, unit)
    if torch.any(local[..., 2] <= 0):
        raise ValueError(
            "Directions lie outside the visible hemisphere of the scene chart."
        )
    radius = torch.as_tensor(solar_radius_m, dtype=unit.dtype, device=unit.device)
    if not torch.isfinite(radius).all() or torch.any(radius <= 0):
        raise ValueError("solar_radius_m must be finite and strictly positive.")
    return radius * local[..., :2] / local[..., 2:3] / 1.0e6


def chart_to_direction(
    chart_xy_mm: torch.Tensor,
    scene_basis: torch.Tensor,
    solar_radius_m: torch.Tensor | float,
) -> torch.Tensor:
    """Invert :func:`direction_to_chart_mm`."""

    chart = torch.as_tensor(chart_xy_mm)
    if chart.shape[-1] != 2 or not torch.isfinite(chart).all():
        raise ValueError("chart_xy_mm must end in a finite two-vector.")
    basis = validate_scene_basis(scene_basis).to(chart)
    radius = torch.as_tensor(solar_radius_m, dtype=chart.dtype, device=chart.device)
    if not torch.isfinite(radius).all() or torch.any(radius <= 0):
        raise ValueError("solar_radius_m must be finite and strictly positive.")
    local = torch.cat((chart * 1.0e6 / radius, torch.ones_like(chart[..., :1])), dim=-1)
    return _unit(torch.einsum("ij,...j->...i", basis.T, local), name="chart direction")


def chart_height_to_position_m(
    chart_xy_mm: torch.Tensor,
    height_m: torch.Tensor,
    scene_basis: torch.Tensor,
    solar_radius_m: torch.Tensor | float,
) -> torch.Tensor:
    direction = chart_to_direction(chart_xy_mm, scene_basis, solar_radius_m)
    height = torch.as_tensor(height_m, dtype=direction.dtype, device=direction.device)
    if height.shape == direction.shape[:-1] + (1,):
        height = height[..., 0]
    if height.shape != direction.shape[:-1] or not torch.isfinite(height).all():
        raise ValueError("height_m must match the chart leading dimensions.")
    radius = torch.as_tensor(
        solar_radius_m, dtype=direction.dtype, device=direction.device
    )
    if torch.any(radius + height <= 0):
        raise ValueError("solar_radius_m + height_m must remain strictly positive.")
    return direction * (radius + height)[..., None]


def position_to_chart_height(
    position_m: torch.Tensor,
    scene_basis: torch.Tensor,
    solar_radius_m: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    position = torch.as_tensor(position_m)
    if position.shape[-1] != 3 or not torch.isfinite(position).all():
        raise ValueError("position_m must end in a finite three-vector.")
    radial_distance = torch.linalg.vector_norm(position, dim=-1)
    chart = direction_to_chart_mm(position, scene_basis, solar_radius_m)
    radius = torch.as_tensor(
        solar_radius_m, dtype=position.dtype, device=position.device
    )
    if not torch.isfinite(radius).all() or torch.any(radius <= 0):
        raise ValueError("solar_radius_m must be finite and strictly positive.")
    return chart, radial_distance - radius


def intersect_sphere_near_side(
    observer_m: torch.Tensor,
    direction: torch.Tensor,
    radius_m: torch.Tensor | float,
) -> torch.Tensor:
    """Return the near-side distance along ``observer + s * direction``."""

    observer = torch.as_tensor(observer_m)
    ray = _unit(torch.as_tensor(direction).to(observer), name="ray direction")
    radius = torch.as_tensor(radius_m, dtype=observer.dtype, device=observer.device)
    if not torch.isfinite(radius).all() or torch.any(radius <= 0):
        raise ValueError("radius_m must be finite and strictly positive.")
    projection = (observer * ray).sum(dim=-1)
    # Keep the solar-scale impact parameter explicit.  The algebraically
    # equivalent projection**2 - (|observer|**2 - radius**2) subtracts two
    # quantities of order AU**2 and can lose the grazing-ray discriminant.
    impact = torch.linalg.vector_norm(
        torch.linalg.cross(observer.expand_as(ray), ray, dim=-1), dim=-1
    )
    discriminant = radius.square() - impact.square()
    if torch.any(discriminant < 0):
        raise ValueError("At least one ray misses the requested spherical surface.")
    distance = -projection - torch.sqrt(discriminant.clamp_min(0.0))
    if not torch.isfinite(distance).all() or torch.any(distance <= 0):
        raise ValueError("Sphere intersection must lie in front of the observer.")
    return distance


def intersect_sphere_near_side_from_local_point(
    point_rsun: torch.Tensor,
    direction: torch.Tensor,
    radius_rsun: torch.Tensor | float,
) -> torch.Tensor:
    """Return a signed near-side offset from a solar-local point.

    The point may lie inside the requested sphere, so the line parameter may
    be negative. Keeping point and radius in solar-radius units avoids
    subtracting spacecraft-scale distances in float32.
    """

    point = torch.as_tensor(point_rsun)
    ray = _unit(torch.as_tensor(direction).to(point), name="ray direction")
    radius = torch.as_tensor(radius_rsun, dtype=point.dtype, device=point.device)
    if not torch.isfinite(radius).all() or torch.any(radius <= 0):
        raise ValueError("radius_rsun must be finite and strictly positive.")
    projection = (point * ray).sum(dim=-1)
    impact = torch.linalg.vector_norm(
        torch.linalg.cross(point.expand_as(ray), ray, dim=-1), dim=-1
    )
    discriminant = radius.square() - impact.square()
    if torch.any(discriminant < 0):
        raise ValueError(
            "At least one local ray misses the requested spherical surface."
        )
    offset = -projection - torch.sqrt(discriminant.clamp_min(0.0))
    if not torch.isfinite(offset).all():
        raise ValueError("Local sphere intersection must be finite.")
    return offset


def project_vectors_to_stokes(
    vector: torch.Tensor,
    stokes_basis: torch.Tensor,
) -> torch.Tensor:
    """Project global vectors onto row bases ``[+Q, +U, toward observer]``."""

    value = torch.as_tensor(vector)
    basis = torch.as_tensor(stokes_basis, dtype=value.dtype, device=value.device)
    if value.shape[-1] != 3 or basis.shape[-2:] != (3, 3):
        raise ValueError("Vector and Stokes basis must end in [3] and [3, 3].")
    if not torch.isfinite(value).all() or not torch.isfinite(basis).all():
        raise ValueError("Vector and Stokes basis must contain only finite values.")
    while basis.ndim < value.ndim + 1:
        basis = basis.unsqueeze(-3)
    return torch.matmul(basis, value.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True)
class RayTraceResult:
    position_m: torch.Tensor
    distance_m: torch.Tensor
    chart_xy_mm: torch.Tensor
    geometric_height_m: torch.Tensor
