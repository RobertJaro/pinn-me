"""Characteristic geometry and a fixed, nonsingular polarization chart.

Directions always point along photon propagation. No transport solution is
performed here: a characteristic is just a straight line between two spheres.
"""

from __future__ import annotations

import torch

from prom3theus.core.validation import require_tensor, tensor_validation_enabled

from .geometry import validate_scene_basis


def _scene_components(vector, basis):
    # Small geometry projections need ordinary float32, even when CUDA permits
    # TF32 for the atmosphere network's matrix multiplications.
    return (vector[..., None, :] * basis).sum(dim=-1)


def _chart_from_local_position(local, radius):
    require_tensor(
        lambda: (local[..., 2] > 0).all(),
        'Directions lie outside the visible hemisphere of the scene chart.')
    return (radius / 1e6) * local[..., :2] / local[..., 2:3]


def unit_direction(direction: torch.Tensor) -> torch.Tensor:
    direction = torch.as_tensor(direction)
    if not direction.is_floating_point():
        direction = direction.to(torch.get_default_dtype())
    if direction.shape[-1:] != (3,):
        raise ValueError('Radiation directions must be finite three-vectors.')
    require_tensor(
        lambda: torch.isfinite(direction).all(),
        'Radiation directions must be finite three-vectors.')
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    require_tensor(
        lambda: (norm > 0).all(),
        'Radiation directions must be nonzero.')
    return direction / norm


def canonical_basis(
    direction: torch.Tensor, reference_axis: torch.Tensor | None = None
) -> torch.Tensor:
    """Return right-handed rows [e1,e2,n] for the supplied reference axes.

    Each ray keeps its reference axis through depth. Independent rays may use
    different axes; their emergent Q/U must be rotated into the detector basis.
    """
    n = unit_direction(direction)
    a = n.new_tensor((1.0, 0.0, 0.0)) if reference_axis is None else unit_direction(
        torch.as_tensor(reference_axis).to(n)
    )
    transverse = a - (a * n).sum(-1, keepdim=True) * n
    norm = torch.linalg.vector_norm(transverse, dim=-1, keepdim=True)
    require_tensor(
        lambda: (norm >= 0.001).all(),
        'Direction approaches the polarization-chart singularity.')
    e1 = transverse / norm
    e2 = torch.linalg.cross(n, e1, dim=-1)
    return torch.stack((e1, e2, n), dim=-2)


def rotate_stokes_basis(
    stokes: torch.Tensor, source_basis: torch.Tensor, target_basis: torch.Tensor
) -> torch.Tensor:
    """Passive spin-2 rotation of [...,4,W] into another right-handed basis."""
    if stokes.shape[-2] != 4:
        raise ValueError("Stokes values must have shape [...,4,W].")
    source, target = source_basis.to(stokes), target_basis.to(stokes)
    require_tensor(
        lambda: torch.isclose(source[..., 2, :], target[..., 2, :], atol=2e-05, rtol=0).all(),
        'A basis rotation cannot reverse photon direction.')
    cosine = (target[..., 0, :] * source[..., 0, :]).sum(-1)[..., None]
    sine = (target[..., 0, :] * source[..., 1, :]).sum(-1)[..., None]
    c2, s2 = cosine.square() - sine.square(), 2 * sine * cosine
    intensity, q, u, v = stokes.unbind(-2)
    return torch.stack((intensity, c2 * q + s2 * u, -s2 * q + c2 * u, v), -2)


def shell_characteristic(
    coordinates: torch.Tensor,
    ray_direction: torch.Tensor,
    *,
    scene_basis: torch.Tensor,
    solar_radius_m: float | torch.Tensor,
    height_bounds_m: torch.Tensor | tuple[float, float],
    chart_bounds_megameter: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Trace an outward characteristic from its real lower shell to its exit.

    Coordinates are [gnomonic x/megameter,y/megameter,time/hour] on the nominal solar sphere.
    Input ray direction follows the observation convention (observer to Sun).
    Signed offsets from the nominal sphere preserve thin photospheric shells
    in float32. Use :func:`sample_shell_characteristic` for local chart/height
    coordinates; heights cannot be recovered accurately from global float32
    positions by subtracting the solar radius.
    """
    coords = torch.as_tensor(coordinates)
    if coords.dtype not in (torch.float32, torch.float64):
        coords = coords.float()
    if coords.shape[-1:] != (3,):
        raise ValueError('coordinates must be finite [x/megameter,y/megameter,time/hour].')
    require_tensor(
        lambda: torch.isfinite(coords).all(),
        'coordinates must be finite [x/megameter,y/megameter,time/hour].')
    ray = unit_direction(torch.as_tensor(ray_direction).to(coords))
    if ray.shape != coords.shape:
        raise ValueError("ray_direction must match coordinates.")
    radius = torch.as_tensor(solar_radius_m, device=coords.device, dtype=coords.dtype)
    require_tensor(
        lambda: (torch.isfinite(radius) & (radius > 0)).all(),
        'solar_radius_m must be finite and strictly positive.')
    heights = torch.as_tensor(height_bounds_m, device=coords.device, dtype=coords.dtype)
    if heights.shape != (2,):
        raise ValueError('Radiation shell bounds must be [inner,outer] metres.')
    require_tensor(
        lambda: torch.isfinite(heights).all() & (heights[0] < heights[1]),
        'Radiation shell bounds must be [inner,outer] metres.')
    require_tensor(
        lambda: (radius + heights[0] > 0).all(),
        'Radiation shell must have positive radius.')
    basis = validate_scene_basis(torch.as_tensor(scene_basis).to(coords))
    local = torch.cat((coords[..., :2] * (1e6 / radius), torch.ones_like(coords[..., :1])), dim=-1)
    local_radial = unit_direction(local)
    radial = unit_direction((local_radial[..., :, None] * basis).sum(dim=-2))
    local_ray = _scene_components(ray, basis)
    mu = -(radial * ray).sum(-1)
    require_tensor(
        lambda: (mu > 0).all(),
        'Only outward observed characteristics are supported.')

    def intersection(height):
        delta = (height / radius) * (2 + height / radius)
        discriminant = mu.square() + delta
        require_tensor(
            lambda: (discriminant > 0).all(),
            'Unsupported grazing characteristic: ray does not reach the true lower thermal boundary. A tangent point is not a lower boundary.')
        return -height * (2 + height / radius) / (mu + discriminant.sqrt())

    inner_offset, outer_offset = intersection(heights[0]), intersection(heights[1])
    inflow = radius * radial + inner_offset[..., None] * ray
    outgoing = radius * radial + outer_offset[..., None] * ray
    length = inner_offset - outer_offset
    require_tensor(
        lambda: (length > 0).all(),
        'Radiation characteristic has no positive path length.')
    if chart_bounds_megameter is not None and tensor_validation_enabled():
        bounds = torch.as_tensor(chart_bounds_megameter).to(coords)
        offset_scale = torch.maximum(inner_offset.abs(), outer_offset.abs()).amax() / 1e6
        tolerance = (4 * torch.finfo(coords.dtype).eps * (
            bounds.abs().amax() + offset_scale
        )).clamp_min(1e-8)
        # Gnomonic coordinates along a straight segment are fractional linear;
        # in the supported hemisphere their extrema occur at the endpoints.
        for offset in (inner_offset, outer_offset):
            point = local_radial + (offset / radius)[..., None] * local_ray
            chart = _chart_from_local_position(point, radius)
            require_tensor(
                lambda: ((chart >= bounds[0] - tolerance) & (chart <= bounds[1] + tolerance)).all(),
                'Characteristic crosses the lateral atmosphere boundary; enlarge synthesis lateral_halo_megameter or restrict the observed domain. Lateral Planck/zero inflow is not silently imposed.')
    return {
        "outgoing_position_m": outgoing,
        "inflow_position_m": inflow,
        "direction": -ray,
        "path_length_m": length,
        "time_hours": coords[..., 2],
        "reference_direction": radial,
        "reference_direction_scene": local_radial,
        "ray_direction_scene": local_ray,
        "outgoing_offset_m": outer_offset,
    }


def sample_shell_characteristic(
    geometry: dict[str, torch.Tensor],
    distance_m: torch.Tensor,
    *,
    solar_radius_m: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local chart, height and global position at observer-to-bottom nodes.

    ``distance_m`` has shape ``[..., depth]`` and increases from the shell exit.
    The rationalized height formula retains the local displacement instead of
    subtracting two solar-scale radii. The nominal reference direction is
    treated as unit length, so its floating-point normalization residual is
    never multiplied by the solar radius.
    """
    radial = geometry["reference_direction"]
    ray = -geometry["direction"]
    distance = torch.as_tensor(distance_m, device=radial.device, dtype=radial.dtype)
    radius = torch.as_tensor(solar_radius_m, device=radial.device, dtype=radial.dtype)
    offset = geometry["outgoing_offset_m"][..., None] + distance
    scaled_offset = offset / radius
    projection = (radial * ray).sum(dim=-1)[..., None]
    radial_increment = scaled_offset * (2 * projection + scaled_offset)
    height = offset * (2 * projection + scaled_offset) / (
        torch.sqrt(1 + radial_increment) + 1
    )
    # Retain the nominal chart directly: rotating a global radial vector back
    # into the scene would amplify its float32 roundoff by the solar radius.
    local_position = (geometry["reference_direction_scene"][..., None, :]
        + scaled_offset[..., None] * geometry["ray_direction_scene"][..., None, :])
    chart = _chart_from_local_position(local_position, radius)
    # Global coordinates remain useful for smooth rotation/vector projections,
    # but must not be used to reconstruct these local atmospheric heights.
    position = radius * radial[..., None, :] + offset[..., None] * ray[..., None, :]
    return chart, height, position


__all__ = ["canonical_basis", "rotate_stokes_basis", "sample_shell_characteristic",
           "shell_characteristic", "unit_direction"]
