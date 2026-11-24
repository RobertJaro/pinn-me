"""Straight observing rays through a rigid Cartesian MURaM box and solar shells.

Geometry stays float64: subtracting solar-scale positions in transfer precision
would lose native spatial resolution. The cube is never reprojected or warped.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import product
import math

import torch

from .geometry import SOLAR_RADIUS_M


BOUNDARY_TYPES = {0: "miss", 1: "bottom_thermal", 2: "side_vacuum", 3: "shell_vacuum", 4: "top_vacuum"}


@dataclass(frozen=True)
class BoxPlacement:
    bounds_m: torch.Tensor  # [lower/upper, local x/y/z], cell-center support
    basis: torch.Tensor  # rows: local west, north, upward in global Cartesian
    origin_m: torch.Tensor  # global location of local (0,0,0)
    solar_radius_m: float

    @classmethod
    def from_bounds(cls, bounds_m, *, bottom_height_megameters, longitude_deg=0., latitude_deg=0.,
                    solar_radius_m=SOLAR_RADIUS_M):
        bounds = torch.as_tensor(bounds_m, dtype=torch.float64).clone()
        if bounds.shape != (2, 3) or not torch.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
            raise ValueError("bounds_m must contain finite, strictly ordered Cartesian bounds [2,3].")
        if not all(math.isfinite(v) for v in (bottom_height_megameters, longitude_deg, latitude_deg, solar_radius_m)):
            raise ValueError("Volume placement must be finite.")
        if abs(latitude_deg) > 90 or solar_radius_m <= 0 or solar_radius_m + bottom_height_megameters * 1e6 <= 0:
            raise ValueError("Volume placement requires a positive radius and latitude in [-90,90].")
        if bounds[0, 2] != 0:
            raise ValueError("Local z=0 must be the bottom native cell center.")
        lon, lat = math.radians(longitude_deg), math.radians(latitude_deg)
        sl, cl, sb, cb = math.sin(lon), math.cos(lon), math.sin(lat), math.cos(lat)
        basis = torch.tensor([[-sl, cl, 0.], [-sb*cl, -sb*sl, cb], [cb*cl, cb*sl, sb]], dtype=torch.float64)
        origin = (solar_radius_m + bottom_height_megameters * 1e6) * basis[2]
        return cls(bounds, basis, origin, float(solar_radius_m))

    def local_to_global(self, points):
        return torch.as_tensor(points, dtype=torch.float64) @ self.basis + self.origin_m

    def global_to_local(self, points):
        return (torch.as_tensor(points, dtype=torch.float64) - self.origin_m) @ self.basis.T

    def vectors_to_global(self, vectors):
        return torch.as_tensor(vectors, dtype=torch.float64) @ self.basis

    def metadata(self):
        return {
            "geometry": "rigid Cartesian box; no spherical reprojection",
            "length_unit": "m",
            "bounds_m": self.bounds_m.tolist(),
            "origin_global_m": self.origin_m.tolist(),
            "local_basis_rows_global": self.basis.tolist(),
            "local_axes": ["west", "north", "upward"],
            "height_origin": "bottom native cell center",
            "bottom_height_megameters": (float(torch.linalg.vector_norm(self.origin_m)) - self.solar_radius_m) / 1e6,
            "solar_radius_m": self.solar_radius_m,
            "enclosing_shell_height_bounds_megameters": list(self.shell_height_bounds_megameters),
            "support": "native cell-center bounds; no extrapolation",
        }

    @property
    def corners_m(self):
        return torch.stack([self.bounds_m[list(indices), torch.arange(3)] for indices in product((0, 1), repeat=3)])

    @property
    def shell_height_bounds_megameters(self):
        # The nearest radius may occur inside a face, not at a corner.
        center = self.origin_m @ self.basis.T
        nearest = torch.maximum(torch.minimum(-center, self.bounds_m[1]), self.bounds_m[0])
        inner = torch.linalg.vector_norm(center + nearest)
        outer = torch.linalg.vector_norm(self.local_to_global(self.corners_m), dim=-1).max()
        return ((float(inner) - self.solar_radius_m) / 1e6,
                (float(outer) - self.solar_radius_m) / 1e6)


@dataclass(frozen=True)
class RayImage:
    spatial_shape: tuple[int, int]  # y,x
    origins_local_m: torch.Tensor  # flattened y,x pixels
    directions_local: torch.Tensor  # observer toward Sun
    stokes_basis: torch.Tensor  # [west-like, north-like, toward observer]
    pixel_size_m: tuple[float, float]  # detector x,y
    image_x_m: torch.Tensor
    image_y_m: torch.Tensor

    def metadata(self):
        return {
            "projection": "orthographic; parallel straight rays",
            "pixel_order": "YX; flattened y * nx + x",
            "spatial_shape": list(self.spatial_shape),
            "pixel_size_xy_m": list(self.pixel_size_m),
            "image_x_range_m": [float(self.image_x_m[0]), float(self.image_x_m[-1])],
            "image_y_range_m": [float(self.image_y_m[0]), float(self.image_y_m[-1])],
            "stokes_basis_rows_global": self.stokes_basis.tolist(),
            "stokes_basis_convention": "right-handed [+Q,+Q_perpendicular,toward observer]",
            "ray_direction_convention": "observer toward Sun",
            "observer_direction_global": self.stokes_basis[2].tolist(),
        }


def build_ray_image(placement, *, pixel_size_m, observer_direction=(1., 0., 0.)):
    """Cover the projected volume with physical-size orthographic pixels.

observer_direction points from the Sun toward a distant observer. At disk center
the detector lattice equals the original native x/y cell-center lattice.
    """
    dx, dy = map(float, pixel_size_m)
    if not all(math.isfinite(v) and v > 0 for v in (dx, dy)):
        raise ValueError("Pixel sizes must be finite and positive.")
    n = torch.as_tensor(observer_direction, dtype=torch.float64)
    if n.shape != (3,) or not torch.isfinite(n).all() or torch.linalg.vector_norm(n) == 0:
        raise ValueError("observer_direction must be a finite nonzero three-vector.")
    n = n / torch.linalg.vector_norm(n)
    reference = torch.tensor([0., 1., 0.], dtype=torch.float64)
    if abs(float(n @ reference)) > .99:
        reference = torch.tensor([0., 0., 1.], dtype=torch.float64)
    e1 = reference - (reference @ n) * n
    e1 /= torch.linalg.vector_norm(e1)
    e2 = torch.linalg.cross(n, e1)
    basis = torch.stack((e1, e2, n))
    corners = placement.local_to_global(placement.corners_m)
    if not ((corners @ n) > 0).all():
        raise ValueError("The entire box must be on the observer-facing solar hemisphere.")
    projected = corners @ basis.T

    def lattice(axis, spacing):
        lower, upper = projected[:, axis].min(), projected[:, axis].max()
        # Keep ray centers within the projected support. The centered remainder
        # is smaller than half a pixel, so pixel footprints still cover it.
        count = math.floor(float((upper - lower) / spacing) + 1e-10) + 1
        return (torch.arange(count, dtype=torch.float64) - (count - 1) / 2) * spacing + (lower + upper) / 2

    x, y = lattice(0, dx), lattice(1, dy)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    plane = torch.linalg.vector_norm(corners, dim=-1).max() + 1.
    origins = xx.reshape(-1, 1) * e1 + yy.reshape(-1, 1) * e2 + plane * n
    return RayImage((y.numel(), x.numel()), placement.global_to_local(origins),
                    (-n @ placement.basis.T).expand(len(origins), 3).clone(), basis,
                    (dx, dy), x, y)


@dataclass(frozen=True)
class RaySamples:
    local_position_m: torch.Tensor
    global_position_m: torch.Tensor
    distance_m: torch.Tensor  # monotone inward, relative to supported entry
    valid_mask: torch.Tensor
    bottom_boundary_mask: torch.Tensor
    boundary_type: torch.Tensor
    interval_mask: torch.Tensor

    def select(self, indices):
        return type(self)(**{field.name: getattr(self, field.name)[indices] for field in fields(self)})


def sample_ray_batch(placement, origins_local_m, directions_local, *, depth_samples=128,
                     shell_height_bounds_megameters=None):
    """Sample shell-distance nodes and clip them exactly to box support.

The nominal grid is evenly spaced between the near-side spherical intersections.
Clamping exterior nodes to the exact box endpoints creates zero-width padding,
not extrapolated atmosphere. Every positive-width interval stays inside the box.
Rays missing the inner sphere terminate at their tangent plane; only the visible
half of the shell is used. Missed pixels have no valid intervals.
    """
    if type(depth_samples) is not int or depth_samples < 2:
        raise ValueError("depth_samples must be an integer >=2.")
    origin = torch.as_tensor(origins_local_m, dtype=torch.float64)
    direction = torch.as_tensor(directions_local, dtype=torch.float64)
    if origin.ndim != 2 or origin.shape[-1] != 3 or direction.shape != origin.shape:
        raise ValueError("Ray origins and directions must have matching [batch,3] shapes.")
    if not torch.isfinite(origin).all() or not torch.isfinite(direction).all():
        raise ValueError("Ray geometry must be finite.")
    norm = torch.linalg.vector_norm(direction, dim=-1)
    if (norm == 0).any():
        raise ValueError("Ray directions must be nonzero.")
    direction = direction / norm[:, None]
    global_origin = placement.local_to_global(origin)
    global_direction = placement.vectors_to_global(direction)
    if not ((global_origin * global_direction).sum(-1) < 0).all():
        raise ValueError("Rays must point inward from the observer side.")
    heights = torch.as_tensor(placement.shell_height_bounds_megameters if shell_height_bounds_megameters is None else shell_height_bounds_megameters,
                              dtype=torch.float64)
    if heights.shape != (2,) or not torch.isfinite(heights).all() or heights[0] >= heights[1]:
        raise ValueError("Shell heights must be finite and ordered [inner,outer] in megameters.")
    radii = placement.solar_radius_m + heights * 1e6
    if radii[0] <= 0:
        raise ValueError("Shell radii must be positive.")
    projection = (global_origin * global_direction).sum(-1)
    impact2 = torch.linalg.vector_norm(torch.linalg.cross(global_origin, global_direction), dim=-1).square()
    outer_discriminant = radii[1].square() - impact2
    inner_discriminant = radii[0].square() - impact2
    shell_entry = -projection - outer_discriminant.clamp_min(0).sqrt()
    shell_exit = -projection - inner_discriminant.clamp_min(0).sqrt()
    parallel = direction.abs() < 1e-14
    safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
    first = (placement.bounds_m[0] - origin) / safe_direction
    second = (placement.bounds_m[1] - origin) / safe_direction
    lower = torch.minimum(first, second).masked_fill(parallel, -torch.inf)
    upper = torch.maximum(first, second).masked_fill(parallel, torch.inf)
    outside_parallel = (parallel & ((origin < placement.bounds_m[0]) | (origin > placement.bounds_m[1]))).any(-1)
    box_entry, box_exit = lower.max(-1).values, upper.min(-1).values
    entry, exit = torch.maximum(shell_entry, box_entry), torch.minimum(shell_exit, box_exit)
    valid = (outer_discriminant > 0) & ~outside_parallel & torch.isfinite(entry) & torch.isfinite(exit) & (exit > entry)
    entry = torch.where(valid, entry, torch.zeros_like(entry))
    exit = torch.where(valid, exit, entry)
    fraction = torch.linspace(0, 1, depth_samples, dtype=torch.float64)
    nominal = shell_entry[:, None] + (shell_exit - shell_entry)[:, None] * fraction
    distances = torch.maximum(torch.minimum(nominal, exit[:, None]), entry[:, None])
    local = origin[:, None, :] + distances[..., None] * direction[:, None, :]
    # Roundoff at an exact face must not turn valid samples into extrapolation.
    local = torch.maximum(torch.minimum(local, placement.bounds_m[1]), placement.bounds_m[0])
    local = torch.where(valid[:, None, None], local, placement.bounds_m.mean(0))
    endpoint = local[:, -1]
    tolerance = max(1e-5, float(placement.bounds_m.abs().max()) * 1e-12)
    bottom = valid & (endpoint[:, 2] - placement.bounds_m[0, 2]).abs().le(tolerance)
    shell_end = valid & (shell_exit < box_exit - tolerance)
    top = valid & (endpoint[:, 2] - placement.bounds_m[1, 2]).abs().le(tolerance)
    boundary = torch.where(valid, 2, 0).to(torch.uint8)
    boundary[shell_end] = 3
    boundary[top] = 4
    boundary[bottom] = 1
    relative = distances - entry[:, None]
    return RaySamples(local, placement.local_to_global(local), relative, valid, bottom, boundary,
                      valid[:, None] & (relative.diff(dim=-1) > 0))
