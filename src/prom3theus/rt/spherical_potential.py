"""Exterior spherical Neumann field from piecewise-constant photospheric Br.

The kernel is -grad of the exterior Neumann Green function (equations 3--6
of https://doi.org/10.1093/gji/ggy515), including the monopole. Sources are on
r=1; the solution decays at infinity. No projected/buried source is used.
"""

from functools import lru_cache

import numpy as np
import torch


def spherical_neumann_kernel(points, directions):
    """Return [query, source, xyz] weights per unit source solid angle.

    Coordinates are in solar radii. Only exterior queries (r > 1) are allowed.
    The boundary value is recovered in the limit r -> 1 from above, away from
    discontinuities of the prescribed Br. Use integrated cells, not point
    quadrature, for numerical evaluation close to that surface.
    """
    radius = points.norm(dim=-1, keepdim=True)
    if not torch.isfinite(points).all() or (radius <= 1).any():
        raise ValueError("Spherical potential queries must be strictly above r=1")
    unit = points / radius
    q = radius.reciprocal()[:, :, None]
    u, v = unit[:, None], directions[None]
    delta = v - u
    one_minus_c = delta.square().sum(-1, keepdim=True) / 2
    distance = ((1 - q).square() + 2 * q * one_minus_c).sqrt()
    tangent = delta + one_minus_c * u
    radial = q.square() * (1 - q) * (1 + q) / distance.pow(3)
    horizontal = -q.pow(3) * (
        2 / distance.pow(3) - 1 / (distance * (1 - q + q * one_minus_c + distance))
    )
    return (radial * u + horizontal * tangent) / (4 * torch.pi)


def spherical_source_cells(longitude_bounds, latitude_bounds, grid_size):
    """Equal-solid-angle rectangular cells: [longitude, sin(latitude)] bounds."""
    longitude = np.linspace(*longitude_bounds, grid_size + 1)
    mu = np.linspace(*np.sin(latitude_bounds), grid_size + 1)
    i, j = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    return np.stack((longitude[i], longitude[i + 1], mu[j], mu[j + 1]), -1).reshape(
        -1, 4
    )


def _directions(longitude, mu):
    cosine = np.sqrt(np.maximum(1 - mu**2, 0))
    return np.stack((cosine * np.cos(longitude), cosine * np.sin(longitude), mu), -1)


@lru_cache(maxsize=16)
def _quadrature_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    weights = (weights[:, None] * weights[None, :] / 4).reshape(-1)
    return x, y, weights


def source_cell_quadrature(cells, order):
    """Directions [cell,node,3] and normalized weights [node] for cell means."""
    x, y, weights = _quadrature_rule(order)
    longitude = (cells[:, :1] + cells[:, 1:2]) / 2 + (
        cells[:, 1:2] - cells[:, :1]
    ) * x.reshape(1, -1) / 2
    mu = (cells[:, 2:3] + cells[:, 3:4]) / 2 + (
        cells[:, 3:4] - cells[:, 2:3]
    ) * y.reshape(1, -1) / 2
    return _directions(longitude, mu), weights


def spherical_potential_matrix(
    cells, points, *, quadrature_order=6, resolution_ratio=0.8
):
    """Integrate the Green kernel over each cell, returning [query,xyz,cell].

    Geometry-only CPU float64 quadrature is performed once. Cells near a query
    are recursively subdivided, preserving their parent Br, until quadrature
    resolves the source-query separation. This avoids the false weakening or
    spikes produced by treating photospheric cells as point sources. The
    resulting float32 matrix can be reused on the training device at refreshes.
    """
    cells = np.asarray(cells, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if cells.ndim != 2 or cells.shape[1] != 4 or not np.isfinite(cells).all():
        raise ValueError("Source cells must be finite [N,4] angular bounds")
    if (
        np.any(cells[:, 1] <= cells[:, 0])
        or np.any(cells[:, 3] <= cells[:, 2])
        or np.any(np.abs(cells[:, 2:]) > 1)
    ):
        raise ValueError(
            "Source cells require increasing longitude and sin(latitude) bounds"
        )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or np.any(np.linalg.norm(points, axis=-1) <= 1)
    ):
        raise ValueError("Spherical potential queries must be strictly above r=1")
    if quadrature_order < 2 or not 0 < resolution_ratio <= 1:
        raise ValueError("Potential quadrature requires order >= 2 and ratio in (0,1]")
    matrix = np.zeros((len(points), 3, len(cells)), dtype=np.float32)
    root_directions, root_weights = source_cell_quadrature(cells, quadrature_order)
    for index, point in enumerate(points):
        radius = np.linalg.norm(point)
        unit, q = point / radius, 1 / radius
        pending, parents = cells, np.arange(len(cells))
        integral = np.zeros((len(cells), 3), dtype=np.float64)
        for depth in range(33):
            center = _directions(pending[:, :2].mean(-1), pending[:, 2:].mean(-1))
            distance = np.linalg.norm(point - center, axis=-1)
            # An upper bound on each cell's angular diameter, including poles.
            nearest_mu = np.where(
                pending[:, 2] * pending[:, 3] <= 0,
                0,
                np.minimum(np.abs(pending[:, 2]), np.abs(pending[:, 3])),
            )
            diameter = (
                (pending[:, 1] - pending[:, 0]) * np.sqrt(1 - nearest_mu**2)
                + np.arcsin(pending[:, 3])
                - np.arcsin(pending[:, 2])
            )
            refine = diameter > resolution_ratio * distance
            leaves, leaf_parents = pending[~refine], parents[~refine]
            if len(leaves):
                if depth == 0:
                    directions, weights = root_directions[~refine], root_weights
                else:
                    directions, weights = source_cell_quadrature(
                        leaves, quadrature_order
                    )
                # Evaluate components separately: avoid allocating several large
                # [cell,node,3] temporaries for every query. Retain squared
                # differences (rather than 1-dot) for near-surface accuracy.
                dx = directions[..., 0] - unit[0]
                dy = directions[..., 1] - unit[1]
                dz = directions[..., 2] - unit[2]
                one_minus_c = (dx * dx + dy * dy + dz * dz) / 2
                separation = np.sqrt((1 - q) ** 2 + 2 * q * one_minus_c)
                radial = q**2 * (1 - q) * (1 + q) / separation**3
                horizontal = -(q**3) * (
                    2 / separation**3
                    - 1 / (separation * (1 - q + q * one_minus_c + separation))
                )
                area = (leaves[:, 1] - leaves[:, 0]) * (leaves[:, 3] - leaves[:, 2])
                weighted_horizontal = horizontal * weights[None]
                radial_integral = (
                    (radial + horizontal * one_minus_c) * weights[None]
                ).sum(1)
                value = (
                    np.stack(
                        [
                            radial_integral * unit[k]
                            + (weighted_horizontal * delta).sum(1)
                            for k, delta in enumerate((dx, dy, dz))
                        ],
                        axis=-1,
                    )
                    * (area / (4 * np.pi))[:, None]
                )
                np.add.at(integral, leaf_parents, value)
            if not refine.any():
                break
            if depth == 32:
                raise ValueError(
                    "Potential quadrature cannot resolve a query this close to the photosphere"
                )
            parent = pending[refine]
            lon_mid = parent[:, :2].mean(-1)
            mu_mid = parent[:, 2:].mean(-1)
            pending = np.concatenate(
                [
                    np.stack((a, b, c, d), -1)
                    for a, b in ((parent[:, 0], lon_mid), (lon_mid, parent[:, 1]))
                    for c, d in ((parent[:, 2], mu_mid), (mu_mid, parent[:, 3]))
                ]
            )
            parents = np.tile(parents[refine], 4)
        matrix[index] = integral.T
    return torch.from_numpy(matrix)


def spherical_photosphere_matrix(cells, *, relative_height=1e-4):
    """Exterior surface limit at source-cell centres, [cell,xyz,source].

    Br is imposed exactly by the Neumann data (the identity operator). The
    tangential principal-value integral is evaluated by extrapolating two
    exterior evaluations to r=1. Their numerical heights are 1e-4 and 5e-5 of
    the smallest angular cell width, NOT a displaced physical boundary. Using
    cell centres avoids the discontinuities of piecewise-constant Br.
    """
    cells = np.asarray(cells, dtype=np.float64)
    centres = source_cell_quadrature(cells, 1)[0][:, 0]
    width = min(
        np.min(cells[:, 1] - cells[:, 0]),
        np.min(np.arcsin(cells[:, 3]) - np.arcsin(cells[:, 2])),
    )
    epsilon = float(width * relative_height)
    if not 0 < relative_height <= 0.01 or epsilon < 1e-12:
        raise ValueError(
            "Surface-limit height must be positive and numerically resolvable"
        )
    # Bound temporary float64 allocations: the persistent result is float32.
    matrix = torch.empty(len(cells), 3, len(cells), dtype=torch.float32)
    for start in range(0, len(cells), 64):
        stop = min(start + 64, len(cells))
        points = centres[start:stop]
        far = spherical_potential_matrix(cells, points * (1 + epsilon)).double()
        near = spherical_potential_matrix(cells, points * (1 + epsilon / 2)).double()
        limit = 2 * near - far
        unit = torch.from_numpy(points)
        limit -= unit[:, :, None] * (limit * unit[:, :, None]).sum(1, keepdim=True)
        limit[torch.arange(stop - start), :, torch.arange(start, stop)] += unit
        matrix[start:stop] = limit.float()
    return matrix
