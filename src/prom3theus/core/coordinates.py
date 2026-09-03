"""Generic Cartesian, spherical, and supplied-observer coordinate transforms.

The spherical convention is ``[radius, colatitude, longitude]`` with vector
components ``[radial, colatitude, longitude]``.  Colatitude increases
southward and longitude follows the right-handed Cartesian ``atan2(y, x)``
direction.  Observer bases are explicit right-handed row matrices; this module
does not encode an instrument, WCS, position angle, or orthographic camera.
"""

from __future__ import annotations

import numpy as np
import torch


def _backend(value, backend=None):
    if backend is not None:
        if backend not in (np, torch):
            raise TypeError("backend must be numpy or torch.")
        return backend
    return torch if isinstance(value, torch.Tensor) else np


def _stack(values, backend):
    return (
        backend.stack(values, dim=-1)
        if backend is torch
        else backend.stack(values, axis=-1)
    )


def _swap_last_two(value, backend):
    return value.transpose(-1, -2) if backend is torch else np.swapaxes(value, -1, -2)


def spherical_to_cartesian(spherical, backend=None):
    """Convert ``[..., radius, colatitude, longitude]`` to Cartesian points."""

    backend = _backend(spherical, backend)
    radius, colatitude, longitude = (
        spherical[..., 0],
        spherical[..., 1],
        spherical[..., 2],
    )
    sin_colatitude = backend.sin(colatitude)
    return _stack(
        (
            radius * sin_colatitude * backend.cos(longitude),
            radius * sin_colatitude * backend.sin(longitude),
            radius * backend.cos(colatitude),
        ),
        backend,
    )


def cartesian_to_spherical(cartesian, backend=None):
    """Convert non-zero Cartesian points to ``[radius, colatitude, longitude]``.

    ``atan2(hypot(x, y), z)`` remains exact at both poles and avoids the
    artificial angular displacement introduced by clamped ``acos(z/r)``.
    Longitude is deterministically zero on the polar axis, where it is
    geometrically undefined.
    """

    backend = _backend(cartesian, backend)
    x, y, z = cartesian[..., 0], cartesian[..., 1], cartesian[..., 2]
    horizontal = (
        backend.sqrt(x.square() + y.square()) if backend is torch else np.hypot(x, y)
    )
    radius = (
        backend.sqrt(horizontal.square() + z.square())
        if backend is torch
        else np.hypot(horizontal, z)
    )
    if backend is torch:
        if torch.any(radius <= 0) or not torch.isfinite(radius).all():
            raise ValueError("Cartesian positions must be finite and non-zero.")
        longitude = torch.atan2(y, x)
        colatitude = torch.atan2(horizontal, z)
    else:
        if np.any(radius <= 0) or not np.isfinite(radius).all():
            raise ValueError("Cartesian positions must be finite and non-zero.")
        longitude = np.arctan2(y, x)
        colatitude = np.arctan2(horizontal, z)
    return _stack((radius, colatitude, longitude), backend)


def cartesian_to_spherical_matrix(spherical, backend=None):
    """Return the row matrix mapping Cartesian vectors to local spherical components."""

    backend = _backend(spherical, backend)
    colatitude, longitude = spherical[..., 1], spherical[..., 2]
    sin_t, cos_t = backend.sin(colatitude), backend.cos(colatitude)
    sin_p, cos_p = backend.sin(longitude), backend.cos(longitude)
    zeros = backend.zeros_like(longitude)
    rows = (
        _stack((sin_t * cos_p, sin_t * sin_p, cos_t), backend),
        _stack((cos_t * cos_p, cos_t * sin_p, -sin_t), backend),
        _stack((-sin_p, cos_p, zeros), backend),
    )
    return backend.stack(rows, dim=-2) if backend is torch else np.stack(rows, axis=-2)


def spherical_to_cartesian_matrix(spherical, backend=None):
    """Return the inverse local-spherical-to-Cartesian vector matrix."""

    backend = _backend(spherical, backend)
    return _swap_last_two(cartesian_to_spherical_matrix(spherical, backend), backend)


def project_cartesian_to_spherical(vector, spherical, backend=None):
    """Project Cartesian vectors into the local ``[r, theta, phi]`` basis."""

    backend = _backend(vector, backend)
    matrix = cartesian_to_spherical_matrix(spherical, backend)
    return backend.einsum("...ij,...j->...i", matrix, vector)


def project_spherical_to_cartesian(vector, spherical, backend=None):
    """Project local spherical vectors into the global Cartesian basis."""

    backend = _backend(vector, backend)
    matrix = spherical_to_cartesian_matrix(spherical, backend)
    return backend.einsum("...ij,...j->...i", matrix, vector)


def spherical_to_observer_matrix(spherical, observer_basis, backend=None):
    """Map local spherical vectors into any supplied Cartesian row basis."""

    backend = _backend(observer_basis, backend)
    spherical_to_cartesian_basis = spherical_to_cartesian_matrix(spherical, backend)
    while observer_basis.ndim < spherical_to_cartesian_basis.ndim:
        observer_basis = (
            observer_basis.unsqueeze(-3)
            if backend is torch
            else np.expand_dims(observer_basis, axis=-3)
        )
    return backend.matmul(observer_basis, spherical_to_cartesian_basis)


def project_spherical_to_observer(vector, spherical, observer_basis, backend=None):
    """Project local spherical vectors into an explicit observer row basis."""

    backend = _backend(vector, backend)
    matrix = spherical_to_observer_matrix(spherical, observer_basis, backend)
    return backend.einsum("...ij,...j->...i", matrix, vector)


__all__ = [
    "cartesian_to_spherical",
    "cartesian_to_spherical_matrix",
    "project_cartesian_to_spherical",
    "project_spherical_to_cartesian",
    "project_spherical_to_observer",
    "spherical_to_cartesian",
    "spherical_to_cartesian_matrix",
    "spherical_to_observer_matrix",
]
