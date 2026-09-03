import numpy as np
import pytest
import torch

from prom3theus.core import (
    cartesian_to_spherical,
    cartesian_to_spherical_matrix,
    project_cartesian_to_spherical,
    project_spherical_to_observer,
    spherical_to_cartesian,
    spherical_to_cartesian_matrix,
    spherical_to_observer_matrix,
)


@pytest.mark.parametrize("backend", [np, torch])
def test_coordinate_round_trip_is_exact_at_axes_and_away_from_them(backend):
    values = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.3, -0.4, 0.5),
        ),
        dtype=np.float32,
    )
    cartesian = torch.from_numpy(values) if backend is torch else values
    spherical = cartesian_to_spherical(cartesian, backend)
    recovered = spherical_to_cartesian(spherical, backend)
    if backend is torch:
        torch.testing.assert_close(recovered, cartesian, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(spherical[[2, 3], 1], torch.tensor((0.0, torch.pi)))
    else:
        np.testing.assert_allclose(recovered, cartesian, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(spherical[[2, 3], 1], (0.0, np.pi), rtol=1e-7)


def test_zero_radius_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        cartesian_to_spherical(torch.zeros(1, 3), torch)
    with pytest.raises(ValueError, match="non-zero"):
        cartesian_to_spherical(np.zeros((1, 3)), np)


def test_spherical_vector_basis_is_right_handed_and_invertible_float32():
    spherical = torch.tensor(
        ((1.0, 0.4, -2.0), (1.01, 1.2, 0.7), (0.99, 2.4, 2.8)),
        dtype=torch.float32,
    )
    cartesian_to_spherical_basis = cartesian_to_spherical_matrix(spherical, torch)
    spherical_to_cartesian_basis = spherical_to_cartesian_matrix(spherical, torch)
    identity = torch.eye(3).expand(3, 3, 3)
    torch.testing.assert_close(
        cartesian_to_spherical_basis @ spherical_to_cartesian_basis,
        identity,
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        torch.linalg.det(cartesian_to_spherical_basis),
        torch.ones(3),
        rtol=2e-6,
        atol=2e-6,
    )


def test_spherical_intermediate_matches_direct_supplied_observer_projection():
    position = torch.tensor(
        ((0.2, -0.3, 0.9), (-0.7, 0.4, 0.5), (0.6, 0.7, -0.2)),
        dtype=torch.float32,
    )
    position /= torch.linalg.vector_norm(position, dim=-1, keepdim=True)
    spherical = cartesian_to_spherical(position, torch)
    vector = torch.tensor(
        ((20.0, -3.0, 8.0), (-4.0, 11.0, 2.0), (7.0, 5.0, -9.0)),
        dtype=torch.float32,
    )
    observer_basis = torch.tensor(
        (
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ),
        dtype=torch.float32,
    )
    spherical_vector = project_cartesian_to_spherical(vector, spherical, torch)
    via_spherical = project_spherical_to_observer(
        spherical_vector, spherical, observer_basis, torch
    )
    direct = torch.einsum("...ij,...j->...i", observer_basis, vector)
    torch.testing.assert_close(via_spherical, direct, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        spherical_to_observer_matrix(spherical, observer_basis, torch)
        @ cartesian_to_spherical_matrix(spherical, torch),
        observer_basis,
        rtol=2e-6,
        atol=2e-6,
    )
