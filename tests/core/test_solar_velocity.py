import numpy as np
import pytest
import torch

from prom3theus.core import (
    CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
    cartesian_to_spherical,
    carrington_rotation_velocity_cartesian,
    project_cartesian_to_spherical,
)


@pytest.mark.parametrize("backend", (np, torch))
def test_carrington_rotation_is_omega_cross_r(backend):
    values = np.asarray(
        (
            (695_700_000.0, 0.0, 0.0),
            (0.0, 695_700_000.0, 0.0),
            (0.0, 0.0, 695_700_000.0),
        ),
        dtype=np.float32,
    )
    position = torch.from_numpy(values) if backend is torch else values
    velocity = carrington_rotation_velocity_cartesian(position, backend)
    speed = CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S * values[0, 0]
    expected_values = np.asarray(
        ((0.0, speed, 0.0), (-speed, 0.0, 0.0), (0.0, 0.0, 0.0)),
        dtype=np.float32,
    )
    if backend is torch:
        torch.testing.assert_close(velocity, torch.from_numpy(expected_values))
    else:
        np.testing.assert_allclose(velocity, expected_values)


def test_carrington_rotation_is_positive_spherical_phi():
    position = torch.tensor(((695_700_000.0, 0.0, 0.0),), dtype=torch.float32)
    spherical = cartesian_to_spherical(position, torch)
    velocity = carrington_rotation_velocity_cartesian(position, torch)
    spherical_velocity = project_cartesian_to_spherical(velocity, spherical, torch)
    assert spherical_velocity[0, 2] > 1_900.0
    torch.testing.assert_close(spherical_velocity[0, :2], torch.zeros(2))
