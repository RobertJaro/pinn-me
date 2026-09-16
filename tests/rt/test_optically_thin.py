from __future__ import annotations

import pytest
import torch

from prom3theus.rt.optically_thin import (
    SI_EMISSION_MEASURE_TO_CGS,
    integrate_ne2_response,
    integrate_optically_thin,
)


def test_uniform_ne2_slab_has_exact_si_to_cgs_count_rate():
    # ne = 3e15 m^-3, K = 2e-25 DN s^-1 pix^-1 cm^5, L = 4e6 m.
    # 1e-10 * ne^2 * K * L = 720 DN s^-1 pix^-1.
    response = torch.full((2, 1), 2.0e-25, dtype=torch.float64)
    electron_density = torch.full((2,), 3.0e15, dtype=torch.float64)
    distance = torch.tensor([0.0, 4.0e6], dtype=torch.float64)

    actual = integrate_ne2_response(response, electron_density, distance)

    assert SI_EMISSION_MEASURE_TO_CGS == 1.0e-10
    assert actual.shape == (1,)
    assert actual.item() == pytest.approx(720.0, rel=1.0e-15)


@pytest.mark.parametrize("response_value", [0.0, 2.0e-25])
def test_dense_float32_ne2_quadrature_stays_finite(response_value):
    response = torch.full((2, 1), response_value, requires_grad=True)
    density = torch.full((2,), 1.0e20, requires_grad=True)
    distance = torch.tensor([0.0, 4.0e6])
    assert torch.isinf(density.square()).all()

    actual = integrate_ne2_response(response, density, distance)
    expected = integrate_ne2_response(
        response.detach().double(), density.detach().double(), distance.double()
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float())
    assert torch.isfinite(actual).all()
    actual.sum().backward()
    assert torch.isfinite(density.grad).all()
    assert torch.isfinite(response.grad).all()
    torch.testing.assert_close(
        density.grad.double(),
        1.0e-10 * response.detach().double()[:, 0] * density.detach().double() * 4.0e6,
    )


def test_optically_thin_quadrature_supports_per_ray_distance_and_gradients():
    emissivity = torch.tensor(
        [[[1.0], [2.0], [3.0]], [[2.0], [2.0], [2.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    distance = torch.tensor([[0.0, 1.0, 3.0], [0.0, 2.0, 5.0]], dtype=torch.float64)

    result = integrate_optically_thin(emissivity, distance, sample_dim=-2)
    result.sum().backward()

    assert result[:, 0].tolist() == pytest.approx([6.5, 10.0])
    assert emissivity.grad is not None
    assert torch.isfinite(emissivity.grad).all()


def test_optically_thin_quadrature_rejects_reversed_ray_and_shape_mismatch():
    values = torch.ones((2, 3, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="increase strictly"):
        integrate_optically_thin(
            values,
            torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64),
            sample_dim=-2,
        )
    with pytest.raises(ValueError, match=r"shape\[:-1\]"):
        integrate_ne2_response(
            values,
            torch.ones((2, 2), dtype=torch.float64),
            torch.arange(3, dtype=torch.float64),
        )


def test_line_elements_preserve_close_float64_distances_with_float32_fields():
    from prom3theus.rt.optically_thin import trapezoid_weights

    distance = torch.tensor([1e8, 1e8 + 0.001, 1e8 + 1.0], dtype=torch.float64)
    assert (
        torch.diff(distance.float())[0] == 0
    )  # Downcasting positions loses this layer.
    response = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    density = torch.ones(3)
    actual = integrate_ne2_response(response, density, distance)
    expected = (1e-10 * torch.trapezoid(response.double(), distance, dim=0)).float()
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    weights = trapezoid_weights(distance)
    torch.testing.assert_close(response.grad[:, 0], (1e-10 * weights).float())
    assert torch.all(weights > 0)
    torch.testing.assert_close(weights.sum(), distance[-1] - distance[0])


@pytest.mark.parametrize("sample_dim", [0, 1, -1])
def test_shared_line_elements_integrate_linear_emissivity_exactly(sample_dim):
    from prom3theus.rt.optically_thin import trapezoid_weights

    distance = torch.tensor(
        [[0.0, 0.01, 0.7, 2.0], [0.0, 0.6, 1.9, 3.0]], dtype=torch.float64
    )
    if sample_dim == 0:
        distance = distance.T
    values = 2 + 3 * distance
    weights = trapezoid_weights(distance, sample_dim=sample_dim)
    length = distance.select(sample_dim, distance.shape[sample_dim] - 1)
    actual = integrate_optically_thin(values, distance, sample_dim=sample_dim)
    torch.testing.assert_close(actual, 2 * length + 1.5 * length.square())
    torch.testing.assert_close(weights.sum(sample_dim), length)
