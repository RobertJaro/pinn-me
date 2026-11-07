"""Polynomial backward agrees with ordinary autograd, including complex inputs."""

import pytest
import torch

from prom3theus.core.special import polyval


def reference(value, coefficients):
    result = torch.zeros_like(value)
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_fixed_polynomial_matches_values_and_two_derivatives(dtype):
    torch.manual_seed(42)
    value = (.3 * torch.randn(5, dtype=dtype)).requires_grad_()
    coefficients = torch.randn(32, dtype=torch.float64)
    actual, expected = polyval(value, coefficients), reference(value, coefficients)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = [torch.autograd.grad(result.abs().square().sum(), value,
                 create_graph=True, retain_graph=True)[0] for result in (actual, expected)]
    torch.testing.assert_close(*gradients)
    second = [torch.autograd.grad(gradient.abs().square().sum(), value,
              retain_graph=True)[0] for gradient in gradients]
    torch.testing.assert_close(*second)
    assert torch.autograd.gradcheck(lambda v: polyval(v, coefficients), (value,))
    assert torch.autograd.gradgradcheck(lambda v: polyval(v, coefficients), (value,))


def test_trainable_coefficients_retain_their_gradients():
    value = torch.tensor([.2, .7], dtype=torch.float64, requires_grad=True)
    coefficients = torch.tensor([.5, -2., 3.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(polyval, (value, coefficients))
