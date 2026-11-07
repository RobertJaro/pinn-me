"""Independent value and derivative checks for the analytic LTE propagator."""

import pytest
import torch

from prom3theus.rt.polarized_exponential import polarized_matrix_exp


def propagation(parameters):
    diagonal, a, b, c, x, y, z = parameters.unbind(-1)
    return torch.stack((diagonal, a, b, c, a, diagonal, z, -y,
                        b, -z, diagonal, x, c, y, -x, diagonal), -1).reshape(
                            *parameters.shape[:-1], 4, 4)


def test_analytic_propagator_and_two_derivatives_match_general_exponential():
    torch.manual_seed(42)
    parameters = torch.randn(128, 7, dtype=torch.float64)
    parameters[:, 0] = -parameters[:, 1:4].norm(dim=-1) - .2
    parameters *= torch.logspace(-8, 4, 128, dtype=torch.float64)[:, None]
    parameters.requires_grad_()
    matrix = propagation(parameters)
    actual, expected = polarized_matrix_exp(matrix), torch.matrix_exp(matrix)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    gradients = [torch.autograd.grad(value.square().sum(), parameters,
                 create_graph=True, retain_graph=True)[0] for value in (actual, expected)]
    torch.testing.assert_close(*gradients, rtol=1e-9, atol=1e-11)
    second = [torch.autograd.grad(gradient.square().sum(), parameters,
              retain_graph=True)[0] for gradient in gradients]
    torch.testing.assert_close(*second, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("values", [
    [0., 0., 0., 0., 0., 0., 0.],  # vacuum / zero-field degeneracy
    [-2., 1., 0., 0., 0., 1., 0.],  # null generator, nonzero nilpotent part
    [-3., 0., 0., 0., 0., 0., 2.],  # pure rotation
    [-2., 1., 0., 0., 0., 0., 0.],  # pure dichroism
    [-1., .1, .2, .3, .2, -.1, .3],
])
def test_degenerate_propagators_have_finite_first_and_second_derivatives(values):
    parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    function = lambda p: polarized_matrix_exp(propagation(p))
    assert torch.autograd.gradcheck(function, (parameters,))
    assert torch.autograd.gradgradcheck(function, (parameters,))


def test_thick_dichroism_avoids_overflow_and_retains_transmitting_mode():
    parameters = torch.tensor([-1e4, 1e4, 0., 0., 0., 0., 0.], dtype=torch.float64)
    expected = torch.zeros(4, 4, dtype=torch.float64)
    expected[:2, :2] = .5
    torch.testing.assert_close(polarized_matrix_exp(propagation(parameters)), expected,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('values', [
    [0., 0., 0., 0., 0., 0., 0.],
    [-2., 1., 0., 0., 0., 1., 0.],
    [-1., .1, .2, .3, .2, -.1, .3],
])
def test_stable_exponential_increment_matches_two_derivatives(values):
    parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    function = lambda p: polarized_matrix_exp(propagation(p), return_increment=True)[1]
    expected = torch.matrix_exp(propagation(parameters)) - torch.eye(4, dtype=torch.float64)
    torch.testing.assert_close(function(parameters), expected, rtol=1e-11, atol=1e-12)
    assert torch.autograd.gradcheck(function, (parameters,))
    assert torch.autograd.gradgradcheck(function, (parameters,))
