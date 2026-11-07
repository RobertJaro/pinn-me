"""The faster recurrence preserves spectra and first/second derivatives."""
import pytest
import torch

from prom3theus.validation.reference_transfer import PolarizedFormalSolver, RayDistancePath


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 1)])
def test_unbound_recurrence_matches_indexed_reference_and_gradients(dtype, batch_shape):
    torch.manual_seed(41)
    depth, wavelengths = 6, 3
    matrix = (.03 * torch.randn(*batch_shape, depth, wavelengths, 4, 4, dtype=dtype)
              + torch.eye(4, dtype=dtype)).requires_grad_()
    source = torch.randn(*batch_shape, depth, wavelengths, 4, dtype=dtype, requires_grad=True)
    distance = torch.linspace(0, 2, depth, dtype=dtype).expand(*batch_shape, depth).clone().requires_grad_()
    boundary = torch.randn(*batch_shape, wavelengths, 4, dtype=dtype, requires_grad=True)
    inputs = (matrix, source, distance, boundary)
    actual = PolarizedFormalSolver()(matrix, source, torch.arange(depth, dtype=dtype),
                                    path=RayDistancePath(distance), bottom_boundary=boundary)
    midpoint = .5 * (matrix[..., :-1, :, :, :] + matrix[..., 1:, :, :, :])
    attenuation = torch.matrix_exp(-midpoint * (distance[..., 1:]-distance[..., :-1])[..., None, None, None])
    mean_source = .5 * (source[..., :-1, :, :] + source[..., 1:, :, :])
    expected = boundary
    for i in range(depth - 2, -1, -1):
        layer = mean_source[..., i, :, :]
        expected = layer + (attenuation[..., i, :, :, :] @ (expected-layer).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs, create_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs, create_graph=True)
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e)
    actual_second = torch.autograd.grad(sum(g.square().sum() for g in actual_grad), inputs)
    expected_second = torch.autograd.grad(sum(g.square().sum() for g in expected_grad), inputs)
    for a, e in zip(actual_second, expected_second):
        torch.testing.assert_close(a, e)
