import pytest
import torch

from prom3theus.instruments.aia_euv import AIAEmissionOperator


RESOURCE = "aia_euv_v1:aia_temperature_response"


@pytest.mark.parametrize("density_value", [1.0e20, 1.0e22, 1.0e25])
def test_normalized_table_preserves_dense_float32_temperature_gradients(density_value):
    operator = AIAEmissionOperator((171, 193, 211), response_resource=RESOURCE).float()
    temperature = torch.full((1, 2), 1.0e6, requires_grad=True)
    density = torch.full_like(temperature, density_value, requires_grad=True)
    distance = torch.tensor([0.0, 1.0e7])
    prediction = operator(temperature, density, distance, sample_dim=1)
    gradients = torch.autograd.grad(prediction.sum(), (temperature, density))

    # Independent physical-unit float64 slab oracle, not the scaled integrator.
    reference_operator = AIAEmissionOperator((171, 193, 211), response_resource=RESOURCE)
    reference_t = temperature.detach().double().requires_grad_()
    reference_n = density.detach().double().requires_grad_()
    local = reference_operator.local_response(reference_t) * reference_n.square()[..., None]
    expected = 1.0e-10 * torch.trapezoid(local, distance.double(), dim=1)
    expected_gradients = torch.autograd.grad(expected.sum(), (reference_t, reference_n))
    assert prediction.dtype == torch.float32
    torch.testing.assert_close(prediction, expected.float(), rtol=2e-5, atol=0)
    for actual, reference in zip(gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference.float(), rtol=2e-5, atol=0)


def test_aia_emission_operator_matches_uniform_slab_and_keeps_gradients():
    operator = AIAEmissionOperator((171, 193, 211), response_resource=RESOURCE)
    temperature = torch.full(
        (2, 5), 1.0e6, dtype=torch.float64, requires_grad=True
    )
    electron_density = torch.full(
        (2, 5), 1.0e15, dtype=torch.float64, requires_grad=True
    )
    distance = torch.linspace(0.0, 2.0e6, 5, dtype=torch.float64).expand(2, -1)

    prediction = operator(
        temperature,
        electron_density,
        distance,
        sample_dim=1,
    )
    expected = (
        1.0e-10
        * electron_density[:, 0].square()[:, None]
        * 2.0e6
        * operator.local_response(temperature[:, 0])
    )
    torch.testing.assert_close(prediction, expected)
    prediction.sum().backward()
    assert torch.isfinite(temperature.grad).all()
    assert torch.isfinite(electron_density.grad).all()
    assert electron_density.grad.abs().sum() > 0


def test_aia_operator_zeroes_response_not_out_of_domain_state():
    operator = AIAEmissionOperator((171,), response_resource=RESOURCE)
    temperature = torch.tensor([[1.0e3, 1.0e3]], dtype=torch.float64)
    electron_density = torch.full_like(temperature, 1.0e15)
    prediction = operator(
        temperature,
        electron_density,
        torch.tensor([0.0, 1.0e6], dtype=torch.float64),
        sample_dim=1,
    )
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))
    torch.testing.assert_close(temperature, torch.full_like(temperature, 1.0e3))


def test_dense_float32_aia_samples_and_diagnostics_remain_finite():
    operator = AIAEmissionOperator((171, 193, 211), response_resource=RESOURCE)
    temperature = torch.tensor(
        [[6000.0, 6000.0], [1.0e6, 1.0e6]], requires_grad=True
    )
    density = torch.full_like(temperature, 1.0e20, requires_grad=True)
    distance = torch.tensor([0.0, 1.0e6])
    local = operator.local_emission_measure_integrand(temperature, density)
    prediction = operator(temperature, density, distance, sample_dim=1)
    assert local.dtype == prediction.dtype == torch.float32
    assert torch.isfinite(local).all()
    torch.testing.assert_close(local[0], torch.zeros_like(local[0]))
    torch.testing.assert_close(prediction[0], torch.zeros_like(prediction[0]))
    assert (prediction[1] > 0).all()
    torch.testing.assert_close(prediction, 1.0e-10 * local[:, 0] * 1.0e6)
    prediction.sum().backward()
    assert torch.isfinite(temperature.grad).all()
    assert torch.isfinite(density.grad).all()
