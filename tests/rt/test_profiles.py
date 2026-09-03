import numpy as np
import pytest
from scipy.special import wofz
import torch

from prom3theus.rt import VoigtFaraday


def test_voigt_faraday_matches_scipy_reference():
    offset = torch.linspace(-8.0, 8.0, 257, dtype=torch.float64)
    damping = torch.logspace(-4, 0, 257, dtype=torch.float64)
    absorption, dispersion = VoigtFaraday()(offset, damping)
    expected = wofz(offset.numpy() + 1j * damping.numpy()) / np.sqrt(np.pi)
    torch.testing.assert_close(
        absorption, torch.from_numpy(expected.real), rtol=2e-9, atol=2e-11
    )
    torch.testing.assert_close(
        dispersion, torch.from_numpy(expected.imag), rtol=2e-9, atol=2e-11
    )


def test_voigt_faraday_symmetry_normalization_and_gradients():
    positive = torch.linspace(0.0, 100.0, 20_001, dtype=torch.float64)
    damping = torch.full_like(positive, 0.1)
    absorption, dispersion = VoigtFaraday()(positive, damping)
    negative_absorption, negative_dispersion = VoigtFaraday()(-positive, damping)
    torch.testing.assert_close(negative_absorption, absorption, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(negative_dispersion, -dispersion, rtol=1e-10, atol=1e-11)
    full_integral = 2.0 * torch.trapezoid(absorption, positive)
    torch.testing.assert_close(
        full_integral, positive.new_tensor(1.0), rtol=0, atol=1e-3
    )

    inputs = torch.tensor(
        [-4.0, -0.2, 0.7, 5.0], dtype=torch.float64, requires_grad=True
    )
    damping_inputs = torch.full_like(inputs, 0.07, requires_grad=True)
    phi, psi = VoigtFaraday()(inputs, damping_inputs)
    (phi.square().sum() + psi.square().sum()).backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert damping_inputs.grad is not None and torch.isfinite(damping_inputs.grad).all()


def test_voigt_faraday_rejects_nonphysical_or_unsafe_numeric_inputs():
    profile = VoigtFaraday()
    with pytest.raises(ValueError, match="non-negative"):
        profile(torch.tensor([0.0]), torch.tensor([-0.1]))
    with pytest.raises(ValueError, match="strictly positive"):
        profile(torch.tensor([0.0]), torch.tensor([0.1]), doppler_width=0.0)
    with pytest.raises(TypeError, match="float32 and float64"):
        profile(
            torch.tensor([0.0], dtype=torch.float16),
            torch.tensor([0.1], dtype=torch.float16),
        )
