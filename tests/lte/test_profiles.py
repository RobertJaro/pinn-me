import numpy as np
from scipy.special import wofz
import torch


def _evaluate_profiles(x, damping):
    from pme.lte import profiles

    if hasattr(profiles, "voigt_faraday"):
        result = profiles.voigt_faraday(x, damping)
    else:
        result = profiles.VoigtFaraday()(x, damping)
    if isinstance(result, dict):
        voigt = result.get("voigt", result.get("H", result.get("phi")))
        faraday = result.get("faraday", result.get("F", result.get("psi")))
        return voigt, faraday
    if isinstance(result, (tuple, list)):
        return tuple(result)
    if isinstance(result, torch.Tensor) and torch.is_complex(result):
        return result.real, result.imag
    raise TypeError(f"Unsupported profile return type: {type(result)!r}")


def test_voigt_faraday_matches_faddeeva_reference():
    x = torch.linspace(-8.0, 8.0, 257, dtype=torch.float64)
    damping = torch.logspace(-4, 0, 257, dtype=torch.float64)
    voigt, faraday = _evaluate_profiles(x, damping)
    expected = wofz(x.numpy() + 1j * damping.numpy()) / np.sqrt(np.pi)

    torch.testing.assert_close(voigt, torch.from_numpy(expected.real), rtol=2e-9, atol=2e-11)
    torch.testing.assert_close(faraday, torch.from_numpy(expected.imag), rtol=2e-9, atol=2e-11)


def test_voigt_faraday_symmetries():
    positive = torch.linspace(0.0, 9.0, 181, dtype=torch.float64)
    damping = torch.full_like(positive, 0.15)
    voigt_positive, faraday_positive = _evaluate_profiles(positive, damping)
    voigt_negative, faraday_negative = _evaluate_profiles(-positive, damping)

    torch.testing.assert_close(voigt_negative, voigt_positive, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(faraday_negative, -faraday_positive, rtol=1e-10, atol=1e-11)
    assert torch.all(voigt_positive > 0)


def test_dimensionless_voigt_profile_integrates_to_unity():
    x = torch.linspace(-100.0, 100.0, 20001, dtype=torch.float64)
    voigt, _ = _evaluate_profiles(x, torch.tensor(0.1, dtype=torch.float64))
    integral = torch.trapezoid(voigt, x)
    torch.testing.assert_close(integral, torch.tensor(1.0, dtype=x.dtype), rtol=0, atol=1e-3)


def test_voigt_faraday_has_finite_input_gradients():
    x = torch.tensor([-4.0, -0.2, 0.0, 0.7, 5.0], dtype=torch.float64, requires_grad=True)
    damping = torch.full_like(x, 0.07, requires_grad=True)
    voigt, faraday = _evaluate_profiles(x, damping)
    (voigt.square().sum() + faraday.square().sum()).backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert damping.grad is not None and torch.isfinite(damping.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    assert torch.count_nonzero(damping.grad) > 0
