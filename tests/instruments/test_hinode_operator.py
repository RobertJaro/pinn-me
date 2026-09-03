"""Strict Gaussian-only Hinode spectral response."""

from __future__ import annotations

import pytest
import torch

from prom3theus.instruments.hinode_sp.operator import (
    HinodeSpectralPSF,
    _build_hinode_sp,
)


def test_gaussian_psf_preserves_constant_profiles_and_gradients():
    operator = HinodeSpectralPSF(fwhm_angstrom=0.025, oversample=3, truncate_sigma=4.0)
    observed = torch.linspace(6301.0, 6303.0, 31)
    synthesis = operator.synthesis_grid(observed)
    operator.prepare(synthesis, observed)
    stokes = torch.ones(2, 4, synthesis.numel(), requires_grad=True)
    result = operator(stokes, synthesis, observed)
    torch.testing.assert_close(result, torch.ones_like(result), atol=2e-6, rtol=0)
    result.square().mean().backward()
    assert stokes.grad is not None
    assert torch.isfinite(stokes.grad).all()


def test_factory_accepts_only_complete_gaussian_configuration():
    operator = _build_hinode_sp(
        spectral_psf={
            "type": "gaussian",
            "fwhm_angstrom": 0.025,
            "oversample": 4,
            "truncate_sigma": 4.0,
        },
    )
    assert operator.metadata()["type"] == "normalized_gaussian"

    with pytest.raises(ValueError, match="must be 'gaussian'"):
        _build_hinode_sp(
            spectral_psf={
                "type": "tabulated",
                "fwhm_angstrom": 0.025,
                "oversample": 4,
                "truncate_sigma": 4.0,
            },
        )
    with pytest.raises(TypeError, match="contain exactly"):
        _build_hinode_sp(
            spectral_psf={
                "type": "gaussian",
                "fwhm_angstrom": 0.025,
                "oversample": 4,
                "truncate_sigma": 4.0,
                "response_weights": [1.0],
            },
        )


def test_operator_rejects_a_mismatched_forward_grid():
    operator = HinodeSpectralPSF(fwhm_angstrom=0.025)
    observed = torch.linspace(6301.0, 6303.0, 31)
    synthesis = operator.synthesis_grid(observed)
    operator.prepare(synthesis, observed)
    stokes = torch.ones(1, 4, synthesis.numel())
    with pytest.raises(ValueError, match="do not match"):
        operator(stokes, synthesis + 0.1, observed)
