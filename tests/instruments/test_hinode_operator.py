"""Strict Gaussian-only Hinode spectral response."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from prom3theus.components.forward import _stokes_instrument_options
from prom3theus.config.schema import HinodeInstrumentConfig, SpectralPSFConfig
from prom3theus.instruments.hinode_sp.operator import (
    HinodeSpectralPSF,
    _build_hinode_sp,
)


def test_inversion_uses_its_own_oversampling_for_stored_hinode_profiles():
    instrument = HinodeInstrumentConfig(
        type="hinode_sp", spectral_psf=SpectralPSFConfig(
            type="gaussian", fwhm_angstrom=0.025, oversample=2, truncate_sigma=4.0,
        ),
    )
    stream = SimpleNamespace(data_term=SimpleNamespace(instrument=instrument))
    stored_psf = {**instrument.spectral_psf.to_dict(), "oversample": 4}
    observation = SimpleNamespace(instrument_options={"spectral_psf": stored_psf})

    resolved, *_ = _stokes_instrument_options(stream, observation)
    assert resolved["spectral_psf"] == instrument.spectral_psf.to_dict()
    assert stored_psf["oversample"] == 4

    stored_psf["fwhm_angstrom"] = 0.05
    with pytest.raises(ValueError, match="spectral_psf.*conflicts"):
        _stokes_instrument_options(stream, observation)


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


def test_precise_grid_survives_float_profiles_and_module_dtype_roundtrip():
    operator = HinodeSpectralPSF(fwhm_angstrom=1e-4, oversample=3)
    observed = torch.linspace(6302.5000, 6302.5004, 5, dtype=torch.float64)
    synthesis = operator.synthesis_grid(observed)
    operator.prepare(synthesis, observed)
    profile = torch.exp(-((synthesis - 6302.5002) / 1e-4).square())
    stokes = profile.expand(1, 4, -1).clone()
    expected = operator(stokes, synthesis, observed)
    operator.float()
    actual = operator(stokes.float(), synthesis, observed)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-8)
    operator.double()
    torch.testing.assert_close(operator(stokes, synthesis, observed), expected, rtol=0, atol=0)
