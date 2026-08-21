import pytest
import torch

from pme.lte.instrument import HinodeSpectralPSF


def test_hinode_spectral_grid_oversamples_observed_grid():
    observed = torch.linspace(6300.8, 6303.2, 112, dtype=torch.float64)
    instrument = HinodeSpectralPSF(oversample=4)
    assert instrument.fwhm_angstrom == 0.025
    synthesis = instrument.synthesis_grid(observed)

    assert synthesis.ndim == 1
    assert synthesis.numel() > observed.numel()
    assert torch.all(synthesis[1:] > synthesis[:-1])
    assert synthesis[0] <= observed[0]
    assert synthesis[-1] >= observed[-1]


def test_hinode_spectral_psf_preserves_constants_and_gradients():
    dtype = torch.float64
    observed = torch.linspace(6300.8, 6303.2, 112, dtype=dtype)
    instrument = HinodeSpectralPSF(oversample=4)
    synthesis = instrument.synthesis_grid(observed)
    channel_values = torch.tensor([1.0, -0.2, 0.03, 0.4], dtype=dtype)
    stokes = channel_values[None, :, None].expand(2, 4, synthesis.numel()).clone()
    stokes.requires_grad_()

    degraded = instrument(stokes, synthesis, observed)
    expected = channel_values[None, :, None].expand(2, 4, observed.numel())
    torch.testing.assert_close(degraded, expected, rtol=2e-12, atol=2e-12)
    degraded.square().mean().backward()
    assert stokes.grad is not None and torch.isfinite(stokes.grad).all()
    assert torch.count_nonzero(stokes.grad) > 0


def test_hinode_spectral_psf_smooths_a_delta():
    dtype = torch.float64
    observed = torch.linspace(6301.9, 6303.0, 112, dtype=dtype)
    instrument = HinodeSpectralPSF(oversample=5)
    synthesis = instrument.synthesis_grid(observed)
    stokes = torch.zeros(1, 4, synthesis.numel(), dtype=dtype)
    stokes[..., synthesis.numel() // 2] = 1.0

    degraded = instrument(stokes, synthesis, observed)
    assert degraded.shape == (1, 4, 112)
    assert torch.isfinite(degraded).all()
    assert torch.all(degraded >= 0)
    assert torch.count_nonzero(degraded[0, 0]) > 1


def test_tabulated_response_is_normalized_and_requires_provenance():
    observed = torch.linspace(6302.0, 6302.4, 21, dtype=torch.float64)
    offsets = (-0.04, -0.02, 0.0, 0.02, 0.04)
    weights = (0.0, 0.15, 1.0, 0.35, 0.0)
    with pytest.raises(ValueError, match="provenance"):
        HinodeSpectralPSF(
            response_offsets_angstrom=offsets,
            response_weights=weights,
        )

    instrument = HinodeSpectralPSF(
        oversample=2,
        response_offsets_angstrom=offsets,
        response_weights=weights,
        response_provenance={"fixture": "synthetic asymmetric response"},
    )
    synthesis = instrument.synthesis_grid(observed)
    constant = torch.ones(2, 4, synthesis.numel(), dtype=torch.float64)
    degraded = instrument(constant, synthesis, observed)
    torch.testing.assert_close(degraded, torch.ones_like(degraded), rtol=1e-12, atol=1e-12)
    assert instrument.metadata()["type"] == "tabulated_spectral_response"
    metadata = instrument.metadata()
    instrument.float()
    assert instrument.metadata() == metadata
