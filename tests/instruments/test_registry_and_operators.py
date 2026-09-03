"""Focused public tests for independent LTE instrument operators."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from prom3theus.config import load_config
from prom3theus.instruments import (
    HMIFilterProfiles,
    HinodeSpectralPSF,
    build_instrument,
    get_instrument_registration,
    resolve_instrument_config,
)
from prom3theus.observations import get_observation_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_only_explicit_lte_instrument_and_observation_discriminators_are_registered():
    assert get_instrument_registration("hinode_sp").name == "hinode_sp"
    assert (
        get_instrument_registration("hmi_filter_profiles").name == "hmi_filter_profiles"
    )
    assert get_observation_adapter("hinode_sp").name == "hinode_sp"
    assert get_observation_adapter("hmi_stokes").name == "hmi_stokes"


def test_hinode_operator_preserves_constant_profiles_and_gradients():
    observed = torch.linspace(6301.0, 6301.4, 31, dtype=torch.float64)
    operator = HinodeSpectralPSF(fwhm_angstrom=0.025, oversample=3)
    synthesis = operator.synthesis_grid(observed)
    operator.prepare(synthesis, observed)
    stokes = torch.tensor([1.0, -0.2, 0.03, 0.4], dtype=torch.float64)[None, :, None]
    stokes = stokes.expand(2, 4, synthesis.numel()).clone().requires_grad_()
    result = operator(stokes, synthesis, observed)
    torch.testing.assert_close(
        result,
        torch.tensor([1.0, -0.2, 0.03, 0.4], dtype=torch.float64)[
            None, :, None
        ].expand_as(result),
    )
    result.square().mean().backward()
    assert stokes.grad is not None and torch.isfinite(stokes.grad).all()


def test_hmi_operator_integrates_batch_local_profiles_and_preserves_gradients():
    quadrature = torch.linspace(6173.0, 6173.6, 4, dtype=torch.float64)
    operator = HMIFilterProfiles(
        quadrature_wavelength_angstrom=quadrature,
        inner_half_width_angstrom=0.65,
    )
    observed = torch.linspace(6173.1, 6173.5, 6, dtype=torch.float64)
    synthesis = operator.synthesis_grid(observed)
    stokes = torch.ones(
        2, 4, synthesis.numel(), dtype=torch.float64, requires_grad=True
    )
    spectral = torch.zeros(2, 6, 4, dtype=torch.float64)
    spectral[..., 1] = 0.75
    continuum = torch.full((2, 6), 0.25, dtype=torch.float64)
    result = operator(
        stokes,
        synthesis,
        observed,
        spectral_weights=spectral,
        continuum_weights=continuum,
    )
    expected = torch.full_like(result, 0.75)
    expected[..., 0, :] = 1.0  # only Stokes I receives the unpolarized continuum
    torch.testing.assert_close(result, expected)
    result.square().mean().backward()
    assert stokes.grad is not None and torch.isfinite(stokes.grad).all()

    invalid = spectral.clone()
    invalid[..., 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        operator(
            stokes,
            synthesis,
            observed,
            spectral_weights=invalid,
            continuum_weights=continuum,
        )


def test_public_factory_requires_an_exact_discriminator():
    config = {
        "type": "hinode_sp",
        "spectral_psf": {
            "type": "gaussian",
            "fwhm_angstrom": 0.025,
            "oversample": 4,
            "truncate_sigma": 4.0,
        },
    }
    assert resolve_instrument_config(config) == config
    assert isinstance(build_instrument(config), HinodeSpectralPSF)
    with pytest.raises(ValueError, match="Unknown instrument type"):
        get_instrument_registration("unknown")
    with pytest.raises(ValueError, match="Unknown observation type"):
        get_observation_adapter("unknown")


@pytest.mark.parametrize(
    ("filename", "data_module_name"),
    [
        ("hinode_lte_mhs.yaml", "HinodeDataModule"),
        ("hmi_lte_subframe.yaml", "HMIDataModule"),
    ],
)
def test_shipped_observation_sections_construct_their_exact_adapter(
    filename, data_module_name
):
    config = load_config(PROJECT_ROOT / "configs" / filename)
    observation = config.observation.to_dict()
    adapter = get_observation_adapter(observation.pop("type"))

    data = adapter.build(observation)

    assert type(data).__name__ == data_module_name
