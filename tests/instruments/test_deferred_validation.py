"""Instrument forwards keep validity flags on device until the pipeline boundary."""
from unittest.mock import patch

import pytest
import torch

from prom3theus.core.validation import deferred_validation
from prom3theus.instruments.hinode_sp.operator import HinodeSpectralPSF
from prom3theus.instruments.hmi.operator import HMIFilterProfiles


@pytest.mark.parametrize('instrument', ['hinode', 'hmi'])
def test_prepared_instrument_does_not_read_tensor_booleans_inside_forward(instrument):
    if instrument == 'hinode':
        operator = HinodeSpectralPSF(fwhm_angstrom=.025)
        observed = torch.linspace(6301., 6301.4, 31, dtype=torch.float64)
        synthesis = operator.synthesis_grid(observed)
        operator.prepare(synthesis, observed)
        response = {}
    else:
        operator = HMIFilterProfiles(torch.linspace(6173., 6173.6, 4, dtype=torch.float64),
                                     inner_half_width_angstrom=.65, magnetic_azimuth_offset_deg=90.)
        observed = torch.linspace(6173.1, 6173.5, 6, dtype=torch.float64)
        synthesis = operator.synthesis_grid(observed)
        response = dict(spectral_weights=torch.full((2, 6, 4), .25, dtype=torch.float64),
                        continuum_weights=torch.zeros(2, 6, dtype=torch.float64))
    stokes = torch.ones(2, 4, synthesis.numel(), dtype=torch.float64, requires_grad=True)
    expected = operator(stokes, synthesis, observed, **response)
    with deferred_validation():
        with patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host scalar read')):
            actual = operator(stokes, synthesis, observed, **response)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.isfinite(stokes.grad).all()
    invalid = stokes.detach().clone()
    invalid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match='finite real floating-point'):
        with deferred_validation():
            operator(invalid, synthesis, observed, **response)
