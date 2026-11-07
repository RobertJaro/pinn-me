"""Fixed isotropic features and their coordinate derivatives."""

import pytest
import torch
from torch.nn import functional as F

from prom3theus.core.nn import IsotropicPositionalEncoding


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("include_input", [False, True])
def test_features_match_spinnme_block_convention(dynamic, include_input):
    counts, maxima = ([32, 8], [8., 2.]) if dynamic else ([32], [8.])
    encoding = IsotropicPositionalEncoding(3 + dynamic, counts, max_frequencies=maxima,
                                           include_input=include_input, seed=7)
    assert encoding.B.dtype == torch.float32
    # Use double for algebraic parity: packed and separate GEMMs can round
    # large spatial phases differently in float32.
    encoding.double()
    coordinates = torch.randn(2, 5, 3 + dynamic, dtype=torch.float64)
    directions = F.normalize(torch.randn(32, 3, generator=torch.Generator().manual_seed(7)), dim=-1)
    spatial = directions * (2 ** torch.linspace(0, 8, 32) * torch.pi)[:, None]
    phase = F.linear(coordinates[..., :3], spatial.double())
    features = [coordinates] if include_input else []
    features.extend((phase.sin(), phase.cos()))
    if dynamic:
        phase = coordinates[..., 3:] * (2 ** torch.linspace(0, 2, 8) * torch.pi).double()
        features.extend((phase.sin(), phase.cos()))
    expected = torch.cat(features, dim=-1)
    torch.testing.assert_close(encoding(coordinates), expected, rtol=0, atol=1e-12)
    assert encoding.out_dim == expected.shape[-1]
    assert not list(encoding.parameters())


def test_directions_are_reproducible_saved_and_leave_global_rng_unchanged():
    rng = torch.get_rng_state()
    encoding = IsotropicPositionalEncoding(4, [32, 8], max_frequencies=[8., 2.])
    assert torch.equal(rng, torch.get_rng_state())
    same = IsotropicPositionalEncoding(4, [32, 8], max_frequencies=[8., 2.])
    assert torch.equal(encoding.B, same.B)
    restored = IsotropicPositionalEncoding(4, [32, 8], max_frequencies=[8., 2.], seed=1)
    assert not torch.equal(encoding.B, restored.B)
    restored.load_state_dict(encoding.state_dict())
    inputs = torch.randn(3, 4)
    torch.testing.assert_close(encoding(inputs), restored(inputs), rtol=0, atol=0)


def test_coordinate_first_and_second_derivatives():
    # Double precision is only used by this numerical finite-difference check.
    encoding = IsotropicPositionalEncoding(4, [4, 2], max_frequencies=[2., 1.]).double()
    coordinates = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(encoding, (coordinates,))
    assert torch.autograd.gradgradcheck(encoding, (coordinates,))


@pytest.mark.parametrize("options", [
    {"num_frequencies": 0}, {"num_frequencies": True},
    {"num_frequencies": [8, 4]}, {"max_frequencies": float("inf")},
    {"max_frequencies": -1.}, {"max_frequencies": 200.}, {"spatial_dim": 4},
])
def test_invalid_options_fail_during_construction(options):
    with pytest.raises((ValueError, TypeError)):
        IsotropicPositionalEncoding(3, **options)
