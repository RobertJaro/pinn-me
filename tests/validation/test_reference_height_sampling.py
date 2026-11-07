"""Physical-height coverage independent of learned/refinement opacity."""

from types import SimpleNamespace

import pytest
import torch

from prom3theus.validation.reference_configuration import resolve_depth_sampling
from prom3theus.inversion.depth_sampling import reference_height_grid


def test_uniform_height_sampling_covers_depressed_layers_without_reference_opacity():
    # No reference atmosphere is needed: initial quiet-Sun opacity must not
    # decide whether a depressed umbral formation layer has coarse samples.
    model = SimpleNamespace(
        solar_radius_m=torch.tensor(695700000., dtype=torch.float64),
        line_formation_height_bounds_megameter=(1.5, -1.0),
    )
    q = torch.linspace(-5, 1, 128, dtype=torch.float64)
    height = reference_height_grid(model, q, height_sampling="uniform")
    torch.testing.assert_close(height, torch.linspace(1.5e6, -1e6, 128, dtype=q.dtype))
    assert (height[:-1] - height[1:]).max() < 20e3
    # Native core continuum formation intervals measured by the ray audit.
    assert ((height >= -690e3) & (height <= -548e3)).sum() >= 7
    assert ((height >= -651e3) & (height <= -554e3)).sum() >= 5
    with pytest.raises(ValueError, match="sampling bounds"):
        reference_height_grid(model, torch.tensor([-6., 1.]), height_sampling="uniform")


@pytest.mark.parametrize("mode", ["reference", "uniform"])
def test_height_sampling_configuration_roundtrips(mode):
    options = resolve_depth_sampling({
        "sample_count": 128,
        "coarse_to_fine": {"enabled": True, "fine_sample_count": 64, "uniform_weight_floor": .05},
        "height_sampling": mode,
    })
    assert options.height_sampling == mode
    assert resolve_depth_sampling(options.model_configuration()) == options


@pytest.mark.parametrize("mode", [None, True, "optical_depth", ""])
def test_height_sampling_rejects_ambiguous_mode(mode):
    with pytest.raises(ValueError, match="height_sampling"):
        resolve_depth_sampling({"height_sampling": mode})
