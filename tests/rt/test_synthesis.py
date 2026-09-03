import pytest
import torch

from prom3theus.core import SPEED_OF_LIGHT
from prom3theus.rt import (
    LTESynthesizer,
    OpticalDepthPath,
    StratifiedAtmosphere,
)


def test_lte_synthesizer_requires_an_explicit_line_selection():
    with pytest.raises(ValueError, match="exactly one"):
        LTESynthesizer(log_tau500=torch.tensor([-4.0, 0.0]))


def test_lte_synthesizer_runs_from_packaged_stic_resources():
    grid = torch.tensor([-4.0, -2.0, 0.0])
    atmosphere = StratifiedAtmosphere(
        log_tau500=grid,
        temperature=torch.full((1, 3), 5_500.0),
        velocity_field=torch.zeros(1, 3, 3),
        microturbulence=torch.full((1, 3), 1_000.0),
        magnetic_field=torch.zeros(1, 3, 3),
        gas_pressure=torch.logspace(1.0, 4.0, 3).reshape(1, 3),
    )
    synthesizer = LTESynthesizer(
        log_tau500=grid,
        line_ids=("FeI_6301.5008", "FeI_6302.4932"),
    )
    wavelength = torch.linspace(6301.3, 6302.7, 9)
    stokes, diagnostics = synthesizer(
        atmosphere,
        wavelength,
        path=OpticalDepthPath(mu=0.9),
        return_diagnostics=True,
    )
    assert stokes.shape == (1, 4, 9)
    assert torch.isfinite(stokes).all()
    assert diagnostics.alpha500.shape == (1, 3)
    assert diagnostics.cumulative_tau500_along_path is None


def test_lte_synthesizer_rejects_unordered_wavelengths_and_superluminal_los():
    grid = torch.tensor([-4.0, -2.0, 0.0])
    base = {
        "log_tau500": grid,
        "temperature": torch.full((1, 3), 5_500.0),
        "microturbulence": torch.full((1, 3), 1_000.0),
        "magnetic_field": torch.zeros(1, 3, 3),
        "gas_pressure": torch.logspace(1.0, 4.0, 3).reshape(1, 3),
    }
    synthesizer = LTESynthesizer(
        log_tau500=grid,
        line_ids=("FeI_6302.4932",),
    )
    atmosphere = StratifiedAtmosphere(
        velocity_field=torch.zeros(1, 3, 3),
        **base,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        synthesizer(
            atmosphere,
            torch.tensor([6302.6, 6302.5]),
            path=OpticalDepthPath(),
        )

    velocity = torch.zeros(1, 3, 3)
    velocity[..., 2] = -1.01 * SPEED_OF_LIGHT
    atmosphere = StratifiedAtmosphere(velocity_field=velocity, **base)
    with pytest.raises(ValueError, match=r"\|v_los\| < c"):
        synthesizer(
            atmosphere,
            torch.tensor([6302.4, 6302.6]),
            path=OpticalDepthPath(),
        )
