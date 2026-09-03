"""Focused checks for diagnostic figure construction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from prom3theus.diagnostics.stokes_rendering import StokesPlotter


def test_stokes_plotter_accepts_the_canonical_observation_contract():
    wavelength = torch.tensor((6300.0, 6301.0, 6302.0, 6303.0))
    reference = torch.linspace(-0.02, 1.0, 64).reshape(4, 4, 4)
    prediction = reference * 0.95
    pixel_index = torch.tensor(((0, 0), (0, 1), (1, 0), (1, 1)))
    coordinates = torch.tensor(
        (
            ((1.0, 10.0, 0.0), (2.0, 10.0, 0.0)),
            ((1.0, 20.0, 0.0), (2.0, 20.0, 0.0)),
        )
    )
    outputs = {
        "stokes_pred": prediction,
        "stokes_reference": reference,
        "pixel_index": pixel_index,
        "integrated_prediction": prediction.abs().sum(dim=-1),
        "integrated_reference": reference.abs().sum(dim=-1),
    }

    figure = StokesPlotter().validation_figure(
        outputs,
        SimpleNamespace(coordinates=coordinates),
        wavelength,
        "test",
        rows=np.array((0, 1)),
        columns=np.array((0, 1)),
    )

    assert len(figure.axes) >= 12
    figure.clear()
