"""Focused checks for diagnostic figure construction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from prom3theus.diagnostics.atmosphere_rendering import AtmospherePlotter
from prom3theus.diagnostics.evaluation import (
    AtmosphereEvaluator,
    AtmosphereSampling,
    MAGNETIC_FIELDS,
    VELOCITY_FIELDS,
)
from prom3theus.diagnostics.rendering import AtmosphereRenderer
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


def test_stokes_plotter_accepts_hmi_six_filter_validation_profiles():
    """HMI uses the same I/Q/U/V validation figure as resolved spectra."""

    wavelength = torch.tensor(
        (6173.0732, 6173.1419, 6173.2095, 6173.2776, 6173.3452, 6173.4139)
    )
    reference = torch.linspace(-0.02, 1.0, 96).reshape(4, 4, 6)
    prediction = reference * 0.95
    pixel_index = torch.tensor(((0, 0), (0, 1), (1, 0), (1, 1)))
    coordinates = torch.tensor(
        (
            ((1.0, 10.0, 0.0), (2.0, 10.0, 0.0)),
            ((1.0, 20.0, 0.0), (2.0, 20.0, 0.0)),
        )
    )
    integrated_reference = reference.abs().sum(dim=-1) * torch.tensor(
        (0.2, 0.01, 0.001, 0.01)
    )
    outputs = {
        "stokes_pred": prediction,
        "stokes_reference": reference,
        "pixel_index": pixel_index,
        "integrated_prediction": integrated_reference * 0.95,
        "integrated_reference": integrated_reference,
    }

    figure = StokesPlotter().validation_figure(
        outputs,
        SimpleNamespace(coordinates=coordinates),
        wavelength,
        "hmi-test",
        rows=np.array((0, 1)),
        columns=np.array((0, 1)),
        line_centers_angstrom=(6173.3352,),
    )

    assert len(figure.axes) >= 12
    for axis in figure.axes[8:12]:
        assert len(axis.lines) >= 2
    figure.canvas.draw()
    colorbar_axes = [
        axis for axis in figure.axes if axis.get_xlabel().startswith("fit-window")
    ]
    assert len(colorbar_axes) == 4
    assert all(axis.xaxis.get_offset_text().get_text() == "" for axis in colorbar_axes)
    figure.clear()


def test_validation_figures_are_published_to_flat_wandb_panel(tmp_path):
    calls = []

    class _Logger:
        def log_image(self, **kwargs):
            calls.append(kwargs)

    trainer = SimpleNamespace(logger=_Logger(), global_step=37)
    renderer = AtmosphereRenderer(
        tmp_path,
        dpi=70,
        evaluator=AtmosphereEvaluator(AtmosphereSampling.from_options()),
    )
    figure = Figure()
    FigureCanvasAgg(figure)

    path = renderer._save_figure(
        trainer,
        figure,
        "validation_test.png",
        "Parameters",
    )

    assert path.is_file()
    assert calls == [
        {
            "key": "Parameters",
            "images": [str(path)],
            "step": 37,
        }
    ]


def test_slice_heights_use_log_spacing_and_preserve_shell_boundaries():
    heights = AtmosphereRenderer.log_spaced_slice_heights(-1.0e5, 2.0e7, 6)

    assert heights[0] == -1.0e5
    assert heights[1] == 0.0
    assert heights[-1] == 2.0e7
    assert np.all(np.diff(heights) > 0.0)
    shifted = heights[1:] + 1.0e5
    np.testing.assert_allclose(
        shifted[1:] / shifted[:-1],
        np.full(4, shifted[1] / shifted[0]),
        rtol=1.0e-6,
    )


def test_vector_panels_render_magnetic_angles_and_los_velocity():
    shape = (2, 2, 2)
    longitude = np.broadcast_to(np.array((10.0, 11.0))[None, :, None], shape)
    latitude = np.broadcast_to(np.array((-2.0, -1.0))[:, None, None], shape)
    evaluated = {
        "solar_radius_m": 695_700_000.0,
        "shell_height_levels_m": np.array((1.0e6, 0.0)),
        "map_longitude_deg": longitude,
        "map_latitude_deg": latitude,
        "map_fields": {
            name: np.ones(shape, dtype=np.float32)
            for name in (*MAGNETIC_FIELDS, *VELOCITY_FIELDS)
        },
    }

    for fields, expected_axis_count in (
        (MAGNETIC_FIELDS, 18),
        (VELOCITY_FIELDS, 12),
    ):
        figure = AtmospherePlotter().field_panel_figure(
            evaluated,
            [0, 1],
            "test",
            field_names=fields,
            title="vector frames",
        )
        assert len(figure.axes) == expected_axis_count
        assert "=1" in figure.axes[0].texts[0].get_text()
        assert "=0" in figure.axes[len(fields)].texts[0].get_text()
        row_label_axes = figure.axes[0 : 2 * len(fields) : len(fields)]
        assert all(
            "mathrm{Mm}" in axis.texts[0].get_text() for axis in row_label_axes
        )
        figure.clear()


def test_observer_magnetic_panels_use_requested_angle_and_strength_styles():
    shape = (2, 2, 1)
    longitude = np.broadcast_to(np.array((10.0, 11.0))[None, :, None], shape)
    latitude = np.broadcast_to(np.array((-2.0, -1.0))[:, None, None], shape)
    evaluated = {
        "solar_radius_m": 695_700_000.0,
        "shell_height_levels_m": np.array((0.0,)),
        "map_longitude_deg": longitude,
        "map_latitude_deg": latitude,
        "map_fields": {
            "field_strength": np.array(((10.0, 100.0), (1_000.0, 100.0)))[
                ..., None
            ],
            "inclination": np.array(((0.0, 60.0), (120.0, 180.0)))[..., None],
            "azimuth": np.array(((-180.0, -90.0), (90.0, 180.0)))[..., None],
        },
    }

    figure = AtmospherePlotter().field_panel_figure(
        evaluated,
        [0],
        "test",
        field_names=("field_strength", "inclination", "azimuth"),
        title="observer magnetic geometry",
    )

    strength, inclination, azimuth = (
        axis.collections[0] for axis in figure.axes[:3]
    )
    assert strength.cmap.name == "viridis"
    assert isinstance(strength.norm, LogNorm)
    assert inclination.cmap.name == "PiYG"
    assert (inclination.norm.vmin, inclination.norm.vmax) == (0.0, 180.0)
    assert azimuth.cmap.name == "twilight"
    assert (azimuth.norm.vmin, azimuth.norm.vmax) == (-180.0, 180.0)
    figure.clear()


def test_thermodynamic_panels_show_explicit_log10_values():
    shape = (2, 2, 2)
    longitude = np.broadcast_to(np.array((10.0, 11.0))[None, :, None], shape)
    latitude = np.broadcast_to(np.array((-2.0, -1.0))[:, None, None], shape)
    evaluated = {
        "solar_radius_m": 695_700_000.0,
        "shell_height_levels_m": np.array((1.0e6, 0.0)),
        "map_longitude_deg": longitude,
        "map_latitude_deg": latitude,
        "map_fields": {
            "temperature": np.full(shape, 1.0e4),
            "density": np.full(shape, 1.0e-8),
            "pressure": np.full(shape, 1.0e2),
            "microturbulence": np.full(shape, 1.0e1),
        },
    }

    figure = AtmospherePlotter().field_panel_figure(
        evaluated,
        [0, 1],
        "test",
        field_names=("temperature", "density", "pressure", "microturbulence"),
        title="thermodynamics",
    )

    expected = (4.0, -8.0, 2.0, 1.0)
    for axis, value in zip(figure.axes[:4], expected, strict=True):
        np.testing.assert_allclose(np.asarray(axis.collections[0].get_array()), value)
    colorbar_labels = [axis.get_xlabel() for axis in figure.axes[8:]]
    assert all("log" in label for label in colorbar_labels)
    figure.clear()
