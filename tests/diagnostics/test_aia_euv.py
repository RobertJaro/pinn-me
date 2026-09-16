"""Lifecycle-independent AIA diagnostic routing and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from prom3theus.diagnostics.aia_euv import AIAEUVDiagnosticRenderer
from prom3theus.diagnostics.registry import (
    DiagnosticRegistry,
    default_diagnostic_registry,
)
from prom3theus.inversion.data_terms.base import DataTermBatchResult
from prom3theus.observations.image_contracts import ImageObservationRaster


SOLAR_RADIUS_M = 6.957e8


def _raster(
    channel: int,
    *,
    image_offset: float = 0.0,
    shape: tuple[int, int] = (2, 3),
) -> ImageObservationRaster:
    ray = torch.zeros((*shape, 3), dtype=torch.float32)
    ray[..., 2] = -1.0
    surface = torch.zeros((*shape, 3), dtype=torch.float32)
    surface[..., 2] = SOLAR_RADIUS_M
    intensity = (
        torch.arange(1, 1 + shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        + channel
        + image_offset
    )
    return ImageObservationRaster(
        intensity=intensity,
        uncertainty=torch.full(shape, 0.25),
        ray_direction=ray,
        surface_position_m=surface,
        valid_mask=torch.ones(shape, dtype=torch.bool),
        absolute_tai_seconds=1_700_000_000.0 + image_offset,
        channel_angstrom=channel,
        exposure_group="validation-exposure",
        metadata={
            "intensity_unit": "DN s-1 pixel-1",
            "ray_geometry": {"solar_radius_m": SOLAR_RADIUS_M},
        },
    )


def _result(
    raster: ImageObservationRaster, image_index: int, intensity_scale: float = 1.0
) -> DataTermBatchResult:
    pixels = torch.nonzero(raster.valid_mask, as_tuple=False)
    target = raster.intensity[pixels[:, 0], pixels[:, 1]] / intensity_scale
    raw = target * 0.8
    calibrated = raw * 1.2
    residual = calibrated - target
    contribution = torch.stack((target * 0.2, target * 0.3, target * 0.5), dim=1)
    height = torch.tensor((0.0, 1.0e6, 2.0e6)).expand_as(contribution)
    return DataTermBatchResult(
        likelihood_loss=torch.tensor(float(image_index + 1)),
        component_losses={
            f"channel_{raster.channel_angstrom}": torch.tensor(float(image_index + 1))
        },
        metrics={"residual_bias": residual.mean()},
        sample_count=len(pixels),
        diagnostics={
            "prediction_raw": raw,
            "prediction_calibrated": calibrated,
            "target": target,
            "uncertainty": torch.full_like(target, 0.25),
            "channel_angstrom": torch.full(
                (len(pixels),), raster.channel_angstrom, dtype=torch.long
            ),
            "channel_index": torch.full((len(pixels),), image_index, dtype=torch.long),
            "pixel_index": pixels,
            "image_index": torch.full((len(pixels),), image_index, dtype=torch.long),
            "standardized_residual": residual / 0.25,
            "asinh_residual": torch.asinh(calibrated) - torch.asinh(target),
            "fractional_residual": residual / target,
            "contribution": contribution,
            "height_m": height,
        },
    )


def _render(directory: Path) -> dict:
    rasters = [
        _raster(171, shape=(1, 2)),
        _raster(193, shape=(2, 2)),
        _raster(211, shape=(2, 3)),
    ]
    intensity_scales = {171: 1000.0, 193: 2000.0, 211: 3000.0}
    results = [
        _result(raster, index, intensity_scales[raster.channel_angstrom])
        for index, raster in enumerate(rasters)
    ]
    contribution_fields = {
        "contribution",
        "height_m",
        "channel_angstrom",
        "channel_index",
        "pixel_index",
        "image_index",
    }
    contribution_payloads = [
        {
            name: value[:1]
            for name, value in result.diagnostics.items()
            if name in contribution_fields
        }
        for result in results
    ]
    image_results = [
        DataTermBatchResult(
            likelihood_loss=result.likelihood_loss,
            component_losses=result.component_losses,
            metrics=result.metrics,
            sample_count=result.sample_count,
            diagnostics={
                name: value
                for name, value in result.diagnostics.items()
                if name not in {"contribution", "height_m"}
            },
        )
        for result in results
    ]
    registry = default_diagnostic_registry(
        aia_max_pixels_per_image=6,
        aia_max_contribution_rays=3,
        dpi=72,
    )
    return registry.render_results(
        {"aia": image_results},
        stream_kinds={"aia": "aia_euv"},
        output_directory=directory,
        label="setup only",
        contexts={
            "aia": {
                "rasters": rasters,
                "asinh_scales": {171: 0.01, 193: 0.01, 211: 0.01},
                "intensity_scales": intensity_scales,
                "contribution_payloads": contribution_payloads,
                "likelihood_component_weights": {
                    "channel_171": 1.0,
                    "channel_193": 1.0,
                    "channel_211": 1.0,
                },
            }
        },
    )


def test_comparison_uses_two_rows_of_native_pixels(tmp_path, monkeypatch):
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    calls = []
    colorbars = []
    original = Axes.imshow
    original_colorbar = Figure.colorbar

    def imshow(axis, values, *args, **kwargs):
        calls.append((axis, np.asarray(values).copy(), kwargs))
        return original(axis, values, *args, **kwargs)

    def colorbar(figure, mappable, **kwargs):
        colorbars.append((mappable.norm, kwargs))
        return original_colorbar(figure, mappable, **kwargs)

    monkeypatch.setattr(Axes, "imshow", imshow)
    monkeypatch.setattr(Figure, "colorbar", colorbar)
    _render(tmp_path)
    assert len(calls) == 6
    assert len(colorbars) == 3
    for column, (channel, shape) in enumerate(
        ((171, (1, 2)), (193, (2, 2)), (211, (2, 3)))
    ):
        observed, predicted = calls[2 * column : 2 * column + 2]
        expected = _raster(channel, shape=shape).intensity.numpy()
        scale = 0.01
        expected = expected / {171: 1000.0, 193: 2000.0, 211: 3000.0}[channel]
        np.testing.assert_array_equal(
            observed[1], (torch.asinh(torch.from_numpy(expected) / scale) / torch.asinh(torch.tensor(scale).reciprocal())).numpy()
        )
        np.testing.assert_array_equal(
            predicted[1],
            (torch.asinh(torch.from_numpy(expected) * 0.8 * 1.2 / scale) / torch.asinh(torch.tensor(scale).reciprocal())).numpy(),
        )
        assert observed[2]["norm"] is predicted[2]["norm"]
        colorbar_norm, options = colorbars[column]
        assert colorbar_norm is observed[2]["norm"]
        assert options["ax"] == [observed[0], predicted[0]]
        assert options["orientation"] == "horizontal"
        assert options["label"] == "asinh(I / a) / asinh(1 / a)"
        for row, (axis, values, kwargs) in enumerate((observed, predicted)):
            assert values.shape == shape
            assert kwargs["interpolation"] == "none"
            assert kwargs["cmap"].name == f"SDO AIA {float(channel)} Angstrom"
            assert axis.get_subplotspec().rowspan.start == row
            assert axis.get_subplotspec().colspan.start == column


def test_subsampled_comparison_displays_matching_compact_native_grids(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    import numpy as np
    from matplotlib.axes import Axes
    from prom3theus.diagnostics.sampling import subsample_grid

    raster = _raster(193, shape=(9, 13))
    mask = raster.valid_mask.clone()
    mask[4, 8] = False
    raster = replace(raster, valid_mask=mask)
    rows, columns = subsample_grid(*raster.spatial_shape, maximum=12)
    result = _result(raster, 0, intensity_scale=1000.0)
    payload = result.diagnostics
    pixels = payload["pixel_index"]
    selected = torch.isin(pixels[:, 0], torch.as_tensor(rows)) & torch.isin(
        pixels[:, 1], torch.as_tensor(columns)
    )
    payload = {
        name: value[selected]
        for name, value in payload.items()
        if name not in {"contribution", "height_m"}
    }
    images = []
    original = Axes.imshow

    def imshow(axis, values, *args, **kwargs):
        images.append(np.asarray(values).copy())
        assert kwargs["interpolation"] == "none"
        return original(axis, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", imshow)
    AIAEUVDiagnosticRenderer(max_pixels_per_image=12, dpi=72).render(
        [payload],
        output_directory=tmp_path,
        label="sampled",
        context={
            "rasters": [raster],
            "asinh_scales": {193: 0.01},
            "intensity_scales": {193: 1000.0},
        },
    )
    expected = raster.intensity[rows][:, columns] / 1000.0
    valid = raster.valid_mask[rows][:, columns].numpy()
    assert len(images) == 2
    assert images[0].shape == images[1].shape == (len(rows), len(columns))
    assert images[0].size <= 12
    np.testing.assert_array_equal(
        images[0][valid], (torch.asinh(expected / 0.01) / torch.asinh(torch.tensor(.01).reciprocal())).numpy()[valid]
    )
    np.testing.assert_array_equal(
        images[1][valid], (torch.asinh(expected * 0.8 * 1.2 / 0.01) / torch.asinh(torch.tensor(.01).reciprocal())).numpy()[valid]
    )
    assert np.isnan(images[0][~valid]).all()
    assert np.isnan(images[1][~valid]).all()


def test_plot_transform_matches_objective_and_preserves_target_contrast(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from prom3theus.inversion.data_terms.aia_objective import AsinhMSEImageObjective

    intensity = torch.tensor([[-2.0, 0.0, 100.0], [10.0, 40.0, 1.0e6]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    raster = replace(_raster(193), intensity=intensity, valid_mask=valid)
    scale = 7.0  # Deliberately not this image's median.
    prediction = torch.ones_like(intensity) * 30.0
    prediction[~valid] = torch.nan
    calls = []
    original = Axes.imshow

    def imshow(axis, values, *args, **kwargs):
        calls.append((np.asarray(values).copy(), kwargs["norm"]))
        return original(axis, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", imshow)
    renderer = AIAEUVDiagnosticRenderer(dpi=72)
    renderer._render_comparison(
        tmp_path / "normal.png",
        "test",
        [(raster, prediction)],
        {193: scale},
        {193: 1.0},
    )
    objective = AsinhMSEImageObjective(
        [193], asinh_scales=[scale], channel_weights=[1.0]
    )
    _, _, residual = objective(
        prediction[valid], intensity[valid], torch.zeros(5, dtype=torch.long)
    )
    observed, synthesis = calls[:2]
    np.testing.assert_allclose(
        synthesis[0][valid] - observed[0][valid], residual.numpy(), rtol=0, atol=0
    )
    assert np.isnan(observed[0][~valid]).all()
    assert observed[0][0, 0] < 0
    assert type(observed[1]) is Normalize  # No double asinh stretch.
    assert observed[1] is synthesis[1]
    renderer._render_comparison(
        tmp_path / "extreme.png",
        "test",
        [(raster, prediction * 1.0e10)],
        {193: scale},
        {193: 1.0},
    )
    np.testing.assert_array_equal(calls[2][0], observed[0])
    assert (calls[2][1].vmin, calls[2][1].vmax) == (observed[1].vmin, observed[1].vmax)
    with pytest.raises(ValueError, match="training asinh scale"):
        renderer._render_comparison(
            tmp_path / "missing.png", "test", [(raster, prediction)], {}, {193: 1.0}
        )


def test_aia_renderer_writes_deterministic_native_grid_pngs_and_json_report(tmp_path):
    first = _render(tmp_path / "first")
    second = _render(tmp_path / "second")

    json.dumps(first, allow_nan=False)
    diagnostics = first["streams"]["aia"]["diagnostics"]
    assert first["streams"]["aia"]["sample_count"] == 12
    assert first["streams"]["aia"]["losses"]["likelihood"] == pytest.approx(2.0)
    assert first["streams"]["aia"]["losses"]["likelihood_reduction"] == {
        "type": "configured_component_balanced",
        "normalized_component_weights": {
            "channel_171": pytest.approx(1.0 / 3.0),
            "channel_193": pytest.approx(1.0 / 3.0),
            "channel_211": pytest.approx(1.0 / 3.0),
        },
    }
    assert [item["channel_angstrom"] for item in diagnostics["images"]] == [
        171,
        193,
        211,
    ]
    assert diagnostics["images"][0]["metrics"]["sampled_fraction"] == 1.0
    assert diagnostics["images"][0]["metrics"][
        "contribution_peak_height_m_median"
    ] == pytest.approx(2.0e6)
    assert diagnostics["contribution_ray_count"] == 3
    assert [item["ray_count"] for item in diagnostics["contribution_profiles"]] == [
        1,
        1,
        1,
    ]

    first_paths = [Path(path) for path in diagnostics["paths"]]
    second_paths = [
        Path(path) for path in second["streams"]["aia"]["diagnostics"]["paths"]
    ]
    assert [path.name for path in first_paths] == [
        "setup_only_aia_validation-exposure_comparison.png",
    ]
    for first_path, second_path in zip(first_paths, second_paths):
        assert first_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert first_path.read_bytes() == second_path.read_bytes()
    first_contribution_paths = [
        Path(path) for path in diagnostics["contribution_paths"]
    ]
    second_contribution_paths = [
        Path(path)
        for path in second["streams"]["aia"]["diagnostics"]["contribution_paths"]
    ]
    assert len(first_contribution_paths) == 3
    for first_path, second_path in zip(
        first_contribution_paths, second_contribution_paths
    ):
        assert first_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert first_path.read_bytes() == second_path.read_bytes()


def test_aia_contribution_collection_enforces_independent_ray_bound(tmp_path):
    raster = _raster(171)
    aia_result = _result(raster, 0)
    registry = DiagnosticRegistry(
        (
            AIAEUVDiagnosticRenderer(
                max_pixels_per_image=6,
                max_contribution_rays=5,
                dpi=72,
            ),
        )
    )
    with pytest.raises(ValueError, match="contain 6 rays.*configured bound is 5"):
        registry.render_results(
            {"aia": aia_result},
            stream_kinds={"aia": "aia_euv"},
            output_directory=tmp_path,
            label="bounded profiles",
            contexts={"aia": {"rasters": [raster], "intensity_scales": {171: 1.0}}},
        )


def test_registry_uses_configured_channel_weights_across_unequal_chunks(tmp_path):
    class Renderer:
        observation_kind = "aia_euv"

        def render(self, payloads, *, output_directory, label, context=None):
            del payloads, output_directory, label, context
            return {"ok": True}

    def result(component: str, value: float, sample_count: int):
        scalar = torch.tensor(value)
        return DataTermBatchResult(
            # A single-channel diagnostic batch normalizes its likelihood to
            # the active channel, as the concrete AIA objective does.
            likelihood_loss=scalar,
            component_losses={component: scalar},
            metrics={},
            sample_count=sample_count,
            diagnostics={"present": torch.ones(())},
        )

    batches = (
        result("channel_171", 1.0, 2),
        result("channel_193", 2.0, 1),
        result("channel_193", 4.0, 3),
        result("channel_211", 8.0, 10),
    )
    report = DiagnosticRegistry((Renderer(),)).render_results(
        {"aia": batches},
        stream_kinds={"aia": "aia_euv"},
        output_directory=tmp_path,
        label="unequal chunks",
        contexts={
            "aia": {
                "likelihood_component_weights": {
                    "channel_171": 0.5,
                    "channel_193": 0.3,
                    "channel_211": 0.2,
                }
            }
        },
    )

    losses = report["streams"]["aia"]["losses"]
    # channel_193 is first reduced over its 1+3 pixels: (2 + 3*4) / 4 = 3.5.
    assert losses["components"]["channel_193"] == pytest.approx(3.5)
    assert losses["likelihood"] == pytest.approx(0.5 * 1.0 + 0.3 * 3.5 + 0.2 * 8.0)
    assert losses["likelihood"] != pytest.approx(
        (2 * 1.0 + 1 * 2.0 + 3 * 4.0 + 10 * 8.0) / 16
    )


def test_aia_renderer_enforces_bounded_collection_and_native_target(tmp_path):
    raster = _raster(171)
    result = _result(raster, 0)
    registry = DiagnosticRegistry(
        (AIAEUVDiagnosticRenderer(max_pixels_per_image=5, dpi=72),)
    )
    with pytest.raises(ValueError, match="configured bound is 5"):
        registry.render_results(
            {"aia": result},
            stream_kinds={"aia": "aia_euv"},
            output_directory=tmp_path,
            label="bounded",
            contexts={"aia": {"rasters": [raster], "intensity_scales": {171: 1.0}}},
        )

    changed = dict(result.diagnostics)
    changed["target"] = changed["target"] + 1.0
    mismatched = DataTermBatchResult(
        likelihood_loss=result.likelihood_loss,
        component_losses=result.component_losses,
        metrics=result.metrics,
        sample_count=result.sample_count,
        diagnostics=changed,
    )
    registry = DiagnosticRegistry(
        (AIAEUVDiagnosticRenderer(max_pixels_per_image=6, dpi=72),)
    )
    with pytest.raises(ValueError, match="does not match native raster"):
        registry.render_results(
            {"aia": mismatched},
            stream_kinds={"aia": "aia_euv"},
            output_directory=tmp_path,
            label="mismatch",
            contexts={"aia": {"rasters": [raster], "intensity_scales": {171: 1.0}}},
        )


def test_registry_rejects_duplicate_kinds_and_nonserializable_renderer(tmp_path):
    class Renderer:
        observation_kind = "test"

        def render(self, payloads, *, output_directory, label, context=None):
            del payloads, output_directory, label, context
            return {"bad": object()}

    registry = DiagnosticRegistry((Renderer(),))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Renderer())
    result = DataTermBatchResult(
        likelihood_loss=torch.zeros(()),
        component_losses={},
        metrics={},
        sample_count=1,
        diagnostics={"anything": torch.ones(())},
    )
    with pytest.raises(TypeError, match="JSON-serializable"):
        registry.render_results(
            {"stream": result},
            stream_kinds={"stream": "test"},
            output_directory=tmp_path,
            label="serializable",
        )


@pytest.mark.parametrize("stream_count", [1, 2])
def test_aia_provider_writes_and_logs_validation_images(tmp_path, stream_count):
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import Mock
    from prom3theus.config import load_config
    from prom3theus.diagnostics.providers import render_diagnostics
    from prom3theus.application.joint_training import _log_validation

    config = load_config(
        Path(__file__).resolve().parents[2] / "configs/hmi_aia_dynamic.yaml"
    )
    stream = config.streams[1]
    rasters = [_raster(channel) for channel in (171, 193, 211)]
    batches = [_result(raster, index) for index, raster in enumerate(rasters)]
    combined = DataTermBatchResult(
        likelihood_loss=torch.tensor(2.0),
        component_losses={
            key: value
            for batch in batches
            for key, value in batch.component_losses.items()
        },
        metrics={},
        sample_count=sum(batch.sample_count for batch in batches),
        diagnostics={
            key: torch.cat([batch.diagnostics[key] for batch in batches])
            for key in batches[0].diagnostics
        },
    )
    term = torch.nn.Module()
    term.channels_angstrom = (171, 193, 211)
    term.objective = SimpleNamespace(asinh_scales=torch.ones(3))
    term.intensity_scales = torch.ones(3)
    model = torch.nn.Module()
    streams = tuple(replace(stream, id=f"aia_{index}") for index in range(stream_count))
    model.terms = torch.nn.ModuleDict({item.id: term for item in streams})
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            streams=streams,
            diagnostics=replace(
                config.diagnostics,
                visualization=replace(
                    config.diagnostics.visualization, contribution_ray_count=18, dpi=60
                ),
            ),
        ),
        model=model,
        streams={
            item.id: SimpleNamespace(rasters=rasters, data_module=SimpleNamespace())
            for item in streams
        },
    )
    report = render_diagnostics(
        runtime,
        SimpleNamespace(global_step=1000),
        SimpleNamespace(streams={item.id: combined for item in streams}),
        tmp_path / "validation" / "step_00001000",
    )
    images = [Path(artifact["path"]) for artifact in report["artifacts"]]
    assert (
        len(images) == 4 * stream_count
    )  # One three-channel comparison and three contribution panels.
    assert len(set(images)) == len(images)
    assert all(path.is_file() and path.stat().st_size > 1000 for path in images)
    logger = Mock()
    _log_validation(logger, {"metrics": {"loss": 2.0}, "rendering": report}, 1000)
    assert logger.log_image.call_count == 1
    assert (
        logger.log_image.call_args.kwargs["key"] == "AIA observation comparison"
    )
    assert {
        path
        for call in logger.log_image.call_args_list
        for path in call.kwargs["images"]
    } == {
        artifact["path"]
        for artifact in report["artifacts"]
        if artifact.get("media_key") is not None
    }
