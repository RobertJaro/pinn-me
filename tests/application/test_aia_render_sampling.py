"""AIA image rendering shares LTE's bounded native rectangular grid."""

from types import SimpleNamespace

import pytest
import torch

from prom3theus.application.joint_rendering import _aia_render_results
from prom3theus.diagnostics.sampling import subsample_grid
from prom3theus.inversion.data_terms import DataTermBatchResult
from prom3theus.observations.image_contracts import ImageObservationRaster
from prom3theus.observations.image_dataset import (
    ImageObservationBatchCollator,
    ImagePixelDataset,
)


class _RecordingTerm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def evaluate_diagnostic_batch(self, batch):
        assert not torch.is_grad_enabled()
        self.batches.append(batch)
        target = batch["intensity"]
        return DataTermBatchResult(
            likelihood_loss=target.sum() * 0,
            component_losses={},
            metrics={},
            sample_count=len(target),
            diagnostics={
                "prediction_raw": target,
                "prediction_calibrated": target,
                "target": target,
                "asinh_residual": torch.zeros_like(target),
                "fractional_residual": torch.zeros_like(target),
                **{
                    name: batch[name]
                    for name in (
                        "channel_angstrom",
                        "channel_index",
                        "pixel_index",
                        "image_index",
                    )
                },
            },
        )


def _dataset(channel, channel_index, shape):
    ray = torch.zeros(*shape, 3)
    ray[..., 2] = -1
    surface = -ray * 6.957e8
    valid = torch.ones(shape, dtype=torch.bool)
    valid[0, 0] = False
    valid[-1, -1] = False
    intensity = 1000 * torch.arange(1, 1 + valid.numel()).reshape(shape).float()
    raster = ImageObservationRaster(
        intensity=intensity,
        ray_direction=ray,
        surface_position_m=surface,
        valid_mask=valid,
        absolute_tai_seconds=1_300_000_000.0,
        channel_angstrom=channel,
        exposure_group="validation",
        metadata={"ray_geometry": {"solar_radius_m": 6.957e8}},
    )
    return ImagePixelDataset(
        raster,
        image_index=channel_index,
        channel_index=channel_index,
        intensity_scale=float(intensity.max()),
    )


@pytest.mark.parametrize("maximum", [6, 100])
def test_aia_render_uses_shared_lte_grid_and_normalized_targets(maximum):
    datasets = {
        171: _dataset(171, 0, (5, 11)),
        193: _dataset(193, 1, (4, 8)),
    }
    term = _RecordingTerm()
    runtime = SimpleNamespace(
        streams={
            "aia": SimpleNamespace(
                data_module=SimpleNamespace(
                    validation_datasets=datasets,
                    collator=ImageObservationBatchCollator(),
                )
            )
        },
        model=SimpleNamespace(terms={"aia": term}),
        config=SimpleNamespace(
            diagnostics=SimpleNamespace(
                visualization=SimpleNamespace(
                    ray_sampling=SimpleNamespace(max_pixels=maximum, batch_size=3),
                    contribution_ray_count=0,
                )
            )
        ),
        device=torch.device("cpu"),
    )

    results, contributions = _aia_render_results(runtime, None, "aia")

    assert contributions == ()
    assert len(results) == len(term.batches)
    assert all(1 <= result.sample_count <= 3 for result in results)
    for channel, dataset in datasets.items():
        rows, columns = subsample_grid(*dataset.raster.spatial_shape, maximum)
        lattice = torch.cartesian_prod(torch.as_tensor(rows), torch.as_tensor(columns))
        expected = lattice[dataset.raster.valid_mask[lattice[:, 0], lattice[:, 1]]]
        channel_results = [
            result.diagnostics
            for result in results
            if int(result.diagnostics["channel_angstrom"][0]) == channel
        ]
        actual = torch.cat([payload["pixel_index"] for payload in channel_results])
        torch.testing.assert_close(actual, expected)
        assert len(actual) <= maximum
        target = torch.cat([payload["target"] for payload in channel_results])
        torch.testing.assert_close(
            target,
            dataset.raster.intensity[expected[:, 0], expected[:, 1]]
            / dataset.intensity_scale,
        )
        assert target.max() <= 1
        assert not target.requires_grad
        assert target.device.type == "cpu"
