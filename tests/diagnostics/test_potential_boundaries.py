from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from prom3theus.application.joint_training import PotentialBoundaryDiagnostics
from prom3theus.config.schema import PotentialBoundaryConfig, PotentialPhotosphereConfig
from prom3theus.diagnostics.potential_boundaries import potential_boundary_figure
from prom3theus.inversion.data_terms.potential_boundary import (
    ProgressivePotentialBoundary,
)
from prom3theus.inversion.sampling import SphericalShellDomain


def reference():
    options = PotentialBoundaryConfig(
        enabled=True,
        grid_size=4,
        top_grid_size=3,
        side_horizontal_points=4,
        side_height_points=3,
        photosphere=PotentialPhotosphereConfig(
            enabled=True, start_step=1000, ramp_steps=2000
        ),
    ).to_dict()
    domain = SphericalShellDomain(
        0.6, (-0.1, 0.1), (-0.45, -0.2), (0.0, 2.0), (0.0, 50.0), 696e6
    )
    term = ProgressivePotentialBoundary(
        options, domain, domain, torch.eye(3), observation_times_hours=[0.0, 2.0]
    )
    r = term.positions / term.positions.norm(dim=-1, keepdim=True)
    lon, lat = torch.atan2(r[:, 1], r[:, 0]), torch.asin(r[:, 2])
    theta = torch.stack((lat.sin() * lon.cos(), lat.sin() * lon.sin(), -lat.cos()), -1)
    phi = torch.stack((-lon.sin(), lon.cos(), torch.zeros_like(lon)), -1)
    for i, (name, (begin, end)) in enumerate(term.surface_ranges.items()):
        amplitude = 10.0 * (i + 1)
        term.targets[0, begin:end] = amplitude * (
            r[begin:end] + 0.25 * theta[begin:end] - 0.5 * phi[begin:end]
        )
    # A very different second time must not affect the first-anchor plot/ranges.
    term.targets[1] = term.targets[0] * 100
    term.last_update, term.update_count = 500, 1
    return term


def test_all_six_surfaces_three_spherical_rows_independent_column_ranges():
    term = reference()
    figure = potential_boundary_figure(term)
    try:
        axes = np.asarray(figure.axes[:18], dtype=object).reshape(3, 6)
        names = ["photosphere", "top", "west", "east", "north", "south"]
        for column, name in enumerate(names):
            amplitude = 10.0 * (list(term.surface_ranges).index(name) + 1)
            norm = axes[0, column].collections[0].norm
            assert (
                abs(norm.vmax - amplitude) < 1e-4 and abs(norm.vmin + amplitude) < 1e-4
            )
            for row, ratio in enumerate([1.0, 0.25, -0.5]):
                artist = axes[row, column].collections[0]
                assert artist.norm is norm
                np.testing.assert_allclose(
                    artist.get_array(), amplitude * ratio, rtol=1e-5, atol=1e-5
                )
        assert axes[0, 0].get_title() == "Bottom (photosphere)"
        assert all(ax._colorbar.orientation == "horizontal" for ax in figure.axes[18:])
        assert len(figure.axes) == 24  # 18 panels plus one colorbar per boundary.
    finally:
        figure.clear()


def test_callback_logs_each_cached_revision_once_and_on_resume(tmp_path, monkeypatch):
    import prom3theus.application.joint_training as training
    import prom3theus.diagnostics.potential_boundaries as plotting

    term = reference()
    model = SimpleNamespace(shared_objectives={"potential_boundary": term})
    module = SimpleNamespace(model=model)
    trainer = SimpleNamespace(logger=Mock(), is_global_zero=True, global_step=501)
    callback = PotentialBoundaryDiagnostics(tmp_path)
    # The renderer must not query a model, create/refresh references, or access
    # the costly geometry operator. The model stub deliberately has no evaluator.
    term.prepare = Mock(side_effect=AssertionError("No reference refresh"))
    term._refresh = Mock(side_effect=AssertionError("No potential construction"))
    before = term.targets.clone()
    callback.on_train_start(trainer, module)
    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert trainer.logger.log_image.call_count == 1
    call = trainer.logger.log_image.call_args.kwargs
    assert call["key"] == "Potential field/potential_boundary"
    assert call["step"] == 501
    assert (tmp_path / "potential_boundary_step_00000500.png").exists()
    term.last_update, term.update_count = 1000, 2
    trainer.global_step = 1001
    callback.on_train_batch_end(trainer, module, None, None, 1)
    assert trainer.logger.log_image.call_count == 2
    resumed = training.PotentialBoundaryDiagnostics(tmp_path)
    resumed.on_train_start(trainer, module)
    assert trainer.logger.log_image.call_count == 3
    torch.testing.assert_close(term.targets, before)

    def forbidden(*args, **kwargs):
        raise AssertionError("No rendering on non-primary ranks or without logging")

    monkeypatch.setattr(plotting, "potential_boundary_figure", forbidden)
    trainer.is_global_zero = False
    resumed.on_train_start(trainer, module)
    trainer.is_global_zero = True
    trainer.logger = None
    resumed.on_train_start(trainer, module)
    trainer.logger = Mock()
    term.last_update = -1
    resumed.on_train_start(trainer, module)
