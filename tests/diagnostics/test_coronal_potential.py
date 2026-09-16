"""Independent physics checks for the alpha=0 coronal baseline exporter."""

import json

import numpy as np
import pytest
import torch

from prom3theus.diagnostics.coronal_potential import (
    SphericalPotentialField,
    differential_metrics,
    export_potential_corona,
    infer_photospheric_source,
)
from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.rt.spherical_potential import spherical_source_cells


def test_full_sphere_monopole_and_cartesian_differential_residuals():
    radius = 696e6
    cells = spherical_source_cells((-np.pi, np.pi), (-np.pi / 2, np.pi / 2), 12)
    field = SphericalPotentialField(cells, np.full(len(cells), 80.0), radius)
    points = np.array([[1.3, 0.2, -0.1], [-0.8, 1.0, 0.4], [0.1, -0.4, 1.2]])
    norms = np.linalg.norm(points, axis=-1, keepdims=True)
    expected = 80 * points / norms**3
    # The existing cell integrator uses sixth-order quadrature and float32
    # operators; its polar-cell integration error is approximately 2e-4 here.
    np.testing.assert_allclose(field(points * radius), expected, rtol=3e-4, atol=1e-6)
    metrics = differential_metrics(field, points * radius, step_megameter=0.05)
    for check in metrics["step_checks"]:
        assert check["max_normalized_divergence"] < 1e-5
        assert check["max_normalized_curl"] < 1e-5


class RadialAtmosphere(torch.nn.Module):
    def __init__(self, strength=150.0):
        super().__init__()
        self.strength = torch.nn.Parameter(torch.tensor(strength, dtype=torch.float64))

    def evaluate_position_rsun(self, position_rsun, time_hours=None):
        return {"magnetic_field": self.strength * position_rsun /
                torch.linalg.vector_norm(position_rsun, dim=-1, keepdim=True)}


def domain():
    return SphericalShellDomain(0.8, (-0.012, 0.012), (-0.01, 0.01),
                                (0.0, 0.0), (-0.1, 1.5), 696e6)


def test_source_is_radial_at_photosphere_and_preserves_model_training_mode():
    model = RadialAtmosphere().train()
    field, time = infer_photospheric_source(model, domain(), grid_size=4)
    np.testing.assert_allclose(field.radial_field_gauss, 150.0, atol=1e-10)
    assert time == 0
    assert model.training
    assert model.strength.grad is None


class PolynomialRadialAtmosphere(RadialAtmosphere):
    def evaluate_position_rsun(self, position_rsun, time_hours=None):
        direction = position_rsun / torch.linalg.vector_norm(position_rsun, dim=-1, keepdim=True)
        longitude = (torch.atan2(direction[:, 1], direction[:, 0]) - 0.8) / 0.012
        mu = direction[:, 2] / np.sin(0.01)
        br = self.strength * (1 + longitude**6 + mu**4)
        return {"magnetic_field": br[:, None] * direction}


def test_source_quadrature_resolves_varying_radial_field_cell_means():
    model = PolynomialRadialAtmosphere()
    high_order, _ = infer_photospheric_source(model, domain(), grid_size=3)
    low_order, _ = infer_photospheric_source(model, domain(), grid_size=3,
                                            source_quadrature_order=2)
    longitude = (high_order.cells[:, :2] - 0.8) / 0.012
    mu = high_order.cells[:, 2:] / np.sin(0.01)
    mean_sixth = (longitude[:, 1]**7 - longitude[:, 0]**7) / (7 * np.diff(longitude)[:, 0])
    mean_fourth = (mu[:, 1]**5 - mu[:, 0]**5) / (5 * np.diff(mu)[:, 0])
    expected = 150 * (1 + mean_sixth + mean_fourth)
    # Four Gauss nodes integrate the degree-six/four source variation exactly;
    # two nodes miss within-cell structure even on the same source grid.
    np.testing.assert_allclose(high_order.radial_field_gauss, expected, atol=1e-9, rtol=1e-12)
    assert np.max(np.abs(low_order.radial_field_gauss - expected)) > 0.1


@pytest.mark.parametrize("order", [True, False, 4.0, "4", None])
def test_source_quadrature_rejects_noninteger_order(order):
    with pytest.raises(TypeError, match="source_quadrature_order"):
        infer_photospheric_source(RadialAtmosphere(), domain(), source_quadrature_order=order)


@pytest.mark.parametrize("order", [-1, 0, 1])
def test_source_quadrature_rejects_insufficient_order(order, tmp_path):
    with pytest.raises(ValueError, match="source_quadrature_order"):
        export_potential_corona(RadialAtmosphere(), domain(), tmp_path, source_quadrature_order=order)


def test_export_validates_nonzero_corona_boundary_and_connectivity(tmp_path):
    result = export_potential_corona(RadialAtmosphere(), domain(), tmp_path,
        source_grid_size=4, horizontal_points=4, heights_megameter=(0.5, 1.0, 3.0),
        trace_count=2, make_figure=False)
    report = result["report"]
    assert report["valid"], report["checks"]
    assert report["layers"][-1]["rms_field_gauss"] > 1
    assert max(report["field_lines"]["maximum_heights_megameter"]) >= 2.99
    assert report["surface_max_radial_error_gauss"] < 1e-4
    assert report["grid_geometry"]["source_cell_mean_quadrature_order"] == 4
    assert report["grid_geometry"]["source_samples_per_cell"] == 16
    stored = np.load(result["paths"]["arrays"], allow_pickle=False)
    assert stored["magnetic_field_gauss"].shape == (3, 4, 4, 3)
    assert stored["line_offsets"][-1] == len(stored["line_positions_m"])
    assert json.loads((tmp_path / "coronal-validation.json").read_text())["valid"]


def test_zero_source_cannot_pass_validation(tmp_path):
    result = export_potential_corona(RadialAtmosphere(0.0), domain(), tmp_path,
        source_grid_size=2, horizontal_points=2, heights_megameter=(0.5, 3.0),
        trace_count=1, make_figure=False, source_quadrature_order=3)
    assert not result["report"]["valid"]
    assert not result["report"]["checks"]["nonzero_field_at_top"]
    assert not result["report"]["checks"]["traced_field_reaches_corona"]
    assert result["report"]["grid_geometry"]["source_cell_mean_quadrature_order"] == 3
