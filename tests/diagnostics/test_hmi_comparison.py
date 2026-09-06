from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import prom3theus.diagnostics.hmi_comparison as comparison
from prom3theus.diagnostics.hmi_comparison import (
    _angular_residual_deg,
    _azimuth_convention_metrics,
    _comparison_metrics,
    _discover_segments,
    _observer_components,
    _spherical_components,
)


def test_observer_components_follow_hmi_azimuth_and_inclination():
    components = _observer_components(
        np.array([100.0, 200.0, 300.0]),
        np.array([90.0, 90.0, 0.0]),
        np.array([0.0, 90.0, 45.0]),
    )
    np.testing.assert_allclose(
        components,
        np.array([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0], [0.0, 0.0, 300.0]]),
        atol=1.0e-12,
    )


def test_director_and_signed_azimuth_residuals_have_distinct_periods():
    left = np.array([179.0, 181.0, 350.0])
    right = np.array([1.0, 359.0, 10.0])
    np.testing.assert_allclose(
        _angular_residual_deg(left, right, 180.0), np.array([-2.0, 2.0, -20.0])
    )
    np.testing.assert_allclose(
        _angular_residual_deg(left, right, 360.0), np.array([178.0, -178.0, -20.0])
    )


def test_metrics_separate_director_fit_from_disambiguation_branch():
    raw = np.array([[[100.0, 0.0, 10.0], [100.0, 0.0, 10.0]]])
    model_observer = np.array([[[100.0, 0.0, 10.0], [-100.0, 0.0, 10.0]]])
    model_spherical = np.array([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
    reference_spherical = model_spherical.copy()
    flips = np.array([[False, True]])
    mask = np.array([[True, True]])
    metrics = _comparison_metrics(
        model_observer,
        model_spherical,
        raw,
        reference_spherical,
        flips,
        mask,
        mask,
    )
    assert metrics["director_mae_deg"] == 0.0
    assert metrics["branch_agreement_fraction"] == 1.0
    assert metrics["signed_azimuth_mae_deg"] == 0.0
    assert metrics["components_all_valid"]["b_phi"]["rmse_gauss"] == 0.0
    assert metrics["components_strong_transverse"]["b_phi"]["rmse_gauss"] == 0.0


def test_azimuth_convention_search_identifies_reversed_rotated_angle():
    field = np.array([[300.0, 420.0, 510.0], [260.0, 390.0, 610.0]])
    inclination = np.array([[35.0, 70.0, 110.0], [50.0, 95.0, 140.0]])
    azimuth = np.array([[12.0, 48.0, 103.0], [157.0, 221.0, 306.0]])
    flips = np.array([[False, True, False], [True, False, True]])
    basis = np.broadcast_to(np.eye(3), (*field.shape, 3, 3)).copy()
    surface = np.array(
        [
            [[1.0, 0.2, 0.1], [1.0, -0.3, 0.4], [0.8, 0.5, -0.2]],
            [[0.9, -0.4, -0.1], [0.7, 0.6, 0.3], [1.0, 0.1, -0.5]],
        ]
    )
    reference = _observer_components(field, inclination, -azimuth + 90.0)
    reference[..., :2] *= np.where(flips, -1.0, 1.0)[..., None]
    model_spherical = _spherical_components(reference, surface)
    mask = np.ones(field.shape, dtype=bool)

    result = _azimuth_convention_metrics(
        reference,
        model_spherical,
        field,
        inclination,
        azimuth,
        flips,
        basis,
        surface,
        mask,
        mask,
    )

    assert result["best_configuration"] == "negative_chi_plus_90"
    best = result["configurations"]["negative_chi_plus_90"]
    assert best["mean_btheta_bphi_correlation_strong_transverse"] == 1.0
    assert len(result["configurations"]) == 8


def test_segment_discovery_requires_one_coherent_record(tmp_path: Path):
    for segment in ("field", "inclination", "azimuth", "disambig"):
        (tmp_path / f"hmi.b_720s.20240324_010000_TAI.{segment}.fits").touch()
    paths = _discover_segments(tmp_path)
    assert set(paths) == {"field", "inclination", "azimuth", "disambig"}


def test_comparison_routes_save_state_queries_through_p3s_loader(tmp_path, monkeypatch):
    basis = torch.eye(3).expand(1, 2, 3, 3).clone()
    raster = SimpleNamespace(
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        stokes_basis=basis,
        surface_position_m=torch.tensor([[[6.96e8, 0.0, 0.0], [6.96e8, 1.0e5, 0.0]]]),
    )
    selection = SimpleNamespace(raster=raster, name="hmi_0000", index=0)
    calls = {}
    target = object()
    cache_path = tmp_path / "observation-cache"
    cache_path.mkdir()

    class FakeLoader:
        def __init__(self, path, *, device):
            calls["init"] = (path, device)
            self.path = Path(path).resolve()
            self.observation = SimpleNamespace(
                spec=SimpleNamespace(observation_type="hmi_stokes")
            )
            self.raster_count = 1
            self.raster_names = ("hmi_0000",)
            self.observation_cache_path = cache_path

        def match_time(self, value, *, tolerance_seconds):
            calls["time"] = (value, tolerance_seconds)
            return 0, 0.25

        def select_raster(self, *, index):
            calls["selection"] = index
            return selection

        def raster_fields_at_height(self, height_m, **options):
            calls["height"] = (height_m, options)
            return {
                "magnetic_field_gauss": np.array(
                    [[[100.0, 0.0, 10.0], [100.0, 0.0, 10.0]]],
                    dtype=np.float32,
                )
            }

    paths = {
        name: tmp_path / f"record.{name}.fits"
        for name in ("field", "inclination", "azimuth", "disambig")
    }
    header = {"T_OBS": "target", "T_REC": "2024.03.24_00:00:00_TAI"}
    monkeypatch.setattr(comparison, "P3SLoader", FakeLoader)
    monkeypatch.setattr(comparison, "_discover_segments", lambda path: paths)
    monkeypatch.setattr(
        comparison, "_reference_header", lambda selected: (header, (4096, 4096))
    )
    monkeypatch.setattr(comparison, "parse_hmi_tai_time", lambda value: target)
    monkeypatch.setattr(
        comparison,
        "_reference_pixel_indices",
        lambda *args: (
            np.array([[1, 1]]),
            np.array([[2, 3]]),
            np.ones((1, 2), dtype=bool),
            0.01,
        ),
    )

    def sample(path, *args):
        del args
        if path == paths["field"]:
            return np.full((1, 2), 100.0)
        if path == paths["inclination"]:
            return np.full((1, 2), 90.0)
        return np.zeros((1, 2))

    monkeypatch.setattr(comparison, "_sample_segment", sample)
    monkeypatch.setattr(comparison, "_save_component_figure", lambda *a, **k: None)
    monkeypatch.setattr(comparison, "_save_azimuth_figure", lambda *a, **k: None)
    monkeypatch.setattr(
        comparison, "_save_azimuth_convention_figure", lambda *a, **k: None
    )
    save_state = tmp_path / "state.p3s"

    metrics = comparison.compare_hmi_save_state(
        save_state,
        tmp_path,
        tmp_path / "comparison",
        height_km=0.5,
        minimum_transverse_gauss=0.0,
        time_tolerance_seconds=2.0,
        device="cpu",
    )

    assert calls["init"] == (save_state, "cpu")
    assert calls["time"] == (target, 2.0)
    assert calls["selection"] == 0
    assert calls["height"] == (500.0, {"batch_size": 4096, "raster_index": 0})
    assert metrics["save_state"] == str(save_state.resolve())
    assert metrics["raster_name"] == "hmi_0000"
    assert set(metrics["outputs"]) == {
        "spherical_components",
        "azimuth_disambiguation",
        "azimuth_conventions",
        "metrics",
    }
    assert not list((tmp_path / "comparison").glob("*.npz"))
