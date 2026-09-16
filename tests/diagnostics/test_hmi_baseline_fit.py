import numpy as np
import pytest

from prom3theus.diagnostics.hmi_baseline_fit import _chosen_tunings, _comparison_context, _load_comparison, _sample_grid


def test_figure_cannot_claim_success_when_representative_fit_fails():
    context = _comparison_context({"global_step": 2000, "fit": {"passed": True},
                                   "representative_fit": {"passed": False}})
    assert "step 2000" in context
    assert "fit checks failed" in context


def test_sample_grid_uses_detector_indices_without_reshaping_order():
    values = np.array([4., 1., 2.])
    pixels = np.array([[24, 40], [8, 8], [8, 40]])
    grid, rows, columns = _sample_grid(values, pixels)
    np.testing.assert_array_equal(rows, [8, 24])
    np.testing.assert_array_equal(columns, [8, 40])
    np.testing.assert_allclose(grid, [[1., 2.], [np.nan, 4.]], equal_nan=True)


def test_map_tuning_is_one_observation_selected_channel_per_component():
    target = np.zeros((3, 4, 6))
    target[:, 0] = 1
    target[:, 0, 2] = .5
    target[:, 1, 4] = -.1
    target[:, 2, 1] = .2
    target[:, 3, 5] = -.3
    assert _chosen_tunings(target) == [2, 4, 1, 5]


def test_comparison_rejects_nonfinite_or_ambiguous_samples(tmp_path):
    array = np.ones((3, 4, 6))
    path = tmp_path / "bad.npz"
    np.savez(path, initial=array, prediction=array, target=array,
             pixel_index=np.array([[0, 0], [0, 0], [1, 1]]))
    with pytest.raises(ValueError, match="Repeated"):
        _load_comparison(path)
    array[1, 2, 3] = np.nan
    np.savez(path, initial=array, prediction=array, target=array,
             pixel_index=np.array([[0, 0], [1, 0], [1, 1]]))
    with pytest.raises(ValueError, match="finite"):
        _load_comparison(path)
