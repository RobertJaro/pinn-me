"""Continuation diagnostics retain their real initialization and exact sample."""

from copy import deepcopy
import json

import numpy as np
import pytest

from prom3theus.diagnostics.initial_reference import preserve_initial_reference


def arrays():
    target = np.arange(48, dtype=np.float32).reshape(2, 4, 6) / 100
    return {
        "initial": target * 0.5, "target": target,
        "pixel_index": np.array([[0, 0], [16, 32]], dtype=np.int64),
        "representative_initial": target[:1] * 0.75,
        "representative_target": target[:1].copy(),
        "representative_pixel_index": np.array([[0, 0]], dtype=np.int64),
    }


def test_resume_preserves_original_predictions_and_provenance_without_writing(tmp_path):
    path = tmp_path / "initial-reference.npz"
    original = arrays()
    provenance = {"method": "weights_only", "source_global_step": 4000, "fresh_optimizer": True}
    preserve_initial_reference(path, original, provenance, resume=False)
    content = path.read_bytes()
    with np.load(path, allow_pickle=False) as stored:
        assert stored["initialization_json"].shape == ()
        assert json.loads(stored["initialization_json"].item()) == provenance
    current = deepcopy(original)
    current["initial"][:] = -10
    current["representative_initial"][:] = -20
    restored, initialization = preserve_initial_reference(
        path, current, {"method": "fresh_random_model"}, resume=True,
    )
    for name in original:
        np.testing.assert_array_equal(restored[name], original[name])
    assert initialization == provenance
    assert path.read_bytes() == content


@pytest.mark.parametrize("name", ["target", "pixel_index", "representative_target", "representative_pixel_index"])
def test_resume_rejects_changed_observations_or_sample_without_writing(tmp_path, name):
    path = tmp_path / "initial-reference.npz"
    current = arrays()
    preserve_initial_reference(path, current, {"seed": 0}, resume=False)
    content = path.read_bytes()
    current[name] = current[name].copy()
    current[name].flat[0] += 1
    with pytest.raises(ValueError, match=f"reference {name} differs"):
        preserve_initial_reference(path, current, {}, resume=True)
    assert path.read_bytes() == content


def test_resume_rejects_missing_reference_and_changed_dtype(tmp_path):
    path = tmp_path / "initial-reference.npz"
    current = arrays()
    with pytest.raises(FileNotFoundError, match="initial diagnostic reference is missing"):
        preserve_initial_reference(path, current, {}, resume=True)
    assert not path.exists()
    preserve_initial_reference(path, current, {}, resume=False)
    current["target"] = current["target"].astype(np.float64)
    with pytest.raises(ValueError, match="reference target differs"):
        preserve_initial_reference(path, current, {}, resume=True)


def test_new_run_refuses_to_overwrite_existing_reference(tmp_path):
    path = tmp_path / "initial-reference.npz"
    preserve_initial_reference(path, arrays(), {}, resume=False)
    content = path.read_bytes()
    with pytest.raises(FileExistsError):
        preserve_initial_reference(path, arrays(), {"different": True}, resume=False)
    assert path.read_bytes() == content
