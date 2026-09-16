from types import SimpleNamespace

import pytest
import torch

from prom3theus.application.joint_streams import _observation_times_hours


def test_observation_times_are_unique_sorted_shared_scene_hours():
    scene = SimpleNamespace()

    def support(actual_scene, chunk_size):
        assert actual_scene is scene
        assert chunk_size > 1
        for values in ([0.75, 0.75, -0.25], [1.5, 0.75], [0.0, 0.0]):
            yield torch.zeros(len(values), 3), torch.tensor(values)

    streams = {"stokes": SimpleNamespace(sampling_support=support)}
    assert _observation_times_hours(streams, scene) == [-0.25, 0.0, 0.75, 1.5]


@pytest.mark.parametrize("values", [[], [float("nan")]])
def test_observation_times_reject_empty_or_nonfinite_support(values):
    def support(scene, chunk_size):
        yield torch.zeros(len(values), 3), torch.tensor(values)

    with pytest.raises(ValueError):
        _observation_times_hours(
            {"stokes": SimpleNamespace(sampling_support=support)}, None
        )
