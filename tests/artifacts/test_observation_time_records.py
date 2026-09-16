"""Cached observation reloads compare canonical containers and scale casing."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from prom3theus.artifacts import loader as implementation
from prom3theus.artifacts.errors import ArtifactExportError


@pytest.mark.parametrize("cached_time,cached_scale,matching", [
    ("2024-03-23T22:11:58.388", "tai", True),
    ("2024-03-23T22:11:58.388", "TAI", True),
    ("2024-03-23T22:11:58.389", "tai", False),
    ("2024-03-23T22:11:58.388", "utc", False),
])
def test_observation_reload_normalizes_representation_but_preserves_time_identity(
    tmp_path, monkeypatch, cached_time, cached_scale, matching,
):
    recorded_times = [{"values": ["2024-03-23T22:11:58.388"], "scale": "TAI"}]
    original = deepcopy(recorded_times)
    bounds = {"time_hours": [0.0, 0.0]}
    metadata = {
        "adapter": "hmi_stokes", "observation": {}, "raster_names": ["hmi"],
        "validation_raster_index": 0, "times": recorded_times, "bounds": bounds,
    }
    raster = SimpleNamespace(metadata={
        "times": [cached_time], "coordinates": {"time_scale": cached_scale},
    })
    data = SimpleNamespace(
        rasters=[raster], raster_names=["hmi"], validation_raster_index=0,
        observation_sampling_bounds=bounds, setup=lambda stage: None,
    )
    options = {"type": "hmi_stokes", "loader": {
        "batch_size": 4, "validation_batch_size": 4, "validation_stride": 1,
    }}
    loaded = implementation.P3SLoader.__new__(implementation.P3SLoader)
    loaded._data_module = None
    loaded.stream_id = "hmi"
    loaded._config = SimpleNamespace(streams=[SimpleNamespace(
        id="hmi", observation=SimpleNamespace(to_dict=lambda: options),
    )])
    loaded.state = SimpleNamespace(scene=None, context={"streams": {"hmi": {
        "store_path": tmp_path / "cache", "source_signature": "source",
        "specification": {}, "store_metadata": metadata,
    }}})
    monkeypatch.setattr(implementation, "_parse_observation_spec",
                        lambda raw: SimpleNamespace(metadata=lambda: {}))
    monkeypatch.setattr(implementation.ObservationStore, "load_sequence",
                        lambda *a, **kw: ([raster], ["hmi"], metadata))
    monkeypatch.setattr("prom3theus.observations.scene.rebase_stokes_raster",
                        lambda value, scene: value)
    monkeypatch.setattr(implementation, "StoredObservationDataModule", lambda *a, **kw: data)

    if matching:
        assert loaded._observations() is data
        assert loaded.observation.times == ({"values": original[0]["values"], "scale": "tai"},)
    else:
        with pytest.raises(ArtifactExportError, match="Reloaded observation times differ"):
            loaded._observations()
        assert loaded._data_module is None
    assert recorded_times == original
