"""One YAML UTC interval selects HMI TAI headers and AIA exposure times."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from prom3theus.config import load_config
from prom3theus.config.joint_schema import TimeWindowConfig
from prom3theus.components import observations as joint_streams


CONFIG = Path(__file__).resolve().parents[2] / "configs/hmi_aia_dynamic.yaml"


def test_yaml_window_and_timezone_conversion():
    config = load_config(CONFIG)
    window = config.scene.time_window
    assert window.start == "2011-02-14T00:00:00+00:00"
    assert window.end == "2011-02-14T02:00:00+00:00"
    assert TimeWindowConfig("2011-02-13T19:00:00-05:00", "2011-02-13T21:00:00-05:00") == window


@pytest.mark.parametrize("start,end", [("bad", "2011-02-14"),
                                       ("2011-02-14", "2011-02-14"),
                                       ("2011-02-15", "2011-02-14")])
def test_invalid_window(start, end):
    with pytest.raises(ValueError):
        TimeWindowConfig(start, end)


@pytest.mark.parametrize("indices,expected", [(None, (1, 2)), ((0, 2, 3), (2,))])
def test_hmi_filters_actual_tai_times_before_loading(tmp_path, indices, expected):
    config = load_config(CONFIG)
    stream = config.streams[0]
    times = Time(["2011-02-13T23:59:59", "2011-02-14T00:00:00",
                  "2011-02-14T01:59:59", "2011-02-14T02:00:00"], scale="utc")
    for time in times.tai:
        date = time.to_datetime()
        header = fits.Header({"T_OBS": date.strftime("%Y.%m.%d_%H:%M:%S_TAI"),
                              "T_REC": date.strftime("%Y.%m.%d_%H:%M:%S_TAI"),
                              "CAMERA": 1, "HCAMID": 3})
        for stokes in "IQUV":
            for wavelength in range(6):
                filename = f"hmi.S_720s.{date:%Y%m%d_%H%M%S}_TAI.1.{stokes}{wavelength}.fits"
                fits.writeto(tmp_path / filename, np.zeros((1, 1)), header)
    stream = replace(stream, observation=replace(
        stream.observation, directory=tmp_path,
        selection=replace(stream.observation.selection, acquisition_indices=indices),
    ))
    bounds = tuple(float(Time(datetime.fromisoformat(value)).tai.to_value("unix_tai"))
                   for value in (config.scene.time_window.start, config.scene.time_window.end))
    selected = joint_streams._select_hmi_time_window(stream, bounds)
    assert selected.observation.selection.acquisition_indices == expected
    assert selected.observation.selection.validation_raster == expected[0]
    with pytest.raises(ValueError, match="No HMI acquisitions"):
        joint_streams._select_hmi_time_window(stream, (bounds[1] + 1, bounds[1] + 2))


def test_default_loader_passes_same_absolute_window_to_each_instrument(monkeypatch, tmp_path):
    config = load_config(CONFIG)
    calls = []
    monkeypatch.setattr(joint_streams, "_select_hmi_time_window",
                        lambda stream, bounds: calls.append(("hmi", bounds)) or stream)
    monkeypatch.setattr(joint_streams, "_load_stokes_stream", lambda *args, **kwargs: "hmi")
    monkeypatch.setattr(joint_streams, "_load_image_stream",
                        lambda *args, time_bounds_tai: calls.append(("aia", time_bounds_tai)) or "aia")
    for stream in config.streams:
        joint_streams._default_stream_loader(stream, {}, tmp_path, rebuild_observations=False,
                                             time_window=config.scene.time_window)
    assert calls[0][1] == calls[1][1]
    assert calls[0][1][1] - calls[0][1][0] == 7200
