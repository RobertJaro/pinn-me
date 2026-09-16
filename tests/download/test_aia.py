"""AIA downloads delegate directly to DRMS."""

from types import SimpleNamespace

import pandas as pd
import pytest

from prom3theus.download import aia


@pytest.mark.parametrize("start,end", [
    ("February 14, 2011", "2011/02/15"),
    ("2011-02-13T18:00:00-06:00", "2011-02-14T18:00:00-06:00"),
])
def test_direct_drms_download_for_each_channel(tmp_path, monkeypatch, start, end):
    calls = []
    def export(query, *, method, protocol):
        calls.append((query, method, protocol))
        def download(directory):
            assert directory == str(tmp_path)
            return pd.DataFrame({"download": [f"{directory}/{len(calls)}.fits"]})
        return SimpleNamespace(wait=lambda: None, download=download)
    monkeypatch.setattr(aia.drms, "Client", lambda **kwargs: SimpleNamespace(export=export))
    paths = aia.download_aia_observations(
        output_directory=tmp_path, email="test@example.org", start=start, end=end,
        channels=(171, 193, 211), cadence_seconds=720,
    )
    assert calls == [
        (f"aia.lev1_euv_12s[2011-02-14T00:00:00_UTC/86400s@720s][{channel}]{{image}}", "url", "fits")
        for channel in (171, 193, 211)
    ]
    assert len(paths) == 3
