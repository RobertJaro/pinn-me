"""HMI downloads use DRMS directly."""

from types import SimpleNamespace

import pandas as pd
import pytest

from prom3theus.download import hmi


@pytest.mark.parametrize("start,end,interval", [
    ("February 14, 2011", "2011/02/15", "2011-02-14T00:00:00_TAI/86400s"),
    ("2024-03-23T22:11:23Z", "2024-03-23T16:23:23-06:00", "2024-03-23T22:12:00_TAI/720s"),
])
def test_direct_drms_download(tmp_path, monkeypatch, start, end, interval):
    calls = []
    files = [str(tmp_path / "downloaded.fits")]

    def download(directory):
        calls.append(("download", directory))
        return pd.DataFrame({"download": files})

    def export(query, *, method, protocol):
        calls.append(("export", query, method, protocol))
        return SimpleNamespace(wait=lambda: calls.append(("wait",)), download=download)

    def client(*, email):
        assert email == "test@example.org"
        return SimpleNamespace(export=export)

    monkeypatch.setattr(hmi.drms, "Client", client)
    result = hmi.download_hmi_stokes(
        output_directory=tmp_path, email="test@example.org", start=start, end=end,
    )
    segments = ",".join(f"{component}{i}" for component in "IQUV" for i in range(6))
    assert calls == [
        ("export", f"hmi.S_720s[{interval}]{{{segments}}}", "url", "fits"),
        ("wait",), ("download", str(tmp_path)),
    ]
    assert result == files


@pytest.mark.parametrize("time", ["2011-02-14T01:00:00", "2011-02-14T00:59:26Z"])
def test_comparison_download_selects_one_vector_record(tmp_path, monkeypatch, time):
    segments = ("field", "inclination", "azimuth", "disambig")
    files = [tmp_path / f"hmi.b_720s.20110214_010000_TAI.{s}.fits" for s in segments]
    calls = []

    def export(query, *, method, protocol):
        calls.append((query, method, protocol))
        return SimpleNamespace(
            wait=lambda: None,
            urls=pd.DataFrame({"record": ["record"] * 4, "filename": [p.name for p in files]}),
            download=download,
        )

    def download(directory, *, fname_from_rec):
        assert directory == str(tmp_path)
        assert fname_from_rec is False
        for path in files:
            path.touch()
        return pd.DataFrame({"download": files})

    monkeypatch.setattr(hmi.drms, "Client", lambda **kw: SimpleNamespace(export=export))
    assert hmi.download_hmi_comparison(
        output_directory=tmp_path, email="test@example.org", time=time,
    ) == [str(p) for p in files]
    assert calls == [(
        "hmi.B_720s[2011-02-14T01:00:00_TAI]{field,inclination,azimuth,disambig}",
        "url", "fits",
    )]


def test_comparison_download_rejects_failed_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(hmi.drms, "Client", lambda **kw: SimpleNamespace(export=lambda *a, **k: None))
    monkeypatch.setattr(hmi, "download_fits_export", lambda *a: pd.DataFrame({"download": [None] * 4}))
    with pytest.raises(RuntimeError, match="all four FITS"):
        hmi.download_hmi_comparison(
            output_directory=tmp_path, email="test@example.org", time="2011-02-14T01:00:00",
        )
