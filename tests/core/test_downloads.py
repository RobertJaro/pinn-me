"""Safe replacement of staged download bundles."""

import pytest

from prom3theus.core import downloads


@pytest.mark.parametrize("filename", ["", " ", ".", "..", "../outside.fits", "/outside.fits", None])
def test_invalid_drms_filenames_are_rejected_before_transfer(tmp_path, monkeypatch, filename):
    import drms
    import pandas as pd

    request = drms.ExportRequest(
        {"status": 0, "method": "url", "protocol": "fits", "requestid": "JSOC_TEST"},
        client=None,
    )
    request._download_urls_cache = pd.DataFrame([{
        "record": "hmi.S_720s[2011.02.14_00:00:00_TAI][3]{I0}",
        "filename": filename, "url": "https://example.invalid/unused",
    }])
    monkeypatch.setattr(request, "download", lambda *a, **kw: pytest.fail("unexpected transfer"))
    with pytest.raises(RuntimeError, match="JSOC_TEST.*invalid FITS filename.*hmi.S_720s"):
        downloads.download_fits_export(request, tmp_path)
    assert not list(tmp_path.iterdir())


def test_publish_failure_restores_existing_download(tmp_path, monkeypatch):
    output = tmp_path / "dataset"
    staging = tmp_path / "staging"
    output.mkdir()
    staging.mkdir()
    (output / "old.txt").write_text("previous data")
    replace = downloads.os.replace

    def fail_publication(source, destination):
        if source == staging:
            raise OSError("publish failed")
        return replace(source, destination)

    monkeypatch.setattr(downloads.os, "replace", fail_publication)
    with pytest.raises(OSError, match="publish failed"):
        downloads.publish_download(staging, output, overwrite=True)
    assert (output / "old.txt").read_text() == "previous data"
    assert staging.is_dir()


def test_publish_does_not_replace_an_output_without_overwrite(tmp_path):
    output = tmp_path / "dataset"
    staging = tmp_path / "staging"
    output.mkdir()
    staging.mkdir()
    with pytest.raises(FileExistsError):
        downloads.publish_download(staging, output, overwrite=False)
    assert output.is_dir()
    assert staging.is_dir()
