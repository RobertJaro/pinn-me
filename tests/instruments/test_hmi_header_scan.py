"""Parallel response-header discovery keeps deterministic acquisition identity."""

from threading import Barrier, get_ident

from prom3theus.instruments.hmi import acquisition


def test_response_header_scan_is_parallel_and_deduplicated(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PROM3THEUS_PREP_WORKERS", "3")
    paths = [tmp_path / f"{index}.I0.fits" for index in range(3)]
    monkeypatch.setattr(acquisition, "_resolve_input_paths", lambda inputs: paths)
    barrier = Barrier(3, timeout=10)
    threads = set()

    def read_header(path):
        threads.add(get_ident())
        barrier.wait()
        index = paths.index(path)
        return {"acquisition_key": ["b", "a", "b"][index], "path": str(path)}

    monkeypatch.setattr(acquisition, "read_acquisition_header", read_header)
    result = acquisition.discover_hmi_acquisitions(tmp_path)
    assert len(threads) == 3
    assert [item["acquisition_key"] for item in result] == ["a", "b"]
    assert result[1]["path"] == str(paths[0])
    assert "HMI response headers" in capsys.readouterr().err
