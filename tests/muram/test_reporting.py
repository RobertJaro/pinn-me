"""Progress reflects finished work; command logging does not leak configuration."""

from io import StringIO
import logging

import numpy as np
import pytest

from prom3theus.muram import columns, fits, reporting


class Terminal(StringIO):
    def isatty(self):
        return True


def test_command_logging_is_timestamped_filtered_and_restored(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(reporting.sys, "stderr", stream)
    package = logging.getLogger("prom3theus.muram")
    before = (package.level, package.propagate, list(package.handlers))
    logger = logging.getLogger("prom3theus.muram.test")
    for _ in range(2):
        with reporting.cli_logging("INFO"):
            logger.info("stage %s", "started")
            logger.debug("hidden debug")
    assert (package.level, package.propagate, package.handlers) == before
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert all("Z INFO stage started" in line for line in lines)
    assert "hidden debug" not in stream.getvalue()
    with pytest.raises(ValueError, match="bad snapshot"):
        with reporting.cli_logging("DEBUG"):
            raise ValueError("bad snapshot")
    assert (package.level, package.propagate, package.handlers) == before
    assert "ERROR ValueError: bad snapshot" in stream.getvalue()


@pytest.mark.parametrize("terminal,enabled,disabled", [(True, True, False), (True, False, True), (False, True, True)])
def test_progress_respects_terminal_and_disable_flag(monkeypatch, terminal, enabled, disabled):
    stream = Terminal() if terminal else StringIO()
    monkeypatch.setattr(reporting.sys, "stderr", stream)
    with reporting.progress_bar(total=2, description="Columns", unit="column", enabled=enabled) as bar:
        assert bar.disable is disabled
        bar.update(2)
    if disabled:
        assert not stream.getvalue()


def test_logs_are_written_through_tqdm_above_live_bars(monkeypatch):
    stream = Terminal()
    monkeypatch.setattr(reporting.sys, "stderr", stream)
    writes = []
    original = reporting.tqdm.write

    def record(message, **kwargs):
        writes.append((message, kwargs["file"]))
        return original(message, **kwargs)

    monkeypatch.setattr(reporting.tqdm, "write", record)
    with reporting.cli_logging():
        with reporting.progress_bar(total=2, description="Live", unit="column") as bar:
            bar.update()
            logging.getLogger("prom3theus.muram.test").info("saved maps")
            bar.update()
    assert len(writes) == 1
    assert writes[0][1] is stream
    assert "saved maps" in writes[0][0]


def test_column_progress_counts_after_processing_and_logs_batch_job_eta(monkeypatch, caplog):
    now = [0.0]
    monkeypatch.setattr(columns, "perf_counter", lambda: now[0])
    monkeypatch.setattr(reporting.sys, "stderr", StringIO())
    source = columns.ArrayColumns({"temperature": np.ones((3, 1, 2)), "ds_m": (1., 1., 1.)})
    caplog.set_level(logging.INFO, logger="prom3theus.muram.columns")
    batches = columns.column_batches(source, 1, progress=True, description="Iteration 42")
    next(batches)
    assert not caplog.records  # Loaded/prefetched work is not finished work.
    now[0] += 31
    next(batches)
    assert "1/3 columns (33.3%)" in caplog.text
    assert "ETA 62 s" in caplog.text
    now[0] += 31
    next(batches)
    now[0] += 31
    with pytest.raises(StopIteration):
        next(batches)
    assert "processed 3/3 columns in 93.0 s" in caplog.text
    assert "processing 93.0 s" in caplog.text


def test_interrupted_column_progress_does_not_claim_unfinished_batch(caplog):
    caplog.set_level(logging.INFO, logger="prom3theus.muram.columns")
    source = columns.ArrayColumns({"temperature": np.ones((3, 1, 2)), "ds_m": (1., 1., 1.)})
    batches = columns.column_batches(source, 1, progress=False)
    next(batches)
    batches.close()
    assert "processed 0/3 columns" in caplog.text
    assert "ETA" not in caplog.text


def test_timed_stage_reports_failure_without_success(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(OSError, match="full disk"):
        with reporting.timed_stage(logging.getLogger("prom3theus.muram.test"), "Saving iteration %d", 42):
            raise OSError("full disk")
    assert "Saving iteration 42 failed after" in caplog.text
    assert "OSError: full disk" in caplog.text
    assert "finished" not in caplog.text


def test_aia_preparation_receives_no_progress(monkeypatch, tmp_path):
    received = []

    def prepare(*args, **kwargs):
        received.append(kwargs["progress"])
        kwargs["source_files_sha256"]()
        return tmp_path

    monkeypatch.setattr(fits, "prepare_aia_observation_store", prepare)
    monkeypatch.setattr(reporting.sys, "stderr", Terminal())
    assert fits.prepare_aia_fits([], tmp_path, progress=False) == tmp_path
    assert received == [False]
    assert not reporting.sys.stderr.getvalue()
