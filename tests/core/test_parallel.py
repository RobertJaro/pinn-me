from threading import Barrier, get_ident

import pytest

from prom3theus.core.parallel import parallel_map


def test_workers_share_memory_and_preserve_result_order(monkeypatch, capsys):
    monkeypatch.setenv("PROM3THEUS_PREP_WORKERS", "3")
    barrier = Barrier(3, timeout=10)
    shared_table = object()
    threads = set()

    def work(index):
        threads.add(get_ident())
        barrier.wait()
        return index, shared_table

    result = parallel_map(work, range(3), description="Shared table test")
    assert len(threads) == 3
    assert [index for index, _ in result] == [0, 1, 2]
    assert all(table is shared_table for _, table in result)
    assert "Shared table test" in capsys.readouterr().err


def test_worker_exception_propagates(monkeypatch):
    monkeypatch.setenv("PROM3THEUS_PREP_WORKERS", "1")

    def fail(_):
        raise ValueError("invalid image")

    with pytest.raises(ValueError, match="invalid image"):
        parallel_map(fail, [1], description="Failure")
