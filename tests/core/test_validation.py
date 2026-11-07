"""Numerical checks preserve public failures and batch production host reads."""

import pytest
import torch

from prom3theus.core.validation import deferred_validation, require_tensor


def test_public_checks_raise_immediately():
    with pytest.raises(ValueError, match='invalid temperature'):
        require_tensor(torch.tensor(False), 'invalid temperature')


def test_nested_pipeline_checks_flush_once(monkeypatch):
    copies = []
    original = torch.Tensor.cpu
    def cpu(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    with deferred_validation():
        require_tensor(torch.tensor(True), 'outer')
        with deferred_validation():
            require_tensor(torch.tensor(True), 'inner')
        assert copies == []
    assert copies == [2]


def test_deferred_failure_reports_constraint_and_clears_context():
    reached_end = False
    with pytest.raises(ValueError, match='density'):
        with deferred_validation():
            require_tensor(torch.tensor(True), 'temperature')
            require_tensor(torch.tensor(False), 'density')
            reached_end = True
    assert reached_end
    with pytest.raises(ValueError, match='immediate'):
        require_tensor(torch.tensor(False), 'immediate')


def test_computation_exception_clears_context_without_hiding_original():
    with pytest.raises(RuntimeError, match='shape'):
        with deferred_validation():
            require_tensor(torch.tensor(False), 'invalid input')
            raise RuntimeError('shape')
    with deferred_validation():
        require_tensor(torch.tensor(True), 'next call')


def test_deferred_checks_preserve_exception_type():
    with pytest.raises(RuntimeError, match='composition'):
        with deferred_validation():
            require_tensor(torch.tensor(False), 'composition', error=RuntimeError)
