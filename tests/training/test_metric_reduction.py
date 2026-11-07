"""Distributed metric moments for unequal and padded observation batches."""

from types import SimpleNamespace

import pytest
import torch

from prom3theus.training.metrics import reduce_training_metrics


@pytest.mark.parametrize('left_count,padded,expected_rms', [(1, False, 2**.5), (3, False, 1.), (3, True, 0.)])
def test_metrics_preserve_rms_max_and_valid_sample_weighting(monkeypatch, left_count, padded, expected_rms):
    # Rank zero observes zeros. Rank one observes a single value of two, unless
    # its batch is padding. Supply that rank's known moments to the collectives.
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    count = 0. if padded else 1.
    calls = []

    def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM):
        calls.append(op)
        if op == torch.distributed.ReduceOp.MAX:
            tensor.copy_(torch.maximum(tensor, tensor.new_tensor([-torch.inf if padded else 2.])))
        else:
            tensor.add_(tensor.new_tensor([[2., 2., 4.*count, 2.*count, 4.*count],
                                           [1., 1., count, count, count]]))

    monkeypatch.setattr(torch.distributed, 'all_reduce', all_reduce)
    metrics = {'loss': torch.tensor(1.),
               'streams.stokes.data_loss': torch.tensor(1.),
               'streams.stokes.stokes_I_raw_residual_rms': torch.tensor(0.),
               'streams.stokes.stokes_outgoing_admissibility_max': torch.tensor(0.),
               'streams.stokes.component_mse.I': torch.tensor(0.)}
    evaluation = SimpleNamespace(loss=metrics['loss'],
        streams={'stokes': SimpleNamespace(sample_count=left_count)})
    result = reduce_training_metrics(metrics, evaluation)
    torch.testing.assert_close(result['loss'], torch.tensor(1.5))
    torch.testing.assert_close(result['streams.stokes.data_loss'], torch.tensor(1.5))
    torch.testing.assert_close(result['streams.stokes.stokes_I_raw_residual_rms'], torch.tensor(expected_rms))
    torch.testing.assert_close(result['streams.stokes.component_mse.I'], torch.tensor(expected_rms**2))
    assert result['streams.stokes.stokes_outgoing_admissibility_max'] == (0. if padded else 2.)
    assert calls == [torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.MAX]


def test_padded_local_rank_has_no_effect_on_observation_statistics(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    metric = 'streams.stokes.stokes_I_raw_residual_rms'
    metrics = {'loss': torch.tensor(0.), metric: torch.tensor(99.)}
    evaluation = SimpleNamespace(loss=metrics['loss'],
        streams={'stokes': SimpleNamespace(sample_count=10)})

    def all_reduce(tensor):
        # Invalid local observation moments/counts are zero; loss still averages
        # equally because its DDP normalization already accounts for validity.
        torch.testing.assert_close(tensor, tensor.new_tensor([[0., 0.], [1., 0.]]))
        tensor.add_(tensor.new_tensor([[4., 12.], [1., 3.]]))

    monkeypatch.setattr(torch.distributed, 'all_reduce', all_reduce)
    result = reduce_training_metrics(metrics, evaluation, data_valid=False)
    assert result[metric] == 2.
    assert result['loss'] == 2.
