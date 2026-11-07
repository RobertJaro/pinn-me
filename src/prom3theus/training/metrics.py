"""Distributed display metrics, separate from differentiable loss reduction."""

import torch


def reduce_training_metrics(metrics, evaluation, *, data_valid=True):
    """Combine scalar metrics with their sample counts and reduction semantics.

    The objective and weighted data losses already compensate for DDP gradient
    averaging and require a plain rank mean. Per-stream fit metrics instead use
    valid observation counts; RMS values combine in squared units, and maxima
    remain maxima. Padded batches do not contribute observation statistics.
    """
    if not torch.distributed.is_initialized():
        return metrics
    names = tuple(metrics)
    values = torch.stack([metrics[name].detach().to(evaluation.loss).reshape(())
                          for name in names])
    counts, rms_indices, max_indices = [], [], []
    for index, name in enumerate(names):
        parts = name.split(".", 2)
        count = 1
        if (len(parts) == 3 and parts[0] == "streams"
                and parts[2] not in {"data_loss", "weight"}):
            count = evaluation.streams[parts[1]].sample_count * int(data_valid)
        counts.append(count)
        if name.endswith(("_rms", "_rmse")):
            rms_indices.append(index)
        elif name.endswith("_max"):
            max_indices.append(index)
    weights = values.new_tensor(counts)
    numerator = values.clone()
    numerator[rms_indices] = values[rms_indices].square()
    numerator = torch.where(weights > 0, numerator * weights, 0.)
    sums = torch.stack((numerator, weights))
    torch.distributed.all_reduce(sums)
    reduced = sums[0] / sums[1].clamp_min(1)
    reduced[rms_indices] = reduced[rms_indices].clamp_min(0).sqrt()
    if max_indices:
        maxima = torch.where(weights[max_indices] > 0, values[max_indices], -torch.inf)
        torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
        reduced[max_indices] = torch.where(sums[1, max_indices] > 0, maxima, 0.)
    return dict(zip(names, reduced.unbind(), strict=True))
