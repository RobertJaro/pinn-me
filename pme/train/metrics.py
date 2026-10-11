"""Small, transient GPU accumulator for infrequent packed metric reductions."""

import torch
import torch.distributed as dist


class TrainingMetricWindow:
    """Average batch metrics over a window and all DDP ranks in one collective.

    Inputs are scalar batch means, matching the previous per-step rank average.
    Accumulation is detached and does not retain training graphs or checkpoints.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.keys = None
        self.total = None
        self.count = 0

    @torch.no_grad()
    def add(self, metrics):
        keys = tuple(sorted(metrics))
        if self.keys is not None and keys != self.keys:
            raise ValueError('Training metric names must remain constant within a logging window.')
        values = []
        for key in keys:
            value = metrics[key].detach().float()
            values.append(value.reshape(()) if value.numel() == 1 else value.mean())
        values = torch.stack(values)
        if self.total is None:
            self.keys, self.total = keys, values
        else:
            self.total.add_(values)
        self.count += 1

    @torch.no_grad()
    def drain(self):
        if self.count == 0:
            return {}
        # Include the count so partial windows retain the correct denominator.
        packed = torch.cat((self.total, self.total.new_tensor([self.count])))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        means = packed[:-1] / packed[-1]
        metrics = dict(zip(self.keys, means.unbind()))
        self.clear()
        return metrics
