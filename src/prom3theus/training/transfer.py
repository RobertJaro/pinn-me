"""One-batch CUDA lookahead with explicit copy/compute ownership."""

import torch


def _map(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: _map(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_map(item, function) for item in value)
    if isinstance(value, list):
        return [_map(item, function) for item in value]
    return value


def prefetch_cuda(batches, device):
    """Retain pinned source allocations until their asynchronous copies finish.

    The iterator owns transfers. Lightning must not perform a second transfer.
    Current-batch readiness is captured before enqueuing the next copy, allowing
    current compute to overlap that next copy on the dedicated stream.
    """
    stream = torch.cuda.Stream(device=device)
    pending = []

    def transfer(host):
        with torch.cuda.stream(stream):
            device_batch = _map(host, lambda x: x.to(device, non_blocking=True))
            ready = torch.cuda.Event()
            ready.record(stream)
        pending.append((ready, host))
        return device_batch, ready

    try:
        try:
            current, ready = transfer(next(batches))
        except StopIteration:
            return
        while True:
            torch.cuda.current_stream(device).wait_event(ready)
            _map(current, lambda x: x.record_stream(torch.cuda.current_stream(device)))
            try:
                future = transfer(next(batches))
            except StopIteration:
                future = None
            pending[:] = [(event, host) for event, host in pending if not event.query()]
            yield current
            if future is None:
                return
            current, ready = future
    finally:
        # Teardown only. No device-wide barrier in steady training.
        stream.synchronize()
        pending.clear()
        batches.close()
