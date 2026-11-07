"""Per-batch distributed likelihood normalization, independent of trainer APIs."""

from contextlib import contextmanager
from contextvars import ContextVar
import torch

_VALID = ContextVar("distributed_data_valid", default=None)


@contextmanager
def distributed_batch(valid=True):
    token = _VALID.set(bool(valid))
    try:
        yield
    finally:
        _VALID.reset(token)


def distributed_means(sums, counts):
    valid = _VALID.get()
    if valid is None or not torch.distributed.is_initialized():
        return sums / counts.clamp_min(1), counts
    local_counts = counts * int(valid)
    total_counts = local_counts.clone()
    torch.distributed.all_reduce(total_counts)
    # DDP averages gradients; compensate to obtain the true global sample mean.
    means = (
        sums
        * int(valid)
        * torch.distributed.get_world_size()
        / total_counts.clamp_min(1)
    )
    return means, total_counts


def distributed_mean(mean, count):
    if _VALID.get() is None or not torch.distributed.is_initialized():
        return mean
    counts = torch.tensor(count, device=mean.device, dtype=torch.long)
    return distributed_means(mean * counts, counts)[0]


def initialize_distributed(device, *, devices="auto"):
    """Launch native Lightning DDP before any data/model initialization.

    Fabric supplies the public Lightning process-launch API; the Trainer owns
    model wrapping, gradient synchronization and optimization afterwards.
    Native child processes re-enter here through the same Python entry point.
    """
    import os
    from datetime import timedelta

    if devices == "auto":
        # CUDA_VISIBLE_DEVICES is already respected by PyTorch's device count.
        devices = torch.cuda.device_count() if device.type == "cuda" and device.index is None else 1
        if devices < 1:
            raise RuntimeError("Automatic CUDA device selection found no visible GPUs")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.distributed.is_initialized() and devices > 1 and "RANK" not in os.environ:
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Distributed training requires CPU or CUDA devices")
        if device.index is not None:
            raise ValueError("Use device: cuda with devices > 1, not an indexed CUDA device")
        from lightning_fabric import Fabric

        Fabric(accelerator=device.type, devices=devices, strategy="ddp").launch()
    if world_size > 1 and not torch.distributed.is_initialized():
        if device.type == "cuda":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(minutes=30),
        )
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1
