"""Coverage and real two-process collective tests; no CUDA requirement."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch

from prom3theus.training.streams import StreamBatchScheduler


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("length", [1, 5, 8])
def test_rank_partition_and_resume_cover_each_reference_once(world_size, length):
    schedules = [
        StreamBatchScheduler(
            {"a": list(range(length)), "b": list(range(3))},
            "a",
            rank=rank,
            world_size=world_size,
            explicit_commit=True,
        )
        for rank in range(world_size)
    ]
    for epoch in range(2):
        observed = []
        for rank, scheduler in enumerate(schedules):
            iterator = iter(scheduler)
            first = next(iterator)
            # Delivered but uncommitted work is replayed on this rank.
            restored = StreamBatchScheduler(
                scheduler.loaders,
                "a",
                rank=rank,
                world_size=world_size,
                explicit_commit=True,
            )
            restored.load_state_dict(scheduler.state_dict())
            replay = iter(restored)
            assert next(replay) == first
            replay.close()
            for batch in [first, *iterator]:
                scheduler.commit()
                if batch.get("_data_valid", True):
                    observed.append(batch["a"])
                    assert batch["b"] == (epoch * length + batch["a"]) % 3
        assert sorted(observed) == list(range(length))


def _collective_worker(rank, root):
    from prom3theus.core.distributed import (
        distributed_batch,
        distributed_mean,
        distributed_means,
    )
    from prom3theus.observations.persistence import (
        JointDataModule,
        restore_or_create_data_module,
    )

    root = Path(root)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:

        def build():
            assert rank == 0
            (root / "built").write_text("once")
            return JointDataModule({"x": torch.arange(9)}, None)

        module = restore_or_create_data_module(root / "module.pt", build)
        generation = module.generation
        module = restore_or_create_data_module(
            root / "module.pt", lambda: pytest.fail("rebuilt")
        )
        assert generation == module.generation
        assert module.streams["x"][2].item() == 2
        for padded in (False, True):
            weight = torch.tensor(1.0, requires_grad=True)
            target = torch.tensor([1.0, 3.0, 5.0]) if rank == 0 else torch.tensor([9.0])
            with distributed_batch(not padded or rank == 0):
                loss = distributed_mean((weight * target).mean(), len(target))
            loss.backward()
            torch.distributed.all_reduce(weight.grad)
            weight.grad /= 2
            assert weight.grad.item() == (3.0 if padded else 4.5)
        # Unequal per-channel counts, including an absent local channel.
        weight = torch.ones(2, requires_grad=True)
        sums = weight * (
            torch.tensor([6.0, 0.0]) if rank == 0 else torch.tensor([10.0, 8.0])
        )
        counts = torch.tensor([2, 0]) if rank == 0 else torch.tensor([1, 2])
        with distributed_batch():
            means, total = distributed_means(sums, counts)
        means.sum().backward()
        torch.distributed.all_reduce(weight.grad)
        weight.grad /= 2
        torch.testing.assert_close(weight.grad, torch.tensor([16 / 3, 4.0]))
        assert total.tolist() == [3, 2]
        # Exercise Lightning DDP and checkpoint callbacks, not only collectives.
        import os
        from pytorch_lightning import LightningModule, Trainer
        from prom3theus.application.joint_training import CommitStreamBatch

        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
            LOCAL_WORLD_SIZE="2",
            GROUP_RANK="0",
            TORCHELASTIC_RUN_ID="data-test",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="12999",
        )

        class Model(LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.stream_scheduler = StreamBatchScheduler(
                    {"a": [torch.tensor(float(x)) for x in range(1, 6)]},
                    "a",
                    rank=rank,
                    world_size=2,
                    explicit_commit=True,
                )

            def training_step(self, batch, batch_idx):
                with distributed_batch(batch["_data_valid"]):
                    return distributed_mean(self.weight * batch["a"], 1)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

            def on_save_checkpoint(self, checkpoint):
                checkpoint["cursor"] = self.stream_scheduler.state_dict()

            def on_load_checkpoint(self, checkpoint):
                self.stream_scheduler.load_state_dict(checkpoint["cursor"])

        def trainer(steps):
            return Trainer(
                accelerator="cpu",
                devices=2,
                strategy="ddp",
                max_steps=steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                use_distributed_sampler=False,
                callbacks=[CommitStreamBatch()],
            )

        model = Model()
        fit = trainer(3)
        fit.fit(model, train_dataloaders=model.stream_scheduler)
        torch.testing.assert_close(model.weight, torch.tensor(0.0), atol=1e-6, rtol=0)
        checkpoint = root / "training.ckpt"
        fit.save_checkpoint(checkpoint)
        resumed = Model()
        trainer(4).fit(
            resumed, train_dataloaders=resumed.stream_scheduler, ckpt_path=checkpoint
        )
        torch.testing.assert_close(
            resumed.weight, torch.tensor(-0.15), atol=1e-6, rtol=0
        )
        assert resumed.stream_scheduler.completed == 4

        def broken():
            raise ValueError("deliberate build error")

        with pytest.raises((ValueError, RuntimeError), match="deliberate build error"):
            restore_or_create_data_module(root / "broken.pt", broken)
    finally:
        torch.distributed.destroy_process_group()


def test_two_process_publication_and_global_loss_weighting(tmp_path):
    torch.multiprocessing.spawn(
        _collective_worker, args=(str(tmp_path),), nprocs=2, join=True
    )
