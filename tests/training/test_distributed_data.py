"""Coverage and real two-process collective tests; no CUDA requirement."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch

from prom3theus.training.streams import StreamBatchScheduler


@pytest.mark.parametrize("device,visible,expected", [
    ("cuda", 4, 4), ("cuda", 1, 1), ("cuda:2", 4, 1), ("cpu", 4, 1),
])
def test_auto_devices_uses_visible_gpus_and_native_ddp(monkeypatch, device, visible, expected):
    import lightning_fabric
    from prom3theus.core.distributed import initialize_distributed

    calls = []
    active = [False]

    class Launcher:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def launch(self):
            active[0] = True

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible)
    monkeypatch.setattr(lightning_fabric, "Fabric", Launcher)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: active[0])
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: expected)
    assert initialize_distributed(torch.device(device)) == (0, expected)
    assert calls == ([dict(accelerator='cuda', devices=expected, strategy='ddp')]
                     if expected > 1 else [])


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Requires Gloo")
def test_native_ddp_launches_before_sampling_and_synchronizes_gradients(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    worker = tmp_path / "native_ddp.py"
    worker.write_text(textwrap.dedent('''
        from pathlib import Path
        import torch
        from prom3theus.core.distributed import initialize_distributed
        from prom3theus.observations.persistence import JointDataModule, restore_or_create_data_module
        from prom3theus.training.streams import StreamBatchScheduler

        if __name__ == '__main__':
            torch.set_num_threads(1)
            rank, world = initialize_distributed(torch.device('cpu'), devices=2)
            assert world == 2
            calls = torch.zeros(2, dtype=torch.long)
            def build():
                assert rank == 0, 'worker must never construct the data module'
                calls[0] += 1
                return JointDataModule({'values': torch.arange(8)}, None)
            def finalize(module):
                assert rank == 0, 'worker must never prepare training pools'
                calls[1] += 1
                return False
            path = Path(__file__).parent / 'data_module.pt'
            module = restore_or_create_data_module(path, build, finalize=finalize)
            first_generation = module.generation
            module = restore_or_create_data_module(path, build, finalize=finalize)
            assert module.generation == first_generation
            assert calls.tolist() == ([1, 2] if rank == 0 else [0, 0])
            torch.distributed.barrier()
            module = restore_or_create_data_module(path, build, finalize=finalize, rebuild=True)
            assert module.generation != first_generation
            assert calls.tolist() == ([2, 3] if rank == 0 else [0, 0])
            generations = [None] * world
            torch.distributed.all_gather_object(generations, module.generation)
            assert len(set(generations)) == 1
            torch.testing.assert_close(module.streams['values'][:], torch.arange(8))
            batches = StreamBatchScheduler({'x': [1., 3., 5., 7.]}, 'x', rank=rank, world_size=world)
            local = [batch['x'] for batch in batches]
            assert local == ([1., 5.] if rank == 0 else [3., 7.])
            model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
            loss = model(torch.tensor([[local[0]]])).sum()
            loss.backward()
            torch.testing.assert_close(model.module.weight.grad, torch.tensor([[2.]]))
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            optimizer.step()
            copies = [torch.empty_like(model.module.weight) for _ in range(world)]
            torch.distributed.all_gather(copies, model.module.weight.detach())
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
            torch.distributed.barrier()
            if rank == 0:
                print('NATIVE_DDP_OK', flush=True)
            torch.distributed.destroy_process_group()
    '''))
    environment = dict(os.environ, MASTER_ADDR="127.0.0.1", OMP_NUM_THREADS="1")
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_PORT',
                'NODE_RANK', 'GROUP_RANK', 'TORCHELASTIC_RUN_ID'):
        environment.pop(key, None)
    root = Path(__file__).resolve().parents[2]
    environment['PYTHONPATH'] = str(root / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, str(worker)], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'NATIVE_DDP_OK' in result.stdout


@pytest.mark.parametrize("old_world,new_world", [(1, 4), (4, 2), (2, 1)])
def test_changed_world_size_restarts_coverage_without_changing_data_contract(old_world, new_world):
    loaders = {"a": list(range(8))}
    old = StreamBatchScheduler(loaders, "a", world_size=old_world)
    list(old)
    saved = old.state_dict()
    observed = []
    for rank in range(new_world):
        current = StreamBatchScheduler(loaders, "a", rank=rank, world_size=new_world)
        current.load_state_dict(saved)
        assert current.completed == 0
        assert current.order_migrated
        observed.extend(batch['a'] for batch in current if batch.get('_data_valid', True))
    assert sorted(observed) == list(range(8))
    changed = StreamBatchScheduler({"a": list(range(9))}, "a", world_size=new_world)
    with pytest.raises(ValueError, match="Checkpoint stream schedule differs"):
        changed.load_state_dict(saved)


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
