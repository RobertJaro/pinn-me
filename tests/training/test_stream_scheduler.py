from prom3theus.training.streams import StreamBatchScheduler
import pytest


def test_reference_defines_epoch_and_other_stream_continues():
    scheduler = StreamBatchScheduler(
        {"hmi": [1, 2], "aia": [10, 11, 12, 13, 14]}, "hmi"
    )
    assert len(scheduler) == 2
    assert list(scheduler) == [{"hmi": 1, "aia": 10}, {"hmi": 2, "aia": 11}]
    assert list(scheduler) == [{"hmi": 1, "aia": 12}, {"hmi": 2, "aia": 13}]
    assert list(scheduler) == [{"hmi": 1, "aia": 14}, {"hmi": 2, "aia": 10}]


@pytest.mark.parametrize(
    "loaders,reference", [({"a": []}, "a"), ({"a": [1]}, "missing")]
)
def test_scheduler_rejects_invalid_reference_or_empty_stream(loaders, reference):
    with pytest.raises(ValueError):
        StreamBatchScheduler(loaders, reference)


def test_only_completed_batches_are_checkpointed():
    def create():
        return StreamBatchScheduler(
            {"hmi": list(range(7)), "aia": list(range(3))}, "hmi", explicit_commit=True
        )

    scheduler = create()
    iterator = iter(scheduler)
    assert next(iterator) == {"hmi": 0, "aia": 0}
    scheduler.commit()
    next(iterator)  # Delivered but not trained, plus any speculative prefetch.
    state = scheduler.state_dict()
    assert state["completed"] == 1
    restored = create()
    restored.load_state_dict(state)
    assert next(iter(restored)) == {"hmi": 1, "aia": 1}
    iterator.close()


def test_resume_seeks_without_reading_consumed_payloads():
    class Seekable:
        def __len__(self):
            return 1_000_000

        def iter_from(self, start, count):
            assert start == 999_998
            return iter(range(start, start + count))

    loader = StreamBatchScheduler({"a": Seekable()}, "a")
    state = loader.state_dict()
    state["completed"] = 999_998
    loader.load_state_dict(state)
    assert list(loader) == [{"a": 999_998}, {"a": 999_999}]


def test_legacy_checkpoint_requires_explicit_migration():
    loader = StreamBatchScheduler({"a": [1]}, "a")
    with pytest.raises(ValueError, match="migrate_data_order"):
        loader.load_state_dict({"streams": {}})
    loader.load_state_dict({"streams": {}}, migrate=True)
    assert loader.completed == 0


def test_lightning_mid_epoch_checkpoint_commits_before_save(tmp_path):
    import torch
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from prom3theus.application.joint_training import CommitStreamBatch

    class Module(LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.stream_scheduler = StreamBatchScheduler(
                {"a": [torch.tensor(float(i)) for i in range(5)]},
                "a",
                explicit_commit=True,
            )
            self.seen = []

        def training_step(self, batch, batch_idx):
            self.seen.append(int(batch["a"]))
            return (self.weight - batch["a"]).square()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

        def on_save_checkpoint(self, checkpoint):
            checkpoint["cursor"] = self.stream_scheduler.state_dict()

        def on_load_checkpoint(self, checkpoint):
            self.stream_scheduler.load_state_dict(checkpoint["cursor"])

    def trainer(steps, path):
        return Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=steps,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                CommitStreamBatch(),
                ModelCheckpoint(
                    dirpath=path, every_n_train_steps=1, save_last=True, save_top_k=0
                ),
            ],
        )

    original = Module()
    trainer(3, tmp_path / "first").fit(
        original, train_dataloaders=original.stream_scheduler
    )
    checkpoint = tmp_path / "first/last.ckpt"
    assert torch.load(checkpoint, weights_only=False)["cursor"]["completed"] == 3
    restored = Module()
    trainer(7, tmp_path / "second").fit(
        restored, train_dataloaders=restored.stream_scheduler, ckpt_path=checkpoint
    )
    assert original.seen + restored.seen == [0, 1, 2, 3, 4, 0, 1]


def test_shuffle_contract_requires_explicit_migration_and_preserves_other_checks():
    class Loader(list):
        def __init__(self, shuffle=None, batch_size=2):
            super().__init__([1, 2])
            self.shuffle, self.batch_size = shuffle, batch_size

        def contract(self):
            result = {"batch_size": self.batch_size}
            if self.shuffle is not None:
                result["shuffle"] = self.shuffle
            return result

    original = StreamBatchScheduler({"a": Loader()}, "a")
    list(original)
    state = original.state_dict()
    changed = StreamBatchScheduler({"a": Loader({"seed": 0})}, "a")
    with pytest.raises(ValueError, match="schedule differs"):
        changed.load_state_dict(state)
    changed.load_state_dict(state, migrate=True)
    assert changed.completed == 0
    incompatible = StreamBatchScheduler({"a": Loader({"seed": 0}, batch_size=3)}, "a")
    with pytest.raises(ValueError, match="schedule differs"):
        incompatible.load_state_dict(state, migrate=True)
