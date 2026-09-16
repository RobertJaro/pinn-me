"""Fixed reference epochs with directly seekable, completed-batch cursors."""

from collections.abc import Sequence
from copy import deepcopy


class StreamBatchScheduler:
    version = 3

    def __init__(self, loaders, reference_stream, *, explicit_commit=False, rank=0, world_size=1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed schedule rank/world_size")
        self.rank, self.world_size = rank, world_size
        self.loaders = dict(loaders)
        if reference_stream not in self.loaders:
            raise ValueError("Training reference must name a configured stream")
        if any(len(loader) == 0 for loader in self.loaders.values()):
            raise ValueError("Every training stream must contain batches")
        self.reference_stream = reference_stream
        self.explicit_commit = explicit_commit
        self.completed = self.delivered = 0
        self._active = []
        self.device = None

    def __len__(self):
        import math
        return math.ceil(len(self.loaders[self.reference_stream]) / self.world_size)

    def _iterator(self, loader, start, count):
        if hasattr(loader, "iter_from"):
            return loader.iter_from(start, count)
        if isinstance(loader, Sequence):
            return iter(
                loader[index % len(loader)] for index in range(start, start + count)
            )
        # Generic in-memory plugin DataLoaders can seek deterministic batch
        # descriptors without replaying any payload. Raster loaders never use it.
        from torch.utils.data import DataLoader, SequentialSampler

        if isinstance(loader, DataLoader) and isinstance(
            loader.sampler, SequentialSampler
        ):

            def selected_batches():
                for number in range(start, start + count):
                    index = number % len(loader)
                    if loader.batch_size is None:
                        yield loader.collate_fn(loader.dataset[index])
                    else:
                        begin = index * loader.batch_size
                        yield loader.collate_fn(
                            [
                                loader.dataset[i]
                                for i in range(
                                    begin,
                                    min(begin + loader.batch_size, len(loader.dataset)),
                                )
                            ]
                        )

            return selected_batches()
        if start % len(loader):
            raise ValueError("Resume requires a seekable loader implementing iter_from")

        def batches():
            emitted = 0
            while emitted < count:
                for batch in loader:
                    yield batch
                    emitted += 1
                    if emitted == count:
                        return

        return batches()

    def _selected_iterator(self, loader, indices):
        if hasattr(loader, "iter_indices"):
            return loader.iter_indices(indices)
        if isinstance(loader, Sequence):
            return iter(loader[index % len(loader)] for index in indices)
        def selected():
            for index in indices:
                iterator = self._iterator(loader, index, 1)
                try:
                    yield next(iterator)
                finally:
                    close = getattr(iterator, "close", None)
                    if close:
                        close()
        return selected()

    def __iter__(self):
        self.close()
        start = self.completed
        self.delivered = start
        count = len(self) - start % len(self)
        # All background producers start before the first blocking receive.
        validity = [True] * count
        if self.world_size == 1:
            self._active = [self._iterator(loader, start, count) for loader in self.loaders.values()]
        else:
            reference_count = len(self.loaders[self.reference_stream])
            epoch, within = divmod(start, len(self))
            slots = [(within + i) * self.world_size + self.rank for i in range(count)]
            validity = [slot < reference_count for slot in slots]
            indices = [epoch * reference_count + min(slot, reference_count - 1) for slot in slots]
            self._active = [self._selected_iterator(loader, indices) for loader in self.loaders.values()]

        def host_batches():
            for valid in validity:
                batch = {name: next(iterator) for name, iterator in zip(self.loaders, self._active, strict=True)}
                if self.world_size > 1:
                    batch["_data_valid"] = valid
                yield batch

        batches = host_batches()
        if self.device is not None:
            from .transfer import prefetch_cuda

            batches = prefetch_cuda(batches, self.device)
        try:
            for batch in batches:
                self.delivered += 1
                if not self.explicit_commit:
                    self.completed = self.delivered
                yield batch
        finally:
            batches.close()
            self.close()

    def commit(self):
        if self.completed >= self.delivered:
            raise RuntimeError("Cannot commit a batch that has not been delivered")
        self.completed += 1

    def _contract(self):
        return {
            "world_size": self.world_size,
            "reference_stream": self.reference_stream,
            "lengths": {name: len(loader) for name, loader in self.loaders.items()},
            "loaders": {
                name: loader.contract() if hasattr(loader, "contract") else None
                for name, loader in self.loaders.items()
            },
        }

    def state_dict(self):
        return deepcopy(
            {"version": self.version, "completed": self.completed, **self._contract()}
        )

    def load_state_dict(self, state, *, migrate=False):
        self.order_migrated = False
        state = deepcopy(state)
        if state.get("version") == 2 and self.world_size == 1:
            state.update(version=self.version, world_size=1)
        if state.get("version") != self.version:
            if not migrate:
                raise ValueError(
                    "Legacy checkpoint: set migrate_data_order=true "
                    "to restart with the current sampling order"
                )
            self.order_migrated = True
            self.completed = self.delivered = 0
            self.close()
            return
        contract = self._contract()
        if state.get("world_size", 1) != self.world_size:
            if not migrate:
                raise ValueError("World size changed; explicitly migrate_data_order to restart coverage")
            state["world_size"] = self.world_size
            state["completed"] = 0
            self.order_migrated = True
        if any(state.get(key) != value for key, value in contract.items()):
            # A batch-size change changes chunk boundaries. Resume optimization
            # but restart coverage with the new chunks; retain data/seed checks.
            old_layout = {key: deepcopy(state.get(key)) for key in contract}
            new_layout = deepcopy(contract)
            batch_changed = False
            for name, loader in (new_layout.get("loaders") or {}).items():
                saved_loader = (old_layout.get("loaders") or {}).get(name, {})
                if isinstance(loader, dict) and isinstance(saved_loader, dict):
                    batch_changed |= saved_loader.get("batch_size") != loader.get("batch_size")
            for layout in (old_layout, new_layout):
                layout.pop("lengths", None)
                for loader in (layout.get("loaders") or {}).values():
                    if isinstance(loader, dict):
                        loader.pop("batch_size", None)
            if batch_changed and old_layout == new_layout:
                self.close()
                self.order_migrated = True
                self.completed = self.delivered = 0
                print("Resuming with changed batch size: restarting data coverage using the new chunks.", flush=True)
                return
        if any(state.get(key) != value for key, value in contract.items()):
            # An explicit order migration may reset coverage, but cannot hide
            # changes to the data, batch sizes, or reference stream.
            saved = {key: deepcopy(state.get(key)) for key in contract}
            current = deepcopy(contract)
            native_contracts = all(
                isinstance(loader, dict) and "sequences" in loader
                for schedule in (saved, current)
                for loader in (schedule.get("loaders") or {}).values()
            )
            for schedule in (saved, current):
                # Different batch-order policies may count channel tails differently.
                # Dataset identity and configured batch sizes remain checked below.
                if native_contracts:
                    schedule.pop("lengths", None)
                for loader in (schedule.get("loaders") or {}).values():
                    if isinstance(loader, dict):
                        loader.pop("shuffle", None)
            if not migrate or saved != current:
                raise ValueError("Checkpoint stream schedule differs from this run")
            self.close()
            self.order_migrated = True
            self.completed = self.delivered = 0
            return
        if type(state["completed"]) is not int or state["completed"] < 0:
            raise ValueError("Invalid completed stream cursor")
        self.close()
        self.completed = self.delivered = state["completed"]

    def close(self):
        for iterator in self._active:
            close = getattr(iterator, "close", None)
            if close:
                close()
        self._active.clear()

    def dataloader(self):
        return self
