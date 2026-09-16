"""One trusted, versioned data module; only rank zero publishes generations."""

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import torch

from .snapshot import (
    BatchDataModule,
    BatchSequence,
    SamplingSupport,
    SnapshotWriter,
    array_references,
    restore,
)


@dataclass
class JointDataModule:
    streams: dict
    scene: object
    train_loaders: dict | None = None
    summaries: dict | None = None
    contract: dict | None = None
    version: int = 2
    generation: str = ""
    root: str = ""
    encoded: bool = False

    def setup(self, stage=None):
        """A restored module is already prepared."""

    def train_dataloader(self, *, batch_sizes=None, **options):
        if self.train_loaders is None:
            raise ValueError("This data module has no prepared training pools")
        from .tensor_loader import PackedLoaderSpec

        return {
            name: spec.bind(batch_size=(batch_sizes or {}).get(name), **options)
            if isinstance(spec, PackedLoaderSpec)
            else spec
            for name, spec in self.train_loaders.items()
        }


def _snapshot_stream(stream, scene, writer, training_loader=None):
    from .data import StoredObservationDataModule
    from .image_data import StoredImageDataModule
    from prom3theus.application.joint_contracts import LoadedJointStream

    if not isinstance(stream, LoadedJointStream):
        return stream
    data = stream.data_module
    if not isinstance(
        data, (StoredObservationDataModule, StoredImageDataModule, BatchDataModule)
    ):
        # Extension providers cross the persistence boundary as finite batch specs,
        # never as local classes or closures. Native providers use their disk fields.
        from prom3theus.application.batches import _validation_loaders

        def capture(loader):
            if isinstance(loader, BatchSequence):
                return loader
            return BatchSequence(tuple(writer.encode_batch(batch) for batch in loader))

        train = training_loader
        if train is None and hasattr(data, "train_dataloader"):
            train = data.train_dataloader()
        data = BatchDataModule(
            None if train is None else capture(train),
            tuple(capture(loader) for loader in _validation_loaders(data)),
            data.run_metadata(),
        )
    support = stream.sampling_support
    if support is not None and not isinstance(support, SamplingSupport):
        support = SamplingSupport(
            tuple(
                writer.encode_batch((position, times))
                for position, times in support(scene, 262144)
            )
        )
    return replace(
        stream,
        prepared=replace(stream.prepared, data_module=data),
        sampling_support=support,
    )


def _snapshot_loader(loader, stream, writer, directory):
    from .tensor_loader import PersistentTensorLoader, PackedLoaderSpec

    data = getattr(stream, "data_module", None)
    if isinstance(loader, PersistentTensorLoader):
        return loader.snapshot(directory)
    if isinstance(data, BatchDataModule):
        return data.training
    if isinstance(loader, (BatchSequence, PackedLoaderSpec)):
        return loader
    return BatchSequence(tuple(writer.encode_batch(batch) for batch in loader))


def persist_prepared_stream(stream, scene, directory, training_loader=None):
    """Finish one stream and release its tensors before loading the next stream."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        writer = SnapshotWriter(directory)
        normalized = _snapshot_stream(stream, scene, writer, training_loader)
        spec = (
            None
            if training_loader is None
            else _snapshot_loader(
                training_loader, normalized, writer, directory / "training"
            )
        )
        state = writer.encode((normalized, spec), payload=True)
        return restore(state)
    except BaseException:
        shutil.rmtree(directory)
        raise
    finally:
        if training_loader is not None:
            close = getattr(training_loader, "close", None)
            if close is not None:
                close()


def save_data_module(module, path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="data-arrays-", dir=path.parent))
    try:
        writer = SnapshotWriter(directory)
        streams = {
            name: _snapshot_stream(
                stream, module.scene, writer, (module.train_loaders or {}).get(name)
            )
            for name, stream in module.streams.items()
        }
        loaders = (
            None
            if module.train_loaders is None
            else {
                name: _snapshot_loader(
                    loader, streams.get(name), writer, directory / f"training-{index}"
                )
                for index, (name, loader) in enumerate(module.train_loaders.items())
            }
        )
        snapshot = JointDataModule(
            writer.encode(streams, payload=True),
            writer.encode(module.scene),
            writer.encode(loaders),
            writer.encode(module.summaries),
            module.contract,
            generation=uuid.uuid4().hex,
            root=str(path.parent),
            encoded=True,
        )
        temporary = directory / "module.tmp"
        torch.save(snapshot, temporary)
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(directory)
        raise


def _rebase_paths(value, old_root, new_root):
    from .arrays import ArrayRef
    from .snapshot import ObjectRecord

    if isinstance(value, ArrayRef):
        path = Path(value.path)
        if path.is_relative_to(old_root):
            return replace(value, path=str(new_root / path.relative_to(old_root)))
        return value
    if isinstance(value, ObjectRecord):
        return replace(value, state=_rebase_paths(value.state, old_root, new_root))
    if isinstance(value, Path) and value.is_relative_to(old_root):
        return new_root / value.relative_to(old_root)
    if isinstance(value, dict):
        return {
            key: _rebase_paths(item, old_root, new_root) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return type(value)(_rebase_paths(item, old_root, new_root) for item in value)
    return value


def load_data_module(path):
    path = Path(path).resolve()
    try:
        module = torch.load(path, map_location="cpu", weights_only=False)
    except (AttributeError, ModuleNotFoundError) as error:
        raise ValueError(
            f"Saved data module uses an unavailable legacy/provider class ({error}); explicitly rebuild observations"
        ) from error
    if (
        not isinstance(module, JointDataModule)
        or module.version != 2
        or not module.encoded
    ):
        raise ValueError(
            "Unsupported data module schema; explicitly rebuild this legacy module"
        )
    for name in ("streams", "scene", "train_loaders", "summaries"):
        state = _rebase_paths(getattr(module, name), Path(module.root), path.parent)
        for ref in set(array_references(state)):
            ref.validate()
        setattr(module, name, restore(state))
    module.root, module.encoded = str(path.parent), False
    return module


def global_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if name in os.environ:
            return int(os.environ[name])
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Distributed loading requires a global RANK, not LOCAL_RANK")
    return int(os.environ.get("LOCAL_RANK", "0"))


def _publication_path(path):
    token = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("PROM3THEUS_RUN_ID")
    if not token:
        return None
    token += ":" + os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    return path.with_suffix(
        "." + hashlib.sha256(token.encode()).hexdigest()[:16] + ".status.json"
    )


def restore_or_create_data_module(
    path, build, *, rebuild=False, timeout=1800, finalize=None
):
    """Reuse existing data; optional finalize prepares only missing derived pools."""
    path = Path(path).resolve()
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    rank = global_rank()
    status_path = _publication_path(path)
    if (
        not distributed
        and (rebuild or int(os.environ.get("WORLD_SIZE", "1")) > 1)
        and rank != 0
        and status_path is None
    ):
        raise RuntimeError(
            "Worker publication requires an initialized process group or a unique PROM3THEUS_RUN_ID"
        )
    status = None
    if rank == 0:
        try:
            import fcntl

            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if rebuild or not path.is_file():
                    module = build()
                    if finalize is not None:
                        finalize(module)
                    save_data_module(module, path)
                else:
                    module = load_data_module(path)
                    if finalize is not None and finalize(module):
                        save_data_module(module, path)
                module = load_data_module(path)
            status = {"generation": module.generation, "error": None}
        except BaseException as error:
            status = {"generation": None, "error": f"{type(error).__name__}: {error}"}
            if distributed:
                torch.distributed.broadcast_object_list([status], src=0)
            elif status_path is not None:
                _publish_status(status_path, status)
            raise
        if distributed:
            torch.distributed.broadcast_object_list([status], src=0)
        elif status_path is not None:
            _publish_status(status_path, status)
        return module
    if distributed:
        message = [None]
        torch.distributed.broadcast_object_list(message, src=0)
        status = message[0]
    else:
        started = time.monotonic()
        while True:
            if status_path is not None and status_path.is_file():
                status = json.loads(status_path.read_text())
                break
            if status_path is None and path.is_file():
                break
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"Rank zero did not publish {path}")
            time.sleep(0.1)
    if status is not None and status["error"]:
        raise RuntimeError(f"Data preparation failed on rank zero: {status['error']}")
    module = load_data_module(path)
    if status is not None and module.generation != status["generation"]:
        raise RuntimeError("Data generation changed during distributed startup")
    return module


def _publish_status(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status))
    os.replace(temporary, path)
