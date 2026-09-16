"""Prepare data once; restore descriptors independently of model/trainer setup."""

import uuid
from contextlib import contextmanager
from time import perf_counter
from types import MappingProxyType
from dataclasses import replace

from pytorch_lightning.utilities.rank_zero import rank_zero_info

from prom3theus.observations import SceneContract
from prom3theus.observations.persistence import (
    JointDataModule,
    restore_or_create_data_module,
)
from .joint_contracts import LoadedJointStream
from .joint_streams import _joint_observation_bounds, _observation_times_hours
from .runtime import _json_value


@contextmanager
def _startup_stage(name):
    started = perf_counter()
    print(f"Startup: {name}", flush=True)
    try:
        yield
    finally:
        print(
            f"Startup: {name} finished after {perf_counter() - started:.2f}s",
            flush=True,
        )


def _prepare_training_tensors(loaders):
    from prom3theus.observations.tensor_loader import PersistentTensorLoader

    for name, loader in loaders.items():
        if isinstance(loader, PersistentTensorLoader):
            rank_zero_info(f"Preparing training tensor files for {name}")
            loader.prepare()
    rank_zero_info("Training tensor files ready")


def prepare_training_loaders(config, streams, scene):
    settings = config.training
    loaders = {}
    from prom3theus.observations.bulk import BulkBatchLoader, configure_readers

    configure_readers(settings.reader_workers)
    for stream_name, stream in streams.items():
        loader = stream.data_module.train_dataloader()
        if isinstance(loader, BulkBatchLoader):
            from prom3theus.observations.tensor_loader import PersistentTensorLoader

            if isinstance(loader, PersistentTensorLoader):
                from dataclasses import asdict

                from prom3theus.observations.identity import implementation_signature

                observation = next(
                    stream.observation.to_dict()
                    for stream in config.streams
                    if stream.id == stream_name
                )
                observation.pop("loader", None)
                loader.tensor_cache_directory = (
                    config.solver.work_directory / "training-tensors" / stream_name
                )
                loader.tensor_cache_identity = _json_value(
                    {
                        "source": streams[stream_name].prepared.source_signature,
                        "observation": observation,
                        "scene": asdict(scene),
                        "setup": streams[stream_name].setup_metadata,
                        "implementation": implementation_signature(),
                    }
                )
            loader.stream_id = stream_name
            loader.block_bytes = settings.loader_block_bytes
            loader.cache_bytes = settings.loader_cache_bytes
            loader.prefetch_batches = settings.loader_prefetch_batches
            loader.max_batch_bytes = settings.loader_max_batch_bytes
            loader.seed = settings.loader_seed
            loader.pin_memory = False
        _prepare_training_tensors({stream_name: loader})
        loaders[stream_name] = loader
    return loaders


def data_contract(config, resources):
    return _json_value(
        {
            "scene": config.scene.to_dict(),
            "geometry": config.atmosphere.geometry.to_dict(),
            "streams": [
                {
                    "id": stream.id,
                    "validation_stride": getattr(
                        getattr(stream.observation, "loader", None),
                        "validation_stride",
                        None,
                    ),
                    "observation": {
                        key: value
                        for key, value in stream.observation.to_dict().items()
                        if key != "loader"
                    },
                }
                for stream in config.streams
            ],
            "resources": resources,
        }
    )


def prepare_joint_data(
    config,
    resources,
    load_stream,
    build_scene,
    *,
    rebuild_observations,
    for_training,
    compute_summaries,
):
    from prom3theus.observations import ObservationRaster, ImageObservationRaster
    from prom3theus.observations.scene import rebase_stokes_raster
    from prom3theus.observations.persistence import persist_prepared_stream

    loaded: dict[str, LoadedJointStream] = {}
    training = {} if for_training else None
    scene = None
    # The reference establishes the shared chart/time convention. Other datasets
    # need only this small contract, not one another's payloads.
    ordered = sorted(
        config.streams, key=lambda stream: stream.id != config.scene.reference_stream
    )
    for stream in ordered:
        with _startup_stage(f"load prepared stream {stream.id}"):
            value = load_stream(
                stream,
                resources,
                config.solver.work_directory,
                rebuild_observations=rebuild_observations,
            )
        if not isinstance(value, LoadedJointStream):
            raise TypeError("stream_loader must return LoadedJointStream.")
        if value.prepared.name != stream.id:
            raise ValueError("Loaded stream identity does not match its configuration.")
        with _startup_stage(f"setup stream {stream.id}"):
            value.data_module.setup("fit" if for_training else "validate")
        if scene is None:
            with _startup_stage("establish scene from reference stream"):
                scene = build_scene(config, MappingProxyType({stream.id: value}))
            if not isinstance(scene, SceneContract):
                raise TypeError("scene_builder must return SceneContract.")
        elif value.rasters and all(
            isinstance(raster, ImageObservationRaster) for raster in value.rasters
        ):
            scene.validate_image_rasters(value.rasters)

        if value.rasters and all(
            isinstance(raster, ObservationRaster) for raster in value.rasters
        ):
            with _startup_stage(f"rebase stream {stream.id}"):
                rasters = tuple(
                    rebase_stokes_raster(
                        raster,
                        scene,
                        output_path=config.solver.work_directory
                        / "derived"
                        / f"{stream.id}-{index}-{uuid.uuid4().hex}.npy",
                    )
                    for index, raster in enumerate(value.rasters)
                )
                if any(new is not old for new, old in zip(rasters, value.rasters)):
                    data = value.data_module
                    data.rasters = list(rasters)
                    data.raster = rasters[data.validation_raster_index]
                    data.dataset = None
                    data.setup("fit" if for_training else "validate")
                    value = replace(value, rasters=rasters)
                    del data
                del rasters

        loader = None
        if for_training:
            with _startup_stage(
                f"prepare, shuffle and persist training stream {stream.id}"
            ):
                loader = prepare_training_loaders(config, {stream.id: value}, scene)[
                    stream.id
                ]
        with _startup_stage(f"finish disk-backed stream {stream.id}"):
            finished, spec = persist_prepared_stream(
                value,
                scene,
                config.solver.work_directory / f"data-arrays-{uuid.uuid4().hex}",
                loader,
            )
        loaded[stream.id] = finished
        if training is not None:
            training[stream.id] = spec
        # No unpersisted raster/provider payload survives into the next iteration.
        del value, loader, finished, spec
    loaded = {stream.id: loaded[stream.id] for stream in config.streams}
    if training is not None:
        training = {stream.id: training[stream.id] for stream in config.streams}
    summaries = {}
    if compute_summaries:
        with _startup_stage("derive reusable domain summaries"):
            summaries["observation_bounds"] = _joint_observation_bounds(loaded, scene)
            stokes = {
                name: stream
                for name, stream in loaded.items()
                if stream.prepared.descriptor.observation_kind == "stokes"
            }
            if stokes:
                summaries["stokes_bounds"] = _joint_observation_bounds(stokes, scene)
                summaries["observation_times_hours"] = _observation_times_hours(
                    stokes, scene
                )
    return JointDataModule(
        loaded,
        scene,
        train_loaders=training,
        summaries=summaries,
        contract=data_contract(config, resources),
    )


def load_joint_data(
    config,
    resources,
    load_stream,
    build_scene,
    *,
    rebuild_observations=False,
    for_training=False,
    compute_summaries=True,
):
    contract = data_contract(config, resources)

    def finalize(module):
        if module.contract != contract:
            raise ValueError(
                "Saved data module differs from the requested data/scene/resource contract; explicitly rebuild observations"
            )
        if not for_training or module.train_loaders is not None:
            return False
        module.train_loaders = prepare_training_loaders(
            config, module.streams, module.scene
        )
        return True

    module = restore_or_create_data_module(
        config.solver.work_directory / "data_module.pt",
        lambda: prepare_joint_data(
            config,
            resources,
            load_stream,
            build_scene,
            rebuild_observations=rebuild_observations,
            for_training=for_training,
            compute_summaries=compute_summaries,
        ),
        rebuild=rebuild_observations,
        finalize=finalize,
    )
    for stream in config.streams:
        options = getattr(stream.observation, "loader", None)
        data = module.streams[stream.id].data_module
        if options is not None:
            for source, target in (
                ("batch_size", "batch_size"),
                ("validation_batch_size", "validation_batch_size"),
                ("workers", "num_workers"),
                ("pin_memory", "pin_memory"),
            ):
                value = getattr(options, source)
                if source == "validation_batch_size" and value is None:
                    value = options.batch_size
                setattr(data, target, value)
    if module.contract != contract:
        raise ValueError(
            "Restored module data contract differs from requested configuration"
        )
    return module
