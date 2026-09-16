"""Explicit atmosphere-network initialization for a new local HMI inversion."""

from copy import deepcopy
import hashlib
from pathlib import Path

from prom3theus.artifacts.loader import P3SLoader

from .joint_assembly import _default_atmosphere_builder
from .joint_contracts import JointRunnerFactories


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _single_hmi_stream(config):
    if (len(config.streams) != 1 or config.streams[0].observation.type != "hmi_stokes"
            or config.streams[0].id != config.scene.reference_stream):
        raise ValueError("HMI warm starts require one reference HMI Stokes stream")
    return config.streams[0]


def _compatible_configuration(config, source_config):
    target_stream = _single_hmi_stream(config)
    source_stream = _single_hmi_stream(source_config)
    atmosphere = deepcopy(config.atmosphere.to_dict())
    source_atmosphere = deepcopy(source_config.atmosphere.to_dict())
    for options in (atmosphere, source_atmosphere):
        options["parameters"]["magnetic_field"].pop("reference_height_megameter", None)
    if atmosphere != source_atmosphere:
        raise ValueError("Warm-start atmosphere/network must match except magnetic reference height")
    if config.scene.to_dict() != source_config.scene.to_dict():
        raise ValueError("Warm-start scene configuration must match the source")
    observation, source_observation = target_stream.observation.to_dict(), source_stream.observation.to_dict()
    for options in (observation, source_observation):
        options.pop("loader", None)
    if observation != source_observation:
        raise ValueError("Warm-start HMI observation, crop selection and calibration must match the source")


def build_hmi_warm_start_factories(config, source_path):
    """Initialize network weights only, preserving the new atmosphere semantics.

    Pass the returned factories to both setup evaluation and training. The new
    run must have its own output directory; ordinary checkpoint resume is a
    separate operation. Returned provenance belongs in the new run report.
    """
    source_path = Path(source_path).expanduser().resolve()
    if config.training is None or config.training.resume_from_checkpoint is not None:
        raise ValueError("A weights-only warm start requires training with no explicit resume checkpoint")
    output = config.solver.output_directory.resolve()
    if (output / "last.ckpt").exists():
        raise ValueError("Warm-start destination already contains a checkpoint (last.ckpt); use a new output directory")
    if source_path.is_relative_to(output):
        raise ValueError("Warm-start source and destination must be separate")
    checksum = _sha256(source_path)
    source = P3SLoader(source_path, device="cpu")
    if checksum != _sha256(source_path):
        raise ValueError("Warm-start source changed while it was being loaded")
    if output == source.config.solver.output_directory.resolve():
        raise ValueError("Warm-start source and destination must be separate")
    _compatible_configuration(config, source.config)
    source_coordinates = dict(source.state.scene.atmosphere_coordinate_metadata)
    network_state = {
        name: value.detach().cpu().clone()
        for name, value in source.module.atmosphere_model.network.state_dict().items()
    }

    def atmosphere_builder(current_config, scene, resources):
        if current_config.solver.output_directory.resolve() != output:
            raise ValueError("Warm-start factories belong to their declared destination")
        if current_config.training.resume_from_checkpoint is not None or (output / "last.ckpt").exists():
            raise ValueError("Warm-start factories cannot be combined with checkpoint resume")
        _compatible_configuration(current_config, source.config)
        if dict(scene.atmosphere_coordinate_metadata) != source_coordinates:
            raise ValueError("Warm-start scene/coordinate geometry must match the source")
        model = _default_atmosphere_builder(current_config, scene, resources)
        model.network.load_state_dict(network_state, strict=True)
        return model

    provenance = {
        "method": "atmosphere_network_weights_only",
        "source_path": str(source_path), "source_sha256": checksum,
        "source_global_step": source.global_step, "source_epoch": source.epoch,
        "source_stream_id": source.stream_id,
        "source_signature": source.observation.source_signature,
        "source_reference_height_megameter": source.config.atmosphere.parameters.magnetic_field.reference_height_megameter,
        "target_reference_height_megameter": config.atmosphere.parameters.magnetic_field.reference_height_megameter,
        "fresh_optimizer": True,
    }
    return JointRunnerFactories(atmosphere_builder=atmosphere_builder), provenance


__all__ = ["build_hmi_warm_start_factories"]
