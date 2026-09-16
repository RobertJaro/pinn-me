"""Atomic, observation-independent evaluation snapshots for stream inversions."""
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping
from copy import deepcopy
import logging
import os
import torch
from torch import nn
from prom3theus.components.forward import reconstruct_term
from prom3theus.observations import SceneContract
from prom3theus.rt import StratifiedAtmosphereModel
from .errors import ArtifactExportError

P3S_FORMAT = "prom3theus.stream_state"
P3S_VERSION = 1


def snapshot_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): snapshot_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot_value(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


class EvaluationModel(nn.Module):
    """Shared atmosphere and arbitrary observation terms, without training state."""

    def __init__(self, atmosphere, terms):
        super().__init__()
        self.atmosphere_model = atmosphere
        self.terms = nn.ModuleDict(terms)


def snapshot_context(runtime):
    return snapshot_value(
        {
            "configuration": runtime.config.to_dict(),
            "resources": runtime.resources,
            "scene": asdict(runtime.scene),
            "atmosphere": runtime.model.atmosphere_model.construction,
            "terms": {
                name: term.construction for name, term in runtime.model.terms.items()
            },
            "streams": {
                name: {
                    "store_path": loaded.prepared.store_path,
                    "source_signature": loaded.prepared.source_signature,
                    "descriptor": loaded.prepared.descriptor.metadata(),
                    "specification": loaded.specification.metadata(),
                    "store_metadata": {
                        **loaded.store_metadata,
                        "raster_names": list(
                            getattr(loaded.data_module, "raster_names", ())
                        ),
                        "validation_raster_index": getattr(
                            loaded.data_module, "validation_raster_index", 0
                        ),
                        "times": [
                            {
                                "values": raster.metadata.get("times", []),
                                "scale": raster.metadata.get("coordinates", {}).get(
                                    "time_scale", "utc"
                                ),
                            }
                            for raster in loaded.rasters
                        ],
                        "bounds": getattr(
                            loaded.data_module, "observation_sampling_bounds", {}
                        ),
                    },
                }
                for name, loaded in runtime.streams.items()
            },
        }
    )


def save_state(path, model, context, *, epoch, global_step):
    path = Path(path)
    state = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(("atmosphere_model.", "terms."))
    }
    state = snapshot_value(state)
    validate_tensors(state)
    payload = dict(
        format=P3S_FORMAT,
        version=P3S_VERSION,
        context=context,
        state_dict=state,
        epoch=int(epoch),
        global_step=int(global_step),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_tensors(state):
    if not isinstance(state, Mapping) or not all(
        isinstance(v, torch.Tensor) for v in state.values()
    ):
        raise ArtifactExportError("Snapshot state must contain tensors only.")
    if any(
        not torch.isfinite(v).all()
        for v in state.values()
        if v.is_floating_point() or v.is_complex()
    ):
        raise ArtifactExportError("Snapshot contains non-finite state.")


@dataclass(frozen=True)
class ValidatedSaveState:
    path: Path
    context: dict
    module: EvaluationModel
    scene: SceneContract
    epoch: int
    global_step: int


def _mse_evaluation_context(context):
    """Translate legacy objective metadata for evaluation without altering weights."""
    context = dict(context)
    configuration = deepcopy(context["configuration"])
    terms = deepcopy(context["terms"])
    changed = False

    def convert(objective):
        nonlocal changed
        replacements = {"huber": "mse", "asinh_huber": "asinh_mse"}
        if objective.get("type") in replacements:
            objective["type"] = replacements[objective["type"]]
            objective.pop("huber_delta", None)
            changed = True

    def direct_weights(objective, weights):
        nonlocal changed
        if "stokes_sigmas" not in objective:
            return
        sigmas = objective.pop("stokes_sigmas")
        total = sum(weights.values())
        for name, value in list(weights.items()):
            weights[name] = value / total / (sigmas[name] ** 2 if sigmas else 1.0)
        changed = True

    for stream in configuration.get("streams", []):
        objective = stream.get("data_term", {}).get("objective", {})
        convert(objective)
        if stream.get("data_term", {}).get("type") == "lte_stokes":
            direct_weights(objective, objective["stokes_weights"])
    for contract in terms.values():
        options = contract["options"]
        if contract["type"] == "lte_stokes":
            convert(options["objective_config"])
            direct_weights(options["objective_config"], options["weight_config"])
        elif contract["type"] == "aia_optically_thin" and "huber_delta" in options:
            options.pop("huber_delta")
            changed = True
    if changed:
        logging.getLogger(__name__).warning(
            "Legacy snapshot objective metadata migrated for evaluation; "
            "stored model weights are unchanged."
        )
        context.update(configuration=configuration, terms=terms)
    return context


def load_validated_save_state(path, *, map_location="cpu"):
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        set(payload)
        != {"format", "version", "context", "state_dict", "epoch", "global_step"}
        or payload["format"] != P3S_FORMAT
        or payload["version"] != P3S_VERSION
    ):
        raise ArtifactExportError(
            "Unsupported P3S format. Only stream-state snapshots are accepted."
        )
    validate_tensors(payload["state_dict"])
    context = _mse_evaluation_context(payload["context"])
    from prom3theus.resources import validate_resource_sets

    resources = validate_resource_sets(context["resources"])
    if snapshot_value(resources) != context["resources"]:
        raise ArtifactExportError(
            "Snapshot resources do not match installed scientific resources."
        )
    scene = SceneContract(**context["scene"])
    atmosphere = StratifiedAtmosphereModel(**context["atmosphere"]).float()
    terms = {
        name: reconstruct_term(contract, atmosphere, scene)
        for name, contract in context["terms"].items()
    }
    module = EvaluationModel(atmosphere, terms)
    expected = module.state_dict()
    # Old sigma buffers become direct component coefficients during migration.
    # Learned atmosphere tensors remain untouched.
    for name in list(payload["state_dict"]):
        if name.endswith(".objective.stokes_sigmas"):
            prefix = name.removesuffix(".objective.stokes_sigmas")
            del payload["state_dict"][name]
            weight_key = prefix + ".stokes_weights"
            payload["state_dict"][weight_key] = expected[weight_key]
    if set(expected) != set(payload["state_dict"]) or any(
        expected[n].shape != v.shape or expected[n].dtype != v.dtype
        for n, v in payload["state_dict"].items()
    ):
        raise ArtifactExportError("Snapshot model tensor contract differs.")
    module.load_state_dict(payload["state_dict"], strict=True)
    module.to(map_location).eval()
    if any(
        type(payload[k]) is not int or payload[k] < 0 for k in ("epoch", "global_step")
    ):
        raise ArtifactExportError("Snapshot progress must be nonnegative integers.")
    return ValidatedSaveState(
        path, context, module, scene, payload["epoch"], payload["global_step"]
    )
