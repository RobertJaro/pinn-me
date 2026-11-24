"""Training provenance, resumable optimizer state and best validation checkpoint."""

import json
import math
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import torch

from prom3theus.resources import file_sha256, resource_path
from prom3theus.rt.coefficient_mapper import FORMAT

from .config import TrainingConfig
from .run_files import atomic_save, logger


def resumed_mapper_options(config):
    """Recover the parameterization before building the model and its optimizer."""
    if not config.resume:
        return {}
    manifest = config.output_dir / "run.json"
    if manifest.is_file():
        spec = json.loads(manifest.read_text())["contract"]["spec"]
    else:
        last = config.output_dir / "last.pt"
        if not last.is_file():
            return {}
        spec = torch.load(last, map_location="cpu", weights_only=True, mmap=True)["spec"]
    return dict(spectral_frame=spec.get("spectral_frame", "observer"),
                linear_residual=spec.get("linear_residual", False))


def training_contract(
    config, model, teacher, snapshot, datasets, detector, radiance_scale
):
    """Identify every input that must agree when a training run is resumed."""
    teacher_modules = (
        "local_coefficients",
        "polarization",
        "opacity",
        "plasma",
        "eos",
        "profiles",
        "wavelength",
    )
    contract = dict(
        spec=model.spec,
        detector=detector,
        radiance_scale=radiance_scale,
        torch_version=str(torch.__version__),
        teacher=teacher.metadata(),
        native=snapshot.native_reference(),
        teacher_resources=file_sha256(resource_path("sources.json")),
        teacher_sources={
            name: file_sha256(files("prom3theus.rt").joinpath(name + ".py"))
            for name in teacher_modules
        },
        perturbations=asdict(config.perturbations),
        log_tau=list(config.log_tau),
        seed=config.seed,
        bounds_by_split={name: list(data.region) for name, data in datasets.items()},
        split_axis=datasets["train"].split_axis,
        wavelength_count=config.wavelength_count,
        ray_depths=config.ray_depths,
        batch_size=config.batch_size,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        final_learning_rate=config.final_learning_rate,
        learning_rate_decay_steps=config.learning_rate_decay_steps,
    )
    if config.perturbations.magnetic_field_fraction:
        contract["magnetic_jitter_distribution"] = "uniform_vector_scale"
    else:
        # Fixed-width runs predate the relative field setting.
        del contract["perturbations"]["magnetic_field_fraction"]
        contract["perturbations"].update(bx_gauss=0.0, by_gauss=0.0, bz_gauss=0.0)
    if not any(contract["perturbations"].values()):
        # Preserve resume compatibility with the zero-jitter baseline.
        contract["perturbations"]["fraction"] = 0.0
    contract["wavelength_jitter_fraction"] = config.wavelength_jitter_fraction
    if config.wavelength_jitter_fraction:
        contract["wavelength_jitter_distribution"] = "normal"
    if config.replay_fraction:
        contract["replay"] = dict(fraction=config.replay_fraction, capacity=config.replay_buffer_size)
    return contract


def _truncate_metrics(path: Path, step: int):
    """Discard test results and records written after the last durable step."""
    if not path.exists():
        return
    temporary = path.with_suffix(".tmp")
    with path.open() as source, temporary.open("w") as target:
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                break  # The last write may have been interrupted.
            if event.get("step", 0) <= step and not event["stage"].startswith("test"):
                target.write(line.rstrip("\n") + "\n")
    temporary.replace(path)


class Checkpoints:
    def __init__(self, directory: Path, contract: dict):
        self.directory = directory
        self.contract = contract
        self.best_score = math.inf
        self.selected = None

    def restore(self, config: TrainingConfig, model, optimizer, replay=None) -> int | None:
        """Return the last committed step, or None when initialization is required."""
        manifest = self.directory / "run.json"
        last = self.directory / "last.pt"
        if not config.resume:
            if manifest.exists() or last.exists():
                raise ValueError(
                    "Output directory contains a run; use --resume or a fresh directory"
                )
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {"contract": self.contract, "arguments": config.arguments()},
                    default=str,
                    indent=2,
                )
            )
            temporary.replace(manifest)
            return None
        if not last.exists():
            if (
                not manifest.is_file()
                or json.loads(manifest.read_text())["contract"] != self.contract
            ):
                raise ValueError("No compatible run to resume")
            logger.info("Restarting interrupted initialization")
            return None

        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        saved_optimizer = checkpoint["contract"].get("optimizer", "adamw")
        if saved_optimizer != config.optimizer:
            raise ValueError(f"Cannot resume {saved_optimizer} optimizer state with {config.optimizer}")
        if self.contract != checkpoint["contract"]:
            raise ValueError(
                "Resume requires identical snapshot, model, sampling and learning-rate configuration"
            )
        step = checkpoint["step"]
        if step > config.steps:
            raise ValueError("--steps precedes the saved step")
        model.load_state_dict(checkpoint["mapper"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if replay is not None:
            replay.load_state_dict(checkpoint["replay"])
        self.best_score = checkpoint["best_loss"]
        self.selected = checkpoint.get("selected")
        if self.selected is not None:
            atomic_save(self.directory / "best.pt", self.selected)
        _truncate_metrics(self.directory / "metrics.jsonl", step)
        logger.info("Resuming after step %d", step)
        return step

    def select(self, step, model, score):
        if score < self.best_score:
            self.best_score = score
            self.selected = dict(
                format=FORMAT,
                spec=model.spec,
                step=step,
                best_loss=score,
                contract=self.contract,
                mapper={
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            )
            atomic_save(self.directory / "best.pt", self.selected)

    def save_last(self, step, model, optimizer, replay=None):
        atomic_save(
            self.directory / "last.pt",
            dict(
                format=FORMAT,
                spec=model.spec,
                step=step,
                mapper=model.state_dict(),
                optimizer=optimizer.state_dict(),
                **({"replay": replay.state_dict()} if replay is not None else {}),
                best_loss=self.best_score,
                contract=self.contract,
                selected=self.selected,
            ),
        )
