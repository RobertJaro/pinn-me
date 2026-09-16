"""Train the joint forward model using all configured observation streams."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from prom3theus.artifacts.checkpoint import save_state, snapshot_context
from prom3theus.config.joint_schema import JointInversionConfig
from prom3theus.diagnostics.providers import render_diagnostics
from prom3theus.training.joint import JointInversionModule
from prom3theus.training.streams import StreamBatchScheduler

from .joint_assembly import build_joint_runtime
from .joint_contracts import JointRunnerFactories
from .joint_evaluation import _temporary_physics_derivative_graph
from .runtime import _json_value, _preserve_rng, _temporary_eval_mode


class CommitStreamBatch(Callback):
    """Commit before ModelCheckpoint and snapshot callbacks see the batch."""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.stream_scheduler.commit()

    def on_exception(self, trainer, pl_module, exception):
        pl_module.stream_scheduler.close()

    def teardown(self, trainer, pl_module, stage):
        pl_module.stream_scheduler.close()


class JointSaveStateCallback(Callback):
    """Write the single evaluation snapshot at checkpoint intervals and completion."""

    def __init__(self, path, context):
        self.path, self.context = Path(path), context

    def _save(self, trainer, module):
        if trainer.is_global_zero:
            save_state(
                self.path,
                module.model,
                self.context,
                epoch=trainer.current_epoch,
                global_step=trainer.global_step,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % pl_module.settings.checkpoint_every_n_steps == 0:
            self._save(trainer, pl_module)


def _online_logger(config):
    if config.logging.type == "disabled":
        return False
    from pytorch_lightning.loggers import WandbLogger

    logger = WandbLogger(
        save_dir=str(config.solver.work_directory),
        project=config.logging.project,
        name=config.logging.run_name,
        tags=list(config.logging.tags),
        offline=False,
        mode="online",
    )
    logger.log_hyperparams(_json_value(config.to_dict()))
    return logger


def _log_validation(logger, report, step):
    if logger is None:
        return
    logger.log_metrics(
        {f"valid.{key}": value for key, value in report["metrics"].items()},
        step=step,
    )
    images_by_key = {}
    for artifact in report["rendering"]["artifacts"]:
        key = artifact.get("media_key", "Observation comparison")
        if key is not None:
            images_by_key.setdefault(key, []).append(artifact["path"])
    for key, paths in images_by_key.items():
        paths = list(dict.fromkeys(paths))
        logger.log_image(
            key=key,
            images=paths,
            caption=[Path(path).stem for path in paths],
            step=step,
        )


class PotentialBoundaryDiagnostics(Callback):
    """Upload cached potential surfaces once per reference update, also on resume."""

    def __init__(self, output):
        self.output = Path(output)
        self.logged_references = {}

    def _log_references(self, trainer, module):
        logger = trainer.logger
        if not trainer.is_global_zero or not callable(getattr(logger, "log_image", None)):
            return
        from prom3theus.diagnostics.potential_boundaries import potential_boundary_figure
        from prom3theus.inversion.data_terms.potential_boundary import ProgressivePotentialBoundary

        for name, term in module.model.shared_objectives.items():
            if not isinstance(term, ProgressivePotentialBoundary) or term.last_update < 0:
                continue
            revision = (term.last_update, term.update_count)
            if self.logged_references.get(name) == revision:
                continue
            # No model evaluation, Green operator construction or normalization
            # sampling: the plot displays exactly the already-created targets.
            figure = potential_boundary_figure(term)
            try:
                self.output.mkdir(parents=True, exist_ok=True)
                path = self.output / f"{name}_step_{term.last_update:08d}.png"
                figure.savefig(path, dpi=120)
                logger.log_image(key=f"Potential field/{name}", images=[str(path)],
                    caption=[f"All surfaces; first time anchor; source step {term.last_update}"],
                    step=int(trainer.global_step))
            finally:
                figure.clear()
            self.logged_references[name] = revision

    def on_train_start(self, trainer, pl_module):
        self._log_references(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_references(trainer, pl_module)


class JointTrainingDiagnostics(Callback):
    """Periodically evaluate and render the fitted model without changing its RNG."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.last_step = None

    def _evaluate(self, trainer, module):
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if self.last_step == step:
            return
        runtime = self.runtime
        module.set_objective_step()
        with (
            _preserve_rng(),
            _temporary_eval_mode(module.model),
        ):
            with (
                _temporary_physics_derivative_graph(module.model, enabled=False),
                torch.no_grad(),
            ):
                evaluation = module.model.evaluate_batches(runtime.validation_batches)
            report = {
                "format": "prom3theus.joint_training_validation",
                "global_step": step,
                "metrics": {
                    key: float(value.detach().cpu())
                    for key, value in evaluation.validation_metrics().items()
                },
                "diagnostics": {
                    key: float(value.detach().cpu())
                    for key, value in evaluation.scalar_metrics().items()
                },
            }
            output = (
                runtime.config.solver.work_directory / "validation" / f"step_{step:08d}"
            )
            output.mkdir(parents=True, exist_ok=True)
            rank_zero_info(f"Validation step {step}: rendering stream diagnostics")
            report["rendering"] = render_diagnostics(
                runtime, trainer, evaluation, output
            )
            (output / "metrics.json").write_text(
                json.dumps(_json_value(report), indent=2, allow_nan=False) + "\n"
            )
            _log_validation(trainer.logger, report, step)
        self.last_step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % pl_module.settings.validation_every_n_steps == 0:
            self._evaluate(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._evaluate(trainer, pl_module)


@dataclass(frozen=True)
class JointInversionRun:
    module: JointInversionModule
    trainer: Trainer
    checkpoint_path: Path
    save_state_path: Path


def _resume_checkpoint(output: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    checkpoint = output / "last.ckpt"
    return checkpoint if checkpoint.is_file() else None


def run_joint_inversion(
    config: JointInversionConfig,
    *,
    rebuild_observations: bool = False,
    factories: JointRunnerFactories | None = None,
    profiler=None,
) -> JointInversionRun:
    """Run optimization on one device or externally launched distributed ranks; persist resumable Lightning checkpoints."""
    settings = config.training
    if settings is None:
        raise ValueError("Joint inversion requires an explicit training section.")
    from prom3theus.core.distributed import initialize_distributed
    from .runtime import _device
    rank, world_size = initialize_distributed(_device(settings.device))
    output = config.solver.output_directory
    resume_checkpoint = _resume_checkpoint(output, settings.resume_from_checkpoint)
    if resume_checkpoint is not None:
        rank_zero_info(f"Resuming joint inversion from {resume_checkpoint}")
    runtime = build_joint_runtime(
        config,
        rebuild_observations=rebuild_observations,
        factories=factories,
        for_training=True,
    )
    if runtime.device.type not in ("cpu", "cuda"):
        raise ValueError("Joint training currently supports CPU or CUDA devices.")
    output.mkdir(parents=True, exist_ok=True)
    contract = config.to_dict()
    # Operational settings may change on resume; the physical/data contract may not.
    for key in ("training", "solver", "dry_run", "diagnostics", "logging"):
        contract.pop(key, None)
    # Stream weights are the supported render-only/fitting switch, including
    # when resuming. The checkpoint still records their original values above.
    for stream in contract["streams"]:
        stream["data_term"].pop("weight", None)
    context = _json_value(
        {
            "format": "prom3theus.joint_checkpoint",
            "version": 1,
            "configuration": config.to_dict(),
            "contract": {
                "configuration": contract,
                "resources": runtime.resources,
                "streams": {
                    name: loaded.prepared.source_signature
                    for name, loaded in runtime.streams.items()
                },
            },
        }
    )
    module = JointInversionModule(runtime.model, settings, context)
    rank_zero_info("Startup: constructing training loaders")
    loaders = runtime.data_module.train_dataloader(
        batch_sizes={stream.id: stream.observation.loader.batch_size
                     for stream in config.streams if hasattr(stream.observation, "loader")},
        pin_memory=runtime.device.type == "cuda",
        prefetch_batches=settings.loader_prefetch_batches,
        max_batch_bytes=settings.loader_max_batch_bytes,
        seed=settings.loader_seed,
    )
    scheduler = StreamBatchScheduler(
        loaders, settings.reference_stream or config.scene.reference_stream,
        explicit_commit=True, rank=rank, world_size=world_size,
    )
    if runtime.device.type == "cuda" and settings.gpu_prefetch:
        scheduler.device = runtime.device
    module.stream_scheduler = scheduler
    checkpoint = ModelCheckpoint(
        dirpath=output,
        every_n_train_steps=settings.checkpoint_every_n_steps,
        save_top_k=0,
        save_last=True,
        enable_version_counter=False,
    )
    save_state_path = output / "state.p3s"
    save_state_callback = JointSaveStateCallback(save_state_path, snapshot_context(runtime))
    trainer = Trainer(
        accelerator="gpu" if runtime.device.type == "cuda" else "cpu",
        devices=(int(os.environ.get("LOCAL_WORLD_SIZE", world_size)) if world_size > 1
                 else [runtime.device.index or 0] if runtime.device.type == "cuda" else 1),
        num_nodes=max(1, world_size // int(os.environ.get("LOCAL_WORLD_SIZE", world_size))),
        strategy="ddp_find_unused_parameters_true" if world_size > 1 else "auto",
        max_steps=settings.max_steps,
        max_epochs=settings.max_epochs,
        gradient_clip_val=settings.gradient_clip_norm,
        log_every_n_steps=settings.log_every_n_steps,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        logger=_online_logger(config) if rank == 0 else False,
        callbacks=[
            CommitStreamBatch(),
            checkpoint,
            save_state_callback,
            PotentialBoundaryDiagnostics(output / "potential-boundaries"),
            JointTrainingDiagnostics(runtime),
        ],
        inference_mode=False,
        use_distributed_sampler=False,
        profiler=profiler,
    )
    try:
        trainer.fit(
            module,
            train_dataloaders=scheduler.dataloader(),
            ckpt_path=resume_checkpoint,
        )
    finally:
        scheduler.close()
        for loader in loaders.values():
            close = getattr(loader, "close", None)
            if close is not None:
                close()
    final_path = output / "last.ckpt"
    trainer.save_checkpoint(final_path)
    # Lightning skips on_train_end when a restored checkpoint already meets
    # max_steps. Refresh the public snapshot from the actual returned model even
    # when no optimizer step ran, replacing any missing or stale P3S.
    save_state_callback._save(trainer, module)
    return JointInversionRun(module, trainer, final_path, save_state_path)


__all__ = ["JointInversionRun", "run_joint_inversion"]
