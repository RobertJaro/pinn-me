"""Lightning optimization lifecycle for the shared multimodal objective."""

from copy import deepcopy
import math
from collections.abc import Mapping

import torch
from pytorch_lightning import LightningModule

from prom3theus.config.joint_schema import JointTrainingConfig, LossBalanceConfig
from prom3theus.core.validation import skip_tensor_validation
from prom3theus.inversion.joint import JointForwardModel
from prom3theus.training.optimizers import build_optimizer, exponential_lr_factor
from prom3theus.training.metrics import reduce_training_metrics


def _restore_cuda_rng_state(states):
    """Restore the states available on the current CUDA device topology."""

    if states is None or not torch.cuda.is_available():
        return
    device_count = torch.cuda.device_count()
    if len(states) != device_count:
        print(
            "CUDA RNG state count differs from the current visible device count; "
            f"restoring {min(len(states), device_count)} of {len(states)} saved states.",
            flush=True,
        )
    torch.cuda.set_rng_state_all(states[:device_count])


def _scientific_contract(contract):
    """Compare persistent data/model structure, excluding fitting controls."""
    if not isinstance(contract, dict):
        return contract
    contract = deepcopy(contract)

    config = contract.get("configuration", {})
    for key in ("training", "solver", "dry_run", "diagnostics", "logging"):
        config.pop(key, None)
    # Ignore legacy wavelength batching and current training-only numerical controls.
    # Retain the physical model and depth grid in the compatibility contract.
    windows = config.get("synthesis", {}).get("windows", [])
    for window in (*windows, *contract.get("synthesis_windows", {}).values()):
        if window.get("coefficient_backend", "exact_lte") == "exact_lte":
            window.pop("coefficient_backend", None)
            window.pop("mapper_checkpoint", None)
        window.pop("wavelength_chunk_size", None)
        window.pop("jitter_fraction", None)
        window.pop("transfer_precision", None)
        if not (window.get("coarse_to_fine") or {}).get("enabled", False):
            window.pop("coarse_to_fine", None)
    magnetic = config.get("atmosphere", {}).get("parameters", {}).get("magnetic_field", {})
    # None preserves the original height-dependent readout. Old configurations
    # omit this field; numerical reference heights change the physical model.
    if magnetic.get("reference_height_megameter") is None:
        magnetic.pop("reference_height_megameter", None)
    # The default decoder multiplier preserves checkpoints predating this option.
    if magnetic.get("scale_vector_potential", 1.0) == 1.0:
        magnetic.pop("scale_vector_potential", None)
    atmosphere = contract.get("atmosphere", {})
    if atmosphere.get("scale_vector_potential", 1.0) == 1.0:
        atmosphere.pop("scale_vector_potential", None)
    physics = config.get("physics") or {}
    # Stokes-only runs formerly carried a full, inactive collocation configuration.
    if not physics.get("potential_boundary", {}).get("enabled") and not any(
        equation.get("enabled") for equation in physics.get("equations", {}).values()
    ):
        config.pop("physics", None)
    for key in (
        "magnetic_current_free_steps",
        "magnetic_current_free_final_factor",
        "normalization",
        "loss_start_step",
        "loss_ramp_steps",
        "robust_loss_delta",
    ):
        physics.pop(key, None)
    equations = physics.get("equations", {})
    if not any(equations.get(name, {}).get("enabled") for name in (
        "upper_domain_temperature_prior", "upper_domain_microturbulence_prior", "coronal_energy",
    )):
        physics.get("collocation", {}).pop("validation_upper_height_layers", None)
    for name, equation in list(equations.items()):
        equation.pop("weight", None)
        # Plain residual switches carry no learned parameters or fixed buffers.
        if set(equation) == {"enabled"}:
            equation.pop("enabled")
        # New disabled defaults and their absence in older snapshots have the
        # same scientific state. Retain equations with structured options.
        if not equation:
            equations.pop(name)
    for key in list(physics.get("collocation", {})):
        if key.endswith("_per_step") or key in {
            "height_sampling_power",
            "validation_volume_points",
        }:
            physics["collocation"].pop(key)
    potential = physics.get("potential_boundary", {})
    if not potential.get("enabled", False):
        physics.pop("potential_boundary", None)
    else:
        # Query sampling and temporary priors have no learned state. Rebuild
        # their derived buffers on resume when the sampling geometry changes.
        for key in (
            "interior",
            "photosphere",
            "top_points",
            "side_points",
            "normalization_height_count",
            "normalization_points_per_height",
            "top_grid_size",
            "side_horizontal_points",
            "side_height_points",
            "jitter_fraction",
            "seed",
        ):
            potential.pop(key, None)
        for key in (
            "weight",
            "batch_size",
            "start_step",
            "ramp_steps",
            "end_step",
            "freeze_step",
            "update_every_n_steps",
            "blend",
            "field_floor_gauss",
        ):
            potential.pop(key, None)
    for item in config.get("atmosphere_regularization", []):
        for key in list(item):
            if key.endswith("_weight"):
                item.pop(key)
    for stream in config.get("streams", []):
        loader = stream.get("observation", {}).get("loader", {})
        for key in ("workers", "pin_memory", "preparation_workers", "batch_size", "validation_batch_size"):
            loader.pop(key, None)
        term = stream.get("data_term", {})
        term.pop("weight", None)
        term.pop("weight_schedule", None)
        term.pop("objective", None)
        # Fitting a global instrumental offset is an optimization control.
        # Fixed calibration values remain part of the scientific contract.
        term.get("instrument", {}).pop("optimize_line_of_sight_velocity_correction", None)
    terms = contract.get("terms", {})
    if not isinstance(terms, Mapping):
        return contract
    for term_contract in terms.values():
        if not isinstance(term_contract, Mapping):
            continue
        if term_contract.get("type") != "lte_stokes":
            continue
        options = term_contract.get("options", {})
        if not isinstance(options, dict):
            continue
        options.pop("optimize_instrument_line_of_sight_velocity_correction", None)
    return contract


def _contract_differences(previous, current, path="contract"):
    if isinstance(previous, dict) and isinstance(current, dict):
        result = []
        for key in sorted(set(previous) | set(current)):
            result.extend(
                _contract_differences(
                    previous.get(key), current.get(key), f"{path}.{key}"
                )
            )
        return result
    return [] if previous == current else [path]


class GradientLossBalancer:
    """Adapt shared-objective multipliers from gradients on atmosphere weights."""

    def __init__(
        self,
        settings: LossBalanceConfig,
        *,
        reference_stream: str | None = None,
    ) -> None:
        if not isinstance(settings, LossBalanceConfig):
            raise TypeError("loss_balance must be a LossBalanceConfig.")
        self.settings = settings
        # The reference stream is a training-level policy, not part of the
        # loss-balance schema. Keep it separate so the YAML location remains
        # ``training.reference_stream``.
        self.reference_stream = reference_stream
        self._ema_ratios: dict[str, float] = {}

    @staticmethod
    def _probe_parameters(model: JointForwardModel, count: int):
        parameters = [
            parameter
            for parameter in model.atmosphere_model.parameters()
            if parameter.requires_grad
        ]
        return parameters[-count:]

    @staticmethod
    def _objective_sources(
        model: JointForwardModel,
        evaluation,
        *,
        included_objectives=(),
    ):
        included = set(included_objectives)
        sources: dict[str, torch.Tensor] = {}
        for stream_id, value in evaluation.weighted_likelihoods.items():
            key = f"streams.{stream_id}"
            # Keep streams available for reference selection even when fixed.
            sources[key] = value
        for term_id, result in evaluation.shared_terms.items():
            if result.component_losses:
                sources.update(
                    {
                        f"shared.{term_id}.{name}": value
                        for name, value in result.component_losses.items()
                        if f"shared.{term_id}.{name}" in included
                    }
                )
            elif result.active:
                key = f"shared.{term_id}.loss"
                if key in included:
                    sources[key] = result.loss
        return sources

    @staticmethod
    def _gradient_norm(
        loss: torch.Tensor,
        parameters,
        *,
        world_size: int,
    ) -> torch.Tensor:
        if not loss.requires_grad:
            return loss.new_zeros(())
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        norm_square = loss.new_zeros(())
        for gradient in gradients:
            if gradient is not None:
                norm_square = norm_square + gradient.detach().square().sum()
        norm = norm_square.sqrt()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(norm)
            norm = norm / world_size
        return norm

    def update(
        self,
        model: JointForwardModel,
        evaluation,
        step: int,
    ) -> dict[str, torch.Tensor]:
        """Update policy multipliers on scheduled steps and return diagnostics."""

        settings = self.settings
        if step < settings.start_step or (
            (step - settings.start_step) % settings.update_every_n_steps
        ):
            return {}
        parameters = self._probe_parameters(model, settings.probe_parameter_count)
        sources = self._objective_sources(
            model,
            evaluation,
            included_objectives=settings.included_objectives,
        )
        if not parameters or len(sources) < 2:
            return {}
        device_loss = next(iter(sources.values()))
        requested_reference = (
            f"streams.{self.reference_stream}"
            if self.reference_stream
            else None
        )
        reference_key = (
            requested_reference
            if requested_reference in sources
            else next(
                (key for key in sources if key.startswith("streams.")),
                next(iter(sources)),
            )
        )
        sources = {
            key: value for key, value in sources.items()
            if key == reference_key or key in settings.included_objectives
        }
        if len(sources) < 2:
            return {}
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        norms = {
            key: self._gradient_norm(value, parameters, world_size=world_size)
            for key, value in sources.items()
        }
        reference_norm = norms[reference_key]
        epsilon = torch.finfo(reference_norm.dtype).eps
        if not torch.isfinite(reference_norm) or reference_norm <= epsilon:
            return {}
        metrics: dict[str, torch.Tensor] = {
            "loss_balance.reference_gradient_norm": reference_norm.detach(),
        }
        for key, norm in norms.items():
            if key == reference_key:
                continue
            if not torch.isfinite(norm) or norm <= epsilon:
                continue
            observed_ratio = float((norm / reference_norm.clamp_min(epsilon)).detach())
            previous = self._ema_ratios.get(key, observed_ratio)
            ema = settings.ema_decay * previous + (1.0 - settings.ema_decay) * observed_ratio
            self._ema_ratios[key] = ema
            multiplier = model.adaptive_multiplier(key)
            multiplier *= math.exp(
                settings.adaptation_rate
                * (math.log(settings.target_gradient_fraction) - math.log(max(ema, epsilon)))
            )
            multiplier = min(
                settings.max_multiplier,
                max(settings.min_multiplier, multiplier),
            )
            model.set_adaptive_multiplier(key, multiplier)
            label = key.replace(".", "_")
            metrics[f"loss_balance.{label}.gradient_norm"] = norm.detach()
            metrics[f"loss_balance.{label}.gradient_ratio"] = norm.detach() / reference_norm
            metrics[f"loss_balance.{label}.multiplier"] = device_loss.new_tensor(
                multiplier
            )
        return metrics

    def state_dict(self) -> dict:
        return {"ema_ratios": dict(self._ema_ratios)}

    def load_state_dict(self, state: Mapping | None) -> None:
        if not state:
            return
        if not isinstance(state, Mapping):
            raise TypeError("loss_balance checkpoint state must be a mapping.")
        ratios = state.get("ema_ratios", {})
        if not isinstance(ratios, Mapping):
            raise TypeError("loss_balance.ema_ratios must be a mapping.")
        self._ema_ratios = {
            str(key): float(value)
            for key, value in ratios.items()
            if key in self.settings.included_objectives
            and key != f"streams.{self.reference_stream}"
            and math.isfinite(float(value)) and float(value) > 0.0
        }


class JointInversionModule(LightningModule):
    def __init__(
        self, model: JointForwardModel, settings: JointTrainingConfig, context: dict
    ):
        super().__init__()
        self.model = model
        self.settings = settings
        self.context = context
        self._first_batch = True
        balance_settings = getattr(settings, "loss_balance", None)
        self.loss_balancer = (
            GradientLossBalancer(
                balance_settings,
                reference_stream=getattr(settings, "reference_stream", None),
            )
            if isinstance(balance_settings, LossBalanceConfig)
            and balance_settings.enabled
            else None
        )

    def set_objective_step(self):
        set_step = getattr(self.model, "set_step", None)
        if set_step is not None:
            set_step(int(self.global_step))
        for term in (
            *self.model.terms.values(),
            *self.model.shared_objectives.values(),
        ):
            set_step = getattr(term, "set_step", None)
            if set_step is not None:
                set_step(int(self.global_step))

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if getattr(getattr(self, "stream_scheduler", None), "device", None) is not None:
            return batch
        from .transfer import _map

        return _map(batch, lambda value: value.to(device, non_blocking=True))

    def training_step(self, batch, batch_idx):
        del batch_idx
        if self._first_batch:
            print("First training batch received: evaluating objectives", flush=True)
        self.set_objective_step()
        from prom3theus.core.distributed import distributed_batch
        valid = batch.get("_data_valid", True)
        if "_data_valid" in batch:
            batch = {key: value for key, value in batch.items() if key != "_data_valid"}
        cadence = getattr(self.settings, "log_every_n_steps", 10)
        collect_metrics = self.global_step == 0 or (self.global_step + 1) % cadence == 0
        with distributed_batch(valid), skip_tensor_validation():
            evaluation = self.model.evaluate_batches(batch, diagnostics=collect_metrics)
        if self._first_batch:
            print("First training forward pass complete: starting backward", flush=True)
        if not evaluation.loss.requires_grad:
            raise ValueError(
                "Joint training requires at least one active differentiable objective."
            )
        # Adaptive weights still update on their own cadence. Display metrics
        # and their DDP collectives run only when the logger consumes them.
        balance_metrics = (self.loss_balancer.update(self.model, evaluation, int(self.global_step))
                           if self.loss_balancer is not None else {})
        if collect_metrics:
            metrics = evaluation.training_metrics()
            if self.loss_balancer is not None:
                for term_id, result in evaluation.shared_terms.items():
                    metrics.update({f"shared.{term_id}.metrics.{name}": value
                                    for name, value in result.metrics.items()})
                metrics.update(balance_metrics)
            metrics = reduce_training_metrics(metrics, evaluation, data_valid=valid)
            for name, value in metrics.items():
                self.log(
                    f"train.{name}", value.detach(), on_step=True, on_epoch=False,
                    prog_bar=name == "loss", batch_size=1,
                )
        return evaluation.loss

    def on_after_backward(self):
        if self._first_batch:
            print("First training backward pass complete", flush=True)
            self._first_batch = False

    def on_before_optimizer_step(self, optimizer):
        """Log the learning rate shared by every optimizer parameter group."""
        cadence = getattr(self.settings, "log_every_n_steps", 10)
        if self.global_step != 0 and (self.global_step + 1) % cadence:
            return
        self.log(
            "train.optimizer.learning_rate", optimizer.param_groups[0]["lr"],
            on_step=True, on_epoch=False, batch_size=1,
        )

    def learning_rate_factor(self, step):
        """One exponential decay shared by every trainable parameter group."""
        steps = self.settings.learning_rate_decay_steps
        if steps is None:
            steps = self.trainer.estimated_stepping_batches
            # An unbounded run still needs a finite learning-rate horizon.
            # Reaching the final rate does not stop optimization.
            if not math.isfinite(steps):
                steps = 10000
        return exponential_lr_factor(step, self.settings.learning_rate, self.settings.final_learning_rate, steps)

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self, kind=getattr(self.settings, "optimizer", "adam"),
            lr=self.settings.learning_rate, parameter_blocks=None,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self.learning_rate_factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["formal_transfer_format_version"] = 1
        checkpoint["optimizer_kind"] = getattr(self.settings, "optimizer", "adam")
        checkpoint["prom3theus_joint"] = self.context
        if hasattr(self, "stream_scheduler"):
            checkpoint["stream_scheduler"] = self.stream_scheduler.state_dict()
        if hasattr(self, "_data_order_migration"):
            checkpoint["data_order_migration"] = self._data_order_migration
        if self.loss_balancer is not None:
            checkpoint["loss_balance"] = {
                "multipliers": self.model.adaptive_multiplier_state(),
                **self.loss_balancer.state_dict(),
            }
        checkpoint["torch_rng"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()

    def on_train_start(self):
        print("Training ready: waiting for the first batch", flush=True)
        # Optimizer loading restores saved group rates. Reapply the shared
        # schedule unconditionally so old block multipliers cannot survive resume.
        factor = self.learning_rate_factor(self.global_step)
        for optimizer in self.trainer.optimizers:
            for group in optimizer.param_groups:
                group["initial_lr"] = self.settings.learning_rate
                group["lr"] = self.settings.learning_rate * factor
        for config in self.trainer.lr_scheduler_configs:
            config.scheduler.base_lrs = [group["initial_lr"] for group in config.scheduler.optimizer.param_groups]
            config.scheduler._last_lr = [group["lr"] for group in config.scheduler.optimizer.param_groups]
        if hasattr(self, "_resume_rng"):
            from prom3theus.inversion.data_terms.potential_boundary import (
                ProgressivePotentialBoundary,
            )

            for term in self.model.shared_objectives.values():
                if isinstance(term, ProgressivePotentialBoundary):
                    term.frozen = term.last_update >= term.options["freeze_step"]
        cpu, cuda = getattr(self, "_resume_rng", (None, None))
        if cpu is not None:
            torch.random.set_rng_state(cpu.cpu())
        _restore_cuda_rng_state(cuda)

    def _restore_potential_sampling(self, checkpoint):
        from prom3theus.inversion.data_terms.potential_boundary import (
            ProgressivePotentialBoundary,
        )

        saved = checkpoint.get("state_dict", {})
        for name, term in self.named_modules():
            if not isinstance(term, ProgressivePotentialBoundary):
                continue
            prefix = name + "."
            current = term.state_dict()
            previous = {
                key[len(prefix) :]: value
                for key, value in saved.items()
                if key.startswith(prefix)
            }
            version = previous.get("_extra_state", {}).get("version")
            if version not in (7, 8, 9, 10):
                raise ValueError("Unsupported progressive potential checkpoint state")
            same_geometry = set(previous) == set(current) and all(
                torch.equal(previous[key].cpu(), current[key].cpu())
                for key in ("positions", "source_cells", "times", "sample_order")
            )
            if version in (7, 8, 9):
                raise ValueError("Potential boundary objective changed to weighted normalized G^2; start a new run instead of resuming an older normalized-loss checkpoint")
            if same_geometry:
                continue
            # Learned atmosphere parameters and optimizer state are untouched.
            # References are rebuilt from the resumed model at its current step.
            for key in list(saved):
                if key.startswith(prefix):
                    del saved[key]
            saved.update({prefix + key: value for key, value in current.items()})
            print(
                "Potential boundary sampling changed: rebuilding surface references "
                "from the resumed model; retaining model and optimizer state.",
                flush=True,
            )

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("formal_transfer_format_version") != 1:
            raise ValueError("This run requires a formal LTE checkpoint; start a new run for the changed forward model")
        previous = checkpoint.get("prom3theus_joint")
        old = _scientific_contract(previous.get("contract")) if previous else None
        new = _scientific_contract(self.context["contract"])
        if old != new:
            differences = ", ".join(_contract_differences(old, new)[:8])
            raise ValueError(
                f"Checkpoint observation/model contract does not match this run: {differences}"
            )
        self._restore_potential_sampling(checkpoint)
        saved_training = previous.get("configuration", {}).get("training", {})
        saved_optimizer = checkpoint.get(
            "optimizer_kind", saved_training.get("optimizer", "adam")
        )
        if saved_optimizer != getattr(self.settings, "optimizer", "adam"):
            raise ValueError(
                "Checkpoint optimizer differs from training.optimizer; start a new "
                "run instead of restoring incompatible optimizer state."
            )
        # Loss weights/scales are configuration, not learned checkpoint state.
        for name, value in self.state_dict().items():
            if (
                ".objective." in name
                or name.endswith((".stokes_weights", ".wavelength_weights"))
            ) and name in checkpoint.get("state_dict", {}):
                checkpoint["state_dict"][name] = value
        if hasattr(self, "stream_scheduler"):
            if (
                "stream_scheduler" not in checkpoint
                and not self.settings.migrate_data_order
            ):
                raise ValueError(
                    "Unsupported checkpoint: missing stream sampling state."
                )
            self.stream_scheduler.load_state_dict(
                checkpoint.get("stream_scheduler", {}),
                migrate=self.settings.migrate_data_order,
            )
        if getattr(
            getattr(self, "stream_scheduler", None), "order_migrated", False
        ) or (
            self.settings.migrate_data_order
            and checkpoint.get("stream_scheduler", {}).get("version") != 2
        ):
            self._data_order_migration = {
                "policy": "global_tensor_shuffle_random_batches",
                "optimizer_step": checkpoint.get("global_step", 0),
                "previous_sampling_version": checkpoint.get("stream_scheduler", {}).get(
                    "version"
                ),
            }
        elif "data_order_migration" in checkpoint:
            self._data_order_migration = checkpoint["data_order_migration"]
        self._resume_rng = (checkpoint.get("torch_rng"), checkpoint.get("cuda_rng"))
        if self.loss_balancer is not None:
            saved_balance = checkpoint.get("loss_balance", {})
            self.model.load_adaptive_multiplier_state(
                saved_balance.get("multipliers", {})
                if isinstance(saved_balance, Mapping)
                else None
            )
            for key in self.model.adaptive_multiplier_state():
                # Old checkpoints may contain multipliers for objectives that
                # are now fixed. The reference stream must also stay fixed.
                if (
                    key not in self.settings.loss_balance.included_objectives
                    or key == f"streams.{self.loss_balancer.reference_stream}"
                ):
                    self.model.set_adaptive_multiplier(key, 1.0)
            self.loss_balancer.load_state_dict(saved_balance)


__all__ = ["JointInversionModule"]
