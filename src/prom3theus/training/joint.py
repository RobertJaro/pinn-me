"""Lightning optimization lifecycle for the shared multimodal objective."""

from copy import deepcopy
import math
from collections.abc import Mapping

import torch
from pytorch_lightning import LightningModule

from prom3theus.config.joint_schema import JointTrainingConfig, LossBalanceConfig
from prom3theus.inversion.joint import JointForwardModel


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
    magnetic = config.get("atmosphere", {}).get("parameters", {}).get("magnetic_field", {})
    # None preserves the original height-dependent readout. Old configurations
    # omit this field; numerical reference heights change the physical model.
    if magnetic.get("reference_height_megameter") is None:
        magnetic.pop("reference_height_megameter", None)
    physics = config.get("physics", {})
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
            "residual_adaptive_enabled",
            "residual_adaptive_candidate_multiplier",
            "residual_adaptive_fraction",
            "residual_adaptive_start_step",
            "residual_adaptive_update_every_n_steps",
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
        for key in ("workers", "pin_memory", "batch_size", "validation_batch_size"):
            loader.pop(key, None)
        term = stream.get("data_term", {})
        term.pop("weight", None)
        term.pop("weight_schedule", None)
        term.pop("objective", None)
        disambiguation = term.get("disambiguation")
        if not disambiguation or not disambiguation.get("enabled", False):
            # A disabled phase continuation has no scientific state and is
            # equivalent to its absence in older checkpoint contracts.
            term.pop("disambiguation", None)
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
        disambiguation = options.get("disambiguation_config")
        if not disambiguation or not disambiguation.get("enabled", False):
            options.pop("disambiguation_config", None)
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
        excluded_objectives=(),
    ):
        excluded = set(excluded_objectives)
        sources: dict[str, torch.Tensor] = {}
        for stream_id, value in evaluation.weighted_likelihoods.items():
            key = f"streams.{stream_id}"
            if key not in excluded:
                sources[key] = value
        for term_id, result in evaluation.shared_terms.items():
            if result.component_losses:
                sources.update(
                    {
                        f"shared.{term_id}.{name}": value
                        for name, value in result.component_losses.items()
                        if f"shared.{term_id}.{name}" not in excluded
                    }
                )
            elif result.active:
                key = f"shared.{term_id}.loss"
                if key not in excluded:
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
            excluded_objectives=settings.excluded_objectives,
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
            if math.isfinite(float(value)) and float(value) > 0.0
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
        with distributed_batch(valid):
            evaluation = self.model.evaluate_batches(batch)
        if self._first_batch:
            print("First training forward pass complete: starting backward", flush=True)
        if not evaluation.loss.requires_grad:
            raise ValueError(
                "Joint training requires at least one active differentiable objective."
            )
        metrics = evaluation.training_metrics()
        if self.loss_balancer is not None:
            for term_id, result in evaluation.shared_terms.items():
                metrics.update(
                    {
                        f"shared.{term_id}.metrics.{name}": value
                        for name, value in result.metrics.items()
                    }
                )
        if self.loss_balancer is not None:
            metrics.update(
                self.loss_balancer.update(
                    self.model,
                    evaluation,
                    int(self.global_step),
                )
            )
        if torch.distributed.is_initialized():
            values = torch.stack([value.detach().to(evaluation.loss).reshape(())
                                  for value in metrics.values()])
            torch.distributed.all_reduce(values)
            values /= torch.distributed.get_world_size()
            metrics = dict(zip(metrics, values.unbind(), strict=True))
        for name, value in metrics.items():
            self.log(
                f"train.{name}",
                value.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=name == "loss",
                batch_size=1,
            )
        return evaluation.loss

    def on_after_backward(self):
        if self._first_batch:
            print("First training backward pass complete", flush=True)
            self._first_batch = False
        # Read all flags together, rather than synchronizing CUDA once per
        # parameter. Keep the parameter-specific error on the failure path.
        by_device = {}
        for name, parameter in self.named_parameters():
            if parameter.grad is not None:
                by_device.setdefault(parameter.grad.device, []).append(
                    (name, torch.isfinite(parameter.grad).all())
                )
        for checks in by_device.values():
            finite = torch.stack([flag for _, flag in checks]).cpu().tolist()
            for (name, _), valid in zip(checks, finite, strict=True):
                if not valid:
                    raise FloatingPointError(f"Non-finite joint gradient: {name}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.settings.learning_rate)
        ratio = self.settings.final_learning_rate / self.settings.learning_rate
        steps = self.trainer.estimated_stepping_batches

        def learning_rate_factor(step: int) -> float:
            warmup = getattr(self.settings, "warmup_steps", 0)
            if warmup > 0 and step < warmup:
                return max((step + 1) / warmup, 1.0e-3)
            decay_steps = max(steps - warmup, 1)
            progress = min(max((step - warmup) / decay_steps, 0.0), 1.0)
            return ratio**progress

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, learning_rate_factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
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
        if getattr(self, "_resume_optimizer_changed", False):
            # Lightning restores old optimizer/scheduler learning rates after
            # on_load_checkpoint. Reapply the requested schedule at its saved step.
            steps = max(self.trainer.estimated_stepping_batches, 1)
            warmup = getattr(self.settings, "warmup_steps", 0)
            if warmup > 0 and self.global_step < warmup:
                factor = max((self.global_step + 1) / warmup, 1.0e-3)
            else:
                ratio = self.settings.final_learning_rate / self.settings.learning_rate
                progress = min(
                    max((self.global_step - warmup) / max(steps - warmup, 1), 0.0),
                    1.0,
                )
                factor = ratio**progress
            rate = self.settings.learning_rate * factor
            for optimizer in self.trainer.optimizers:
                for group in optimizer.param_groups:
                    group["initial_lr"] = self.settings.learning_rate
                    group["lr"] = rate
            for config in self.trainer.lr_scheduler_configs:
                config.scheduler.base_lrs = [self.settings.learning_rate] * len(
                    config.scheduler.base_lrs
                )
                config.scheduler._last_lr = [rate] * len(config.scheduler.base_lrs)
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
        self._resume_optimizer_changed = any(
            saved_training.get(key) != getattr(self.settings, key, None)
            for key in (
                "learning_rate",
                "final_learning_rate",
                "warmup_steps",
                "max_steps",
                "max_epochs",
            )
        )
        # Loss weights/scales are configuration, not learned checkpoint state.
        for name in list(checkpoint.get("state_dict", {})):
            if name.endswith(".objective.stokes_sigmas"):
                del checkpoint["state_dict"][name]
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
            for key in self.settings.loss_balance.excluded_objectives:
                # A resumed checkpoint may predate an exclusion policy and
                # contain an amplified multiplier for this objective.
                self.model.set_adaptive_multiplier(key, 1.0)
            self.loss_balancer.load_state_dict(saved_balance)


__all__ = ["JointInversionModule"]
