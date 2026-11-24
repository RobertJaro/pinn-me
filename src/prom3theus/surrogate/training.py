"""Fit one coefficient surrogate using exact LTE targets generated on the training device."""

from time import perf_counter

import numpy as np
import torch
from lightning_fabric import Fabric
from torch.utils.data import DataLoader

from prom3theus.core.validation import skip_tensor_validation
from prom3theus.muram.reporting import progress_bar
from prom3theus.rt.coefficient_mapper import CoefficientMapper
from prom3theus.rt.local_coefficients import LTELocalCoefficients
from prom3theus.training.optimizers import build_optimizer, exponential_lr_factor

from .checkpoints import Checkpoints, resumed_mapper_options, training_contract
from .config import TrainingConfig
from .diagnostics import spline_resolution_check
from .evaluation import Evaluator, make_evaluation_data
from .run_files import atomic_save, logger, open_run, write_metrics
from .replay import HardProfileBuffer
from .sampling import FreshPointBatches, load_snapshot_splits
from .spectrum import observing_grid
from .teacher import training_pair


@torch.no_grad()
def _calibrate(model, teacher, points, wavelengths):
    """Estimate input and target scales from a reproducible batch of training states."""
    mapper_points, target = training_pair(teacher, points, wavelengths)
    features = model.features(mapper_points)
    model.input_center.copy_(features.mean(0))
    model.input_scale.copy_(features.std(0, unbiased=False).clamp_min(0.05))
    extinction = target[..., :1].clamp_min(1e-35)
    model.log_extinction_center.copy_(extinction.log().mean())
    model.log_extinction_scale.copy_(extinction.log().std(unbiased=False).clamp_min(1))
    model.relative_scale.copy_(
        (target[..., 1:] / extinction).abs().mean((0, 1)).clamp_min(0.001)
    )
    if model.linear is not None:
        # Express the affine map in calibrated units. Q and U share one amplitude.
        s = model.relative_scale
        model.linear_scale.copy_(torch.stack((s.new_ones(()), s[:2].mean(), s[2], s[3:5].mean(), s[5])))


def _train_step(model, teacher, optimizer, points, wavelengths, step, *, replay=None, replay_count=0,
                wavelength_jitter_fraction=0.0, seed=0):
    # A batch-indexed draw reproduces the grid when resuming from a checkpoint.
    if wavelength_jitter_fraction:
        shift = np.random.default_rng([seed, step]).normal(scale=wavelength_jitter_fraction)
        offset = shift * (wavelengths[1] - wavelengths[0])
        query = (wavelengths + offset).clamp(*model.spec["bounds"])
    else:
        offset = wavelengths.new_zeros(())
        query = wavelengths
    optimizer.zero_grad(set_to_none=True)
    if len(points):
        with torch.no_grad(), skip_tensor_validation():
            fresh_points, fresh_targets = training_pair(teacher, points, query)
    else:
        fresh_points, fresh_targets = replay.points[:0], replay.targets[:0]
    mapper_points, truth = fresh_points, fresh_targets
    indices = None
    if replay_count:
        indices = replay.sample_indices(replay_count, step)
        mapper_points = torch.cat((fresh_points, replay.points[indices]))
        truth = torch.cat((fresh_targets, replay.targets[indices]))
        if wavelength_jitter_fraction:
            offsets = torch.cat((offset.expand(len(points)), replay.offsets[indices]))
            query = (wavelengths + offsets[:, None]).clamp(*model.spec["bounds"])
    prediction = model.coefficients(mapper_points, model.prepare_spectrum(query))
    if replay is None:
        losses = model.loss_channels(prediction, truth)
    else:
        profile_losses = model.profile_loss_channels(prediction, truth)
        losses = profile_losses.mean(0)
    losses.mean().backward()
    optimizer.step()
    if replay is not None:
        replay.update(fresh_points, fresh_targets, offset, profile_losses.detach().mean(-1), indices)
    return losses.detach()


def _fit(config, model, teacher, optimizer, dataset, evaluator, checkpoints, start, replay):
    wavelengths = evaluator.wavelengths
    device = wavelengths.device
    # Each index identifies a complete point batch, including its perturbations.
    loader = DataLoader(
        FreshPointBatches(dataset, config.replay_fraction) if replay is not None else dataset,
        sampler=range(start, config.steps),
        batch_size=None,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    steps = progress_bar(range(start + 1, config.steps + 1), description="Training mapper", unit="step")
    accumulated = torch.zeros(7, device=device)
    last_log = start
    loader_wait = 0.0
    start_time = perf_counter()
    batches = iter(loader)
    for step in steps:
        wait_start = perf_counter()
        batch = next(batches)
        loader_wait += perf_counter() - wait_start
        factor = exponential_lr_factor(step - 1, config.learning_rate, config.final_learning_rate,
                                       config.learning_rate_decay_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * factor
        points = batch.to(device, non_blocking=True)
        accumulated += _train_step(model, teacher, optimizer, points, wavelengths, step,
                                   replay=replay, replay_count=config.batch_size - len(points),
                                   wavelength_jitter_fraction=config.wavelength_jitter_fraction, seed=config.seed)
        if step % config.log_every == 0 or step == config.steps:
            averages = (accumulated / (step - last_log)).tolist()
            elapsed = perf_counter() - start_time
            rate = (step - start) * config.batch_size / elapsed
            loss = sum(averages) / len(averages)
            steps.set_postfix_str(f"loss={loss:.4g}")
            eta = (config.steps - step) * config.batch_size / rate
            write_metrics(
                config.output_dir,
                "train",
                step=step,
                samples_seen=step * config.batch_size,
                channel_loss=averages,
                loss=loss,
                learning_rate=optimizer.param_groups[0]["lr"],
                elapsed_s=elapsed,
                points_per_second=rate,
                loader_wait_ms_per_batch=1000 * loader_wait / (step - last_log),
                eta_s=eta,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
            )
            accumulated.zero_()
            last_log = step
            loader_wait = 0.0
        if step % config.validate_every == 0 or step == config.steps:
            score = evaluator.evaluate(model, teacher, "validation", step)
            checkpoints.select(step, model, score)
        if step % config.checkpoint_every == 0 or step == config.steps:
            checkpoints.save_last(step, model, optimizer, replay)
            logger.info("Saved checkpoint at step %d", step)


def _run(config: TrainingConfig):
    torch.manual_seed(config.seed)
    if config.device == "auto":
        device = Fabric(accelerator="auto", devices=1).device
        if device.type == "mps":
            logger.info("Using CPU because the LTE teacher needs float64 wavelength support")
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    logger.info("Training surrogate on %s", device)
    window, detector = config.window_and_detector
    snapshot, datasets = load_snapshot_splits(config)
    line_ids = window["line_ids"]
    bounds = window["wavelength_bounds_vacuum_angstrom"]
    model = CoefficientMapper(
        line_ids=line_ids, bounds=bounds, knots=config.knots, width=config.width,
        architecture=config.architecture, hidden_layers=config.hidden_layers,
        first_omega_0=config.first_omega_0,
        hidden_omega_0=config.hidden_omega_0, **resumed_mapper_options(config),
    ).to(device)
    teacher = LTELocalCoefficients(line_ids=line_ids).to(device).requires_grad_(False)
    optimizer = build_optimizer(model, kind=config.optimizer, lr=config.learning_rate)
    wavelengths = torch.linspace(
        *bounds, config.wavelength_count, dtype=torch.float64, device=device
    )
    ray_wavelengths, observe = observing_grid(detector, wavelengths, device)
    if (
        float(ray_wavelengths.min()) < bounds[0]
        or float(ray_wavelengths.max()) > bounds[1]
    ):
        raise ValueError("Detector quadrature extends beyond training bounds")
    scale = window.get("radiance_scale", 2.5e13)
    contract = training_contract(
        config, model, teacher, snapshot, datasets, detector, scale
    )
    checkpoints = Checkpoints(config.output_dir, contract)
    replay = (HardProfileBuffer(config.replay_buffer_size, wavelengths, config.seed)
              if config.replay_fraction else None)
    start = checkpoints.restore(config, model, optimizer, replay)
    evaluation_path = config.output_dir / "evaluation.pt"
    if start is None:
        _calibrate(model, teacher, datasets["train"][2**32 - 1].to(device), wavelengths)
        data = make_evaluation_data(datasets, teacher, wavelengths, config.ray_depths)
        basis_points = data["validation"]["points"][:16].to(device)
        write_metrics(
            config.output_dir,
            "basis_resolution",
            **spline_resolution_check(model, teacher, basis_points),
        )
        atomic_save(evaluation_path, data)
        start = 0
        checkpoints.save_last(start, model, optimizer, replay)
    else:
        data = torch.load(evaluation_path, weights_only=True, map_location="cpu")
    evaluator = Evaluator(
        directory=config.output_dir,
        data=data,
        wavelengths=wavelengths,
        ray_wavelengths=ray_wavelengths,
        observe=observe,
        profile_axis=detector["observed_wavelength_angstrom"]
        if detector
        else wavelengths.cpu(),
        wavelength_medium="air" if detector else "vacuum",
        radiance_scale=scale,
    )
    _fit(
        config,
        model,
        teacher,
        optimizer,
        datasets["train"],
        evaluator,
        checkpoints,
        start,
        replay,
    )
    best_path = config.output_dir / "best.pt"
    chosen = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(chosen["mapper"])
    evaluator.evaluate(model, teacher, "test", chosen["step"], gradients=True)
    logger.info(
        "Finished. Selected checkpoint %s at step %d", best_path, chosen["step"]
    )


def train(config: TrainingConfig):
    """Train or resume a surrogate and evaluate the selected checkpoint on held-out rays."""
    with open_run(config.output_dir):
        _run(config)
