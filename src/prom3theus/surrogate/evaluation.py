"""Fixed held-out atmospheres and validation of coefficients and emergent Stokes profiles."""

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

import torch

from .diagnostics import ray_comparison, save_profile_figure
from .run_files import atomic_save, logger, write_metrics
from .teacher import density_inputs, training_pair


@torch.no_grad()
def make_evaluation_data(datasets, teacher, wavelengths, ray_depths):
    """Fix the same exact labels and reference rays for all validation checkpoints."""
    data = {}
    for split in ("validation", "test"):
        dataset = datasets[split]
        points, identifiers = dataset.sample_batch(0, describe=True)
        candidates = [dataset.reference_ray(i + 1, ray_depths) for i in range(8)]
        candidates.sort(key=lambda sample: float(sample[1][:, 4:].norm(dim=-1).mean()))
        mapper_points, reference = training_pair(
            teacher, points.to(wavelengths.device), wavelengths
        )
        data[split] = dict(
            points=points.cpu(),
            mass_density=mapper_points[..., 1].contiguous().cpu(),
            targets=reference.cpu(),
            identifiers=identifiers,
            rays={
                "weaker_field": candidates[0],
                "stronger_field": candidates[-1],
                "inclined": dataset.reference_ray(10, ray_depths, mu=0.85),
            },
        )
    return data


def _group_errors(fixed, absolute):
    identifiers = fixed["identifiers"]
    groups = {
        "native": ~identifiers["perturbed"],
        "perturbed": identifiers["perturbed"],
        "deep": identifiers["log_tau"] >= -1,
        "upper": identifiers["log_tau"] < -1,
        "strong_field": fixed["points"][:, 4:].norm(dim=-1) >= 1000,
    }
    return {
        name: {"count": int(mask.sum()), "mean_abs_m1": absolute[mask].mean(0).tolist()}
        for name, mask in groups.items()
        if mask.any()
    }


@dataclass
class Evaluator:
    directory: Path
    data: dict
    wavelengths: torch.Tensor
    ray_wavelengths: torch.Tensor
    observe: Callable
    profile_axis: list | torch.Tensor
    wavelength_medium: str
    radiance_scale: float

    def evaluate(self, model, teacher, split, step, *, gradients=False):
        fixed = self.data[split]
        device = self.wavelengths.device
        with torch.no_grad():
            prediction = model.coefficients(
                density_inputs(fixed["points"], fixed["mass_density"]).to(device),
                model.prepare_spectrum(self.wavelengths),
            )
            reference = fixed["targets"].to(device)
            loss = model.loss_channels(prediction, reference)
            absolute = (prediction - reference).abs().mean(-2).cpu()
        write_metrics(
            self.directory,
            split,
            step=step,
            coefficient_loss=loss.tolist(),
            coefficient_errors=_group_errors(fixed, absolute),
        )

        for name, (column, ray, distance) in fixed["rays"].items():
            metrics, profiles = ray_comparison(
                model,
                teacher,
                ray.to(device),
                distance.to(device),
                self.ray_wavelengths,
                self.observe,
                self.radiance_scale,
                gradients=gradients,
            )
            write_metrics(
                self.directory,
                split + "_ray",
                step=step,
                ray=name,
                column=column,
                **metrics,
            )
            stem = f"{split}_{name}"
            atomic_save(
                self.directory / f"{stem}.pt",
                dict(
                    reference=profiles[0],
                    prediction=profiles[1],
                    wavelength=self.profile_axis,
                    wavelength_medium=self.wavelength_medium,
                    radiance_scale=self.radiance_scale,
                ),
            )
            if find_spec("matplotlib") is not None:
                save_profile_figure(
                    self.directory / f"{stem}.png", self.profile_axis, *profiles
                )
        score = float(loss.mean())
        logger.info(
            "%s step %d  coefficient loss %.4g", split.capitalize(), step, score
        )
        return score
