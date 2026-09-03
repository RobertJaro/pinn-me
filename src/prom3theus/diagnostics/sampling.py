"""Bounded, deterministic sampling for LTE training diagnostics."""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import torch
import torch.distributed as dist


def subsample_grid(
    height: int,
    width: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a rectangular grid containing at most ``maximum`` pixels."""

    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (height, width, maximum)
    ):
        raise TypeError("Grid dimensions and maximum sample count must be integers.")
    height, width, maximum = int(height), int(width), int(maximum)
    if min(height, width, maximum) < 1:
        raise ValueError("Grid dimensions and maximum sample count must be positive.")
    stride = max(1, int(math.ceil(math.sqrt(height * width / maximum))))
    rows = np.arange(0, height, stride, dtype=np.int64)
    columns = np.arange(0, width, stride, dtype=np.int64)
    while rows.size * columns.size > maximum:
        stride += 1
        rows = np.arange(0, height, stride, dtype=np.int64)
        columns = np.arange(0, width, stride, dtype=np.int64)
    return rows, columns


def display_grid(trainer, raster, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose one validation-derived grid shared by every diagnostic map."""

    data_module = getattr(trainer, "datamodule", None)
    validation = getattr(data_module, "validation_dataset", None)
    evaluation_dataset = getattr(data_module, "_evaluation_dataset", None)
    if evaluation_dataset is None:
        evaluation_dataset = getattr(validation, "dataset", None)
    pixel_indices = getattr(evaluation_dataset, "pixel_indices", None)
    subset_indices = getattr(validation, "indices", None)
    if pixel_indices is None or subset_indices is None:
        return subsample_grid(*raster.spatial_shape, maximum)

    selected = pixel_indices[torch.as_tensor(subset_indices, dtype=torch.long)]
    all_rows = torch.unique(selected[:, 0], sorted=True).cpu().numpy()
    all_columns = torch.unique(selected[:, 1], sorted=True).cpu().numpy()
    stride = max(
        1,
        int(math.ceil(math.sqrt(all_rows.size * all_columns.size / maximum))),
    )
    rows = all_rows[::stride]
    columns = all_columns[::stride]
    while rows.size * columns.size > maximum:
        stride += 1
        rows = all_rows[::stride]
        columns = all_columns[::stride]
    return rows, columns


def map_coordinates(
    raster,
    rows: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two-dimensional Carrington-chart center grids in Mm."""

    coords = raster.coordinates[rows][:, columns].detach().float().cpu().numpy()
    if coords.shape[-1] != 3:
        raise ValueError(
            "Observation coordinates must end in [x_mm, y_mm, time_hours]."
        )
    x_mm = coords[..., 0]
    y_mm = coords[..., 1]
    if not np.isfinite(x_mm).all() or not np.isfinite(y_mm).all():
        raise ValueError("Observation map coordinates must be finite.")
    return x_mm, y_mm


def integrated_stokes(
    stokes: torch.Tensor,
    wavelength_angstrom: torch.Tensor,
    wavelength_weights: torch.Tensor,
) -> torch.Tensor:
    """Integrate absolute Stokes over the loss-weighted wavelength intervals."""

    if stokes.ndim < 2 or stokes.shape[-2] != 4:
        raise ValueError("stokes must use a penultimate [I,Q,U,V] axis.")
    if not stokes.is_floating_point():
        raise TypeError("stokes must use a real floating-point dtype.")
    if not torch.isfinite(stokes).all():
        raise ValueError("stokes must contain only finite values.")
    if wavelength_angstrom.ndim != 1 or wavelength_weights.ndim != 1:
        raise ValueError(
            "wavelength_angstrom and wavelength_weights must be one-dimensional."
        )
    wavelength_count = wavelength_angstrom.numel()
    if wavelength_count < 2 or wavelength_weights.numel() != wavelength_count:
        raise ValueError(
            "Stokes, wavelength_angstrom, and wavelength_weights must share at least "
            "two spectral samples."
        )
    if stokes.shape[-1] != wavelength_count:
        raise ValueError(
            "Stokes, wavelength_angstrom, and wavelength_weights must share at least "
            "two spectral samples."
        )
    wavelength = wavelength_angstrom.to(stokes)
    spectral_weights = wavelength_weights.to(stokes)
    if (
        not torch.isfinite(wavelength).all()
        or not torch.isfinite(spectral_weights).all()
    ):
        raise ValueError("Wavelengths and spectral weights must be finite.")
    if not torch.all(wavelength[1:] > wavelength[:-1]):
        raise ValueError("wavelength_angstrom must be strictly increasing.")
    if torch.any(spectral_weights < 0):
        raise ValueError("wavelength_weights must be non-negative.")
    weighted_delta_wavelength = (wavelength[1:] - wavelength[:-1]) * torch.minimum(
        spectral_weights[1:], spectral_weights[:-1]
    )
    return (
        0.5
        * (stokes[..., 1:].abs() + stokes[..., :-1].abs())
        * weighted_delta_wavelength
    ).sum(dim=-1)


class ValidationSampleCollector:
    """Accumulate complete integrals and a bounded deterministic profile sample."""

    def __init__(self, max_profile_samples: int):
        if isinstance(max_profile_samples, bool) or not isinstance(
            max_profile_samples, Integral
        ):
            raise TypeError("max_profile_samples must be an integer.")
        if max_profile_samples < 1:
            raise ValueError("max_profile_samples must be positive.")
        self.max_profile_samples = int(max_profile_samples)
        self._integrated_outputs: list[dict[str, torch.Tensor]] = []
        self._profile_reservoir: dict[str, torch.Tensor] | None = None

    def reset(self) -> None:
        self._integrated_outputs.clear()
        self._profile_reservoir = None

    def add(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        pixel_index: torch.Tensor,
        wavelength_angstrom: torch.Tensor,
        wavelength_weights: torch.Tensor,
    ) -> None:
        """Add one validation batch without retaining every full profile."""

        integrated_prediction = integrated_stokes(
            prediction, wavelength_angstrom, wavelength_weights
        )
        integrated_reference = integrated_stokes(
            reference, wavelength_angstrom, wavelength_weights
        )
        cpu_pixel_index = pixel_index.detach().to(device="cpu", dtype=torch.long)
        self._integrated_outputs.append(
            {
                "integrated_prediction": integrated_prediction.detach().float().cpu(),
                "integrated_reference": integrated_reference.detach().float().cpu(),
                "pixel_index": cpu_pixel_index,
            }
        )
        candidate = {
            "stokes_pred": prediction.detach().float().cpu(),
            "stokes_reference": reference.detach().float().cpu(),
            "priority": (
                (cpu_pixel_index[:, 0] * 73_856_093)
                ^ (cpu_pixel_index[:, 1] * 19_349_663)
            ).bitwise_and(0x7FFF_FFFF),
        }
        if self._profile_reservoir is not None:
            candidate = {
                key: torch.cat((self._profile_reservoir[key], value), dim=0)
                for key, value in candidate.items()
            }
        count = candidate["priority"].numel()
        if count > self.max_profile_samples:
            selected = torch.topk(
                candidate["priority"],
                self.max_profile_samples,
                largest=False,
                sorted=False,
            ).indices
            candidate = {
                key: value.index_select(0, selected) for key, value in candidate.items()
            }
        self._profile_reservoir = candidate

    def local_payload(self) -> dict[str, torch.Tensor] | None:
        if not self._integrated_outputs or self._profile_reservoir is None:
            return None
        payload = {
            key: torch.cat([item[key] for item in self._integrated_outputs], dim=0)
            for key in (
                "integrated_prediction",
                "integrated_reference",
                "pixel_index",
            )
        }
        payload.update(self._profile_reservoir)
        return payload

    def merge_payloads(self, payloads) -> dict[str, torch.Tensor] | None:
        """Merge process-local payloads while retaining the reservoir bound."""

        available = [payload for payload in payloads if payload is not None]
        if not available:
            return None
        merged = {
            key: torch.cat([payload[key] for payload in available], dim=0)
            for key in (
                "integrated_prediction",
                "integrated_reference",
                "pixel_index",
                "stokes_pred",
                "stokes_reference",
                "priority",
            )
        }
        if merged["priority"].numel() > self.max_profile_samples:
            selected = torch.topk(
                merged["priority"],
                self.max_profile_samples,
                largest=False,
                sorted=False,
            ).indices
            for key in ("stokes_pred", "stokes_reference", "priority"):
                merged[key] = merged[key].index_select(0, selected)
        return merged

    def gather(self) -> dict[str, torch.Tensor] | None:
        """Gather validation samples on distributed rank zero."""

        payload = self.local_payload()
        if not (dist.is_available() and dist.is_initialized()):
            return payload
        rank = dist.get_rank()
        gathered = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
        if rank != 0:
            return None
        return self.merge_payloads(gathered)


__all__ = [
    "ValidationSampleCollector",
    "display_grid",
    "integrated_stokes",
    "map_coordinates",
    "subsample_grid",
]
