"""Stored-input Stokes evaluation for validated LTE artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from prom3theus.observations import (
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    ObservationRaster,
    ObservationSpec,
)
from prom3theus.training.lightning import LTEInversionModule

from .errors import ArtifactExportError
from .evaluation import resolve_storage_dtype
from .validation import removed_solar_los_velocity


_HMI_SPECTRAL_WEIGHTS = "instrument_response:spectral_weights"
_HMI_CONTINUUM_WEIGHTS = "instrument_response:continuum_weights"


def _flatten_auxiliary(
    raster: ObservationRaster,
    name: str,
    flat_indices: torch.Tensor,
) -> torch.Tensor | None:
    value = raster.auxiliary.get(name)
    if value is None:
        return None
    return value.reshape(-1, *value.shape[2:]).index_select(0, flat_indices)


def stokes_batches(
    raster: ObservationRaster,
    flat_indices: torch.Tensor,
    batch_size: int,
    velocity_synthesis_mode: str,
    instrument_type: str,
) -> Iterator[tuple[dict[str, Any], torch.Tensor]]:
    """Yield canonical synthesis inputs for each valid-pixel batch."""

    coordinates = raster.coordinates.reshape(-1, 3).index_select(0, flat_indices)
    rays = raster.ray_direction.reshape(-1, 3).index_select(0, flat_indices)
    bases = raster.stokes_basis.reshape(-1, 3, 3).index_select(0, flat_indices)
    observer_los = _flatten_auxiliary(
        raster, "observer_los_velocity_m_per_s", flat_indices
    )
    response = None
    spectral = _flatten_auxiliary(raster, _HMI_SPECTRAL_WEIGHTS, flat_indices)
    continuum = _flatten_auxiliary(raster, _HMI_CONTINUUM_WEIGHTS, flat_indices)
    if (spectral is None) != (continuum is None):
        raise ArtifactExportError(
            "Canonical observation store must contain both HMI instrument-response arrays."
        )
    if spectral is not None:
        response = {"spectral_weights": spectral, "continuum_weights": continuum}
    if instrument_type == "hmi_filter_profiles" and response is None:
        raise ArtifactExportError(
            "HMI Stokes export requires canonical auxiliary arrays "
            f"{_HMI_SPECTRAL_WEIGHTS!r} and {_HMI_CONTINUUM_WEIGHTS!r}. "
            "Recreate the artifact with a runner that persists its evaluated "
            "per-pixel instrument response."
        )
    if instrument_type != "hmi_filter_profiles" and response is not None:
        raise ArtifactExportError(
            "Only hmi_filter_profiles observations may contain HMI response arrays."
        )
    removed = None
    if velocity_synthesis_mode == CARRINGTON_REGISTERED_RELATIVE_VELOCITY:
        per_column = torch.as_tensor(removed_solar_los_velocity(raster))
        column_indices = flat_indices % raster.spatial_shape[1]
        removed = per_column.index_select(0, column_indices)
    for start in range(0, coordinates.shape[0], batch_size):
        stop = min(start + batch_size, coordinates.shape[0])
        batch: dict[str, Any] = {
            "coordinates": coordinates[start:stop],
            "ray_direction": rays[start:stop],
            "stokes_basis": bases[start:stop],
        }
        if observer_los is not None:
            batch["observer_los_velocity_m_per_s"] = observer_los[start:stop]
        if removed is not None:
            batch["removed_solar_los_velocity_m_per_s"] = removed[start:stop]
        if response is not None:
            batch["instrument_response"] = {
                name: value[start:stop] for name, value in response.items()
            }
        yield batch, flat_indices[start:stop]


@torch.inference_mode()
def evaluate_stokes(
    module: LTEInversionModule,
    raster: ObservationRaster,
    spec: ObservationSpec,
    *,
    batch_size: int,
    storage_dtype: str,
) -> dict[str, np.ndarray]:
    """Synthesize every valid pixel from only canonical stored inputs."""

    if batch_size < 1:
        raise ValueError("stokes_batch_size must be positive.")
    torch_dtype, numpy_dtype = resolve_storage_dtype(storage_dtype)
    parameter = next(module.atmosphere_model.parameters())
    flat_valid = raster.valid_mask.reshape(-1)
    flat_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
    if flat_indices.numel() == 0:
        raise ArtifactExportError("The canonical observation contains no valid pixels.")
    wavelength_count = int(raster.wavelength_angstrom.numel())
    predicted_flat = np.full(
        (int(flat_valid.numel()), 4, wavelength_count),
        np.nan,
        dtype=numpy_dtype,
    )
    was_training = module.training
    module.eval()
    try:
        batches = stokes_batches(
            raster,
            flat_indices,
            batch_size,
            spec.velocity_synthesis_mode.value,
            spec.instrument_type,
        )
        for batch, selected in batches:
            response = batch.get("instrument_response")
            if response is not None:
                response = {
                    name: value.to(parameter) for name, value in response.items()
                }
            prediction = module.synthesize(
                batch["coordinates"].to(parameter),
                ray_direction=batch["ray_direction"].to(parameter),
                stokes_basis=batch["stokes_basis"].to(parameter),
                observer_los_velocity_m_per_s=(
                    None
                    if "observer_los_velocity_m_per_s" not in batch
                    else batch["observer_los_velocity_m_per_s"].to(parameter)
                ),
                removed_solar_los_velocity_m_per_s=(
                    None
                    if "removed_solar_los_velocity_m_per_s" not in batch
                    else batch["removed_solar_los_velocity_m_per_s"].to(parameter)
                ),
                instrument_response=response,
            )["stokes"]
            expected_shape = (int(selected.numel()), 4, wavelength_count)
            if tuple(prediction.shape) != expected_shape:
                raise ArtifactExportError(
                    "Synthesized Stokes shape differs from the canonical observation: "
                    f"expected {expected_shape}, got {tuple(prediction.shape)}."
                )
            predicted_flat[selected.cpu().numpy()] = (
                prediction.detach().to(torch_dtype).cpu().numpy()
            )
    finally:
        module.train(was_training)
    shape = (*raster.spatial_shape, 4, wavelength_count)
    predicted = predicted_flat.reshape(shape)
    observed = raster.stokes.detach().to(torch_dtype).cpu().numpy()
    return {
        "wavelength_angstrom": (
            raster.wavelength_angstrom.detach().to(torch_dtype).cpu().numpy()
        ),
        "predicted_stokes": predicted,
        "observed_stokes": observed,
        "stokes_residual": predicted - observed,
    }


__all__ = ["evaluate_stokes", "stokes_batches"]
