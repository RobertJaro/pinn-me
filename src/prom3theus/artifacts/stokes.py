"""Stored-input outgoing radiation evaluation for validated LTE inversion artifacts."""

from __future__ import annotations

from prom3theus.observations.arrays import materialize_array, read_flat_samples

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from prom3theus.core.nn import temporary_eval_mode
from prom3theus.observations import (
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    ObservationRaster,
    ObservationSpec,
)

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
    return read_flat_samples(value, flat_indices)


def stokes_batches(
    raster: ObservationRaster,
    flat_indices: torch.Tensor,
    batch_size: int,
    velocity_synthesis_mode: str,
    instrument_type: str,
) -> Iterator[tuple[dict[str, Any], torch.Tensor]]:
    """Yield canonical radiation observation inputs for each valid-pixel batch."""

    spectral_present = _HMI_SPECTRAL_WEIGHTS in raster.auxiliary
    continuum_present = _HMI_CONTINUUM_WEIGHTS in raster.auxiliary
    if spectral_present != continuum_present:
        raise ArtifactExportError(
            "Canonical observation store must contain both HMI instrument-response arrays."
        )
    if instrument_type == "hmi_filter_profiles" and not spectral_present:
        raise ArtifactExportError("HMI Stokes export requires canonical instrument-response arrays.")
    if instrument_type != "hmi_filter_profiles" and spectral_present:
        raise ArtifactExportError("Only hmi_filter_profiles observations may contain HMI response arrays.")
    removed = None
    if velocity_synthesis_mode == CARRINGTON_REGISTERED_RELATIVE_VELOCITY:
        removed = torch.as_tensor(removed_solar_los_velocity(raster))
    for start in range(0, len(flat_indices), batch_size):
        selected = flat_indices[start:start + batch_size]
        batch = {
            "coordinates": read_flat_samples(raster.coordinates, selected),
            "ray_direction": read_flat_samples(raster.ray_direction, selected),
            "stokes_basis": read_flat_samples(raster.stokes_basis, selected),
        }
        observer_los = _flatten_auxiliary(raster, "observer_los_velocity_m_per_s", selected)
        if observer_los is not None:
            batch["observer_los_velocity_m_per_s"] = observer_los
        if removed is not None:
            batch["removed_solar_los_velocity_m_per_s"] = removed.index_select(
                0, selected % raster.spatial_shape[1],
            )
        if spectral_present:
            batch["instrument_response"] = {
                "spectral_weights": _flatten_auxiliary(raster, _HMI_SPECTRAL_WEIGHTS, selected),
                "continuum_weights": _flatten_auxiliary(raster, _HMI_CONTINUUM_WEIGHTS, selected),
            }
        yield batch, selected


@torch.inference_mode()
def evaluate_stokes(
    module: Any,
    raster: ObservationRaster,
    spec: ObservationSpec,
    *,
    batch_size: int,
    storage_dtype: str,
) -> dict[str, np.ndarray]:
    """Predict every valid pixel from only canonical stored inputs."""

    if batch_size < 1:
        raise ValueError("stokes_batch_size must be positive.")
    torch_dtype, numpy_dtype = resolve_storage_dtype(storage_dtype)
    parameter = next(module.atmosphere_model.parameters())
    flat_valid = materialize_array(raster.valid_mask).reshape(-1)
    flat_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
    if flat_indices.numel() == 0:
        raise ArtifactExportError("The canonical observation contains no valid pixels.")
    wavelength_count = int(raster.wavelength_angstrom.numel())
    predicted_flat = np.full(
        (int(flat_valid.numel()), 4, wavelength_count),
        np.nan,
        dtype=numpy_dtype,
    )
    # A direct observation term does not own its shared window or atmosphere.
    with temporary_eval_mode(
        module, getattr(module, "synthesis_window", None), module.atmosphere_model
    ):
        batches = stokes_batches(
            raster,
            flat_indices,
            batch_size,
            spec.velocity_synthesis_mode.value,
            spec.instrument_type,
        )
        for batch, selected in batches:
            device_batch = {
                name: ({key: value.to(parameter) for key, value in item.items()}
                       if isinstance(item, dict) else item.to(parameter))
                for name, item in batch.items()
            }
            prediction = module.predict(device_batch)
            expected_shape = (int(selected.numel()), 4, wavelength_count)
            if tuple(prediction.shape) != expected_shape:
                raise ArtifactExportError(
                    "Predicted Stokes shape differs from the canonical observation: "
                    f"expected {expected_shape}, got {tuple(prediction.shape)}."
                )
            predicted_flat[selected.cpu().numpy()] = (
                prediction.detach().to(torch_dtype).cpu().numpy()
            )
    shape = (*raster.spatial_shape, 4, wavelength_count)
    predicted = predicted_flat.reshape(shape)
    observed = materialize_array(raster.stokes).detach().to(torch_dtype).cpu().numpy()
    return {
        "wavelength_angstrom": (
            raster.wavelength_angstrom.detach().double().cpu().numpy()
        ),
        "predicted_stokes": predicted,
        "observed_stokes": observed,
        "stokes_residual": predicted - observed,
    }


__all__ = ["evaluate_stokes", "stokes_batches"]
