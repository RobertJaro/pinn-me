"""Dataset and collation primitives for canonical observation rasters."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch.utils.data import Dataset, default_collate

from .contracts import ObservationBatch, ObservationRaster


class ObservationPixelDataset(Dataset):
    """Flatten valid raster pixels without copying the underlying arrays."""

    def __init__(
        self,
        raster: ObservationRaster,
        *,
        auxiliary_fields: Iterable[str] | None = None,
        include_surface_position: bool = False,
        include_pixel_index: bool = True,
        pixel_indices: torch.Tensor | None = None,
    ) -> None:
        self.raster = raster
        self.include_surface_position = bool(include_surface_position)
        self.include_pixel_index = bool(include_pixel_index)
        self.auxiliary_fields = tuple(
            raster.auxiliary if auxiliary_fields is None else auxiliary_fields
        )
        if any(not isinstance(name, str) or not name for name in self.auxiliary_fields):
            raise ValueError("auxiliary_fields must contain non-empty strings.")
        if len(set(self.auxiliary_fields)) != len(self.auxiliary_fields):
            raise ValueError("auxiliary_fields must be unique.")
        missing = set(self.auxiliary_fields) - set(raster.auxiliary)
        if missing:
            raise KeyError(
                f"Raster does not provide auxiliary fields: {sorted(missing)}."
            )
        if pixel_indices is None:
            indices = torch.nonzero(raster.valid_mask, as_tuple=False)
        else:
            raw_indices = torch.as_tensor(pixel_indices)
            integer_dtypes = {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
            if raw_indices.dtype not in integer_dtypes:
                raise TypeError("pixel_indices must use an integer dtype.")
            indices = raw_indices.to(dtype=torch.long)
        if indices.ndim != 2 or indices.shape[-1] != 2:
            raise ValueError("pixel_indices must have shape [sample, 2].")
        if indices.numel() == 0:
            raise ValueError(
                "An observation dataset requires at least one valid pixel."
            )
        if (
            torch.any(indices < 0)
            or torch.any(indices[:, 0] >= raster.spatial_shape[0])
            or torch.any(indices[:, 1] >= raster.spatial_shape[1])
        ):
            raise IndexError("pixel_indices lie outside the raster.")
        if not torch.all(raster.valid_mask[indices[:, 0], indices[:, 1]]):
            raise ValueError("pixel_indices may only select valid pixels.")
        self.pixel_indices = indices

    def __len__(self) -> int:
        return int(self.pixel_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pixel = self.pixel_indices[index]
        row, column = map(int, pixel)
        sample = {
            "coordinates": self.raster.coordinates[row, column],
            "ray_direction": self.raster.ray_direction[row, column],
            "stokes_basis": self.raster.stokes_basis[row, column],
            "stokes": self.raster.stokes[row, column],
        }
        if self.include_surface_position:
            sample["surface_position_m"] = self.raster.surface_position_m[row, column]
        for name in self.auxiliary_fields:
            sample[name] = self.raster.auxiliary[name][row, column]
        if self.include_pixel_index:
            sample["pixel_index"] = pixel
        return sample


def collate_observation_samples(
    samples: Sequence[dict[str, torch.Tensor]],
) -> ObservationBatch:
    if not samples:
        raise ValueError("Cannot collate an empty observation batch.")
    return default_collate(samples)


class ObservationBatchCollator:
    """Picklable default collator exposed as part of the training boundary."""

    def __call__(self, samples: Sequence[dict[str, torch.Tensor]]) -> ObservationBatch:
        return collate_observation_samples(samples)


class ObservationResponseCollator:
    """Nest colon-qualified response arrays at the instrument boundary."""

    _PREFIX = "instrument_response:"

    def __call__(self, samples: Sequence[dict[str, torch.Tensor]]) -> ObservationBatch:
        batch = collate_observation_samples(samples)
        response = {
            name.removeprefix(self._PREFIX): batch.pop(name)
            for name in tuple(batch)
            if name.startswith(self._PREFIX)
        }
        if response:
            batch["instrument_response"] = response
        return batch


__all__ = [
    "ObservationBatchCollator",
    "ObservationPixelDataset",
    "ObservationResponseCollator",
    "collate_observation_samples",
]
