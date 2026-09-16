"""Dataset primitives for native-grid scalar image observations."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from .bulk import PixelCatalog
from torch.utils.data import Dataset, default_collate

from .image_contracts import (
    ABSOLUTE_TAI_SECONDS,
    CHANNEL_ANGSTROM,
    CHANNEL_INDEX,
    IMAGE_INDEX,
    INTENSITY,
    PIXEL_INDEX,
    RAY_DIRECTION,
    SURFACE_POSITION_M,
    UNCERTAINTY,
    ImageObservationBatch,
    ImageObservationRaster,
)


class ImagePixelDataset(Dataset):
    """Flatten valid pixels, applying one fixed intensity scale at batch access.

    The raster stays in its physical units; intensity and optional uncertainty
    samples are divided by the same scale without modifying its backing arrays.
    """

    def __init__(
        self,
        raster: ImageObservationRaster,
        *,
        image_index: int,
        channel_index: int,
        intensity_scale: float = 1.0,
        include_pixel_index: bool = True,
        pixel_indices: torch.Tensor | None = None,
    ) -> None:
        if type(image_index) is not int or image_index < 0:
            raise ValueError("image_index must be a non-negative integer.")
        if type(channel_index) is not int or channel_index < 0:
            raise ValueError("channel_index must be a non-negative integer.")
        if not math.isfinite(intensity_scale) or intensity_scale <= 0:
            raise ValueError("intensity_scale must be finite and strictly positive.")
        self.raster = raster
        self.image_index = image_index
        self.channel_index = channel_index
        self.intensity_scale = float(intensity_scale)
        self.include_pixel_index = bool(include_pixel_index)
        self._pixel_indices = None
        self._pixel_catalog = getattr(raster, "_bulk_catalog", None) or PixelCatalog(
            raster.valid_mask
        )
        if not len(self._pixel_catalog):
            raise ValueError("A dataset requires at least one valid pixel")
        if pixel_indices is not None:
            raw = torch.as_tensor(pixel_indices)
            if raw.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError("pixel_indices must use an integer dtype.")
            indices = raw.to(dtype=torch.long)
            if indices.ndim != 2 or indices.shape[-1] != 2:
                raise ValueError("pixel_indices must have shape [sample, 2].")
            if not indices.numel():
                raise ValueError("An image dataset requires at least one valid pixel.")
            if (
                torch.any(indices < 0)
                or torch.any(indices[:, 0] >= raster.spatial_shape[0])
                or torch.any(indices[:, 1] >= raster.spatial_shape[1])
            ):
                raise IndexError("pixel_indices lie outside the image raster.")
            if not torch.all(raster.valid_mask[indices[:, 0], indices[:, 1]]):
                raise ValueError("pixel_indices may select only valid image pixels.")
            self.pixel_indices = indices
        self._absolute_time = float(raster.absolute_tai_seconds)
        self._channel = int(raster.channel_angstrom)
        self._channel_index = channel_index
        self._image_index = image_index

    @property
    def pixel_indices(self):
        # Compatibility for explicit diagnostics; training uses compact catalogs.
        if self._pixel_indices is not None:
            from .arrays import materialize_array
            return materialize_array(self._pixel_indices)
        return self._pixel_catalog.pixels(0, len(self._pixel_catalog))

    @pixel_indices.setter
    def pixel_indices(self, value):
        self._pixel_indices = value

    def pixels_at(self, indices):
        if self._pixel_indices is not None:
            return self._pixel_indices[indices]
        if isinstance(indices, slice):
            return self._pixel_catalog.pixels(indices.start, indices.stop)
        # Sparse diagnostics load each containing mask block only once.
        parts = []
        for block in torch.unique(
            torch.searchsorted(
                torch.tensor(self._pixel_catalog.prefix[1:]), indices, right=True
            )
        ):
            first = self._pixel_catalog.prefix[int(block)]
            last = self._pixel_catalog.prefix[int(block) + 1]
            selected = indices[(indices >= first) & (indices < last)]
            parts.append(self._pixel_catalog.pixels(first, last)[selected - first])
        return torch.cat(parts)

    def __len__(self) -> int:
        return (
            len(self._pixel_catalog)
            if self._pixel_indices is None
            else int(self._pixel_indices.shape[0])
        )

    def diagnostic_dataset_indices(self, maximum_count: int) -> torch.Tensor:
        """Return deterministic, evenly distributed indices for image rendering.

        The dataset's valid pixels are stored in native row-major order.  Taking
        evenly spaced ranks across that vector avoids the spatial bias of simply
        taking the first ``maximum_count`` pixels, while returning every pixel
        exactly once when the configured bound permits a complete render.
        """

        if type(maximum_count) is not int or maximum_count < 1:
            raise ValueError("maximum_count must be a positive integer.")
        count = min(maximum_count, len(self))
        if count == len(self):
            return torch.arange(count, dtype=torch.long)
        if count == 1:
            return torch.tensor((len(self) // 2,), dtype=torch.long)
        return (torch.arange(count, dtype=torch.long) * (len(self) - 1)) // (count - 1)

    def bulk_fields(self):
        fields = {
            name: getattr(self.raster, name)
            for name in (INTENSITY, RAY_DIRECTION, SURFACE_POSITION_M)
        }
        if self.raster.uncertainty is not None:
            fields[UNCERTAINTY] = self.raster.uncertainty
        return fields

    def finish_bulk(self, batch, pixels):
        batch[INTENSITY] = batch[INTENSITY] / self.intensity_scale
        if UNCERTAINTY in batch:
            batch[UNCERTAINTY] = batch[UNCERTAINTY] / self.intensity_scale
        for key, value in (
            (ABSOLUTE_TAI_SECONDS, self._absolute_time),
            (CHANNEL_ANGSTROM, self._channel),
            (CHANNEL_INDEX, self._channel_index),
            (IMAGE_INDEX, self._image_index),
        ):
            batch[key] = torch.full((len(pixels),), value, dtype=torch.float64 if key == ABSOLUTE_TAI_SECONDS else torch.long)
        if self.include_pixel_index:
            batch[PIXEL_INDEX] = pixels
        return batch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = index + len(self) if index < 0 else index
        if not 0 <= index < len(self):
            raise IndexError(index)
        pixel = self.pixels_at(slice(index, index + 1))[0]
        row, column = map(int, pixel)
        sample = {
            INTENSITY: self.raster.intensity[row, column] / self.intensity_scale,
            RAY_DIRECTION: self.raster.ray_direction[row, column],
            SURFACE_POSITION_M: self.raster.surface_position_m[row, column],
            ABSOLUTE_TAI_SECONDS: torch.tensor(self._absolute_time, dtype=torch.float64),
            CHANNEL_ANGSTROM: torch.tensor(self._channel, dtype=torch.long),
            CHANNEL_INDEX: torch.tensor(self._channel_index, dtype=torch.long),
            IMAGE_INDEX: torch.tensor(self._image_index, dtype=torch.long),
        }
        if self.include_pixel_index:
            sample[PIXEL_INDEX] = pixel
        if self.raster.uncertainty is not None:
            sample[UNCERTAINTY] = (
                self.raster.uncertainty[row, column] / self.intensity_scale
            )
        return sample


def collate_image_samples(
    samples: Sequence[dict[str, torch.Tensor]],
) -> ImageObservationBatch:
    if not samples:
        raise ValueError("Cannot collate an empty image-observation batch.")
    return default_collate(samples)


class ImageObservationBatchCollator:
    """Picklable image collator for multi-worker data loading."""

    def __call__(
        self, samples: Sequence[dict[str, torch.Tensor]]
    ) -> ImageObservationBatch:
        return collate_image_samples(samples)


def reconstruct_image(
    values: torch.Tensor,
    pixel_indices: torch.Tensor,
    spatial_shape: tuple[int, int],
    *,
    fill_value: float = math.nan,
) -> torch.Tensor:
    """Reconstruct a native-grid image from dataset-derived pixel indices."""

    values = torch.as_tensor(values)
    indices = torch.as_tensor(pixel_indices)
    if values.ndim < 1:
        raise ValueError("values must have a leading sample dimension.")
    if indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("pixel_indices must use an integer dtype.")
    indices = indices.to(device=values.device, dtype=torch.long)
    if indices.shape != (values.shape[0], 2):
        raise ValueError("pixel_indices must have shape [values.shape[0], 2].")
    if (
        not isinstance(spatial_shape, tuple)
        or len(spatial_shape) != 2
        or any(type(size) is not int or size <= 0 for size in spatial_shape)
    ):
        raise ValueError("spatial_shape must contain two positive integers.")
    if (
        torch.any(indices < 0)
        or torch.any(indices[:, 0] >= spatial_shape[0])
        or torch.any(indices[:, 1] >= spatial_shape[1])
    ):
        raise IndexError("pixel_indices lie outside spatial_shape.")
    linear = indices[:, 0] * spatial_shape[1] + indices[:, 1]
    if torch.unique(linear).numel() != linear.numel():
        raise ValueError("pixel_indices must be unique for image reconstruction.")
    if not values.is_floating_point() and math.isnan(float(fill_value)):
        raise ValueError("Integer values require a finite fill_value.")
    output = torch.full(
        (*spatial_shape, *values.shape[1:]),
        fill_value,
        dtype=values.dtype,
        device=values.device,
    )
    output[indices[:, 0], indices[:, 1]] = values
    return output


__all__ = [
    "ImageObservationBatchCollator",
    "ImagePixelDataset",
    "collate_image_samples",
    "reconstruct_image",
]
