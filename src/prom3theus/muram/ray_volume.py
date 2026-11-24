"""Eager Cartesian MURaM fields and bounded trilinear ray interpolation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from numbers import Integral

import numpy as np

from .atmosphere_builder import NATIVE_ATMOSPHERE_FIELDS
from .columns import NativeColumns, snapshot_columns
from .reporting import progress_bar, timed_stage


logger = logging.getLogger(__name__)
DEFAULT_VOLUME_HEIGHT_MEGAMETERS = 3.0


def volume_depth_count(nz, dz_m, height_megameters):
    """Select cell centers from the native bottom up to the requested height."""
    if height_megameters is None:
        return nz
    if isinstance(height_megameters, bool) or not np.isfinite(height_megameters) or height_megameters <= 0:
        raise ValueError("volume_height_megameters must be finite and positive, or None for full height")
    count = min(nz, int(np.floor(height_megameters * 1e6 / dz_m + 1e-9)) + 1)
    if count < 2:
        raise ValueError("volume_height_megameters must include at least two native cell centers")
    return count


def _retained_indices(size, stride):
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        raise ValueError("Volume strides must be positive integers.")
    if size < 2:
        raise ValueError("Ray interpolation requires at least two cells on every axis.")
    indices = np.arange(0, size, stride, dtype=np.int64)
    if indices[-1] != size - 1:
        indices = np.append(indices, size - 1)
    return indices


@dataclass(frozen=True)
class EagerVolume:
    """Read-only float32 fields on their original, possibly nonuniform grid.

    Support extends between native cell centers. No half-cell extrapolation is
    introduced. Striding keeps both ends of every axis, so the last interval may
    be shorter than other intervals. Only the owning synthesis process needs this
    object; the ray DataLoader should contain geometry rather than these arrays.
    """

    fields: dict[str, np.ndarray]
    axes_m: tuple[np.ndarray, np.ndarray, np.ndarray]
    native_reference: dict
    metadata: dict

    @classmethod
    def from_snapshot(cls, snapshot, *, stride=1, vertical_stride=1, progress=True,
                      volume_height_megameters=DEFAULT_VOLUME_HEIGHT_MEGAMETERS):
        """Crop before loading; convert only retained samples into owned arrays."""
        columns = snapshot_columns(snapshot)
        shape = columns.shape
        spacing = tuple(float(value) for value in columns.ds_m)
        if not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError("Native volume spacing must be finite and positive.")
        depth = volume_depth_count(shape[2], spacing[2], volume_height_megameters)
        cropped_shape = (*shape[:2], depth)
        indices = tuple(_retained_indices(size, step) for size, step in
                        zip(cropped_shape, (stride, stride, vertical_stride), strict=True))
        axes = tuple((index - ((size - 1) / 2 if axis < 2 else 0)) * delta
                     for axis, (index, size, delta) in enumerate(zip(indices, shape, spacing, strict=True)))
        fields = {}
        selected_shape = tuple(len(index) for index in indices)
        logger.info("Loading ray volume %s -> %s, %.2f GiB of float32 fields",
                    shape, selected_shape, np.prod(selected_shape) * len(NATIVE_ATMOSPHERE_FIELDS) * 4 / 2**30)
        logger.info("Ray volume: %d/%d native depths, 0–%.3f megameters above bottom cell center",
                    depth, shape[2], (depth - 1) * spacing[2] / 1e6)
        with timed_stage(logger, "Loading complete ray volume"), progress_bar(
            NATIVE_ATMOSPHERE_FIELDS, description="Volume fields", unit="field", enabled=progress,
        ) as names:
            for name in names:
                if isinstance(columns, NativeColumns):
                    dataset = columns.dataset
                    source = dataset._mapped(name)
                    factor = np.float32(dataset.scales[name] * dataset.signs[name])
                else:
                    source = columns.quantities[name]
                    factor = np.float32(1)
                source = source[..., :depth]
                values = np.empty(selected_shape, dtype=np.float32)
                if stride == vertical_stride == 1:
                    np.copyto(values, source, casting="unsafe")
                else:
                    # Limit the gather temporary to a few selected x planes.
                    for start in range(0, selected_shape[0], 8):
                        selection = np.ix_(indices[0][start:start + 8], indices[1], indices[2])
                        values[start:start + 8] = source[selection]
                np.multiply(values, factor, out=values)
                values.flags.writeable = False
                fields[name] = values
        for axis in axes:
            axis.flags.writeable = False
        descriptor = getattr(snapshot, "native_reference", None)
        reference = descriptor(stride=1, binning=1) if descriptor is not None else {}
        metadata = {
            "native_shape": list(shape), "native_spacing_m": list(spacing),
            "selected_shape": list(selected_shape),
            "volume_height_megameters": volume_height_megameters,
            "loaded_height_range_m": [0.0, float(axes[2][-1])],
            "horizontal_stride": int(stride), "vertical_stride": int(vertical_stride),
            "retained_native_indices": [index.tolist() for index in indices],
            "axes_m": [axis.tolist() for axis in axes],
            "support": "closed box between outermost retained native cell centers",
            "height_origin": "bottom native cell center; not the photosphere",
            "vector_frame": "native Cartesian west/north/up; co-rotating velocity",
            "units": {"temperature": "K", "pressure": "Pa", "bx": "G", "by": "G", "bz": "G",
                      "vx": "m/s", "vy": "m/s", "vz": "m/s"},
            "interpolation": "trilinear on retained physical coordinates; no extrapolation",
        }
        return cls(fields, axes, reference, metadata)

    @property
    def shape(self):
        return tuple(len(axis) for axis in self.axes_m)

    @property
    def bounds_m(self):
        return np.array([[axis[0] for axis in self.axes_m], [axis[-1] for axis in self.axes_m]])

    @property
    def nbytes(self):
        return sum(value.nbytes for value in self.fields.values()) + sum(axis.nbytes for axis in self.axes_m)

    def sample(self, points_m):
        """Return ``(fields, valid)`` for CPU points shaped ``[..., 3]``.

        Invalid queries are NaN, never a fabricated vacuum atmosphere. Callers
        must evaluate local physics only for valid points and explicitly assign
        zero opacity/emissivity to vacuum intervals.
        """
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("Interpolation points must have shape [..., 3].")
        output_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        bounds = self.bounds_m
        valid = np.isfinite(flat).all(axis=-1) & (flat >= bounds[0]).all(axis=-1) & (flat <= bounds[1]).all(axis=-1)
        good = flat[valid]
        low, fraction = [], []
        for dimension, axis in enumerate(self.axes_m):
            cell = np.clip(np.searchsorted(axis, good[:, dimension], side="right") - 1, 0, len(axis) - 2)
            low.append(cell)
            fraction.append((good[:, dimension] - axis[cell]) / (axis[cell + 1] - axis[cell]))
        output = {}
        for name, cube in self.fields.items():
            interpolated = np.zeros(len(good), dtype=np.float64)
            for x in (0, 1):
                for y in (0, 1):
                    for z in (0, 1):
                        weight = ((fraction[0] if x else 1 - fraction[0])
                                  * (fraction[1] if y else 1 - fraction[1])
                                  * (fraction[2] if z else 1 - fraction[2]))
                        interpolated += weight * cube[low[0] + x, low[1] + y, low[2] + z]
            result = np.full(len(flat), np.nan, dtype=np.float32)
            result[valid] = interpolated
            output[name] = result.reshape(output_shape)
        return output, valid.reshape(output_shape)


__all__ = ["EagerVolume"]
