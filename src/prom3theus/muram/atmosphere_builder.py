"""Resample native MURaM log-tau labels and their physical heights.

The labels select sampling locations, not the LTE backend's optical depth.
MURaM forward synthesis uses the resampled heights for geometric transfer.
"""

from __future__ import annotations

import numpy as np
import torch

from prom3theus.rt.atmosphere import StratifiedAtmosphere


def _monotonic_log_tau(
    tau_top_to_bottom: np.ndarray, *, floor: float = 1e-30
) -> np.ndarray:
    """Validate integrated optical depth; positive opacity cannot reverse tau."""
    tau = np.asarray(tau_top_to_bottom, dtype=np.float64)
    if not np.isfinite(tau).all() or np.any(tau < 0):
        raise ValueError("tau must be finite and non-negative.")
    tau = np.maximum(tau, floor)
    reversals = np.diff(tau, axis=-1) < -1e-6 * np.maximum(tau[..., 1:], 1e-30)
    if np.any(reversals):
        index = tuple(np.argwhere(reversals)[0])
        following = (*index[:-1], index[-1] + 1)
        raise ValueError(
            "Optical depth must increase toward the bottom of every column; "
            f"tau reverses from {tau[index]:.9g} to {tau[following]:.9g} "
            f"at bracket-relative top-to-bottom depth {index[-1]}."
        )
    log_tau = np.log10(np.maximum(tau, 1e-30))
    return np.maximum.accumulate(log_tau, axis=-1) + np.arange(tau.shape[-1]) * 1e-12


def _interpolation_plan(target_x: np.ndarray, source_x: np.ndarray):
    """Find one shared set of brackets/weights for all atmospheric fields."""
    xp = source_x.reshape(-1, source_x.shape[-1])
    lower = np.stack([np.searchsorted(row, target_x, side="right") - 1 for row in xp])
    lower = np.clip(lower, 0, xp.shape[-1] - 2)
    upper = lower + 1
    x_lo = np.take_along_axis(xp, lower, axis=-1)
    x_hi = np.take_along_axis(xp, upper, axis=-1)
    fraction = np.clip(
        (target_x[None, :] - x_lo) / np.maximum(x_hi - x_lo, 1e-30), 0, 1
    )
    return lower, upper, fraction


class ColumnSamplingError(ValueError):
    """A native column cannot supply a physically ordered sampling interval."""

    def __init__(self, column_index, message):
        self.column_index = column_index
        super().__init__(message)


def _native_tau_interpolation_plan(target, tau):
    """Use the required tau interval and its immediate native brackets only.

    The first crossing of the upper-atmosphere target and the last crossing
    of the lower-atmosphere target bound the interval. Using the last crossing
    deliberately keeps interior spikes/reversals visible rather than selecting
    an arbitrary monotonic branch. No atmospheric samples are sorted or moved.
    """
    flat = np.asarray(tau).reshape(-1, tau.shape[-1])
    lower = np.empty((len(flat), len(target)), dtype=np.intp)
    upper = np.empty_like(lower)
    fraction = np.empty(lower.shape, dtype=np.float64)
    for column, row in enumerate(flat):
        start, stop = 0, len(row) - 1
        try:
            if not np.isfinite(row).all() or np.any(row < 0):
                raise ValueError("tau must be finite and non-negative.")
            log_row = np.log10(np.maximum(row.astype(np.float64), 1e-30))
            above = np.flatnonzero(log_row >= target[0])
            below = np.flatnonzero(log_row <= target[-1])
            if not len(above) or not len(below):
                raise ValueError("Requested log_tau500_grid is not covered.")
            start = max(int(above[0]) - 1, 0)
            stop = min(int(below[-1]) + 1, len(row) - 1)
            if (
                start >= stop
                or log_row[start] > target[0]
                or log_row[stop] < target[-1]
            ):
                raise ValueError("Requested log_tau500_grid is not covered.")
            selected = _monotonic_log_tau(row[start : stop + 1])
            lo, hi, weight = _interpolation_plan(target, selected)
            lower[column], upper[column] = lo[0] + start, hi[0] + start
            fraction[column] = weight[0]
        except ValueError as error:
            raise ColumnSamplingError(
                column,
                f"{error} Column {column}, native top-to-bottom bracket "
                f"[{start}, {stop}], requested log10(tau) "
                f"[{target[0]:g}, {target[-1]:g}]. "
                "Check the tau output and vertical-axis orientation; "
                "reversals inside the sampling interval cannot be interpolated safely.",
            ) from error
    return lower, upper, fraction


def _apply_interpolation(source_y: np.ndarray, plan) -> np.ndarray:
    lower, upper, fraction = plan
    fp = source_y.reshape(-1, source_y.shape[-1])
    y_lo = np.take_along_axis(fp, lower, axis=-1)
    y_hi = np.take_along_axis(fp, upper, axis=-1)
    return (y_lo + fraction * (y_hi - y_lo)).reshape(
        *source_y.shape[:-1], lower.shape[-1]
    )


def _interp_along_last_axis(
    target_x: np.ndarray, source_x: np.ndarray, source_y: np.ndarray
) -> np.ndarray:
    """Batched linear interpolation with endpoint clamping."""
    return _apply_interpolation(source_y, _interpolation_plan(target_x, source_x))


def build_stratified_atmosphere(
    quantities: dict[str, np.ndarray],
    log_tau500_grid,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> StratifiedAtmosphere:
    """Build a batched :class:`StratifiedAtmosphere` from MURaM columns.

    Parameters
    ----------
    quantities:
        Output of :meth:`prom3theus.muram.reader.MURaMSnapshot.synthesis_ready_quantities`:
        arrays shaped ``[nx, ny, nz]`` with ``z`` increasing height, plus
        ``ds_m``. ``bx/by/bz`` are gauss; ``vx/vy/vz`` m/s (box axes are
        treated as the disk-center scene frame: ``x`` -> west, ``y`` ->
        north, ``z`` -> radial/line-of-sight in the local Stokes frame);
        ``pressure`` Pa; ``temperature`` K; ``tau`` is
        MURaM's native continuum optical-depth proxy.
    log_tau500_grid:
        Legacy name for the native log-tau sampling grid (top to bottom), e.g.
        ``(-5, -4, -3, -2, -1, 0, 1)``.

    Returns
    -------
    A :class:`StratifiedAtmosphere` with batch shape ``[nx * ny]`` (flattened
    pixel index) and depth shape matching ``log_tau500_grid``. Microturbulence
    is set to zero (MURaM resolves turbulent motions explicitly; the
    inversion's microturbulence term is not exercised by this validation).
    """

    target = np.asarray(log_tau500_grid, dtype=np.float64)
    if (
        target.ndim != 1
        or target.size < 2
        or not np.isfinite(target).all()
        or not np.all(np.diff(target) > 0)
    ):
        raise ValueError(
            "log_tau500_grid must be one-dimensional and strictly increasing."
        )

    # StratifiedAtmosphere orders depth top (low tau) to bottom (high tau);
    # MURaM's height-increasing z axis runs the opposite way (tau decreases
    # with height), so flip before building the per-column depth coordinate.
    tau = np.flip(quantities["tau"], axis=-1)
    nx, ny, nz = tau.shape
    n_pixels = nx * ny
    plan = _native_tau_interpolation_plan(target, tau)

    def resample(name: str) -> np.ndarray:
        field = (
            np.flip(quantities[name], axis=-1).reshape(n_pixels, nz).astype(np.float64)
        )
        return _apply_interpolation(field, plan)

    temperature = resample("temperature")
    gas_pressure = resample("pressure")
    vx = resample("vx")
    vy = resample("vy")
    vz = resample("vz")
    bx = resample("bx")
    by = resample("by")
    bz = resample("bz")

    depth_coordinate = torch.as_tensor(target, dtype=dtype, device=device)
    temperature_t = torch.as_tensor(temperature, dtype=dtype, device=device)
    gas_pressure_t = torch.as_tensor(gas_pressure, dtype=dtype, device=device)
    velocity_field = torch.as_tensor(
        np.stack([vx, vy, vz], axis=-1), dtype=dtype, device=device
    )
    magnetic_field = torch.as_tensor(
        np.stack([bx, by, bz], axis=-1), dtype=dtype, device=device
    )
    microturbulence = torch.zeros_like(temperature_t)
    height = None
    if "ds_m" in quantities:
        heights = np.broadcast_to(
            np.arange(nz, dtype=np.float64)[::-1] * quantities["ds_m"][2],
            (n_pixels, nz),
        )
        height = torch.as_tensor(
            _apply_interpolation(heights, plan),
            dtype=dtype,
            device=device,
        )

    return StratifiedAtmosphere(
        depth_coordinate=depth_coordinate,
        temperature=temperature_t,
        velocity_field=velocity_field,
        microturbulence=microturbulence,
        magnetic_field=magnetic_field,
        gas_pressure=gas_pressure_t,
        geometric_height_m=height,
    )


NATIVE_ATMOSPHERE_FIELDS = ("temperature", "pressure", "vx", "vy", "vz", "bx", "by", "bz")


def build_native_atmosphere(quantities, *, dtype=torch.float64, device=None):
    """Use every native cell, ordered top to bottom, without tau resampling."""
    nx, ny, nz = quantities["temperature"].shape
    dz = float(quantities["ds_m"][2])
    if nz < 2 or not np.isfinite(dz) or dz <= 0:
        raise ValueError("Native columns require at least two layers and positive finite dz.")

    def field(name):
        # The reader orients z upward. Transfer walks from the top cell to the
        # bottom cell; an explicit contiguous copy supports negative NumPy strides.
        values = np.ascontiguousarray(quantities[name][..., ::-1].reshape(nx * ny, nz))
        return torch.as_tensor(values, dtype=dtype, device=device)

    temperature = field("temperature")
    height = torch.arange(nz - 1, -1, -1, dtype=dtype, device=device) * dz
    return StratifiedAtmosphere(
        # Labels only: GeometricHeightPath supplies all physical path lengths.
        depth_coordinate=torch.arange(nz, dtype=dtype, device=device),
        temperature=temperature,
        gas_pressure=field("pressure"),
        velocity_field=torch.stack([field(name) for name in ("vx", "vy", "vz")], dim=-1),
        magnetic_field=torch.stack([field(name) for name in ("bx", "by", "bz")], dim=-1),
        microturbulence=torch.zeros_like(temperature),
        geometric_height_m=height.expand(nx * ny, -1),
    )
