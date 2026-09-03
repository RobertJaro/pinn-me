"""Shared Matplotlib styles and geometry for LTE diagnostic plots."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter


FIELD_STYLES = {
    "temperature": {
        "label": r"$T$ [K]",
        "cmap": "inferno",
        "signed": False,
        "log_norm": True,
    },
    "density": {
        "label": r"$\rho$ [kg m$^{-3}$]",
        "cmap": "cividis",
        "signed": False,
        "log_norm": True,
    },
    "pressure": {
        "label": r"$P_{gas}$ [Pa]",
        "cmap": "viridis",
        "signed": False,
        "log_norm": True,
    },
    "v_r": {
        "label": r"$v_r$ [km s$^{-1}$]",
        "cmap": "seismic",
        "signed": True,
    },
    "v_theta": {
        "label": r"$v_\theta$ [km s$^{-1}$]",
        "cmap": "seismic",
        "signed": True,
    },
    "v_phi": {
        "label": r"$v_\phi$ [km s$^{-1}$]",
        "cmap": "seismic",
        "signed": True,
    },
    "b_r": {"label": r"$B_r$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_theta": {"label": r"$B_\theta$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_phi": {"label": r"$B_\phi$ [G]", "cmap": "RdBu_r", "signed": True},
    "microturbulence": {
        "label": r"$\xi$ [km s$^{-1}$]",
        "cmap": "magma",
        "signed": False,
    },
}


class DiagnosticPlotter:
    """Common normalization and layout helpers for diagnostic plot families."""

    @staticmethod
    def _limits(values: np.ndarray, *, signed: bool) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        low, high = np.percentile(finite, (2.0, 98.0))
        if signed:
            extent = max(abs(float(low)), abs(float(high)), 1.0e-12)
            return -extent, extent
        if not high > low:
            padding = max(abs(float(low)) * 0.01, 1.0e-12)
            return float(low - padding), float(high + padding)
        return float(low), float(high)

    @classmethod
    def _field_norm(cls, values: np.ndarray, style: Mapping) -> Normalize:
        """Return a shared linear or logarithmic normalization for a field."""

        if not style.get("log_norm", False):
            vmin, vmax = cls._limits(values, signed=style["signed"])
            return Normalize(vmin=vmin, vmax=vmax)

        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size == 0:
            return LogNorm(vmin=1.0, vmax=10.0)
        vmin, vmax = np.percentile(positive, (2.0, 98.0))
        if not vmax > vmin:
            vmin = float(vmin) / 1.01
            vmax = float(vmax) * 1.01
        return LogNorm(
            vmin=max(float(vmin), np.finfo(float).tiny),
            vmax=float(vmax),
        )

    @staticmethod
    def _add_shared_colorbar(
        figure: Figure,
        mappable,
        axes,
        label: str,
    ):
        """Add one horizontal colorbar shared by a panel column."""

        colorbar = figure.colorbar(
            mappable,
            ax=np.asarray(axes).ravel().tolist(),
            location="top",
            orientation="horizontal",
            shrink=0.86,
            pad=0.02,
        )
        colorbar.set_label(label)
        colorbar.ax.set_title("")
        colorbar.locator = MaxNLocator(nbins=4, min_n_ticks=2)
        formatter = ScalarFormatter(useOffset=True, useMathText=True)
        formatter.set_powerlimits((-3, 4))
        colorbar.formatter = formatter
        colorbar.update_ticks()
        colorbar.minorticks_off()
        return colorbar

    @staticmethod
    def _curvilinear_cell_corners(centers: np.ndarray) -> np.ndarray:
        """Construct finite cell corners from a two-dimensional centre grid."""

        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2 or min(centers.shape) < 2:
            raise ValueError(
                "A meridional section needs at least two angular and radial samples."
            )
        if not np.isfinite(centers).all():
            # Invalid atmosphere pixels remain masked in the plotted values.
            # Interpolate only their plotting coordinates along the slit so
            # Matplotlib receives a finite curvilinear mesh without inventing
            # any displayed physical values.
            row = np.arange(centers.shape[0])
            centers = centers.copy()
            for depth_index in range(centers.shape[1]):
                valid = np.isfinite(centers[:, depth_index])
                if valid.sum() < 2:
                    raise ValueError(
                        "A meridional section needs at least two valid angular samples at "
                        "every depth level."
                    )
                centers[:, depth_index] = np.interp(
                    row, row[valid], centers[valid, depth_index]
                )
        padded = np.empty(
            (centers.shape[0] + 2, centers.shape[1] + 2), dtype=np.float32
        )
        padded[1:-1, 1:-1] = centers
        padded[0, 1:-1] = 2.0 * centers[0] - centers[1]
        padded[-1, 1:-1] = 2.0 * centers[-1] - centers[-2]
        padded[:, 0] = 2.0 * padded[:, 1] - padded[:, 2]
        padded[:, -1] = 2.0 * padded[:, -2] - padded[:, -3]
        return 0.25 * (
            padded[:-1, :-1] + padded[1:, :-1] + padded[:-1, 1:] + padded[1:, 1:]
        )


__all__ = ["DiagnosticPlotter", "FIELD_STYLES"]
