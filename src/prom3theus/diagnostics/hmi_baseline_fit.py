"""Review an actual local HMI inversion's fixed Stokes comparison sample.

Only saved observations and predictions are rendered. Pixel indices determine
the spatial support; missing samples remain blank. Tuning indices are used
because the comparison archive does not contain physical filter wavelengths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sample_grid(values, pixel_index):
    """Place shuffled scalar samples on their actual detector row/column grid."""
    pixels = np.asarray(pixel_index)
    values = np.asarray(values)
    if pixels.shape != (len(values), 2) or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError("pixel_index must contain integer [sample, row/column] pairs")
    if len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError("Repeated pixels cannot define a unique comparison map")
    rows, columns = np.unique(pixels[:, 0]), np.unique(pixels[:, 1])
    grid = np.full((len(rows), len(columns)), np.nan)
    grid[np.searchsorted(rows, pixels[:, 0]), np.searchsorted(columns, pixels[:, 1])] = values
    return grid, rows, columns


def _cell_edges(centres):
    centres = np.asarray(centres, dtype=np.float64)
    if len(centres) == 1:
        return np.array([centres[0] - 0.5, centres[0] + 0.5])
    halfway = (centres[:-1] + centres[1:]) / 2
    return np.concatenate(([centres[0] - (centres[1] - centres[0]) / 2], halfway,
                           [centres[-1] + (centres[-1] - centres[-2]) / 2]))


def _load_comparison(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in
                ("initial", "prediction", "target", "pixel_index")}
    shape = data["target"].shape
    if len(shape) != 3 or shape[1] != 4 or min(shape) < 1:
        raise ValueError("Stokes arrays must have nonempty [sample,4,tuning] shape")
    for name in ("initial", "prediction", "target"):
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError("Comparison arrays must have matching shapes and finite values")
    _sample_grid(data["target"][:, 0, 0], data["pixel_index"])
    return data


def _chosen_tunings(target):
    # Line-core I and the strongest observed polarization tuning, fixed across
    # every pixel in a given map. Selecting each pixel's own extremum would
    # mix wavelengths and could hide sign or profile-shape errors.
    return [int(np.argmin(np.mean(target[:, 0], axis=0))),
            *[int(np.argmax(np.sqrt(np.mean(target[:, index]**2, axis=0))))
              for index in (1, 2, 3)]]


def _comparison_context(report):
    if not report:
        return ""
    pieces = []
    step = report.get("global_step", report.get("step"))
    if step is not None:
        pieces.append(f"step {step}")
    if "fit" in report:
        passed = all(report[key]["passed"] for key in ("fit", "representative_fit") if key in report)
        pieces.append("fit checks passed" if passed else "fit checks failed")
    return "" if not pieces else " | " + " | ".join(pieces)


def stokes_fit_maps(data, report=None):
    """Compare observed/fitted maps and all-filter scatter in four Stokes rows."""
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    target, prediction, initial = (data[name] for name in ("target", "prediction", "initial"))
    pixels = data["pixel_index"]
    tunings = _chosen_tunings(target)
    figure = Figure(figsize=(16, 12), layout="constrained")
    axes = figure.subplots(4, 4)
    for component, label in enumerate("IQUV"):
        factor = 1 if component == 0 else 100
        units = "atlas continuum" if component == 0 else "% of atlas continuum"
        channel = tunings[component]
        values = [target[:, component, channel] * factor,
                  prediction[:, component, channel] * factor,
                  (prediction[:, component, channel] - target[:, component, channel]) * factor]
        if component == 0:
            lower, upper = min(np.min(values[0]), np.min(values[1])), max(np.max(values[0]), np.max(values[1]))
            if upper == lower:
                lower, upper = lower - 1e-6, upper + 1e-6
            shared = Normalize(lower, upper)
            cmap = "gray"
        else:
            limit = max(float(np.max(np.abs(values[:2]))), 1e-6)
            shared = Normalize(-limit, limit)
            cmap = "RdBu_r"
        residual_limit = max(float(np.max(np.abs(values[2]))), 1e-6)
        for column in range(3):
            grid, rows, columns = _sample_grid(values[column], pixels)
            artist = axes[component, column].pcolormesh(
                _cell_edges(columns), _cell_edges(rows), grid, shading="flat",
                cmap=cmap if column < 2 else "RdBu_r",
                norm=shared if column < 2 else Normalize(-residual_limit, residual_limit),
                rasterized=True)
            axes[component, column].set_aspect("equal")
            if component == 0:
                axes[component, column].set_title(("Observed", "Fitted", "Fitted − observed")[column])
            if column == 0:
                axes[component, column].set_ylabel(f"{label}, tuning {channel}\nHMI crop row")
            else:
                axes[component, column].tick_params(labelleft=False)
            if component == 3:
                axes[component, column].set_xlabel("HMI crop column")
            if column == 1:
                figure.colorbar(artist, ax=list(axes[component, :2]), label=units, shrink=0.82, pad=0.01)
            if column == 2:
                figure.colorbar(artist, ax=axes[component, column], label=units, shrink=0.82, pad=0.01)
        observed_all, fitted_all = target[:, component].ravel() * factor, prediction[:, component].ravel() * factor
        scatter = axes[component, 3]
        scatter.scatter(observed_all, fitted_all, s=4, alpha=0.22, color="tab:blue", edgecolors="none", rasterized=True)
        bounds = (min(observed_all.min(), fitted_all.min()), max(observed_all.max(), fitted_all.max()))
        if bounds[0] == bounds[1]:
            bounds = (bounds[0] - 1e-6, bounds[1] + 1e-6)
        scatter.plot(bounds, bounds, color="0.3", lw=0.9, ls="--")
        before = np.sqrt(np.mean((initial[:, component] - target[:, component])**2)) * factor
        after = np.sqrt(np.mean((prediction[:, component] - target[:, component])**2)) * factor
        scatter.set(xlabel=f"Observed {label} [{units}]", ylabel=f"Fitted {label}",
                    title=f"All tunings: RMS {before:.3g} → {after:.3g}")
        scatter.grid(alpha=0.15)
    figure.suptitle(f"HMI LTE inversion: observed and fitted Stokes{_comparison_context(report)}\n"
                   f"{len(target)} fixed diagnostic pixels; sample can overlap training; map tuning selected from observations",
                   fontsize=14)
    return figure


def stokes_fit_profiles(data, report=None):
    """Show opposite observed V signs and weak polarization without fitted selection."""
    from matplotlib.figure import Figure

    target, prediction = (data[name] for name in ("target", "prediction"))
    v_tuning = _chosen_tunings(target)[3]
    strength = np.sqrt(np.mean(target[:, 1:]**2, axis=(1, 2)))
    candidates = [int(np.argmax(target[:, 3, v_tuning])), int(np.argmin(target[:, 3, v_tuning])),
                  int(np.argmin(strength))]
    titles = [f"Largest V at tuning {v_tuning}", f"Smallest V at tuning {v_tuning}", "Weakest observed polarization"]
    figure = Figure(figsize=(13, 10), layout="constrained")
    axes = figure.subplots(4, 3, sharex=True)
    tuning = np.arange(target.shape[-1])
    for column, (sample, title) in enumerate(zip(candidates, titles)):
        row, pixel_column = data["pixel_index"][sample]
        axes[0, column].set_title(f"{title}\nPixel row {row}, column {pixel_column}")
        for component, label in enumerate("IQUV"):
            factor = 1 if component == 0 else 100
            ax = axes[component, column]
            ax.plot(tuning, prediction[sample, component] * factor, "-", color="tab:orange", lw=1.5, label="Fitted")
            ax.plot(tuning, target[sample, component] * factor, "o", color="black", ms=4, label="Observed")
            ax.grid(alpha=0.15)
            if component != 0:
                ax.axhline(0, color="0.7", lw=0.7)
            if column == 0:
                units = "atlas continuum" if component == 0 else "% of atlas continuum"
                ax.set_ylabel(f"{label} [{units}]")
            if component == 3:
                ax.set_xlabel("HMI filter tuning index")
                ax.set_xticks(tuning)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"HMI Stokes profiles selected by observed polarization{_comparison_context(report)}\n"
                   "Dots are the calibrated observations; lines connect the six filter measurements/predictions", fontsize=14)
    return figure


def export_hmi_baseline_fit(comparison_path, report_path=None, output_directory=None):
    """Render two reviewable PNGs from a saved, actual Stokes comparison archive."""
    comparison = Path(comparison_path)
    output = comparison.parent if output_directory is None else Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    data = _load_comparison(comparison)
    report = None if report_path is None else json.loads(Path(report_path).read_text())
    paths = {}
    for name, make_figure in (("maps", stokes_fit_maps), ("profiles", stokes_fit_profiles)):
        path = output / f"stokes-fit-{name}.png"
        figure = make_figure(data, report)
        figure.savefig(path, dpi=160)
        paths[name] = str(path.resolve())
    return paths
