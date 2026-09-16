"""Time-resolved spherical diagnostics for compact P3S save states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prom3theus.artifacts.loader import P3SLoader

from .plotting import DiagnosticPlotter


_PANELS = (
    ("b_r", r"$B_r$ [G]", "RdBu_r", True, False),
    ("b_theta", r"$B_\theta$ [G]", "RdBu_r", True, False),
    ("b_phi", r"$B_\phi$ [G]", "RdBu_r", True, False),
    ("v_r", r"$v_r$ [km s$^{-1}$]", "seismic", True, False),
    ("v_theta", r"$v_\theta$ [km s$^{-1}$]", "seismic", True, False),
    ("v_phi", r"$v_\phi$ [km s$^{-1}$]", "seismic", True, False),
    ("field_strength", r"$|\mathbf{B}|$ [G]", "viridis", False, True),
    (
        "integrated_current_density",
        r"$\int |\mathbf{J}|\,dh$ [A m$^{-1}$]",
        "magma",
        False,
        True,
    ),
    (
        "radial_poynting_flux",
        r"$S_r$ [W m$^{-2}$]",
        "PuOr_r",
        True,
        False,
    ),
)


def _norm(values: np.ndarray, *, signed: bool, logarithmic: bool):
    from matplotlib.colors import LogNorm, Normalize

    finite = values[np.isfinite(values)]
    if logarithmic:
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return LogNorm(vmin=1.0, vmax=10.0)
        low, high = np.percentile(positive, (1.0, 99.0))
        if not high > low:
            low, high = float(low) / 1.01, float(high) * 1.01
        return LogNorm(vmin=max(float(low), np.finfo(float).tiny), vmax=float(high))
    low, high = DiagnosticPlotter._limits(values, signed=signed)
    return Normalize(vmin=low, vmax=high)


def _save_frames(series: dict[str, Any], directory: Path, *, dpi: int) -> list[Path]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import ScalarFormatter

    directory.mkdir(parents=True, exist_ok=True)
    fields = series["map_fields"]
    longitude = series["longitude_deg"]
    latitude = series["latitude_deg"]
    extent = (
        float(np.nanmin(longitude)),
        float(np.nanmax(longitude)),
        float(np.nanmin(latitude)),
        float(np.nanmax(latitude)),
    )
    figure = Figure(figsize=(13.2, 10.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(3, 3, sharex=True, sharey=True))
    images = []
    for axis, (name, label, cmap, signed, logarithmic) in zip(
        axes.ravel(), _PANELS, strict=True
    ):
        values = fields[name]
        image = axis.imshow(
            values[0],
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            norm=_norm(values, signed=signed, logarithmic=logarithmic),
            aspect="equal",
        )
        colorbar = figure.colorbar(
            image, ax=axis, location="top", shrink=0.86, pad=0.02
        )
        colorbar.set_label(label)
        if not logarithmic:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 4))
            colorbar.formatter = formatter
            colorbar.update_ticks()
        images.append(image)
    for axis in axes[-1]:
        axis.set_xlabel("Carrington longitude [deg]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Carrington latitude [deg]")
    title = figure.suptitle("")

    paths = []
    for frame in range(len(series["times"])):
        for image, (name, *_rest) in zip(images, _PANELS, strict=True):
            image.set_data(fields[name][frame])
        title.set_text(
            "PROM3THEUS spherical time series at "
            rf"$h={series['height_m'] / 1.0e3:g}\,$km — "
            f"{series['times'][frame]} {series['time_scale'].upper()} "
            f"(t={series['elapsed_hours'][frame]:g} h)"
        )
        timestamp = (
            series["times"][frame]
            .replace("-", "")
            .replace(":", "")
            .replace("T", "_")
            .replace(".", "p")
        )
        path = directory / f"frame_{frame:03d}_{timestamp}_{series['time_scale']}.jpg"
        figure.savefig(
            path,
            dpi=dpi,
            facecolor="white",
            format="jpg",
            pil_kwargs={"quality": 95, "optimize": True},
        )
        paths.append(path)
    figure.clear()
    return paths


def _spatial_rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nanmean(np.asarray(values, dtype=np.float64) ** 2, axis=(1, 2)))


def _save_summary(series: dict[str, Any], path: Path, *, dpi: int) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fields = series["map_fields"]
    time = series["elapsed_hours"]
    figure = Figure(figsize=(12.0, 8.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(2, 2))
    for name, label in (
        ("b_r", r"$B_r$"),
        ("b_theta", r"$B_\theta$"),
        ("b_phi", r"$B_\phi$"),
    ):
        axes[0, 0].plot(time, _spatial_rms(fields[name]), marker="o", label=label)
    axes[0, 0].set_ylabel("spatial RMS [G]")
    axes[0, 0].set_title(r"surface $\mathbf{B}$")
    axes[0, 0].legend()

    for name, label in (
        ("v_r", r"$v_r$"),
        ("v_theta", r"$v_\theta$"),
        ("v_phi", r"$v_\phi$"),
    ):
        axes[0, 1].plot(time, _spatial_rms(fields[name]), marker="o", label=label)
    axes[0, 1].set_ylabel(r"spatial RMS [km s$^{-1}$]")
    axes[0, 1].set_title(r"surface $\mathbf{v}$")
    axes[0, 1].legend()

    current = fields["integrated_current_density"]
    axes[1, 0].plot(time, np.nanmedian(current, axis=(1, 2)), marker="o")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel(r"spatial median [A m$^{-1}$]")
    axes[1, 0].set_title(r"radially integrated $|\mathbf{J}|$")

    poynting = fields["radial_poynting_flux"]
    low, median, high = np.nanpercentile(poynting, (16.0, 50.0, 84.0), axis=(1, 2))
    axes[1, 1].fill_between(time, low, high, alpha=0.25, label="16–84%")
    axes[1, 1].plot(time, median, marker="o", label="median")
    axes[1, 1].axhline(0.0, color="0.4", linewidth=0.8)
    axes[1, 1].set_ylabel(r"$S_r$ [W m$^{-2}$]")
    axes[1, 1].set_title("radial Poynting flux")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.set_xlabel("hours since first acquisition")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "PROM3THEUS temporal evolution — " rf"$h={series['height_m'] / 1.0e3:g}\,$km"
    )
    figure.savefig(path, dpi=dpi, facecolor="white")
    figure.clear()


def evaluate_p3s_time_series(
    save_state: str | Path,
    output_directory: str | Path,
    *,
    height_km: float = 0.0,
    longitude_points: int = 96,
    latitude_points: int = 96,
    current_height_samples: int = 48,
    batch_size: int = 4096,
    device: str = "auto",
    dpi: int = 120,
) -> dict[str, Any]:
    """Evaluate and render one complete P3S model time sequence."""

    if dpi < 50:
        raise ValueError("dpi must be at least 50.")
    loader = P3SLoader(save_state, device=device)
    series = loader.spherical_time_series(
        height_m=float(height_km) * 1.0e3,
        longitude_points=longitude_points,
        latitude_points=latitude_points,
        current_height_samples=current_height_samples,
        batch_size=batch_size,
    )
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame_directory = output / "spherical_time_series_jpg"
    summary_path = output / "temporal_summary.png"
    arrays_path = output / "spherical_time_series.npz"
    metadata_path = output / "metadata.json"
    frame_paths = _save_frames(series, frame_directory, dpi=dpi)
    _save_summary(series, summary_path, dpi=dpi)
    np.savez_compressed(
        arrays_path,
        times=np.asarray(series["times"]),
        time_scale=np.asarray(series["time_scale"]),
        elapsed_hours=series["elapsed_hours"],
        height_m=np.asarray(series["height_m"]),
        integration_height_m=series["integration_height_m"],
        longitude_deg=series["longitude_deg"],
        latitude_deg=series["latitude_deg"],
        **series["map_fields"],
    )
    result = {
        "save_state": str(loader.path),
        "global_step": loader.global_step,
        "frame_count": len(series["times"]),
        "start_time": series["times"][0],
        "end_time": series["times"][-1],
        "time_scale": series["time_scale"],
        "height_km": float(height_km),
        "current_integration_height_km": [
            float(series["integration_height_m"][0] / 1.0e3),
            float(series["integration_height_m"][-1] / 1.0e3),
        ],
        "component_convention": series["component_convention"],
        "outputs": {
            "jpg_series": str(frame_directory),
            "first_frame": str(frame_paths[0]),
            "last_frame": str(frame_paths[-1]),
            "summary": str(summary_path),
            "arrays": str(arrays_path),
            "metadata": str(metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


__all__ = ["evaluate_p3s_time_series"]
