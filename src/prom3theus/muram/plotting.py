"""Render overviews from saved 2D diagnostic maps, never from native cubes."""

import math

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .diagnostic_maps import atomic_path, load_diagnostic_maps


def write_parameter_overview(path, sidecar, *, max_display_pixels=768):
    if max_display_pixels < 1:
        raise ValueError("max_display_pixels must be positive.")
    result = load_diagnostic_maps(sidecar)
    sampling = result.metadata["sampling"]
    ny, nx = result.status.shape
    step = max(1, math.ceil(max(ny, nx) / max_display_pixels))
    maps = {name: values[::step, ::step] for name, values in result.maps.items()}
    stride = sampling["stride"]
    native_nx, native_ny, _ = sampling["native_shape_xyz"]
    dx, dy, _ = sampling["native_spacing_m"]
    extent = ((-(native_nx-1)/2-stride/2)*dx/1e6,
              ((nx-1)*stride-(native_nx-1)/2+stride/2)*dx/1e6,
              (-(native_ny-1)/2-stride/2)*dy/1e6,
              ((ny-1)*stride-(native_ny-1)/2+stride/2)*dy/1e6)
    panels = (("bx", "Bx — west", "G", 1.), ("by", "By — north", "G", 1.),
              ("bz", "Bz — upward", "G", 1.), ("vx", "vx — west", "km/s", .001),
              ("vy", "vy — north", "km/s", .001), ("vz", "vz — upward", "km/s", .001),
              ("temperature", "Temperature", "K", 1.), ("pressure", "Gas pressure", "kPa", .001),
              ("height_m", "Native tau=1 height", "megameters above bottom cell", 1e-6))

    def bounds(names):
        values = np.concatenate([maps[name].ravel() for name in names])
        finite = np.abs(values[np.isfinite(values)])
        return max(float(np.percentile(finite, 99.5)), 1e-6) if finite.size else 1.

    b_limit, v_limit = bounds(("bx", "by", "bz")), bounds(("vx", "vy", "vz")) * .001
    fig = Figure(figsize=(15, 13), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 3)
    try:
        for index, (ax, (name, title, unit, scale)) in enumerate(zip(axes.flat, panels)):
            values = maps[name] * scale
            if index < 6:
                high = b_limit if index < 3 else v_limit
                low, cmap = -high, "RdBu_r"
            else:
                finite = values[np.isfinite(values)]
                low, high = np.percentile(finite, [.5, 99.5]) if finite.size else (0., 1.)
                high, cmap = max(high, low + 1e-6), "magma"
            image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap, vmin=low, vmax=high)
            ax.set(title=title, xlabel="local x [megameters]", ylabel="local y [megameters]")
            fig.colorbar(image, ax=ax, label=unit, fraction=.045, extend="both")
        counts = {label: int(np.count_nonzero(result.status == int(code)))
                  for code, label in result.metadata["status_codes"].items()}
        fig.suptitle(
            f"MURaM {result.metadata['iteration']:06d} — t={result.metadata['time_s']:g} s — native tau=1\n"
            f"Local co-rotating frame; stride {stride}; display step {step}; unbinned numeric maps\n"
            + "; ".join(f"{label}: {count}" for label, count in counts.items()), fontsize=11,
        )
        with atomic_path(path) as temporary:
            fig.savefig(temporary, dpi=130, format="jpg")
    finally:
        fig.clear()
