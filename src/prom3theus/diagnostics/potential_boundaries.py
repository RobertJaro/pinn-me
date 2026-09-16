"""Plot cached potential targets on their existing surfaces; no field queries."""

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

SURFACES = (
    ("photosphere", "Bottom (photosphere)"),
    ("top", "Top"),
    ("west", "West side"),
    ("east", "East side"),
    ("north", "North side"),
    ("south", "South side"),
)
COMPONENTS = (r"$B_r$ (outward)", r"$B_\theta$ (southward)", r"$B_\phi$ (westward)")


def potential_boundary_figure(term):
    """First time anchor, three component rows, independent scale per face.

    Each column shares a symmetric Gauss range across its three components.
    Read only the small cached surface arrays, never the potential operator.
    """
    if term.last_update < 0:
        raise ValueError("Potential reference has not been created")
    fields = term.targets[0].detach().cpu().numpy()
    positions = term.positions.detach().cpu().numpy()
    radial = positions / np.linalg.norm(positions, axis=-1, keepdims=True)
    longitude = np.arctan2(radial[:, 1], radial[:, 0])
    phi = np.stack(
        (-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)), -1
    )
    theta = np.cross(phi, radial)
    spherical = np.stack(
        [np.sum(fields * basis, axis=-1) for basis in (radial, theta, phi)], -1
    )
    figure = Figure(figsize=(25, 10), layout="constrained")
    axes = figure.subplots(3, len(SURFACES), squeeze=False)
    for column, (name, title) in enumerate(SURFACES):
        axes[0, column].set_title(title)
        if name not in term.surface_ranges:
            for ax in axes[:, column]:
                ax.text(
                    0.5,
                    0.5,
                    "Not configured",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            continue
        begin, end = term.surface_ranges[name]
        shape = term.surface_shapes[name]
        values = spherical[begin:end].reshape(*shape, 3)
        limit = max(float(np.abs(values).max()), 1e-6)
        norm = Normalize(-limit, limit)
        domain = term.surface_domains[
            "photosphere" if name == "photosphere" else "outer"
        ]
        lon_bounds = np.rad2deg(
            np.asarray(domain.longitude_offset_bounds_rad) + domain.longitude_center_rad
        )
        if name in ("photosphere", "top"):
            x = np.linspace(*lon_bounds, shape[0] + 1)
            y = np.rad2deg(
                np.arcsin(
                    np.linspace(*np.sin(domain.latitude_bounds_rad), shape[1] + 1)
                )
            )
            xlabel, ylabel = "Carrington longitude [deg]", "Latitude [deg]"
        else:
            x = (
                np.rad2deg(
                    np.arcsin(
                        np.linspace(*np.sin(domain.latitude_bounds_rad), shape[1] + 1)
                    )
                )
                if name in ("east", "west")
                else np.linspace(*lon_bounds, shape[1] + 1)
            )
            y = np.linspace(0, domain.height_bounds_Mm[1], shape[0] + 1)
            xlabel = (
                "Latitude [deg]"
                if name in ("east", "west")
                else "Carrington longitude [deg]"
            )
            ylabel = "Height [Mm]"
        for row, component in enumerate(COMPONENTS):
            ax = axes[row, column]
            data = (
                values[..., row].T
                if name in ("photosphere", "top")
                else values[..., row]
            )
            artist = ax.pcolormesh(
                x, y, data, cmap="RdBu_r", norm=norm, shading="flat", rasterized=True
            )
            ax.set_ylabel(f"{component}\n{ylabel}" if column == 0 else ylabel)
            if row == 2:
                ax.set_xlabel(xlabel)
        figure.colorbar(
            artist,
            ax=list(axes[:, column]),
            label="B [G]",
            orientation="horizontal",
            shrink=0.9,
            pad=0.06,
        )
    figure.suptitle(
        f"Potential boundary targets | t={float(term.times[0]):.6g} h | source update step {term.last_update}\n"
        "Independent range per boundary, shared across its three components"
    )
    return figure
