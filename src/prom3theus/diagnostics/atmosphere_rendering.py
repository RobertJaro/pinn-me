"""Atmosphere-specific LTE diagnostic figures."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from .plotting import DiagnosticPlotter, FIELD_STYLES


class AtmospherePlotter(DiagnosticPlotter):
    """Build shell, meridional, and optical-depth atmosphere figures."""

    def field_panel_figure(
        self,
        evaluated: dict,
        depth_indices: list[int],
        label: str,
        *,
        field_names: tuple[str, ...],
        title: str,
    ) -> Figure:
        """Show selected fields on ordered optical-depth or shell-height layers."""

        depth_axis = evaluated["shell_height_levels_m"]
        solar_radius_m = float(evaluated["solar_radius_m"])
        depth_indices = sorted(
            depth_indices,
            key=lambda index: float(depth_axis[index]),
            reverse=True,
        )
        figure = Figure(
            figsize=(
                max(4.2 * len(field_names), 6.0),
                max(3.2 * len(depth_indices), 5.0),
            ),
            constrained_layout=True,
        )
        FigureCanvasAgg(figure)
        axes = np.asarray(
            figure.subplots(
                len(depth_indices),
                len(field_names),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
        )
        map_longitude_deg = evaluated["map_longitude_deg"]
        map_latitude_deg = evaluated["map_latitude_deg"]
        shared_norms = {
            name: self._field_norm(evaluated["map_fields"][name], style)
            for name, style in ((name, FIELD_STYLES[name]) for name in field_names)
        }
        column_images = [None] * len(field_names)
        for row, depth_index in enumerate(depth_indices):
            depth_value = float(depth_axis[depth_index])
            for column, name in enumerate(field_names):
                style = FIELD_STYLES[name]
                axis = axes[row, column]
                values = evaluated["map_fields"][name][..., depth_index]
                layer_longitude_deg = (
                    map_longitude_deg[..., depth_index]
                    if map_longitude_deg.ndim == 3
                    else map_longitude_deg
                )
                layer_latitude_deg = (
                    map_latitude_deg[..., depth_index]
                    if map_latitude_deg.ndim == 3
                    else map_latitude_deg
                )
                image = axis.pcolormesh(
                    layer_longitude_deg,
                    layer_latitude_deg,
                    values,
                    shading="nearest",
                    cmap=style["cmap"],
                    norm=shared_norms[name],
                    rasterized=True,
                )
                slice_longitude = evaluated.get("slice_longitude_deg")
                if slice_longitude is not None:
                    axis.axvline(
                        slice_longitude,
                        color="black",
                        linewidth=0.7,
                        linestyle="--",
                        alpha=0.8,
                    )
                column_images[column] = image
                axis.set_aspect("equal", adjustable="box")
                if column == 0:
                    radius_Rsun = 1.0 + depth_value / solar_radius_m
                    axis.annotate(
                        rf"$r={radius_Rsun:.6f}\,R_\odot$",
                        xy=(0.01, 0.98),
                        xycoords="axes fraction",
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="black",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.7,
                            "pad": 1.5,
                        },
                    )
                axis.label_outer()
        figure.supxlabel("Carrington longitude [deg]")
        figure.supylabel("Carrington latitude [deg]")
        for column, (name, image) in enumerate(
            zip(field_names, column_images, strict=True)
        ):
            self._add_shared_colorbar(
                figure,
                image,
                axes[:, column],
                FIELD_STYLES[name]["label"],
            )
        figure.suptitle(f"{title} — {label}")
        return figure

    def tau_figure(self, evaluated: dict, label: str) -> Figure:
        """Show continuum optical depth integrated along the traced rays."""

        tau = evaluated["profile_fields"]["tau500_ray"]
        tiny = np.finfo(tau.dtype).tiny
        log_tau = np.log10(np.clip(tau, tiny, None))
        radius_Rsun = 1.0 + (
            evaluated["shell_height_levels_m"] / evaluated["solar_radius_m"]
        )
        # The outer boundary is defined to have tau=0 and therefore has no
        # finite logarithm. Exclude only that endpoint from the profile panel.
        profile_slice = slice(1, None)
        lower, median, upper = np.nanpercentile(
            log_tau[:, profile_slice], (16.0, 50.0, 84.0), axis=0
        )

        figure = Figure(figsize=(11.0, 4.5), constrained_layout=True)
        FigureCanvasAgg(figure)
        profile_axis, map_axis = figure.subplots(1, 2)
        profile_axis.fill_betweenx(
            radius_Rsun[profile_slice],
            lower,
            upper,
            color="tab:blue",
            alpha=0.25,
            label="validation rays: 16–84%",
        )
        profile_axis.plot(
            median,
            radius_Rsun[profile_slice],
            color="tab:blue",
            linewidth=1.6,
            label="validation-ray median",
        )
        profile_axis.plot(
            evaluated["log_tau500"][profile_slice],
            radius_Rsun[profile_slice],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="FALC radial shell labels",
        )
        profile_axis.axvline(0.0, color="0.45", linewidth=0.8, linestyle=":")
        profile_axis.set_xlabel(r"$\log_{10}\tau_{500}$")
        profile_axis.set_ylabel(r"radius $r/R_\odot$")
        profile_axis.set_title(r"derived $\tau_{500}=\int\alpha_{500}\,ds$")
        profile_axis.grid(alpha=0.2)
        profile_axis.legend(loc="best", fontsize=8)

        bottom_index = -1
        bottom_tau = evaluated["map_fields"]["tau500_ray"][..., bottom_index]
        finite = bottom_tau[np.isfinite(bottom_tau) & (bottom_tau > 0)]
        if finite.size:
            bottom_log_tau = np.log10(np.clip(bottom_tau, tiny, None))
            vmin, vmax = self._limits(bottom_log_tau, signed=False)
        else:
            bottom_log_tau = np.full_like(bottom_tau, np.nan)
            vmin, vmax = -1.0, 1.0
        map_x = evaluated["map_longitude_deg"]
        map_y = evaluated["map_latitude_deg"]
        if map_x.ndim == 3:
            map_x = map_x[..., bottom_index]
            map_y = map_y[..., bottom_index]
        image = map_axis.pcolormesh(
            map_x,
            map_y,
            bottom_log_tau,
            shading="nearest",
            cmap="viridis",
            norm=Normalize(vmin=vmin, vmax=vmax),
            rasterized=True,
        )
        map_axis.set_aspect("equal", adjustable="box")
        map_axis.set_xlabel("Carrington longitude [deg]")
        map_axis.set_ylabel("Carrington latitude [deg]")
        bottom_radius_Rsun = radius_Rsun[bottom_index]
        map_axis.set_title(
            rf"bottom layer: $\log_{{10}}\tau_{{500}}$ at "
            rf"$r={bottom_radius_Rsun:.6f}\,R_\odot$"
        )
        colorbar = figure.colorbar(image, ax=map_axis, location="right", shrink=0.86)
        colorbar.set_label(r"$\log_{10}\tau_{500}$")
        figure.suptitle(f"Continuum optical-depth validation — {label}")
        return figure

    def meridional_field_panel_figure(
        self,
        evaluated: dict,
        label: str,
        *,
        field_names: tuple[str, ...],
        title: str,
        norm_evaluated: dict | None = None,
    ) -> Figure:
        """Plot fields on an explicitly sampled constant-longitude radial plane."""

        figure = Figure(
            figsize=(max(4.2 * len(field_names), 7.0), 5.5),
            constrained_layout=True,
        )
        FigureCanvasAgg(figure)
        axes = np.asarray(
            figure.subplots(
                1, len(field_names), squeeze=False, sharex=True, sharey=True
            )
        )[0]
        map_latitude = evaluated["map_latitude_deg"][:, 0]
        latitude_deg = map_latitude if map_latitude.ndim == 2 else map_latitude[:, None]
        vertical = 1.0 + (
            evaluated["map_fields"]["geometric_height"][:, 0, :]
            * 1.0e6
            / evaluated["solar_radius_m"]
        )
        vertical_label = r"radius $r/R_\odot$"
        depth_plane = "physical Carrington latitude-radius"
        latitude_grid_deg = (
            latitude_deg
            if latitude_deg.shape == vertical.shape
            else np.broadcast_to(latitude_deg, vertical.shape)
        )
        latitude_corners_deg = self._curvilinear_cell_corners(latitude_grid_deg)
        vertical_corners = self._curvilinear_cell_corners(vertical)
        longitude_deg = evaluated["map_longitude_deg"][:, 0]

        images = []
        for axis, name in zip(axes, field_names, strict=True):
            style = FIELD_STYLES[name]
            values = evaluated["map_fields"][name][:, 0, :]
            norm_values = (
                norm_evaluated["map_fields"][name]
                if norm_evaluated is not None
                else values
            )
            image = axis.pcolormesh(
                latitude_corners_deg,
                vertical_corners,
                np.ma.masked_invalid(values),
                shading="flat",
                cmap=style["cmap"],
                norm=self._field_norm(norm_values, style),
                rasterized=True,
            )
            images.append(image)
            axis.set_aspect("auto")
            axis.label_outer()
        figure.supxlabel("Carrington latitude [deg]")
        figure.supylabel(vertical_label)
        for axis, name, image in zip(axes, field_names, images, strict=True):
            self._add_shared_colorbar(
                figure,
                image,
                axis,
                FIELD_STYLES[name]["label"],
            )

        finite_x = longitude_deg[np.isfinite(longitude_deg)]
        if finite_x.size:
            x_min, x_max = float(np.min(finite_x)), float(np.max(finite_x))
            x_description = f"Carrington longitude={0.5 * (x_min + x_max):.3f} deg"
        else:
            x_description = "Carrington longitude unavailable"
        figure.suptitle(f"{title} {depth_plane} slice — {x_description} — {label}")
        return figure


__all__ = ["AtmospherePlotter"]
