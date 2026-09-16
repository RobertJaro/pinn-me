"""Stokes-profile validation figures for LTE diagnostics."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .plotting import DiagnosticPlotter
from .sampling import map_coordinates


class StokesPlotter(DiagnosticPlotter):
    """Build observed-versus-predicted Stokes validation figures."""

    def validation_figure(
        self,
        outputs,
        raster,
        wavelength_angstrom: torch.Tensor,
        label: str,
        *,
        rows: np.ndarray,
        columns: np.ndarray,
        objective_config=None,
        exclusion_windows_angstrom=(),
        line_centers_angstrom=(),
    ) -> Figure:
        """Show loss-space Stokes predictions and references without rescaling."""

        wavelength = wavelength_angstrom.detach().float().cpu().numpy()
        finite_wavelength = wavelength[np.isfinite(wavelength)]
        if finite_wavelength.size < 2:
            raise ValueError(
                "Stokes visualization requires at least two finite wavelengths."
            )
        wavelength_limits = (
            float(finite_wavelength.min()),
            float(finite_wavelength.max()),
        )
        line_centers_angstrom = tuple(map(float, line_centers_angstrom))
        if not wavelength_limits[1] > wavelength_limits[0]:
            raise ValueError(
                "Stokes visualization wavelengths must span a finite interval."
            )
        if not isinstance(outputs, dict):
            raise TypeError(
                "Stokes visualization requires the streamed callback payload."
            )
        prediction = outputs["stokes_pred"].numpy()
        reference = outputs["stokes_reference"].numpy()
        pixel_index = outputs["pixel_index"].numpy()
        integrated_prediction = outputs["integrated_prediction"].numpy()
        integrated_reference = outputs["integrated_reference"].numpy()
        sampled_rows = np.asarray(rows)
        sampled_columns = np.asarray(columns)
        selected = np.isin(pixel_index[:, 0], sampled_rows) & np.isin(
            pixel_index[:, 1], sampled_columns
        )
        if not np.any(selected):
            raise ValueError(
                "The shared visualization grid contains no validation pixels."
            )
        pixel_index = pixel_index[selected]
        integrated_prediction = integrated_prediction[selected]
        integrated_reference = integrated_reference[selected]
        compact_rows = np.searchsorted(sampled_rows, pixel_index[:, 0])
        compact_columns = np.searchsorted(sampled_columns, pixel_index[:, 1])
        prediction_maps = np.full(
            (sampled_rows.size, sampled_columns.size, 4), np.nan, dtype=np.float32
        )
        reference_maps = np.full_like(prediction_maps, np.nan)
        prediction_maps[compact_rows, compact_columns] = integrated_prediction
        reference_maps[compact_rows, compact_columns] = integrated_reference
        map_x_mm, map_y_mm = map_coordinates(raster, sampled_rows, sampled_columns)
        # The image values stay on the native, strided detector array. Carrington
        # coordinates are used only to label its outer pixel edges; they do not
        # define a second mesh or trigger spatial interpolation.
        x_centers = np.nanmedian(map_x_mm, axis=0)
        y_centers = np.nanmedian(map_y_mm, axis=1)

        def outer_edges(centers: np.ndarray) -> tuple[float, float]:
            if centers.size == 1:
                return float(centers[0] - 0.5), float(centers[0] + 0.5)
            return (
                float(centers[0] - 0.5 * (centers[1] - centers[0])),
                float(centers[-1] + 0.5 * (centers[-1] - centers[-2])),
            )

        x_edges = outer_edges(x_centers)
        y_edges = outer_edges(y_centers)
        image_extent = (*x_edges, *y_edges)

        # Explicit subfigure spacing is more stable here than constrained layout:
        # the equal-aspect map panels otherwise squeeze the profile row nearly
        # flat when horizontal colorbars are attached above them.
        figure = Figure(figsize=(16.0, 14.0))
        FigureCanvasAgg(figure)
        map_figure, profile_figure, scatter_figure = figure.subfigures(
            3, 1, height_ratios=(2.0, 1.0, 1.0)
        )
        map_axes = map_figure.subplots(2, 4, squeeze=False, sharex=True, sharey=True)
        profile_axes = np.asarray(
            profile_figure.subplots(1, 4, squeeze=False, sharex=True)
        )[0]
        scatter_axes = np.asarray(scatter_figure.subplots(1, 4, squeeze=False))[0]
        map_figure.subplots_adjust(
            left=0.07, right=0.99, bottom=0.12, top=0.88, wspace=0.16, hspace=0.30
        )
        profile_figure.subplots_adjust(
            left=0.07, right=0.99, bottom=0.22, top=0.82, wspace=0.16
        )
        scatter_figure.subplots_adjust(
            left=0.07, right=0.99, bottom=0.22, top=0.82, wspace=0.16
        )
        map_figure.supxlabel("Carrington chart X [Mm]")
        map_figure.supylabel("Carrington chart Y [Mm]")
        map_figure.text(
            0.985, 0.66, "reference", rotation=-90, ha="center", va="center"
        )
        map_figure.text(
            0.985, 0.27, "predicted", rotation=-90, ha="center", va="center"
        )
        profile_figure.supxlabel(r"air wavelength [$\AA$]")
        asinh = (objective_config or {}).get("type") == "asinh_mse"
        comparison_label = (
            f"I linear; Q/U/V normalized asinh\na={objective_config['asinh_scale']:g}"
            if asinh
            else r"Stokes component / $I_{c,\,atlas}(\mu=1)$"
        )
        profile_figure.supylabel(comparison_label)
        scatter_figure.supxlabel(r"reference absolute integral [$\AA$]")
        scatter_figure.supylabel(r"predicted absolute integral [$\AA$]")
        for component, name in enumerate(("I", "Q", "U", "V")):
            display_name = (
                rf"\mathrm{{asinh}}({name}/a)/\mathrm{{asinh}}(1/a)"
                if asinh and component
                else name
            )
            finite = reference_maps[..., component][
                np.isfinite(reference_maps[..., component])
            ]
            if finite.size:
                vmin, vmax = float(finite.min()), float(finite.max())
                if not vmax > vmin:
                    vmax = vmin + max(abs(vmin) * 1.0e-6, 1.0e-12)
            else:
                vmin, vmax = 0.0, 1.0
            map_image = None
            for row, values in enumerate(
                (
                    reference_maps[..., component],
                    prediction_maps[..., component],
                )
            ):
                axis = map_axes[row, component]
                image = axis.imshow(
                    values,
                    origin="lower",
                    extent=image_extent,
                    interpolation="none",
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=True,
                )
                map_image = image
                axis.set_aspect("equal", adjustable="box")
                axis.label_outer()
            self._add_shared_colorbar(
                map_figure,
                map_image,
                map_axes[:, component],
                rf"fit-window $|{display_name}|$ integral [$\AA$]",
                scientific_notation=False,
            )
            profile_axis = profile_axes[component]
            reference_percentiles = np.nanpercentile(
                reference[:, component, :], (16.0, 50.0, 84.0), axis=0
            )
            prediction_percentiles = np.nanpercentile(
                prediction[:, component, :], (16.0, 50.0, 84.0), axis=0
            )
            profile_axis.fill_between(
                wavelength,
                reference_percentiles[0],
                reference_percentiles[2],
                color="black",
                alpha=0.14,
                linewidth=0.0,
            )
            profile_axis.fill_between(
                wavelength,
                prediction_percentiles[0],
                prediction_percentiles[2],
                color="tab:red",
                alpha=0.18,
                linewidth=0.0,
            )
            profile_axis.plot(
                wavelength,
                reference_percentiles[1],
                color="black",
                linewidth=1.25,
                label="reference median",
            )
            profile_axis.plot(
                wavelength,
                prediction_percentiles[1],
                color="tab:red",
                linewidth=1.15,
                label="predicted median",
            )
            profile_axis.axhline(0.0, color="0.75", linewidth=0.6)
            for line_center in line_centers_angstrom:
                if wavelength_limits[0] <= line_center <= wavelength_limits[1]:
                    profile_axis.axvline(
                        line_center,
                        color="tab:blue",
                        alpha=0.25,
                        linewidth=0.8,
                    )
            for lower, upper in exclusion_windows_angstrom:
                profile_axis.axvspan(
                    lower,
                    upper,
                    color="0.65",
                    alpha=0.22,
                    linewidth=0.0,
                    label=("excluded from objective" if component == 0 else None),
                )
            # Instrument-specific line markers must not expand another
            # instrument's observed wavelength range (notably HMI at 6173 A).
            profile_axis.set_xlim(*wavelength_limits)
            profile_axis.set_title(f"signed {name} ensemble (median, 16–84%)")
            profile_axis.grid(alpha=0.18)

            scatter_axis = scatter_axes[component]
            reference_integral = integrated_reference[:, component]
            prediction_integral = integrated_prediction[:, component]
            finite_integrals = np.isfinite(reference_integral) & np.isfinite(
                prediction_integral
            )
            reference_integral = reference_integral[finite_integrals]
            prediction_integral = prediction_integral[finite_integrals]
            scatter_axis.scatter(
                reference_integral,
                prediction_integral,
                s=7,
                alpha=0.35,
                color="tab:blue",
                edgecolors="none",
                rasterized=True,
            )
            if reference_integral.size:
                combined_integrals = np.concatenate(
                    (reference_integral, prediction_integral)
                )
                upper = float(np.nanpercentile(combined_integrals, 99.5))
                if component == 0:
                    # Integrated Stokes I occupies a narrow positive range.
                    # Following it from zero discards nearly all useful scatter
                    # contrast, so use the robust joint prediction/reference
                    # range while retaining equal axes and the identity line.
                    lower = float(np.nanpercentile(combined_integrals, 0.5))
                    span = upper - lower
                    padding = (
                        0.03 * span if span > 0.0 else max(abs(lower) * 1.0e-3, 1.0e-12)
                    )
                    lower -= padding
                    upper += padding
                else:
                    lower = 0.0
                    upper = max(
                        upper,
                        float(reference_integral.max()) * 1.0e-6,
                        1.0e-12,
                    )
                scatter_axis.plot(
                    (lower, upper), (lower, upper), color="black", linewidth=0.8
                )
                scatter_axis.set_xlim(lower, upper)
                scatter_axis.set_ylim(lower, upper)
            scatter_axis.set_title(f"integrated $|{display_name}|$")
            scatter_axis.grid(alpha=0.18)
        profile_axes[0].legend(loc="best", fontsize=8)
        figure.suptitle(
            f"Stokes validation — instrument-integrated samples — {label}\n"
            "Profiles show comparison values; maps show their absolute wavelength integrals"
        )
        return figure

    def phase_figure(
        self,
        outputs,
        raster,
        label: str,
        *,
        rows: np.ndarray,
        columns: np.ndarray,
        step: int,
        cold_steps: int,
        handoff_step: int,
    ) -> Figure:
        """Show the temporary Stokes disambiguation phase on the validation grid.

        The phase is displayed in degrees because the two equivalent
        transverse-field branches are separated by 180 degrees.  The second
        panel shows the exact smooth binary objective used during warmup,
        ``sin(phi)^2``.
        """

        if not isinstance(outputs, dict):
            raise TypeError("Phase visualization requires the streamed callback payload.")

        def as_numpy(value) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        phase = as_numpy(outputs["phase_rad"]).astype(np.float32, copy=False).reshape(-1)
        pixel_index = as_numpy(outputs["pixel_index"])
        if pixel_index.ndim != 2 or pixel_index.shape[1] != 2:
            raise ValueError("Phase visualization pixel_index must have shape [N, 2].")
        if pixel_index.shape[0] != phase.shape[0]:
            raise ValueError(
                "Phase visualization phase and pixel_index must contain the same samples."
            )
        sampled_rows = np.asarray(rows, dtype=np.int64)
        sampled_columns = np.asarray(columns, dtype=np.int64)
        selected = (
            np.isin(pixel_index[:, 0], sampled_rows)
            & np.isin(pixel_index[:, 1], sampled_columns)
            & np.isfinite(phase)
        )
        if not np.any(selected):
            raise ValueError(
                "The shared visualization grid contains no finite phase samples."
            )
        pixel_index = pixel_index[selected]
        phase = phase[selected]
        compact_rows = np.searchsorted(sampled_rows, pixel_index[:, 0])
        compact_columns = np.searchsorted(sampled_columns, pixel_index[:, 1])
        phase_degrees = np.rad2deg(phase)
        # The network output is an unconstrained real phase.  Wrap only for
        # display so equivalent 0/360-degree branches share a cyclic color.
        wrapped_phase_degrees = (phase_degrees + 180.0) % 360.0 - 180.0
        binary_distance = np.square(np.sin(phase))
        phase_map = np.full(
            (sampled_rows.size, sampled_columns.size), np.nan, dtype=np.float32
        )
        binary_map = np.full_like(phase_map, np.nan)
        phase_map[compact_rows, compact_columns] = wrapped_phase_degrees
        binary_map[compact_rows, compact_columns] = binary_distance
        map_x_mm, map_y_mm = map_coordinates(raster, sampled_rows, sampled_columns)
        x_centers = np.nanmedian(map_x_mm, axis=0)
        y_centers = np.nanmedian(map_y_mm, axis=1)

        def outer_edges(centers: np.ndarray) -> tuple[float, float]:
            if centers.size == 1:
                return float(centers[0] - 0.5), float(centers[0] + 0.5)
            return (
                float(centers[0] - 0.5 * (centers[1] - centers[0])),
                float(centers[-1] + 0.5 * (centers[-1] - centers[-2])),
            )

        image_extent = (*outer_edges(x_centers), *outer_edges(y_centers))
        figure = Figure(figsize=(13.0, 6.2))
        FigureCanvasAgg(figure)
        axes = np.asarray(figure.subplots(1, 2, squeeze=False))[0]
        panels = (
            (
                phase_map,
                "twilight",
                -180.0,
                180.0,
                r"wrapped phase $\phi$ [deg] (mod 360°)",
                "degrees",
            ),
            (
                binary_map,
                "viridis",
                0.0,
                1.0,
                r"binary distance $\sin^2(\phi)$",
                "dimensionless",
            ),
        )
        for axis, (values, cmap, vmin, vmax, title, colorbar_label) in zip(
            axes, panels, strict=True
        ):
            image = axis.imshow(
                values,
                origin="lower",
                extent=image_extent,
                interpolation="none",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Carrington chart X [Mm]")
            axis.set_ylabel("Carrington chart Y [Mm]")
            axis.set_title(title)
            axis.grid(alpha=0.15)
            colorbar = figure.colorbar(
                image, ax=axis, orientation="horizontal", pad=0.10, fraction=0.08
            )
            colorbar.set_label(colorbar_label)

        if step < cold_steps:
            state = "cold start: binary penalty off"
        elif step >= handoff_step:
            state = "hard handoff: phase mask bypassed"
        else:
            state = "phase network active; physical field remains unrotated"
        figure.suptitle(
            f"Stokes disambiguation phase — {label} — step {step}\n{state}"
        )
        return figure


__all__ = ["StokesPlotter"]
