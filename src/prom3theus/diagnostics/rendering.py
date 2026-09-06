"""Coordinate evaluation, plotting, and publication of LTE diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from .atmosphere_rendering import AtmospherePlotter
from .evaluation import (
    AtmosphereEvaluator,
    CURRENT_DENSITY_FIELDS,
    MAGNETIC_FIELDS,
    THERMODYNAMIC_FIELDS,
    VELOCITY_FIELDS,
)
from .sampling import display_grid
from .stokes_rendering import StokesPlotter


class AtmosphereRenderer:
    """Render evaluated diagnostics without owning any training lifecycle state."""

    def __init__(
        self,
        output_directory,
        *,
        dpi: int,
        evaluator: AtmosphereEvaluator,
    ):
        if dpi < 50:
            raise ValueError("dpi must be at least 50.")
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.dpi = int(dpi)
        self.evaluator = evaluator
        self.atmosphere_plots = AtmospherePlotter()
        self.stokes_plots = StokesPlotter()

    @staticmethod
    def _log_figure(trainer, key: str, figure: Figure, path: Path) -> None:
        logger = getattr(trainer, "logger", None)
        if logger in (None, False):
            return
        log_image = getattr(logger, "log_image", None)
        if callable(log_image):
            # Upload the already-rendered PNG rather than the live Matplotlib
            # figure. WandB otherwise rasterizes the figure again at its
            # default DPI, making detailed atmosphere maps look softer than
            # the local artifact.
            log_image(
                key=key,
                images=[str(path)],
                step=int(getattr(trainer, "global_step", 0)),
            )
            return
        experiment = getattr(logger, "experiment", None)
        add_figure = getattr(experiment, "add_figure", None)
        if callable(add_figure):
            add_figure(key, figure, global_step=int(getattr(trainer, "global_step", 0)))

    def _save_figure(self, trainer, figure: Figure, filename: str, key: str) -> Path:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.output_directory / filename
        figure.savefig(path, dpi=self.dpi, facecolor="white")
        self._log_figure(trainer, key, figure, path)
        figure.clear()
        return path

    def render_atmosphere(
        self,
        trainer,
        pl_module,
        raster,
        *,
        label: str,
        rows: np.ndarray | None = None,
        columns: np.ndarray | None = None,
    ) -> list[Path]:
        """Evaluate and save one complete visualization snapshot."""

        if (rows is None) != (columns is None):
            raise ValueError("rows and columns must be supplied together.")
        if rows is None or columns is None:
            rows, columns = display_grid(trainer, raster, self.evaluator.max_ray_pixels)
        ray_evaluated = self.evaluator.evaluate_ray_optical_depth(
            pl_module, raster, rows=rows, columns=columns
        )
        paths = []
        outer_height_Mm, inner_height_Mm = (
            pl_module.atmosphere_model.shell_height_bounds_Mm
        )
        slice_heights = self.log_spaced_slice_heights(
            inner_height_Mm * 1.0e6,
            outer_height_Mm * 1.0e6,
            self.evaluator.slice_layer_count,
        )
        evaluated = self.evaluator.evaluate_shell_layers(
            pl_module, raster, slice_heights
        )
        selected_indices = list(range(len(slice_heights)))
        thermodynamic_fields = tuple(
            name for name in THERMODYNAMIC_FIELDS if name in evaluated["map_fields"]
        )
        panels = (
            (
                thermodynamic_fields,
                "Depth-stratified LTE thermodynamic parameters",
                "parameters",
                "Parameters",
            ),
            (
                MAGNETIC_FIELDS,
                r"Depth-stratified magnetic field — spherical components and observer-frame angles",
                "magnetic_field",
                "Magnetic field",
            ),
            (
                VELOCITY_FIELDS,
                r"Depth-stratified velocity — spherical components and observer LOS",
                "velocity",
                "Velocity",
            ),
            (
                CURRENT_DENSITY_FIELDS,
                r"Current-density magnitude $\|\mathbf{J}\|$",
                "current_density",
                "Current density",
            ),
        )
        for field_names, title, filename_suffix, log_key in panels:
            paths.append(
                self._save_figure(
                    trainer,
                    self.atmosphere_plots.field_panel_figure(
                        evaluated,
                        selected_indices,
                        label,
                        field_names=field_names,
                        title=title,
                    ),
                    f"{label}_{filename_suffix}.png",
                    log_key,
                )
            )
        paths.append(
            self._save_figure(
                trainer,
                self.atmosphere_plots.tau_figure(ray_evaluated, label),
                f"{label}_tau500.png",
                "Optical depth",
            )
        )
        meridional_evaluated = (
            self.evaluator.evaluate_meridional_slice(pl_module, raster)
            if self.evaluator.meridional_slice_enabled
            else None
        )
        if self.evaluator.meridional_slice_enabled:
            meridional_panels = (
                (
                    THERMODYNAMIC_FIELDS,
                    "LTE thermodynamic parameters",
                    "meridional_parameters",
                    "Parameters",
                ),
                (
                    MAGNETIC_FIELDS,
                    r"Magnetic field — spherical components and observer-frame angles",
                    "meridional_magnetic_field",
                    "Magnetic field",
                ),
                (
                    VELOCITY_FIELDS,
                    r"Velocity — spherical components and observer LOS",
                    "meridional_velocity",
                    "Velocity",
                ),
                (
                    CURRENT_DENSITY_FIELDS,
                    r"Current-density magnitude $\|\mathbf{J}\|$",
                    "meridional_current_density",
                    "Current density",
                ),
            )
            for field_names, title, filename_suffix, log_key in meridional_panels:
                paths.append(
                    self._save_figure(
                        trainer,
                        self.atmosphere_plots.meridional_field_panel_figure(
                            meridional_evaluated,
                            label,
                            field_names=field_names,
                            title=title,
                            norm_evaluated=evaluated,
                        ),
                        f"{label}_{filename_suffix}.png",
                        f"{log_key} meridional slice",
                    )
                )
        return paths

    @staticmethod
    def log_spaced_slice_heights(
        inner_height_m: float,
        outer_height_m: float,
        count: int,
    ) -> np.ndarray:
        """Sample the lower boundary, then reference-to-top layers logarithmically."""

        if not inner_height_m < 0.0 < outer_height_m or count < 3:
            raise ValueError(
                "Log-spaced shell slices require inner < 0 < outer and count >= 3."
            )
        # The first layer is the exact lower boundary. The remaining layers
        # start exactly at r=R_sun (h=0) and use a 100-km shifted logarithmic
        # scale so both zero and the exact upper boundary can be included.
        offset_m = 1.0e5
        above_reference = (
            np.geomspace(offset_m, outer_height_m + offset_m, count - 1) - offset_m
        )
        return np.concatenate(([inner_height_m], above_reference)).astype(np.float32)

    def render_stokes_validation(
        self,
        trainer,
        outputs,
        raster,
        wavelength_angstrom: torch.Tensor,
        label: str,
        *,
        rows: np.ndarray,
        columns: np.ndarray,
        exclusion_windows_angstrom=(),
        line_centers_angstrom=(),
    ) -> Path:
        """Render and save the validation Stokes comparison."""

        return self._save_figure(
            trainer,
            self.stokes_plots.validation_figure(
                outputs,
                raster,
                wavelength_angstrom,
                label,
                rows=rows,
                columns=columns,
                exclusion_windows_angstrom=exclusion_windows_angstrom,
                line_centers_angstrom=line_centers_angstrom,
            ),
            f"{label}_stokes_validation.png",
            "Stokes comparison",
        )


__all__ = ["AtmosphereRenderer"]
