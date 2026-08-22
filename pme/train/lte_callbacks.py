"""Training visualizations for depth-stratified LTE atmospheres."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from pytorch_lightning import Callback


_FIELD_STYLES = {
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
    "geometric_height": {
        "label": r"$z$ [Mm; $\langle z(q=0)\rangle=0$]",
        "cmap": "terrain",
        "signed": False,
    },
    "v_x": {
        "label": r"$v_x$ [km s$^{-1}$]",
        "cmap": "seismic",
        "signed": True,
    },
    "v_y": {
        "label": r"$v_y$ [km s$^{-1}$]",
        "cmap": "seismic",
        "signed": True,
    },
    "v_z": {
        "label": r"$v_z$ [km s$^{-1}$; toward observer]",
        "cmap": "seismic",
        "signed": True,
    },
    "v_magnitude": {
        "label": r"$|v|$ [km s$^{-1}$]",
        "cmap": "cividis",
        "signed": False,
    },
    "b_x": {"label": r"$B_x$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_y": {"label": r"$B_y$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_los": {"label": r"$B_{LOS}$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_magnitude": {"label": r"$|B|$ [G]", "cmap": "cividis", "signed": False},
    "b_inclination": {
        "label": r"$\gamma_B$ [deg]",
        "cmap": "PiYG",
        "signed": False,
        "limits": (0.0, 180.0),
    },
    "b_azimuth": {
        "label": r"$\chi_B$ [deg]",
        "cmap": "twilight",
        "signed": True,
        "limits": (-180.0, 180.0),
    },
    "microturbulence": {
        "label": r"$\xi$ [km s$^{-1}$]",
        "cmap": "magma",
        "signed": False,
    },
}

_THERMODYNAMIC_FIELDS = (
    "temperature",
    "density",
    "pressure",
    "geometric_height",
    "microturbulence",
)
_MAGNETIC_FIELDS = (
    "b_x",
    "b_y",
    "b_los",
    "b_magnitude",
    "b_inclination",
    "b_azimuth",
)
_VELOCITY_FIELDS = ("v_x", "v_y", "v_z", "v_magnitude")
_YZ_THERMODYNAMIC_FIELDS = (
    "temperature",
    "density",
    "pressure",
    "microturbulence",
)


class LTEAtmosphereVisualizationCallback(Callback):
    """Render inferred LTE atmosphere profiles and optical-depth maps.

    The callback evaluates the atmosphere network plus the pinned STiC
    thermodynamic lookup so density is the same quantity used by geometric
    HSE. It does not repeat polarized synthesis. Full rasters are subsampled before
    evaluation, bounding both callback memory and runtime. Atmosphere and
    Stokes maps share the same two-dimensional helioprojective Solar-X/Solar-Y
    grid in Mm; no detector-index or axis-aligned WCS approximation is used.
    PNG files are always written locally.  When the configured Lightning
    logger implements ``log_image`` (for example WandB), the same figures are
    also uploaded. Integrated Stokes products retain every validation sample;
    complete wavelength profiles are kept in a bounded deterministic reservoir
    so callback memory does not scale as pixels times wavelengths.
    """

    def __init__(
        self,
        output_directory,
        *,
        every_n_epochs: int = 5,
        depth_layer_count: int = 4,
        evaluation_batch_size: int = 8192,
        max_map_pixels: int = 65_536,
        max_profile_samples: int = 4_096,
        dpi: int = 180,
        include_initial: bool = True,
        yz_slice: Mapping | None = None,
    ):
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive.")
        if depth_layer_count < 2:
            raise ValueError("depth_layer_count must be at least two.")
        if evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be positive.")
        if max_map_pixels < 1:
            raise ValueError("max_map_pixels must be positive.")
        if max_profile_samples < 1:
            raise ValueError("max_profile_samples must be positive.")
        if dpi < 50:
            raise ValueError("dpi must be at least 50.")
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.every_n_epochs = int(every_n_epochs)
        self.depth_layer_count = int(depth_layer_count)
        self.evaluation_batch_size = int(evaluation_batch_size)
        self.max_map_pixels = int(max_map_pixels)
        self.max_profile_samples = int(max_profile_samples)
        self.dpi = int(dpi)
        self.include_initial = bool(include_initial)
        yz_config = dict(yz_slice or {})
        self.yz_slice_enabled = bool(yz_config.pop("enabled", False))
        raw_scan_index = yz_config.pop("scan_index", None)
        if yz_config:
            raise TypeError(f"Unknown YZ-slice options: {sorted(yz_config)}")
        if self.yz_slice_enabled and raw_scan_index is None:
            raise ValueError("An enabled YZ slice requires scan_index.")
        self.yz_slice_scan_index = (
            None if raw_scan_index is None else int(raw_scan_index)
        )
        self._last_rendered_step: int | None = None
        self._validation_outputs: list[dict[str, torch.Tensor]] = []
        self._profile_reservoir: dict[str, torch.Tensor] | None = None
        self._collect_validation_epoch = True

    @staticmethod
    def _subsample_indices(
        height: int, width: int, maximum: int
    ) -> tuple[np.ndarray, np.ndarray]:
        stride = max(1, int(math.ceil(math.sqrt(height * width / maximum))))
        rows = np.arange(0, height, stride, dtype=np.int64)
        columns = np.arange(0, width, stride, dtype=np.int64)
        while rows.size * columns.size > maximum:
            stride += 1
            rows = np.arange(0, height, stride, dtype=np.int64)
            columns = np.arange(0, width, stride, dtype=np.int64)
        return rows, columns

    def _display_indices(self, trainer, raster) -> tuple[np.ndarray, np.ndarray]:
        """Use one validation-derived rectangular grid for every map panel."""

        data_module = getattr(trainer, "datamodule", None)
        dataset = getattr(data_module, "dataset", None)
        validation = getattr(data_module, "validation_dataset", None)
        pixel_indices = getattr(dataset, "pixel_indices", None)
        subset_indices = getattr(validation, "indices", None)
        if pixel_indices is None or subset_indices is None:
            return self._subsample_indices(*raster.spatial_shape, self.max_map_pixels)
        selected = pixel_indices[torch.as_tensor(subset_indices, dtype=torch.long)]
        rows = torch.unique(selected[:, 0], sorted=True).cpu().numpy()
        columns = torch.unique(selected[:, 1], sorted=True).cpu().numpy()
        stride = max(
            1,
            int(math.ceil(math.sqrt(rows.size * columns.size / self.max_map_pixels))),
        )
        rows = rows[::stride]
        columns = columns[::stride]
        while rows.size * columns.size > self.max_map_pixels:
            stride += 1
            all_rows = torch.unique(selected[:, 0], sorted=True).cpu().numpy()
            all_columns = torch.unique(selected[:, 1], sorted=True).cpu().numpy()
            rows = all_rows[::stride]
            columns = all_columns[::stride]
        return rows, columns

    def _depth_indices(self, depth_grid: np.ndarray) -> list[int]:
        """Select evenly spaced display levels across the represented range."""

        requested_depths = np.linspace(
            float(depth_grid.min()),
            float(depth_grid.max()),
            self.depth_layer_count,
        )
        selected: list[int] = []
        for requested in requested_depths:
            index = int(np.argmin(np.abs(depth_grid - requested)))
            if index not in selected:
                selected.append(index)
        return selected

    @staticmethod
    def _map_coordinates(
        raster,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the shared 2-D Solar-X/Solar-Y centre grids in Mm."""

        coords = raster.coords[rows][:, columns].detach().double().cpu().numpy()
        if coords.shape[-1] != 3:
            raise ValueError(
                "Hinode map coordinates must end in [time_hours, x_mm, y_mm]."
            )
        x_mm = coords[..., 1]
        y_mm = coords[..., 2]
        if not np.isfinite(x_mm).all() or not np.isfinite(y_mm).all():
            raise ValueError("Hinode map coordinates must be finite.")
        return x_mm, y_mm

    def _evaluate(
        self,
        pl_module,
        raster,
        *,
        rows: np.ndarray | None = None,
        columns: np.ndarray | None = None,
    ) -> dict:
        model = pl_module.atmosphere_model
        parameter = next(model.parameters())
        height, width = raster.spatial_shape
        if rows is None or columns is None:
            rows, columns = self._subsample_indices(height, width, self.max_map_pixels)
        coords = raster.coords[rows][:, columns]
        valid = raster.valid_mask[rows][:, columns]
        flat_coords = coords[valid].to(device=parameter.device, dtype=parameter.dtype)
        if flat_coords.numel() == 0:
            raise ValueError(
                "No valid Hinode pixels remain for atmosphere visualization."
            )

        was_training = model.training
        model.eval()
        chunks = {
            "temperature": [],
            "density": [],
            "pressure": [],
            "v_x": [],
            "v_y": [],
            "v_z": [],
            "v_magnitude": [],
            "b_x": [],
            "b_y": [],
            "b_los": [],
            "b_magnitude": [],
            "b_inclination": [],
            "b_azimuth": [],
            "microturbulence": [],
        }
        has_height_mapping = getattr(model, "height_mapping", None) is not None
        if has_height_mapping:
            chunks.update({"geometric_height": [], "height_metric": []})
        try:
            with torch.no_grad():
                for start in range(0, flat_coords.shape[0], self.evaluation_batch_size):
                    batch_coords = flat_coords[
                        start : start + self.evaluation_batch_size
                    ]
                    atmosphere = model(batch_coords)
                    if has_height_mapping:
                        # Z is a direct coordinate MLP, so its metric is obtained
                        # by differentiation. no_grad is intentionally overridden
                        # only for this small validation derivative.
                        with torch.enable_grad():
                            height_metric = (
                                model.height_mapping.metric_m_per_log_tau(
                                    batch_coords,
                                    model.log_tau500,
                                    create_graph=False,
                                )
                            )
                    gas_pressure = atmosphere.gas_pressure
                    if gas_pressure is None:
                        raise RuntimeError(
                            "LTE atmosphere visualization requires predicted gas pressure."
                        )
                    mass_density = (
                        pl_module.synthesizer.continuum_opacity.reference_mass_density(
                            atmosphere.temperature, gas_pressure
                        )
                    )
                    magnetic = atmosphere.magnetic_field
                    velocity = atmosphere.velocity_field / 1_000.0
                    transverse = torch.linalg.vector_norm(magnetic[..., :2], dim=-1)
                    values = {
                        "temperature": atmosphere.temperature,
                        "density": mass_density,
                        "pressure": gas_pressure,
                        "v_x": velocity[..., 0],
                        "v_y": velocity[..., 1],
                        "v_z": velocity[..., 2],
                        "v_magnitude": torch.linalg.vector_norm(velocity, dim=-1),
                        "b_x": magnetic[..., 0],
                        "b_y": magnetic[..., 1],
                        "b_los": magnetic[..., 2],
                        "b_magnitude": torch.linalg.vector_norm(magnetic, dim=-1),
                        "b_inclination": torch.rad2deg(
                            torch.atan2(transverse, magnetic[..., 2])
                        ),
                        "b_azimuth": torch.rad2deg(
                            torch.atan2(magnetic[..., 1], magnetic[..., 0])
                        ),
                        "microturbulence": atmosphere.microturbulence / 1_000.0,
                    }
                    if has_height_mapping:
                        values.update(
                            {
                                "geometric_height": atmosphere.geometric_height_m / 1.0e6,
                                "height_metric": height_metric / 1.0e3,
                            }
                        )
                    for name, value in values.items():
                        chunks[name].append(value.detach().float().cpu())
        finally:
            model.train(was_training)

        fields = {name: torch.cat(values).numpy() for name, values in chunks.items()}
        depth = next(iter(fields.values())).shape[-1]
        map_fields = {}
        for name, values in fields.items():
            image = np.full((*valid.shape, depth), np.nan, dtype=np.float32)
            image[valid.numpy()] = values
            map_fields[name] = image
        map_x_mm, map_y_mm = self._map_coordinates(raster, rows, columns)
        return {
            "log_tau500": model.log_tau500.detach().float().cpu().numpy(),
            "profile_fields": fields,
            "map_fields": map_fields,
            "rows": rows,
            "columns": columns,
            "map_x_mm": map_x_mm,
            "map_y_mm": map_y_mm,
            "slit_indices": np.asarray(
                raster.metadata.get("slit_indices", np.arange(height))
            )[rows],
            "scan_indices": np.asarray(
                raster.metadata.get("scan_indices", np.arange(width))
            )[columns],
        }

    def _tau_mapping_figure(
        self,
        evaluated: dict,
        depth_indices: list[int],
        label: str,
        *,
        yz_evaluated: dict | None = None,
    ) -> Figure:
        """Show the learned optical-depth geometry and its signed metric.

        Heights use the fixed-FOV gauge ``mean[z(q=0)]=0``. The metric
        is the automatic derivative ``-dz/dlog10(tau500)`` rather than a
        finite-difference estimate, so wrong-sign regions remain visible while
        the optical-depth mapping physics objective is converging.
        """

        depth_indices = sorted(
            depth_indices,
            key=lambda index: float(evaluated["log_tau500"][index]),
            reverse=True,
        )
        rows = 2 if yz_evaluated is not None else 1
        figure = Figure(
            figsize=(11.0, 8.0 if yz_evaluated is not None else 4.2),
            constrained_layout=True,
        )
        FigureCanvasAgg(figure)
        axes = np.asarray(figure.subplots(rows, 2, squeeze=False))
        q = evaluated["log_tau500"]
        height_profiles = evaluated["profile_fields"]["geometric_height"]
        metric_profiles = evaluated["profile_fields"]["height_metric"]

        for axis, profiles, ylabel, logarithmic in (
            (
                axes[0, 0],
                height_profiles,
                r"$z$ [Mm; $\langle z(q=0)\rangle=0$]",
                False,
            ),
            (
                axes[0, 1],
                metric_profiles,
                r"$-dz/d\log_{10}\tau_{500}$ [km dex$^{-1}$]",
                True,
            ),
        ):
            percentiles = np.nanpercentile(
                profiles, (2.0, 16.0, 50.0, 84.0, 98.0), axis=0
            )
            axis.fill_between(
                q, percentiles[0], percentiles[4], color="tab:blue", alpha=0.10
            )
            axis.fill_between(
                q, percentiles[1], percentiles[3], color="tab:blue", alpha=0.24
            )
            axis.plot(
                q,
                percentiles[2],
                color="tab:blue",
                linewidth=1.5,
                label="spatial median",
            )
            for depth_index in depth_indices:
                axis.axvline(q[depth_index], color="0.55", linewidth=0.6, alpha=0.55)
            if logarithmic:
                finite = np.abs(profiles[np.isfinite(profiles)])
                linear_width = (
                    max(float(np.nanpercentile(finite, 10.0)), 1.0)
                    if finite.size
                    else 1.0
                )
                axis.set_yscale("symlog", linthresh=linear_width)
            axis.set_xlabel(r"$\log_{10}\tau_{500}$ [dimensionless]")
            # Display optical depth in the same high-to-low order used by the
            # stratification map rows rather than Matplotlib's numeric order.
            axis.invert_xaxis()
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[0, 0].set_title("learned geometric-height mapping")
        axes[0, 1].set_title("learned signed mapping metric")

        if yz_evaluated is not None:
            yz_q = yz_evaluated["log_tau500"]
            y_mm = yz_evaluated["map_y_mm"][:, 0]
            height_mm = yz_evaluated["map_fields"]["geometric_height"][:, 0, :]
            metric = yz_evaluated["map_fields"]["height_metric"][:, 0, :]
            y_grid_mm = np.broadcast_to(y_mm[:, None], height_mm.shape)
            y_corners_mm = self._curvilinear_cell_corners(y_grid_mm)
            height_corners_mm = self._curvilinear_cell_corners(height_mm)
            log_tau_field = np.broadcast_to(yz_q[None, :], height_mm.shape)
            slice_specs = (
                (
                    log_tau_field,
                    r"optical depth in learned geometry",
                    "plasma",
                    (float(yz_q.min()), float(yz_q.max())),
                    r"$\log_{10}\tau_{500}$ [dimensionless]",
                ),
                (
                    metric,
                    "signed height metric in learned geometry",
                    "RdBu_r",
                    self._limits(metric, signed=True),
                    r"$-dz/d\log_{10}\tau_{500}$ [km dex$^{-1}$]",
                ),
            )
            contour_levels = sorted({float(q[index]) for index in depth_indices})
            for column, (values, title, cmap, limits, colorbar_label) in enumerate(
                slice_specs
            ):
                axis = axes[1, column]
                invalid = ~np.isfinite(height_mm) | ~np.isfinite(values)
                image = axis.pcolormesh(
                    y_corners_mm,
                    height_corners_mm,
                    np.ma.masked_where(invalid, values),
                    shading="flat",
                    cmap=cmap,
                    vmin=limits[0],
                    vmax=limits[1],
                    rasterized=True,
                )
                axis.contour(
                    y_grid_mm,
                    height_mm,
                    np.ma.masked_where(invalid, log_tau_field),
                    levels=contour_levels,
                    colors="white",
                    linewidths=0.55,
                    linestyles="--",
                    alpha=0.7,
                )
                axis.set_title(title, fontsize=9)
                axis.set_xlabel("Solar-Y [Mm]")
                if column == 0:
                    axis.set_ylabel(r"geometric height $z$ [Mm]")
                # Keep the horizontal and vertical display scales independent;
                # their physical extents differ strongly in a stratified slice.
                axis.set_aspect("auto")
                self._add_shared_colorbar(
                    figure,
                    image,
                    axis,
                    colorbar_label,
                    shared_by="column",
                )
            scan_index = int(yz_evaluated["slice_scan_index"])
            axes[1, 0].text(
                0.01,
                0.99,
                f"detector x/scan index {scan_index}",
                transform=axes[1, 0].transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
            )
        figure.suptitle(f"Optical-depth to geometric-height mapping — {label}")
        return figure

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
            vmin, vmax = style.get(
                "limits", cls._limits(values, signed=style["signed"])
            )
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
        *,
        shared_by: str = "column",
    ):
        """Add one labelled colorbar for a common panel row or column."""

        if shared_by not in {"column", "row"}:
            raise ValueError("shared_by must be either 'column' or 'row'.")
        location = "top" if shared_by == "column" else "right"
        colorbar = figure.colorbar(
            mappable,
            ax=np.asarray(axes).ravel().tolist(),
            location=location,
            orientation="horizontal" if location == "top" else "vertical",
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

    def _field_panel_figure(
        self,
        evaluated: dict,
        depth_indices: list[int],
        label: str,
        *,
        field_names: tuple[str, ...],
        title: str,
    ) -> Figure:
        """Show selected fields along x and high-to-low optical depth along y."""

        depth_indices = sorted(
            depth_indices,
            key=lambda index: float(evaluated["log_tau500"][index]),
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
        map_x_mm = evaluated["map_x_mm"]
        map_y_mm = evaluated["map_y_mm"]
        shared_norms = {
            name: self._field_norm(evaluated["map_fields"][name], style)
            for name, style in ((name, _FIELD_STYLES[name]) for name in field_names)
        }
        column_images = [None] * len(field_names)
        for row, depth_index in enumerate(depth_indices):
            log_tau = float(evaluated["log_tau500"][depth_index])
            for column, name in enumerate(field_names):
                style = _FIELD_STYLES[name]
                axis = axes[row, column]
                values = evaluated["map_fields"][name][..., depth_index]
                image = axis.pcolormesh(
                    map_x_mm,
                    map_y_mm,
                    values,
                    shading="nearest",
                    cmap=style["cmap"],
                    norm=shared_norms[name],
                    rasterized=True,
                )
                column_images[column] = image
                axis.set_aspect("equal", adjustable="box")
                if column == 0:
                    axis.annotate(
                        rf"$\log_{{10}}\tau_{{500}}={log_tau:.2f}$",
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
        figure.supxlabel("Solar-X [Mm]")
        figure.supylabel("Solar-Y [Mm]")
        for column, (name, image) in enumerate(
            zip(field_names, column_images, strict=True)
        ):
            self._add_shared_colorbar(
                figure,
                image,
                axes[:, column],
                _FIELD_STYLES[name]["label"],
                shared_by="column",
            )
        figure.suptitle(f"{title} — {label}")
        return figure

    def _yz_slice_column(self, raster) -> int:
        """Resolve the configured detector scan index to a raster column."""

        scan_indices = np.asarray(
            raster.metadata.get("scan_indices", np.arange(raster.spatial_shape[1]))
        )
        matches = np.flatnonzero(scan_indices == self.yz_slice_scan_index)
        if matches.size != 1:
            available = (int(scan_indices.min()), int(scan_indices.max()))
            raise ValueError(
                f"Configured YZ scan_index={self.yz_slice_scan_index} is not present "
                f"in this raster; available detector scan range is {available}."
            )
        return int(matches[0])

    def _evaluate_yz_slice(self, pl_module, raster) -> dict:
        """Evaluate every slit position through depth at one detector x index."""

        column = self._yz_slice_column(raster)
        evaluated = self._evaluate(
            pl_module,
            raster,
            rows=np.arange(raster.spatial_shape[0], dtype=np.int64),
            columns=np.asarray((column,), dtype=np.int64),
        )
        evaluated["slice_scan_index"] = self.yz_slice_scan_index
        return evaluated

    @staticmethod
    def _curvilinear_cell_corners(centers: np.ndarray) -> np.ndarray:
        """Construct finite cell corners from a two-dimensional centre grid."""

        centers = np.asarray(centers, dtype=np.float64)
        if centers.ndim != 2 or min(centers.shape) < 2:
            raise ValueError("A YZ section needs at least two slit and depth samples.")
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
                        "A YZ section needs at least two valid slit samples at "
                        "every optical-depth level."
                    )
                centers[:, depth_index] = np.interp(
                    row, row[valid], centers[valid, depth_index]
                )
        padded = np.empty(
            (centers.shape[0] + 2, centers.shape[1] + 2), dtype=np.float64
        )
        padded[1:-1, 1:-1] = centers
        padded[0, 1:-1] = 2.0 * centers[0] - centers[1]
        padded[-1, 1:-1] = 2.0 * centers[-1] - centers[-2]
        padded[:, 0] = 2.0 * padded[:, 1] - padded[:, 2]
        padded[:, -1] = 2.0 * padded[:, -2] - padded[:, -3]
        return 0.25 * (
            padded[:-1, :-1] + padded[1:, :-1] + padded[:-1, 1:] + padded[1:, 1:]
        )

    def _yz_field_panel_figure(
        self,
        evaluated: dict,
        label: str,
        *,
        field_names: tuple[str, ...],
        title: str,
        vertical_coordinate: str = "geometric_height",
    ) -> Figure:
        """Plot fields on a Y-depth section in tau or geometric height."""

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
        y_mm = evaluated["map_y_mm"][:, 0]
        if vertical_coordinate == "geometric_height":
            if "geometric_height" not in evaluated["map_fields"]:
                raise ValueError(
                    "A geometric-height YZ panel requires a learned height mapping."
                )
            vertical = evaluated["map_fields"]["geometric_height"][:, 0, :]
            vertical_label = r"geometric height $z$ [Mm]"
            depth_plane = "Y-Z"
            invert_vertical_axis = False
        elif vertical_coordinate == "log_tau500":
            vertical = np.broadcast_to(
                evaluated["log_tau500"][None, :],
                (y_mm.shape[0], evaluated["log_tau500"].size),
            )
            vertical_label = r"$\log_{10}\tau_{500}$ [dimensionless]"
            depth_plane = r"Y-$\log\tau_{500}$"
            invert_vertical_axis = True
        else:
            raise ValueError(
                "vertical_coordinate must be 'geometric_height' or 'log_tau500'."
            )
        y_grid_mm = np.broadcast_to(y_mm[:, None], vertical.shape)
        y_corners_mm = self._curvilinear_cell_corners(y_grid_mm)
        vertical_corners = self._curvilinear_cell_corners(vertical)
        solar_x_mm = evaluated["map_x_mm"][:, 0]

        images = []
        contour_sets = []
        log_tau_field = np.broadcast_to(
            evaluated["log_tau500"][None, :], vertical.shape
        )
        contour_levels = sorted(
            {
                float(evaluated["log_tau500"][index])
                for index in self._depth_indices(evaluated["log_tau500"])
            }
        )
        for axis, name in zip(axes, field_names, strict=True):
            style = _FIELD_STYLES[name]
            values = evaluated["map_fields"][name][:, 0, :]
            image = axis.pcolormesh(
                y_corners_mm,
                vertical_corners,
                np.ma.masked_invalid(values),
                shading="flat",
                cmap=style["cmap"],
                norm=self._field_norm(values, style),
                rasterized=True,
            )
            images.append(image)
            if vertical_coordinate == "geometric_height":
                invalid = ~np.isfinite(vertical) | ~np.isfinite(values)
                contours = axis.contour(
                    y_grid_mm,
                    vertical,
                    np.ma.masked_where(invalid, log_tau_field),
                    levels=contour_levels,
                    colors="white",
                    linewidths=0.65,
                    linestyles="--",
                    alpha=0.85,
                )
                contour_sets.append(contours)
            axis.set_aspect("auto")
            axis.label_outer()
        if invert_vertical_axis:
            # The panels share their y-axis, so invert it exactly once. Inverting
            # every panel toggles the shared axis repeatedly and cancels out for
            # the usual even number of validation fields.
            axes[0].invert_yaxis()
        figure.supxlabel("Solar-Y [Mm]")
        figure.supylabel(vertical_label)
        for axis, name, image in zip(axes, field_names, images, strict=True):
            self._add_shared_colorbar(
                figure,
                image,
                axis,
                _FIELD_STYLES[name]["label"],
                shared_by="column",
            )
        if contour_sets:
            axes[0].clabel(
                contour_sets[0],
                fmt=lambda value: rf"$\log\tau={value:g}$",
                inline=True,
                fontsize=7,
            )

        scan_index = int(evaluated["slice_scan_index"])
        finite_x = solar_x_mm[np.isfinite(solar_x_mm)]
        x_description = (
            f"Solar-X={float(np.mean(finite_x)):.3f} Mm"
            if finite_x.size
            else "Solar-X unavailable"
        )
        figure.suptitle(
            f"{title} {depth_plane} slice — detector x/scan index {scan_index}, "
            f"{x_description} — {label}"
        )
        return figure

    def _stokes_validation_figure(
        self,
        outputs,
        raster,
        wavelength_angstrom: torch.Tensor,
        label: str,
        *,
        rows: np.ndarray | None = None,
        columns: np.ndarray | None = None,
        exclusion_windows_angstrom=(),
    ) -> Figure:
        """Show loss-space Stokes predictions and references without rescaling."""

        wavelength = wavelength_angstrom.detach().double().cpu().numpy()
        if not isinstance(outputs, dict):
            raise TypeError(
                "Stokes visualization requires the streamed callback payload."
            )
        prediction = outputs["stokes_pred"].numpy()
        reference = outputs["stokes_reference"].numpy()
        pixel_index = outputs["pixel_index"].numpy()
        integrated_prediction = outputs["integrated_prediction"].numpy()
        integrated_reference = outputs["integrated_reference"].numpy()
        sampled_rows = (
            np.unique(pixel_index[:, 0]) if rows is None else np.asarray(rows)
        )
        sampled_columns = (
            np.unique(pixel_index[:, 1]) if columns is None else np.asarray(columns)
        )
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
            (sampled_rows.size, sampled_columns.size, 4), np.nan, dtype=np.float64
        )
        reference_maps = np.full_like(prediction_maps, np.nan)
        prediction_maps[compact_rows, compact_columns] = integrated_prediction
        reference_maps[compact_rows, compact_columns] = integrated_reference
        map_x_mm, map_y_mm = self._map_coordinates(
            raster, sampled_rows, sampled_columns
        )

        # Explicit subfigure spacing is more stable here than constrained layout:
        # the equal-aspect map panels otherwise squeeze the profile row nearly
        # flat when horizontal colorbars are attached above them.
        figure = Figure(figsize=(16.0, 14.0))
        FigureCanvasAgg(figure)
        map_figure, profile_figure, scatter_figure = figure.subfigures(
            3, 1, height_ratios=(2.0, 1.0, 1.0)
        )
        map_axes = map_figure.subplots(
            2, 4, squeeze=False, sharex=True, sharey=True
        )
        profile_axes = np.asarray(
            profile_figure.subplots(1, 4, squeeze=False, sharex=True)
        )[0]
        scatter_axes = np.asarray(
            scatter_figure.subplots(1, 4, squeeze=False)
        )[0]
        map_figure.subplots_adjust(
            left=0.07, right=0.99, bottom=0.12, top=0.88, wspace=0.16, hspace=0.30
        )
        profile_figure.subplots_adjust(
            left=0.07, right=0.99, bottom=0.22, top=0.82, wspace=0.16
        )
        scatter_figure.subplots_adjust(
            left=0.07, right=0.99, bottom=0.22, top=0.82, wspace=0.16
        )
        map_figure.supxlabel("Solar-X [Mm]")
        map_figure.supylabel("Solar-Y [Mm]")
        map_figure.text(
            0.985, 0.66, "reference", rotation=-90, ha="center", va="center"
        )
        map_figure.text(
            0.985, 0.27, "predicted", rotation=-90, ha="center", va="center"
        )
        profile_figure.supxlabel(r"air wavelength [$\AA$]")
        profile_figure.supylabel(r"Stokes component / $I_{c,\,atlas}(\mu=1)$")
        scatter_figure.supxlabel(
            r"reference integral in atlas-$I_c$ units [$\AA$]"
        )
        scatter_figure.supylabel(
            r"predicted integral in atlas-$I_c$ units [$\AA$]"
        )
        for component, name in enumerate(("I", "Q", "U", "V")):
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
                image = axis.pcolormesh(
                    map_x_mm,
                    map_y_mm,
                    values,
                    shading="nearest",
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
                rf"fit-window $|{name}|$ integral [$\AA$]",
                shared_by="column",
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
            profile_axis.axvline(6301.5008, color="tab:blue", alpha=0.25, linewidth=0.8)
            profile_axis.axvline(6302.4932, color="tab:blue", alpha=0.25, linewidth=0.8)
            for lower, upper in exclusion_windows_angstrom:
                profile_axis.axvspan(
                    lower,
                    upper,
                    color="0.65",
                    alpha=0.22,
                    linewidth=0.0,
                    label=("excluded from objective" if component == 0 else None),
                )
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
            scatter_axis.set_title(f"integrated $|{name}|$ (all validation pixels)")
            scatter_axis.grid(alpha=0.18)
        profile_axes[0].legend(loc="best", fontsize=8)
        figure.suptitle(
            "Stokes validation — loss-space profiles with no plot-time "
            "normalization; absolute calibration in fixed disk-center "
            f"atlas-$I_c$ units — {label}"
        )
        return figure

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

    def render(
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

        if rows is None or columns is None:
            rows, columns = self._display_indices(trainer, raster)
        evaluated = self._evaluate(pl_module, raster, rows=rows, columns=columns)
        paths = []
        depth_grid = evaluated["log_tau500"]
        selected_indices = self._depth_indices(depth_grid)
        thermodynamic_fields = tuple(
            name for name in _THERMODYNAMIC_FIELDS
            if name in evaluated["map_fields"]
        )
        panels = (
            (
                thermodynamic_fields,
                "Depth-stratified LTE thermodynamic parameters",
                "parameters",
                "Parameters",
            ),
            (
                _MAGNETIC_FIELDS,
                "Depth-stratified magnetic field",
                "magnetic_field",
                "Magnetic field",
            ),
            (
                _VELOCITY_FIELDS,
                "Depth-stratified velocity field",
                "velocity",
                "Velocity",
            ),
        )
        for field_names, title, filename_suffix, log_key in panels:
            paths.append(
                self._save_figure(
                    trainer,
                    self._field_panel_figure(
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
        yz_evaluated = (
            self._evaluate_yz_slice(pl_module, raster)
            if self.yz_slice_enabled
            else None
        )
        if "geometric_height" in evaluated["map_fields"]:
            paths.append(
                self._save_figure(
                    trainer,
                    self._tau_mapping_figure(
                        evaluated,
                        selected_indices,
                        label,
                        yz_evaluated=yz_evaluated,
                    ),
                    f"{label}_tau_mapping.png",
                    "Tau mapping",
                )
            )
        if self.yz_slice_enabled:
            yz_panels = (
                (
                    _YZ_THERMODYNAMIC_FIELDS,
                    "LTE thermodynamic parameters",
                    "yz_parameters",
                    "Parameters",
                ),
                (
                    _MAGNETIC_FIELDS,
                    "Magnetic field",
                    "yz_magnetic_field",
                    "Magnetic field",
                ),
                (
                    _VELOCITY_FIELDS,
                    "Velocity field",
                    "yz_velocity",
                    "Velocity",
                ),
            )
            for field_names, title, filename_suffix, log_key in yz_panels:
                paths.append(
                    self._save_figure(
                        trainer,
                        self._yz_field_panel_figure(
                            yz_evaluated,
                            label,
                            field_names=field_names,
                            title=title,
                            vertical_coordinate="log_tau500",
                        ),
                        f"{label}_y_tau_{filename_suffix.removeprefix('yz_')}.png",
                        f"{log_key} Y-tau",
                    )
                )
            if "geometric_height" not in yz_evaluated["map_fields"]:
                return paths
            for field_names, title, filename_suffix, log_key in yz_panels:
                paths.append(
                    self._save_figure(
                        trainer,
                        self._yz_field_panel_figure(
                            yz_evaluated,
                            label,
                            field_names=field_names,
                            title=title,
                            vertical_coordinate="geometric_height",
                        ),
                        f"{label}_{filename_suffix}.png",
                        f"{log_key} YZ",
                    )
                )
        return paths

    @staticmethod
    def _raster(trainer):
        data_module = getattr(trainer, "datamodule", None)
        raster = getattr(data_module, "raster", None)
        if raster is None:
            raise RuntimeError(
                "LTEAtmosphereVisualizationCallback requires a Hinode data module "
                "whose setup() method has populated raster."
            )
        return raster

    def on_fit_start(self, trainer, pl_module) -> None:
        if self.include_initial and bool(getattr(trainer, "is_global_zero", True)):
            step = int(getattr(trainer, "global_step", 0))
            label = "initial" if step == 0 else f"resume_step_{step:08d}"
            self.render(trainer, pl_module, self._raster(trainer), label=label)
            self._last_rendered_step = step

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        del pl_module
        epoch = int(getattr(trainer, "current_epoch", 0))
        self._collect_validation_epoch = trainer is None or (
            not bool(getattr(trainer, "sanity_checking", False))
            and (epoch + 1) % self.every_n_epochs == 0
        )
        self._validation_outputs = []
        self._profile_reservoir = None

    def _update_profile_reservoir(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        pixel_index: torch.Tensor,
    ) -> None:
        """Keep a deterministic, spatially mixed bounded profile sample."""

        pixel_index = pixel_index.detach().to(device="cpu", dtype=torch.long)
        candidate = {
            "stokes_pred": prediction.detach().float().cpu(),
            "stokes_reference": reference.detach().float().cpu(),
            "priority": (
                (pixel_index[:, 0] * 73_856_093) ^ (pixel_index[:, 1] * 19_349_663)
            ).bitwise_and(0x7FFF_FFFF),
        }
        if self._profile_reservoir is not None:
            candidate = {
                key: torch.cat((self._profile_reservoir[key], value), dim=0)
                for key, value in candidate.items()
            }
        count = candidate["priority"].numel()
        if count > self.max_profile_samples:
            selected = torch.topk(
                candidate["priority"],
                self.max_profile_samples,
                largest=False,
                sorted=False,
            ).indices
            candidate = {
                key: value.index_select(0, selected) for key, value in candidate.items()
            }
        self._profile_reservoir = candidate

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        del trainer, batch, batch_idx, dataloader_idx
        if not self._collect_validation_epoch:
            return
        if not isinstance(outputs, dict):
            return
        required = ("stokes_pred", "stokes_reference", "pixel_index")
        if all(name in outputs for name in required):
            prediction = outputs["stokes_pred"]
            reference = outputs["stokes_reference"]
            pixel_index = outputs["pixel_index"]
            wavelength = pl_module.wavelength_angstrom.to(prediction)
            delta_wavelength = wavelength[1:] - wavelength[:-1]
            spectral_weights = pl_module.wavelength_weights.to(prediction)
            segment_weights = torch.minimum(spectral_weights[1:], spectral_weights[:-1])
            weighted_delta_wavelength = delta_wavelength * segment_weights
            integrated_prediction = (
                0.5
                * (prediction[..., 1:].abs() + prediction[..., :-1].abs())
                * weighted_delta_wavelength
            ).sum(dim=-1)
            integrated_reference = (
                0.5
                * (reference[..., 1:].abs() + reference[..., :-1].abs())
                * weighted_delta_wavelength
            ).sum(dim=-1)
            self._validation_outputs.append(
                {
                    "integrated_prediction": integrated_prediction.detach()
                    .float()
                    .cpu(),
                    "integrated_reference": integrated_reference.detach().float().cpu(),
                    "pixel_index": pixel_index.detach().to(
                        device="cpu", dtype=torch.long
                    ),
                }
            )
            self._update_profile_reservoir(prediction, reference, pixel_index)

    def _local_validation_payload(self) -> dict[str, torch.Tensor] | None:
        if not self._validation_outputs or self._profile_reservoir is None:
            return None
        payload = {
            key: torch.cat([item[key] for item in self._validation_outputs], dim=0)
            for key in (
                "integrated_prediction",
                "integrated_reference",
                "pixel_index",
            )
        }
        payload.update(self._profile_reservoir)
        return payload

    def _merge_validation_payloads(self, payloads) -> dict[str, torch.Tensor] | None:
        payloads = [payload for payload in payloads if payload is not None]
        if not payloads:
            return None
        merged = {
            key: torch.cat([payload[key] for payload in payloads], dim=0)
            for key in (
                "integrated_prediction",
                "integrated_reference",
                "pixel_index",
                "stokes_pred",
                "stokes_reference",
                "priority",
            )
        }
        if merged["priority"].numel() > self.max_profile_samples:
            selected = torch.topk(
                merged["priority"],
                self.max_profile_samples,
                largest=False,
                sorted=False,
            ).indices
            for key in (
                "stokes_pred",
                "stokes_reference",
                "priority",
            ):
                merged[key] = merged[key].index_select(0, selected)
        return merged

    def _gather_validation_outputs(self) -> dict[str, torch.Tensor] | None:
        payload = self._local_validation_payload()
        if not (dist.is_available() and dist.is_initialized()):
            return payload
        rank = dist.get_rank()
        gathered = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
        if rank != 0:
            return None
        return self._merge_validation_payloads(gathered)

    def on_validation_end(self, trainer, pl_module) -> None:
        """Publish validation artifacts after Lightning has finalized the loop."""

        if bool(getattr(trainer, "sanity_checking", False)):
            return
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        outputs = self._gather_validation_outputs()
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        label = f"epoch_{epoch + 1:04d}"
        raster = self._raster(trainer)
        rows, columns = self._display_indices(trainer, raster)
        if outputs is None:
            raise RuntimeError(
                "Scheduled LTE validation produced no Stokes visualization payload; "
                "the comparison plot would otherwise be silently omitted."
            )
        self._save_figure(
            trainer,
            self._stokes_validation_figure(
                outputs,
                raster,
                pl_module.wavelength_angstrom,
                label,
                rows=rows,
                columns=columns,
                exclusion_windows_angstrom=(
                    pl_module.wavelength_exclude_windows_angstrom
                ),
            ),
            f"{label}_stokes_validation.png",
            "Validation/Stokes comparison",
        )
        self.render(
            trainer,
            pl_module,
            raster,
            label=label,
            rows=rows,
            columns=columns,
        )
        self._last_rendered_step = int(getattr(trainer, "global_step", 0))

    def on_fit_end(self, trainer, pl_module) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        step = int(getattr(trainer, "global_step", 0))
        if self._last_rendered_step == step:
            return
        completed_epochs = max(int(trainer.current_epoch), 1)
        self.render(
            trainer,
            pl_module,
            self._raster(trainer),
            label=f"final_epoch_{completed_epochs:04d}",
        )
        self._last_rendered_step = step


__all__ = ["LTEAtmosphereVisualizationCallback"]
