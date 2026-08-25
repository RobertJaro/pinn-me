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

from pme.coordinates import (
    cartesian_to_spherical,
    project_cartesian_to_spherical,
    spherical_to_cartesian,
)


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
    "v_magnitude": {
        "label": r"$|v|$ [km s$^{-1}$]",
        "cmap": "cividis",
        "signed": False,
    },
    "b_r": {"label": r"$B_r$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_theta": {"label": r"$B_\theta$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_phi": {"label": r"$B_\phi$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_q": {"label": r"$B_{+Q}$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_u": {"label": r"$B_{+U}$ [G]", "cmap": "RdBu_r", "signed": True},
    "b_toward": {"label": r"$B_{\rm toward}$ [G]", "cmap": "RdBu_r", "signed": True},
    "v_q": {"label": r"$v_{+Q}$ [km s$^{-1}$]", "cmap": "seismic", "signed": True},
    "v_u": {"label": r"$v_{+U}$ [km s$^{-1}$]", "cmap": "seismic", "signed": True},
    "v_toward": {"label": r"$v_{\rm toward}$ [km s$^{-1}$]", "cmap": "seismic", "signed": True},
    "v_rotation_toward": {
        "label": r"$v_{\rm rot,toward}$ [km s$^{-1}$]",
        "cmap": "seismic",
        "signed": True,
    },
    "v_inertial_magnitude": {
        "label": r"$|v_{\rm inertial}|$ [km s$^{-1}$]",
        "cmap": "cividis",
        "signed": False,
    },
    "b_magnitude": {"label": r"$|B|$ [G]", "cmap": "cividis", "signed": False},
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
    "microturbulence",
)
_MAGNETIC_FIELDS = ("b_r", "b_theta", "b_phi")
_VELOCITY_FIELDS = ("v_r", "v_theta", "v_phi")
_MERIDIONAL_THERMODYNAMIC_FIELDS = (
    "temperature",
    "density",
    "pressure",
    "microturbulence",
)


class LTEAtmosphereVisualizationCallback(Callback):
    """Render inferred LTE atmosphere profiles and optical-depth maps.

    The callback evaluates the atmosphere network plus the pinned STiC
    thermodynamic lookup so density is the same quantity used by geometric
    MHS. It does not repeat polarized synthesis. Optical depth uses a bounded
    subsample of observed rays. Atmosphere panels instead evaluate only explicit
    longitude-latitude shell layers and a constant-longitude radial plane in
    physical Carrington space, at an independently configured resolution.
    Stokes maps retain the observer-facing Carrington chart supplied with the raster.
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
        ray_sampling: Mapping | None = None,
        slice_sampling: Mapping | None = None,
        dpi: int = 180,
        include_initial: bool = True,
        meridional_slice: Mapping | None = None,
    ):
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive.")
        ray_config = dict(ray_sampling or {})
        self.ray_evaluation_batch_size = int(ray_config.pop("batch_size", 8192))
        self.max_ray_pixels = int(ray_config.pop("max_pixels", 65_536))
        self.max_profile_samples = int(
            ray_config.pop("max_profile_samples", 4_096)
        )
        if ray_config:
            raise TypeError(f"Unknown ray-sampling options: {sorted(ray_config)}")
        slice_config = dict(slice_sampling or {})
        self.slice_longitude_points = int(
            slice_config.pop("longitude_points", 256)
        )
        self.slice_latitude_points = int(slice_config.pop("latitude_points", 256))
        self.slice_radial_points = int(slice_config.pop("radial_points", 192))
        self.slice_layer_count = int(slice_config.pop("layer_count", 4))
        self.slice_evaluation_batch_size = int(slice_config.pop("batch_size", 8192))
        if slice_config:
            raise TypeError(f"Unknown slice-sampling options: {sorted(slice_config)}")
        positive_counts = {
            "ray_sampling.batch_size": self.ray_evaluation_batch_size,
            "ray_sampling.max_pixels": self.max_ray_pixels,
            "ray_sampling.max_profile_samples": self.max_profile_samples,
            "slice_sampling.longitude_points": self.slice_longitude_points,
            "slice_sampling.latitude_points": self.slice_latitude_points,
            "slice_sampling.radial_points": self.slice_radial_points,
            "slice_sampling.layer_count": self.slice_layer_count,
            "slice_sampling.batch_size": self.slice_evaluation_batch_size,
        }
        invalid = [name for name, value in positive_counts.items() if value < 1]
        if invalid:
            raise ValueError(f"Visualization sample counts must be positive: {invalid}.")
        if min(self.slice_longitude_points, self.slice_latitude_points) < 2:
            raise ValueError("Physical shell slices require at least two angular points.")
        if self.slice_radial_points < 2:
            raise ValueError("A physical radial slice requires at least two radial points.")
        if self.slice_layer_count < 2:
            raise ValueError("Physical shell plots require at least two radial layers.")
        if dpi < 50:
            raise ValueError("dpi must be at least 50.")
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.every_n_epochs = int(every_n_epochs)
        self.dpi = int(dpi)
        self.include_initial = bool(include_initial)
        meridional_config = dict(meridional_slice or {})
        self.meridional_slice_enabled = bool(
            meridional_config.pop("enabled", False)
        )
        raw_longitude_deg = meridional_config.pop("longitude_deg", None)
        if meridional_config:
            raise TypeError(
                "Unknown meridional-slice options: "
                f"{sorted(meridional_config)}"
            )
        if self.meridional_slice_enabled and raw_longitude_deg is None:
            raise ValueError("An enabled meridional slice requires longitude_deg.")
        self.meridional_slice_longitude_deg = (
            None if raw_longitude_deg is None else float(raw_longitude_deg)
        )
        if self.meridional_slice_longitude_deg is not None and not math.isfinite(
            self.meridional_slice_longitude_deg
        ):
            raise ValueError("Meridional-slice longitude_deg must be finite.")
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
            rows, columns = self._subsample_indices(
                *raster.spatial_shape, self.max_ray_pixels
            )
        else:
            selected = pixel_indices[torch.as_tensor(subset_indices, dtype=torch.long)]
            rows = torch.unique(selected[:, 0], sorted=True).cpu().numpy()
            columns = torch.unique(selected[:, 1], sorted=True).cpu().numpy()
            stride = max(
                1,
                int(math.ceil(math.sqrt(rows.size * columns.size / self.max_ray_pixels))),
            )
            rows = rows[::stride]
            columns = columns[::stride]
            while rows.size * columns.size > self.max_ray_pixels:
                stride += 1
                all_rows = torch.unique(selected[:, 0], sorted=True).cpu().numpy()
                all_columns = torch.unique(selected[:, 1], sorted=True).cpu().numpy()
                rows = all_rows[::stride]
                columns = all_columns[::stride]
        return rows, columns

    @staticmethod
    def _map_coordinates(
        raster,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the shared 2-D Carrington-chart centre grids in Mm."""

        coords = raster.coords[rows][:, columns].detach().float().cpu().numpy()
        if coords.shape[-1] != 3:
            raise ValueError(
                "Observation coordinates must end in [time_hours, x_mm, y_mm]."
            )
        x_mm = coords[..., 1]
        y_mm = coords[..., 2]
        if not np.isfinite(x_mm).all() or not np.isfinite(y_mm).all():
            raise ValueError("Observation map coordinates must be finite.")
        return x_mm, y_mm

    def _evaluate_ray_optical_depth(
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
            rows, columns = self._subsample_indices(height, width, self.max_ray_pixels)
        coords = raster.coords[rows][:, columns]
        ray_origin_m = raster.ray_origin_m[rows][:, columns]
        ray_direction = raster.ray_direction[rows][:, columns]
        surface_position_m = raster.surface_position_m[rows][:, columns]
        valid = raster.valid_mask[rows][:, columns]
        flat_coords = coords[valid].to(device=parameter.device, dtype=parameter.dtype)
        flat_ray_origin_m = ray_origin_m[valid].to(device=parameter.device)
        flat_ray_direction = ray_direction[valid].to(device=parameter.device)
        if flat_coords.numel() == 0:
            raise ValueError(
                "No valid observation pixels remain for atmosphere visualization."
            )

        was_training = model.training
        model.eval()
        chunks = {"tau500_ray": [], "geometric_height": []}
        traced_chart_xy = []
        traced_spherical = []
        sampled_depth_grid = None
        try:
            with torch.no_grad():
                for start in range(0, flat_coords.shape[0], self.ray_evaluation_batch_size):
                    batch_coords = flat_coords[
                        start : start + self.ray_evaluation_batch_size
                    ]
                    atmosphere, ray_trace = model.trace_rays(
                        batch_coords,
                        flat_ray_origin_m[start : start + self.ray_evaluation_batch_size],
                        flat_ray_direction[start : start + self.ray_evaluation_batch_size],
                        model.log_tau500,
                    )
                    if bool(getattr(pl_module, "coarse_to_fine_enabled", False)):
                        atmosphere, ray_trace = pl_module._refine_ray_sampling(
                            batch_coords,
                            flat_ray_direction[
                                start : start + self.ray_evaluation_batch_size
                            ],
                            atmosphere,
                            ray_trace,
                        )
                    sampled_depth_grid = atmosphere.log_tau500.detach().float().cpu()
                    traced_chart_xy.append(
                        ray_trace.chart_xy_mm.detach().float().cpu()
                    )
                    gas_pressure = atmosphere.gas_pressure
                    if gas_pressure is None:
                        raise RuntimeError(
                            "LTE atmosphere visualization requires predicted gas pressure."
                        )
                    alpha500 = (
                        pl_module.synthesizer.continuum_opacity.volume_extinction_at_5000(
                            atmosphere.temperature, gas_pressure
                        )
                    )
                    distance_interval_m = (
                        ray_trace.distance_m[..., 1:]
                        - ray_trace.distance_m[..., :-1]
                    )
                    tau_increment = 0.5 * (
                        alpha500[..., :-1] + alpha500[..., 1:]
                    ) * distance_interval_m
                    tau500_ray = torch.cat(
                        (
                            torch.zeros_like(alpha500[..., :1]),
                            torch.cumsum(tau_increment, dim=-1),
                        ),
                        dim=-1,
                    )
                    spherical = cartesian_to_spherical(ray_trace.position_m, torch)
                    scene_center_spherical = cartesian_to_spherical(
                        model.scene_basis[2].to(spherical), torch
                    )
                    longitude_reference = scene_center_spherical[2]
                    longitude = longitude_reference + torch.atan2(
                        torch.sin(spherical[..., 2] - longitude_reference),
                        torch.cos(spherical[..., 2] - longitude_reference),
                    )
                    traced_spherical.append(
                        torch.stack((spherical[..., 0], spherical[..., 1], longitude), dim=-1)
                        .detach().float().cpu()
                    )
                    chunks["tau500_ray"].append(
                        tau500_ray.detach().float().cpu()
                    )
                    chunks["geometric_height"].append(
                        (ray_trace.geometric_height_m / 1.0e6).detach().float().cpu()
                    )
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
        base_spherical = cartesian_to_spherical(surface_position_m, torch)
        scene_center_spherical = cartesian_to_spherical(
            model.scene_basis[2].to(base_spherical), torch
        )
        longitude_reference = scene_center_spherical[2]
        base_longitude = longitude_reference + torch.atan2(
            torch.sin(base_spherical[..., 2] - longitude_reference),
            torch.cos(base_spherical[..., 2] - longitude_reference),
        )
        map_longitude_deg = np.broadcast_to(
            torch.rad2deg(base_longitude).numpy()[..., None], (*valid.shape, depth)
        ).copy()
        map_latitude_deg = np.broadcast_to(
            (90.0 - torch.rad2deg(base_spherical[..., 1])).numpy()[..., None],
            (*valid.shape, depth),
        ).copy()
        if traced_chart_xy:
            chart = torch.cat(traced_chart_xy).numpy()
            # Invalid detector samples are masked in every field. Retain their
            # finite reference-surface chart coordinates solely so Matplotlib
            # can construct a complete curvilinear mesh around those holes.
            map_x_mm = np.broadcast_to(map_x_mm[..., None], (*valid.shape, depth)).copy()
            map_y_mm = np.broadcast_to(map_y_mm[..., None], (*valid.shape, depth)).copy()
            map_x_mm[valid.numpy()] = chart[..., 0]
            map_y_mm[valid.numpy()] = chart[..., 1]
        if traced_spherical:
            spherical = torch.cat(traced_spherical).numpy()
            map_longitude_deg[valid.numpy()] = np.rad2deg(spherical[..., 2])
            map_latitude_deg[valid.numpy()] = 90.0 - np.rad2deg(spherical[..., 1])
        shell_height_levels_m = np.nanmedian(
            fields["geometric_height"], axis=0
        ) * 1.0e6
        if sampled_depth_grid is None:
            raise RuntimeError("Ray optical-depth sampling produced no depth grid.")
        return {
            "log_tau500": sampled_depth_grid.numpy(),
            "solar_radius_m": float(model.solar_radius_m.detach().cpu()),
            "shell_height_levels_m": shell_height_levels_m,
            "profile_fields": fields,
            "map_fields": map_fields,
            "rows": rows,
            "columns": columns,
            "map_x_mm": map_x_mm,
            "map_y_mm": map_y_mm,
            "map_longitude_deg": map_longitude_deg,
            "map_latitude_deg": map_latitude_deg,
            "slit_indices": np.asarray(
                raster.metadata.get("slit_indices", np.arange(height))
            )[rows],
            "scan_indices": np.asarray(
                raster.metadata.get("scan_indices", np.arange(width))
            )[columns],
            "sampling": "subsampled observed rays for continuum optical depth only",
        }

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
            for name, style in ((name, _FIELD_STYLES[name]) for name in field_names)
        }
        column_images = [None] * len(field_names)
        for row, depth_index in enumerate(depth_indices):
            depth_value = float(depth_axis[depth_index])
            for column, name in enumerate(field_names):
                style = _FIELD_STYLES[name]
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
                _FIELD_STYLES[name]["label"],
                shared_by="column",
            )
        figure.suptitle(f"{title} — {label}")
        return figure

    def _tau_figure(self, evaluated: dict, label: str) -> Figure:
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
            radius_Rsun[profile_slice], lower, upper, color="tab:blue", alpha=0.25,
            label="validation rays: 16–84%",
        )
        profile_axis.plot(
            median, radius_Rsun[profile_slice], color="tab:blue", linewidth=1.6,
            label="validation-ray median",
        )
        profile_axis.plot(
            evaluated["log_tau500"][profile_slice],
            radius_Rsun[profile_slice],
            color="black", linestyle="--", linewidth=1.0,
            label="FALC radial shell labels",
        )
        profile_axis.axvline(0.0, color="0.45", linewidth=0.8, linestyle=":")
        profile_axis.set_xlabel(r"$\log_{10}\tau_{500}$")
        profile_axis.set_ylabel(r"radius $r/R_\odot$")
        profile_axis.set_title(r"derived $\tau_{500}=\int\alpha_{500}\,ds$")
        profile_axis.grid(alpha=0.2)
        profile_axis.legend(loc="best", fontsize=8)

        surface_index = int(np.argmin(np.abs(evaluated["shell_height_levels_m"])))
        surface_log_tau = evaluated["map_fields"]["tau500_ray"][..., surface_index]
        finite = surface_log_tau[np.isfinite(surface_log_tau) & (surface_log_tau > 0)]
        if finite.size:
            surface_log_tau = np.log10(np.clip(surface_log_tau, tiny, None))
            vmin, vmax = self._limits(surface_log_tau, signed=False)
        else:
            surface_log_tau = np.full_like(surface_log_tau, np.nan)
            vmin, vmax = -1.0, 1.0
        map_x = evaluated["map_longitude_deg"]
        map_y = evaluated["map_latitude_deg"]
        if map_x.ndim == 3:
            map_x = map_x[..., surface_index]
            map_y = map_y[..., surface_index]
        image = map_axis.pcolormesh(
            map_x, map_y, surface_log_tau, shading="nearest", cmap="viridis",
            norm=Normalize(vmin=vmin, vmax=vmax), rasterized=True,
        )
        map_axis.set_aspect("equal", adjustable="box")
        map_axis.set_xlabel("Carrington longitude [deg]")
        map_axis.set_ylabel("Carrington latitude [deg]")
        surface_radius_Rsun = radius_Rsun[surface_index]
        map_axis.set_title(
            rf"$\log_{{10}}\tau_{{500}}$ at $r={surface_radius_Rsun:.6f}\,R_\odot$"
        )
        colorbar = figure.colorbar(image, ax=map_axis, location="right", shrink=0.86)
        colorbar.set_label(r"$\log_{10}\tau_{500}$")
        figure.suptitle(f"Continuum optical-depth validation — {label}")
        return figure

    @staticmethod
    def _observed_spherical_bounds(model, raster) -> tuple[float, float, float, float]:
        """Return unwrapped longitude and latitude bounds of valid surface points."""

        surface = raster.surface_position_m[raster.valid_mask].detach().float().cpu()
        if surface.shape[0] < 2:
            raise ValueError("Physical slices require at least two valid surface points.")
        spherical = cartesian_to_spherical(surface, torch)
        center = cartesian_to_spherical(model.scene_basis[2].detach().float().cpu(), torch)
        longitude_center = center[2]
        longitude = longitude_center + torch.atan2(
            torch.sin(spherical[:, 2] - longitude_center),
            torch.cos(spherical[:, 2] - longitude_center),
        )
        latitude = 0.5 * math.pi - spherical[:, 1]
        bounds = (
            float(longitude.min()),
            float(longitude.max()),
            float(latitude.min()),
            float(latitude.max()),
        )
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("Observed spherical bounds must be finite.")
        if not bounds[1] > bounds[0] or not bounds[3] > bounds[2]:
            raise ValueError("Observed spherical bounds must span longitude and latitude.")
        return bounds

    def _evaluate_physical_positions(
        self, pl_module, position_m: torch.Tensor
    ) -> dict[str, np.ndarray]:
        """Evaluate plot fields at explicitly supplied Carrington positions."""

        model = pl_module.atmosphere_model
        parameter = next(model.parameters())
        flat_position = position_m.reshape(-1, 3).to(
            device=parameter.device, dtype=parameter.dtype
        )
        chunks = {
            name: []
            for name in (
                *_MERIDIONAL_THERMODYNAMIC_FIELDS,
                *_MAGNETIC_FIELDS,
                *_VELOCITY_FIELDS,
            )
        }
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, flat_position.shape[0], self.slice_evaluation_batch_size):
                    position = flat_position[
                        start : start + self.slice_evaluation_batch_size
                    ]
                    atmosphere = model.evaluate_position_points(position)
                    pressure = atmosphere["gas_pressure"]
                    density = pl_module.synthesizer.continuum_opacity.reference_mass_density(
                        atmosphere["temperature"], pressure
                    )
                    spherical = cartesian_to_spherical(position, torch)
                    magnetic_spherical = project_cartesian_to_spherical(
                        atmosphere["magnetic_field"], spherical, torch
                    )
                    velocity_spherical = project_cartesian_to_spherical(
                        atmosphere["velocity_field"], spherical, torch
                    ) / 1_000.0
                    values = {
                        "temperature": atmosphere["temperature"],
                        "density": density,
                        "pressure": pressure,
                        "microturbulence": atmosphere["microturbulence"] / 1_000.0,
                        "b_r": magnetic_spherical[..., 0],
                        "b_theta": magnetic_spherical[..., 1],
                        "b_phi": magnetic_spherical[..., 2],
                        "v_r": velocity_spherical[..., 0],
                        "v_theta": velocity_spherical[..., 1],
                        "v_phi": velocity_spherical[..., 2],
                    }
                    for name, value in values.items():
                        chunks[name].append(value.detach().float().cpu())
        finally:
            model.train(was_training)
        leading_shape = position_m.shape[:-1]
        return {
            name: torch.cat(values).numpy().reshape(leading_shape)
            for name, values in chunks.items()
        }

    def _evaluate_shell_layers(
        self, pl_module, raster, shell_height_levels_m: np.ndarray
    ) -> dict:
        """Evaluate only requested longitude-latitude layers in physical space."""

        model = pl_module.atmosphere_model
        lon_min, lon_max, lat_min, lat_max = self._observed_spherical_bounds(
            model, raster
        )
        longitude = torch.linspace(lon_min, lon_max, self.slice_longitude_points)
        latitude = torch.linspace(lat_min, lat_max, self.slice_latitude_points)
        latitude_grid, longitude_grid = torch.meshgrid(
            latitude, longitude, indexing="ij"
        )
        heights = torch.as_tensor(shell_height_levels_m, dtype=torch.float32)
        spherical = torch.stack(
            (
                (
                    model.solar_radius_m.detach().float().cpu()
                    + heights[:, None, None]
                ).expand(-1, self.slice_latitude_points, self.slice_longitude_points),
                (0.5 * math.pi - latitude_grid)[None].expand(heights.numel(), -1, -1),
                longitude_grid[None].expand(heights.numel(), -1, -1),
            ),
            dim=-1,
        )
        position = spherical_to_cartesian(spherical, torch).permute(1, 2, 0, 3)
        fields = self._evaluate_physical_positions(pl_module, position)
        map_longitude = np.broadcast_to(
            np.rad2deg(longitude_grid.numpy())[..., None], position.shape[:-1]
        ).copy()
        map_latitude = np.broadcast_to(
            np.rad2deg(latitude_grid.numpy())[..., None], position.shape[:-1]
        ).copy()
        result = {
            "solar_radius_m": float(model.solar_radius_m.detach().cpu()),
            "shell_height_levels_m": np.asarray(shell_height_levels_m),
            "map_fields": fields,
            "map_longitude_deg": map_longitude,
            "map_latitude_deg": map_latitude,
            "sampling": "explicit Carrington longitude-latitude shell layers",
        }
        if self.meridional_slice_enabled:
            result["slice_longitude_deg"] = math.degrees(
                self._meridional_slice_longitude(model)
            )
        return result

    def _meridional_slice_longitude(self, model) -> float:
        """Return the configured longitude on the scene-centred continuous branch."""

        requested = math.radians(self.meridional_slice_longitude_deg)
        center = float(
            cartesian_to_spherical(model.scene_basis[2].detach().float().cpu(), torch)[2]
        )
        return center + math.atan2(
            math.sin(requested - center), math.cos(requested - center)
        )

    def _evaluate_meridional_slice(self, pl_module, raster) -> dict:
        """Evaluate a constant-longitude radial plane in physical space."""

        model = pl_module.atmosphere_model
        longitude = self._meridional_slice_longitude(model)
        _, _, latitude_min, latitude_max = self._observed_spherical_bounds(model, raster)
        latitude = torch.linspace(
            latitude_min, latitude_max, self.slice_latitude_points
        )
        outer_height_m, inner_height_m = (
            float(value) * 1.0e6 for value in model.shell_height_bounds_Mm
        )
        height = torch.linspace(
            outer_height_m, inner_height_m, self.slice_radial_points
        )
        latitude_grid, height_grid = torch.meshgrid(latitude, height, indexing="ij")
        spherical = torch.stack(
            (
                model.solar_radius_m.detach().float().cpu() + height_grid,
                0.5 * math.pi - latitude_grid,
                torch.full_like(latitude_grid, longitude),
            ),
            dim=-1,
        )
        position = spherical_to_cartesian(spherical, torch)
        fields = self._evaluate_physical_positions(pl_module, position)
        shape = (self.slice_latitude_points, 1, self.slice_radial_points)
        return {
            "solar_radius_m": float(model.solar_radius_m.detach().cpu()),
            "map_fields": {
                **{name: values[:, None, :] for name, values in fields.items()},
                "geometric_height": (height_grid / 1.0e6).numpy()[:, None, :],
            },
            "map_latitude_deg": np.rad2deg(latitude_grid.numpy())[:, None, :],
            "map_longitude_deg": np.full(
                shape, math.degrees(longitude), dtype=np.float32
            ),
            "requested_longitude_deg": self.meridional_slice_longitude_deg,
            "slice_longitude_deg": math.degrees(longitude),
            "sampling": "explicit constant-Carrington-longitude radial plane",
        }

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

    def _meridional_field_panel_figure(
        self,
        evaluated: dict,
        label: str,
        *,
        field_names: tuple[str, ...],
        title: str,
        norm_evaluated: dict | None = None,
        reference_depth_indices: list[int] | None = None,
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
            evaluated["map_fields"]["geometric_height"][:, 0, :] * 1.0e6
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
            style = _FIELD_STYLES[name]
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
            for depth_index in reference_depth_indices or ():
                axis.plot(
                    latitude_grid_deg[:, depth_index],
                    vertical[:, depth_index],
                    color="black",
                    linewidth=0.6,
                    linestyle=":",
                    alpha=0.7,
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
                _FIELD_STYLES[name]["label"],
                shared_by="column",
            )

        finite_x = longitude_deg[np.isfinite(longitude_deg)]
        if finite_x.size:
            x_min, x_max = float(np.min(finite_x)), float(np.max(finite_x))
            x_description = f"Carrington longitude={0.5 * (x_min + x_max):.3f} deg"
        else:
            x_description = "Carrington longitude unavailable"
        figure.suptitle(
            f"{title} {depth_plane} slice — {x_description} — {label}"
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

        wavelength = wavelength_angstrom.detach().float().cpu().numpy()
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
            (sampled_rows.size, sampled_columns.size, 4), np.nan, dtype=np.float32
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
        map_figure.supxlabel("Carrington chart X [Mm]")
        map_figure.supylabel("Carrington chart Y [Mm]")
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
        ray_evaluated = self._evaluate_ray_optical_depth(
            pl_module, raster, rows=rows, columns=columns
        )
        paths = []
        outer_height_Mm, inner_height_Mm = pl_module.atmosphere_model.shell_height_bounds_Mm
        slice_heights = np.linspace(
            outer_height_Mm * 1.0e6,
            inner_height_Mm * 1.0e6,
            self.slice_layer_count,
            dtype=np.float32,
        )
        evaluated = self._evaluate_shell_layers(pl_module, raster, slice_heights)
        selected_indices = list(range(len(slice_heights)))
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
                r"Depth-stratified magnetic field components",
                "magnetic_field",
                "Magnetic field",
            ),
            (
                _VELOCITY_FIELDS,
                r"Depth-stratified velocity components",
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
        paths.append(
            self._save_figure(
                trainer,
                self._tau_figure(ray_evaluated, label),
                f"{label}_tau500.png",
                "Optical depth",
            )
        )
        meridional_evaluated = (
            self._evaluate_meridional_slice(pl_module, raster)
            if self.meridional_slice_enabled
            else None
        )
        if self.meridional_slice_enabled:
            meridional_panels = (
                (
                    _MERIDIONAL_THERMODYNAMIC_FIELDS,
                    "LTE thermodynamic parameters",
                    "meridional_parameters",
                    "Parameters",
                ),
                (
                    _MAGNETIC_FIELDS,
                    r"Magnetic field components",
                    "meridional_magnetic_field",
                    "Magnetic field",
                ),
                (
                    _VELOCITY_FIELDS,
                    r"Velocity components",
                    "meridional_velocity",
                    "Velocity",
                ),
            )
            for field_names, title, filename_suffix, log_key in meridional_panels:
                paths.append(
                    self._save_figure(
                        trainer,
                        self._meridional_field_panel_figure(
                            meridional_evaluated,
                            label,
                            field_names=field_names,
                            title=title,
                            norm_evaluated=evaluated,
                            reference_depth_indices=None,
                        ),
                        f"{label}_{filename_suffix}.png",
                        f"{log_key} meridional slice",
                    )
                )
        return paths

    @staticmethod
    def _raster(trainer):
        data_module = getattr(trainer, "datamodule", None)
        raster = getattr(data_module, "raster", None)
        if raster is None:
            raise RuntimeError(
                "LTEAtmosphereVisualizationCallback requires a data module whose "
                "setup() method has populated a compatible ray raster."
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
