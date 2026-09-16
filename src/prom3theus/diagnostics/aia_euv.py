"""Deterministic native-grid diagnostics for AIA EUV image likelihoods."""

from __future__ import annotations

from prom3theus.observations.arrays import materialize_array

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from prom3theus.observations.bulk import read_native_samples
import torch

from prom3theus.core.transforms import normalized_asinh

from prom3theus.observations.image_contracts import ImageObservationRaster
from prom3theus.observations.image_dataset import reconstruct_image
from .sampling import subsample_grid


_REQUIRED_FIELDS = (
    "prediction_raw",
    "prediction_calibrated",
    "target",
    "channel_angstrom",
    "channel_index",
    "pixel_index",
    "image_index",
    "asinh_residual",
    "fractional_residual",
)
_OPTIONAL_PROFILE_FIELDS = ("contribution", "height_m")
_PROFILE_ID_FIELDS = (
    "channel_angstrom",
    "channel_index",
    "pixel_index",
    "image_index",
)


def _safe_label(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("AIA diagnostic label must be a non-empty string.")
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not result:
        raise ValueError("AIA diagnostic label contains no filename-safe characters.")
    return result


def _cpu_tensor(value: Any, *, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(
            f"AIA diagnostic field {name!r} must be tensor-like."
        ) from error
    return tensor


def _sample_vector(
    value: Any,
    *,
    name: str,
    sample_count: int | None = None,
    integer: bool = False,
) -> torch.Tensor:
    tensor = _cpu_tensor(value, name=name)
    if tensor.ndim == 2 and tensor.shape[-1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim == 0 and sample_count is not None:
        tensor = tensor.expand(sample_count)
    if tensor.ndim != 1 or not tensor.numel():
        raise ValueError(f"AIA diagnostic field {name!r} must have shape [sample].")
    if sample_count is not None and tensor.shape[0] != sample_count:
        raise ValueError(
            f"AIA diagnostic field {name!r} has {tensor.shape[0]} samples; "
            f"expected {sample_count}."
        )
    if integer:
        if tensor.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(f"AIA diagnostic field {name!r} must use an integer dtype.")
        return tensor.to(torch.long)
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(
            f"AIA diagnostic field {name!r} must use a real floating-point dtype."
        )
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"AIA diagnostic field {name!r} must be finite.")
    return tensor


def _pixel_indices(value: Any, *, sample_count: int) -> torch.Tensor:
    tensor = _cpu_tensor(value, name="pixel_index")
    if tensor.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("AIA diagnostic field 'pixel_index' must use an integer dtype.")
    if tensor.shape != (sample_count, 2):
        raise ValueError(
            "AIA diagnostic field 'pixel_index' must have shape [sample, 2]."
        )
    return tensor.to(torch.long)


def _profile_field(
    value: Any,
    *,
    name: str,
    sample_count: int,
) -> torch.Tensor:
    tensor = _cpu_tensor(value, name=name)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).expand(sample_count, -1)
    if tensor.ndim != 2 or tensor.shape[0] != sample_count or tensor.shape[1] < 2:
        raise ValueError(
            f"AIA diagnostic field {name!r} must have shape [sample, height] "
            "with at least two heights."
        )
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(
            f"AIA diagnostic field {name!r} must use a real floating-point dtype."
        )
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"AIA diagnostic field {name!r} must be finite.")
    return tensor


def _prepare_payload(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("AIA diagnostic payloads must be mappings.")
    missing = sorted(set(_REQUIRED_FIELDS) - set(payload))
    if missing:
        raise KeyError(f"AIA diagnostic payload is missing fields: {missing}.")
    target = _sample_vector(payload["target"], name="target")
    count = int(target.shape[0])
    prepared = {"target": target}
    for name in (
        "prediction_raw",
        "prediction_calibrated",
        "asinh_residual",
        "fractional_residual",
    ):
        prepared[name] = _sample_vector(payload[name], name=name, sample_count=count)
    for name in ("channel_angstrom", "channel_index", "image_index"):
        prepared[name] = _sample_vector(
            payload[name], name=name, sample_count=count, integer=True
        )
    prepared["pixel_index"] = _pixel_indices(payload["pixel_index"], sample_count=count)
    optional_present = [name in payload for name in _OPTIONAL_PROFILE_FIELDS]
    if any(optional_present) and not all(optional_present):
        raise KeyError(
            "AIA contribution diagnostics require both 'contribution' and 'height_m'."
        )
    if all(optional_present):
        prepared["contribution"] = _profile_field(
            payload["contribution"], name="contribution", sample_count=count
        )
        prepared["height_m"] = _profile_field(
            payload["height_m"], name="height_m", sample_count=count
        )
        if prepared["contribution"].shape != prepared["height_m"].shape:
            raise ValueError(
                "AIA contribution and height_m arrays must have equal shape."
            )
    if torch.any(prepared["prediction_raw"] < 0) or torch.any(
        prepared["prediction_calibrated"] < 0
    ):
        raise ValueError("AIA diagnostic predictions must be non-negative.")
    return prepared


def _prepare_profile_payload(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("AIA contribution payloads must be mappings.")
    required = set(_OPTIONAL_PROFILE_FIELDS) | set(_PROFILE_ID_FIELDS)
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"AIA contribution payload is missing fields: {missing}.")
    channel = _sample_vector(
        payload["channel_angstrom"], name="channel_angstrom", integer=True
    )
    count = int(channel.shape[0])
    prepared = {"channel_angstrom": channel}
    for name in ("channel_index", "image_index"):
        prepared[name] = _sample_vector(
            payload[name], name=name, sample_count=count, integer=True
        )
    prepared["pixel_index"] = _pixel_indices(payload["pixel_index"], sample_count=count)
    for name in _OPTIONAL_PROFILE_FIELDS:
        prepared[name] = _profile_field(payload[name], name=name, sample_count=count)
    if prepared["contribution"].shape != prepared["height_m"].shape:
        raise ValueError("AIA contribution and height_m arrays must have equal shape.")
    return prepared


def _context_profile_payloads(
    context: Mapping[str, Any] | None,
) -> tuple[dict[str, torch.Tensor], ...]:
    if not isinstance(context, Mapping) or "contribution_payloads" not in context:
        return ()
    source = context["contribution_payloads"]
    if isinstance(source, Mapping):
        values = (source,)
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        values = tuple(source)
    else:
        raise TypeError(
            "AIA diagnostic contribution_payloads must be a mapping or sequence."
        )
    return tuple(_prepare_profile_payload(payload) for payload in values)


def _rasters_from_context(
    context: Mapping[str, Any] | None,
) -> dict[int, ImageObservationRaster]:
    if not isinstance(context, Mapping):
        raise ValueError(
            "AIA rendering requires a context containing its native-grid rasters."
        )
    source = context.get("rasters")
    if source is None:
        source = getattr(context.get("data_module"), "rasters", None)
    if isinstance(source, Mapping):
        raster_by_index = dict(source)
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        raster_by_index = dict(enumerate(source))
    else:
        raise ValueError(
            "AIA diagnostic context must provide 'rasters' or a data_module.rasters."
        )
    if not raster_by_index or any(
        type(index) is not int or index < 0 for index in raster_by_index
    ):
        raise ValueError("AIA diagnostic raster indices must be non-negative integers.")
    if any(
        not isinstance(raster, ImageObservationRaster)
        for raster in raster_by_index.values()
    ):
        raise TypeError("AIA diagnostic rasters must be ImageObservationRaster values.")
    return raster_by_index


def _nullable(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _image_metrics(values: Mapping[str, np.ndarray]) -> dict[str, float | int | None]:
    target = values["target"]
    calibrated = values["prediction_calibrated"]
    residual = calibrated - target
    if target.size > 1 and np.std(target) > 0 and np.std(calibrated) > 0:
        correlation = _nullable(np.corrcoef(target, calibrated)[0, 1])
    else:
        correlation = None
    return {
        "sample_count": int(target.size),
        "residual_bias": float(np.mean(residual)),
        "residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "residual_mae": float(np.mean(np.abs(residual))),
        "correlation": correlation,
    }


def _contribution_metrics(
    contribution: np.ndarray,
    height_m: np.ndarray,
) -> dict[str, float | None]:
    clipped = np.clip(contribution, 0.0, None)
    denominator = clipped.sum(axis=1)
    valid = denominator > 0
    if not np.any(valid):
        return {
            "contribution_peak_height_m_median": None,
            "contribution_centroid_height_m_median": None,
        }
    peak_height = np.take_along_axis(
        height_m, np.argmax(clipped, axis=1)[:, None], axis=1
    )[:, 0]
    centroid = (clipped[valid] * height_m[valid]).sum(axis=1) / denominator[valid]
    return {
        "contribution_peak_height_m_median": float(np.median(peak_height[valid])),
        "contribution_centroid_height_m_median": float(np.median(centroid)),
    }


class AIAEUVDiagnosticRenderer:
    """Reconstruct and plot observed and synthesized AIA native-grid images."""

    observation_kind = "aia_euv"

    def __init__(
        self,
        *,
        max_pixels_per_image: int = 65_536,
        max_contribution_rays: int = 16,
        dpi: int = 180,
    ) -> None:
        if type(max_pixels_per_image) is not int or max_pixels_per_image < 1:
            raise ValueError("max_pixels_per_image must be a positive integer.")
        if type(dpi) is not int or dpi < 50:
            raise ValueError("dpi must be an integer of at least 50.")
        if type(max_contribution_rays) is not int or max_contribution_rays < 0:
            raise ValueError("max_contribution_rays must be a nonnegative integer.")
        self.max_pixels_per_image = max_pixels_per_image
        self.max_contribution_rays = max_contribution_rays
        self.dpi = dpi

    def render(
        self,
        payloads: Sequence[Mapping[str, Any]],
        *,
        output_directory: Path,
        label: str,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prepared = tuple(_prepare_payload(payload) for payload in payloads)
        if not prepared:
            raise ValueError("AIA diagnostics require at least one payload.")
        profile_payloads = tuple(
            _prepare_profile_payload(payload)
            for payload in prepared
            if "contribution" in payload
        ) + _context_profile_payloads(context)
        profile_ray_count = sum(
            int(payload["contribution"].shape[0]) for payload in profile_payloads
        )
        if profile_ray_count > self.max_contribution_rays:
            raise ValueError(
                f"AIA contribution diagnostics contain {profile_ray_count} rays; "
                f"the configured bound is {self.max_contribution_rays}."
            )
        merged = {
            name: torch.cat([payload[name] for payload in prepared], dim=0)
            for name in _REQUIRED_FIELDS
        }
        raster_by_index = _rasters_from_context(context)
        safe_label = _safe_label(label)
        directory = Path(output_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)

        indices_by_image: dict[int, list[int]] = defaultdict(list)
        for sample_index, image_index in enumerate(merged["image_index"].tolist()):
            indices_by_image[int(image_index)].append(sample_index)

        paths: list[str] = []
        reports: list[dict[str, Any]] = []
        comparisons = defaultdict(list)
        for image_index in sorted(indices_by_image):
            if image_index not in raster_by_index:
                raise IndexError(
                    f"AIA diagnostic image_index {image_index} has no native raster."
                )
            selection = torch.tensor(indices_by_image[image_index], dtype=torch.long)
            sample_count = int(selection.numel())
            if sample_count > self.max_pixels_per_image:
                raise ValueError(
                    f"AIA diagnostic image {image_index} contains "
                    f"{sample_count} pixels; "
                    f"the configured bound is {self.max_pixels_per_image}."
                )
            raster = raster_by_index[image_index]
            channel_values = torch.unique(merged["channel_angstrom"][selection])
            channel_indices = torch.unique(merged["channel_index"][selection])
            if channel_values.numel() != 1 or channel_indices.numel() != 1:
                raise ValueError(
                    "Each AIA diagnostic image_index must identify exactly one channel."
                )
            channel = int(channel_values[0])
            channel_index = int(channel_indices[0])
            if channel != raster.channel_angstrom:
                raise ValueError(
                    f"AIA diagnostic channel {channel} does not match raster "
                    f"channel {raster.channel_angstrom} for image {image_index}."
                )
            intensity_scale = context["intensity_scales"][channel]

            pixels = merged["pixel_index"][selection]
            vectors = {
                name: merged[name][selection].numpy()
                for name in (
                    "target",
                    "prediction_raw",
                    "prediction_calibrated",
                    "asinh_residual",
                    "fractional_residual",
                )
            }
            target_tensor = torch.from_numpy(vectors["target"])
            expected_target = (
                read_native_samples(raster.intensity, pixels) / intensity_scale
            ).to(dtype=target_tensor.dtype)
            if not torch.allclose(
                target_tensor,
                expected_target.cpu(),
                rtol=1.0e-5,
                atol=1.0e-6,
            ):
                raise ValueError(
                    f"AIA diagnostic target does not match native raster {image_index}."
                )
            prediction = reconstruct_image(
                torch.from_numpy(vectors["prediction_calibrated"]),
                pixels,
                raster.spatial_shape,
            ).numpy()
            comparisons[raster.exposure_group].append((raster, prediction))
            path = (
                directory
                / f"{safe_label}_aia_{_safe_label(raster.exposure_group)}_comparison.png"
            )
            metrics = _image_metrics(vectors)
            metrics["native_pixel_count"] = int(np.prod(raster.spatial_shape))
            metrics["sampled_fraction"] = sample_count / int(
                np.prod(raster.spatial_shape)
            )
            reports.append(
                {
                    "image_index": image_index,
                    "channel_angstrom": channel,
                    "channel_index": channel_index,
                    "exposure_group": raster.exposure_group,
                    "absolute_tai_seconds": raster.absolute_tai_seconds,
                    "intensity_unit": "dimensionless",
                    "intensity_scale_dn_s_pixel": intensity_scale,
                    "path": str(path),
                    "metrics": metrics,
                }
            )

        for group, images in sorted(comparisons.items()):
            path = directory / f"{safe_label}_aia_{_safe_label(group)}_comparison.png"
            self._render_comparison(
                path,
                group,
                images,
                context.get("asinh_scales", {}),
                context["intensity_scales"],
            )
            paths.append(str(path))

        contribution_paths: list[str] = []
        contribution_reports: list[dict[str, Any]] = []
        if profile_payloads:
            profile_merged = {
                name: torch.cat([payload[name] for payload in profile_payloads], dim=0)
                for name in (*_PROFILE_ID_FIELDS, *_OPTIONAL_PROFILE_FIELDS)
            }
            profile_indices_by_image: dict[int, list[int]] = defaultdict(list)
            for sample_index, image_index in enumerate(
                profile_merged["image_index"].tolist()
            ):
                profile_indices_by_image[int(image_index)].append(sample_index)
            image_reports = {item["image_index"]: item for item in reports}
            for image_index in sorted(profile_indices_by_image):
                if (
                    image_index not in raster_by_index
                    or image_index not in image_reports
                ):
                    raise IndexError(
                        f"AIA contribution image_index {image_index} has no rendered "
                        "native raster."
                    )
                selection = torch.tensor(
                    profile_indices_by_image[image_index], dtype=torch.long
                )
                channel_values = torch.unique(
                    profile_merged["channel_angstrom"][selection]
                )
                channel_indices = torch.unique(
                    profile_merged["channel_index"][selection]
                )
                if channel_values.numel() != 1 or channel_indices.numel() != 1:
                    raise ValueError(
                        "Each AIA contribution image_index must identify exactly one "
                        "channel."
                    )
                channel = int(channel_values[0])
                channel_index = int(channel_indices[0])
                raster = raster_by_index[image_index]
                if channel != raster.channel_angstrom:
                    raise ValueError(
                        f"AIA contribution channel {channel} does not match raster "
                        f"channel {raster.channel_angstrom} for image {image_index}."
                    )
                contribution = (
                    profile_merged["contribution"][selection]
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                height_m = (
                    profile_merged["height_m"][selection]
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                path = directory / (
                    f"{safe_label}_aia_{channel:03d}_image_{image_index:04d}_"
                    "contribution.png"
                )
                self._render_contribution(
                    path,
                    channel=channel,
                    contribution=contribution,
                    height_m=height_m,
                )
                metrics = _contribution_metrics(contribution, height_m)
                metrics["contribution_ray_count"] = int(selection.numel())
                image_reports[image_index]["metrics"].update(metrics)
                contribution_paths.append(str(path))
                contribution_reports.append(
                    {
                        "image_index": image_index,
                        "channel_angstrom": channel,
                        "channel_index": channel_index,
                        "ray_count": int(selection.numel()),
                        "quadrature_sample_count": int(contribution.shape[1]),
                        "pixel_indices": profile_merged["pixel_index"][
                            selection
                        ].tolist(),
                        "normalization": "each ray divided by its non-negative sum",
                        "path": str(path),
                        "metrics": metrics,
                    }
                )

        return {
            "paths": paths,
            "images": reports,
            "contribution_paths": contribution_paths,
            "contribution_profiles": contribution_reports,
            "contribution_ray_count": profile_ray_count,
        }

    def _render_contribution(
        self,
        path: Path,
        *,
        channel: int,
        contribution: np.ndarray,
        height_m: np.ndarray,
    ) -> None:
        """Plot a bounded set of response-weighted LOS contributions."""

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        clipped = np.clip(contribution, 0.0, None)
        denominator = clipped.sum(axis=1, keepdims=True)
        normalized = np.divide(
            clipped,
            denominator,
            out=np.zeros_like(clipped),
            where=denominator > 0,
        )
        figure = Figure(figsize=(6.4, 5.0), dpi=self.dpi)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for ray, height in zip(normalized, height_m):
            order = np.argsort(height)
            axis.plot(ray[order], height[order] * 1.0e-6, alpha=0.28, linewidth=0.8)
        if normalized.shape[0] > 1 and np.allclose(height_m, height_m[:1]):
            axis.plot(
                np.median(normalized, axis=0),
                height_m[0] * 1.0e-6,
                color="black",
                linewidth=1.8,
                label="median",
            )
            axis.legend(loc="best", fontsize=8)
        axis.set_xlabel("Normalized response-weighted contribution")
        axis.set_ylabel("Geometric height [Mm]")
        axis.set_title(
            f"AIA {channel} Angstrom contribution functions "
            f"({normalized.shape[0]} rays)"
        )
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(
            path,
            dpi=self.dpi,
            facecolor="white",
            metadata={"Software": "PROM3THEUS"},
        )
        figure.clear()

    def _render_comparison(
        self,
        path: Path,
        group: str,
        images: Sequence,
        asinh_scales: Mapping[int, float],
        intensity_scales: Mapping[int, float],
    ) -> None:
        """Compare native AIA arrays, without reprojection or pixel interpolation."""
        import astropy.units as u
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import Normalize
        from matplotlib.figure import Figure
        from sunpy.visualization.colormaps.color_tables import aia_color_table

        images = sorted(images, key=lambda item: item[0].channel_angstrom)
        figure = Figure(
            figsize=(4.5 * len(images), 7.0), dpi=self.dpi, layout="constrained"
        )
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, len(images), squeeze=False)
        for column, (raster, prediction) in enumerate(images):
            channel = raster.channel_angstrom
            scale = asinh_scales.get(channel)
            if scale is None or not math.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"AIA {channel} rendering requires its training asinh scale."
                )
            cmap = aia_color_table(channel * u.angstrom)
            cmap.set_bad("0.85")
            # Exactly the objective's transform, using its fixed channel scale.
            # The target stays unchanged; calibration gains apply only to synthesis.
            observed = normalized_asinh(
                materialize_array(raster.intensity).detach().cpu() / intensity_scales[channel], scale
            ).numpy()
            prediction = normalized_asinh(torch.as_tensor(prediction), scale).numpy()
            valid = materialize_array(raster.valid_mask).detach().cpu().numpy()
            observed = np.where(valid, observed, np.nan)
            # One linear colour range for both rows, fixed by the observation.
            # No second stretch, percentile clipping, or prediction-dependent limits.
            norm = Normalize(
                vmin=min(0.0, float(np.nanmin(observed))),
                vmax=max(1.0, float(np.nanmax(observed))),
            )
            rows, columns = subsample_grid(
                *raster.spatial_shape, self.max_pixels_per_image
            )
            observed = observed[np.ix_(rows, columns)]
            prediction = prediction[np.ix_(rows, columns)]
            for row, array in enumerate((observed, prediction)):
                image = axes[row, column].imshow(
                    array, origin="lower", interpolation="none", cmap=cmap, norm=norm
                )
                axes[row, column].set_xticks(())
                axes[row, column].set_yticks(())
            axes[0, column].set_title(f"{raster.channel_angstrom} Å")
            figure.colorbar(
                image,
                ax=axes[:, column].tolist(),
                orientation="horizontal",
                fraction=0.05,
                pad=0.025,
                label="asinh(I / a) / asinh(1 / a)",
            )
        axes[0, 0].set_ylabel("Observation")
        axes[1, 0].set_ylabel("Synthesis (gain-adjusted)")
        figure.suptitle(f"AIA | {group} | normalized asinh, training channel scales")
        figure.savefig(
            path, dpi=self.dpi, facecolor="white", metadata={"Software": "PROM3THEUS"}
        )
        figure.clear()


__all__ = ["AIAEUVDiagnosticRenderer"]
