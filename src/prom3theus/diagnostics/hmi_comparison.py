"""Post-training comparison of a P3S model with native HMI vector products."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from prom3theus.artifacts.loader import P3SLoader
from prom3theus.core import cartesian_to_spherical, project_cartesian_to_spherical
from prom3theus.instruments.hmi.acquisition import (
    hmi_observation_wcs_header,
    parse_hmi_tai_time,
)
from prom3theus.instruments.hmi.geometry import build_geometry
from prom3theus.observations import ObservationRaster
from prom3theus.rt.geometry import direction_to_chart_mm


_REQUIRED_SEGMENTS = ("field", "inclination", "azimuth", "disambig")


def _discover_segments(directory: str | Path) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HMI comparison directory not found: {root}")
    selected: dict[str, Path] = {}
    record_prefixes: set[str] = set()
    for segment in _REQUIRED_SEGMENTS:
        matches = sorted(root.glob(f"hmi.[Bb]_720s.*.{segment}.fits"))
        if len(matches) != 1:
            raise ValueError(
                f"HMI comparison requires exactly one {segment!r} FITS file in "
                f"{root}; found {len(matches)}."
            )
        selected[segment] = matches[0]
        record_prefixes.add(matches[0].name.removesuffix(f".{segment}.fits"))
    if len(record_prefixes) != 1:
        raise ValueError("HMI comparison segments do not belong to one B_720s record.")
    return selected


@contextmanager
def _fits_image(path: Path):
    try:
        from astropy.io import fits
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HMI comparison requires the optional observations dependencies."
        ) from error
    with fits.open(path, memmap=True) as hdus:
        images = [hdu for hdu in hdus if int(hdu.header.get("NAXIS", 0)) == 2]
        if len(images) != 1:
            raise ValueError(f"{path} must contain exactly one two-dimensional image.")
        data = images[0].data
        if data is None or data.ndim != 2:
            raise ValueError(f"{path} does not contain a readable image.")
        yield images[0].header.copy(), np.asarray(data)


def _reference_header(paths: Mapping[str, Path]):
    headers = {}
    shapes = {}
    for name, path in paths.items():
        with _fits_image(path) as (header, image):
            headers[name] = header
            shapes[name] = image.shape
    reference = headers["field"]
    identity = (
        "T_OBS",
        "T_REC",
        "CTYPE1",
        "CTYPE2",
        "CDELT1",
        "CDELT2",
        "CRPIX1",
        "CRPIX2",
        "CRVAL1",
        "CRVAL2",
        "CROTA2",
    )
    for name, header in headers.items():
        if shapes[name] != shapes["field"]:
            raise ValueError("HMI comparison segment image shapes differ.")
        differing = [key for key in identity if header.get(key) != reference.get(key)]
        if differing:
            raise ValueError(
                f"HMI comparison segment {name!r} has inconsistent WCS/time keys: "
                f"{differing}."
            )
    if int(reference.get("QUALITY", -1)) != 0:
        raise ValueError("HMI comparison requires a QUALITY=0 B_720s record.")
    return reference, shapes["field"]


def _reference_pixel_indices(
    raster: ObservationRaster,
    header,
    full_disk_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import frames
        from sunpy.map import Map
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HMI comparison requires the optional observations dependencies."
        ) from error
    reference_map = Map(
        np.zeros((1, 1), dtype=np.uint8), hmi_observation_wcs_header(header)
    )
    frame = frames.HeliographicCarrington(
        observer=reference_map.observer_coordinate,
        obstime=reference_map.date,
    )
    position = raster.surface_position_m.detach().cpu().numpy()
    coordinate = SkyCoord(
        x=position[..., 0] * u.m,
        y=position[..., 1] * u.m,
        z=position[..., 2] * u.m,
        representation_type="cartesian",
        frame=frame,
    ).transform_to(reference_map.coordinate_frame)
    pixels = reference_map.world_to_pixel(coordinate)
    column_float = pixels.x.to_value(u.pix)
    row_float = pixels.y.to_value(u.pix)
    columns = np.rint(column_float).astype(np.int64)
    rows = np.rint(row_float).astype(np.int64)
    in_bounds = (
        np.isfinite(row_float)
        & np.isfinite(column_float)
        & (rows >= 0)
        & (rows < full_disk_shape[0])
        & (columns >= 0)
        & (columns < full_disk_shape[1])
    )
    residual = np.hypot(row_float - rows, column_float - columns)
    finite_residual = residual[in_bounds & raster.valid_mask.detach().cpu().numpy()]
    maximum_residual = (
        float(np.max(finite_residual)) if finite_residual.size else float("inf")
    )
    return rows, columns, in_bounds, maximum_residual


def _sample_segment(
    path: Path,
    rows: np.ndarray,
    columns: np.ndarray,
    in_bounds: np.ndarray,
) -> np.ndarray:
    sampled = np.full(rows.shape, np.nan, dtype=np.float64)
    with _fits_image(path) as (_, image):
        sampled[in_bounds] = image[rows[in_bounds], columns[in_bounds]]
    return sampled


def _reference_evaluation_grid(loader: P3SLoader, header, target_time):
    """Build the model grid from B_720s WCS and compact P3S geometry metadata."""

    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        from sunpy.coordinates import frames
        from sunpy.map import Map
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HMI comparison requires the optional observations dependencies."
        ) from error

    wcs_header = hmi_observation_wcs_header(header)
    reference_map = Map(np.zeros((1, 1), dtype=np.uint8), wcs_header)
    bounds = loader.observation.bounds
    center = float(bounds["surface_longitude_center_rad"])
    lon_min, lon_max = (
        center + float(value) for value in bounds["surface_longitude_offset_rad"]
    )
    lat_min, lat_max = map(float, bounds["surface_latitude_rad"])
    edge_samples = 128
    lon_edge = np.linspace(lon_min, lon_max, edge_samples)
    lat_edge = np.linspace(lat_min, lat_max, edge_samples)
    longitude = np.concatenate(
        (
            lon_edge,
            lon_edge,
            np.full(edge_samples, lon_min),
            np.full(edge_samples, lon_max),
        )
    )
    latitude = np.concatenate(
        (
            np.full(edge_samples, lat_min),
            np.full(edge_samples, lat_max),
            lat_edge,
            lat_edge,
        )
    )
    frame = frames.HeliographicCarrington(
        observer=reference_map.observer_coordinate,
        obstime=reference_map.date,
    )
    boundary = SkyCoord(
        lon=longitude * u.rad,
        lat=latitude * u.rad,
        radius=float(bounds["solar_radius_m"]) * u.m,
        frame=frame,
    ).transform_to(reference_map.coordinate_frame)
    boundary_pixels = reference_map.world_to_pixel(boundary)
    x = boundary_pixels.x.to_value(u.pix)
    y = boundary_pixels.y.to_value(u.pix)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise ValueError("Saved P3S bounds do not project onto the HMI reference WCS.")
    full_height = int(header["NAXIS2"])
    full_width = int(header["NAXIS1"])
    padding = 2
    x0 = max(0, int(np.floor(x[finite].min())) - padding)
    x1 = min(full_width - 1, int(np.ceil(x[finite].max())) + padding)
    y0 = max(0, int(np.floor(y[finite].min())) - padding)
    y1 = min(full_height - 1, int(np.ceil(y[finite].max())) + padding)
    if x1 < x0 or y1 < y0:
        raise ValueError("Saved P3S bounds select no HMI detector pixels.")

    cutout_header = wcs_header.copy()
    cutout_header["CRPIX1"] = float(cutout_header["CRPIX1"]) - x0
    cutout_header["CRPIX2"] = float(cutout_header["CRPIX2"]) - y0
    shape = (y1 - y0 + 1, x1 - x0 + 1)
    cutout = Map(np.zeros(shape, dtype=np.uint8), cutout_header)
    candidate = np.ones(shape, dtype=bool)
    _, surface, basis, mu, _, _ = build_geometry(
        cutout,
        candidate,
        np.zeros(3, dtype=np.float64),
        scene_basis_rows=loader.module.hparams["atmosphere_config"][
            "scene_geometry_config"
        ]["scene_basis"],
    )
    radius = np.linalg.norm(surface, axis=-1)
    longitude_grid = np.arctan2(surface[..., 1], surface[..., 0])
    longitude_offset = np.arctan2(
        np.sin(longitude_grid - center), np.cos(longitude_grid - center)
    )
    latitude_grid = np.arctan2(
        surface[..., 2], np.linalg.norm(surface[..., :2], axis=-1)
    )
    valid = (
        np.isfinite(surface).all(axis=-1)
        & np.isfinite(mu)
        & (mu > 0)
        & np.isclose(radius, float(bounds["solar_radius_m"]), rtol=2e-5, atol=1.0)
        & (longitude_offset >= float(bounds["surface_longitude_offset_rad"][0]))
        & (longitude_offset <= float(bounds["surface_longitude_offset_rad"][1]))
        & (latitude_grid >= lat_min)
        & (latitude_grid <= lat_max)
    )
    if not np.any(valid):
        raise ValueError("Saved P3S bounds select no valid HMI surface pixels.")
    scene = torch.as_tensor(
        loader.module.hparams["atmosphere_config"]["scene_geometry_config"][
            "scene_basis"
        ],
        dtype=torch.float64,
    )
    chart = direction_to_chart_mm(
        torch.from_numpy(surface), scene, float(bounds["solar_radius_m"])
    ).numpy()
    first_record = loader.observation.times[0]
    origin = Time(first_record["values"][0], scale=first_record["scale"])
    time_hours = float((target_time - origin).to_value(u.hour))
    coordinates = np.concatenate(
        (chart, np.full((*shape, 1), time_hours, dtype=np.float64)), axis=-1
    )
    rows, columns = np.indices(shape, dtype=np.int64)
    return SimpleNamespace(
        coordinates=torch.from_numpy(coordinates.astype(np.float32)),
        valid_mask=torch.from_numpy(valid),
        stokes_basis=torch.from_numpy(basis.astype(np.float32)),
        surface_position_m=torch.from_numpy(surface.astype(np.float32)),
        rows=rows + y0,
        columns=columns + x0,
    )


def _observer_components(
    field_gauss: np.ndarray,
    inclination_deg: np.ndarray,
    azimuth_deg: np.ndarray,
) -> np.ndarray:
    inclination = np.deg2rad(inclination_deg)
    azimuth = np.deg2rad(azimuth_deg)
    transverse = field_gauss * np.sin(inclination)
    return np.stack(
        (
            transverse * np.cos(azimuth),
            transverse * np.sin(azimuth),
            field_gauss * np.cos(inclination),
        ),
        axis=-1,
    )


def _angular_residual_deg(left: np.ndarray, right: np.ndarray, period: float):
    return (left - right + 0.5 * period) % period - 0.5 * period


def _spherical_components(
    cartesian_field: np.ndarray, surface_position_m: np.ndarray
) -> np.ndarray:
    field = torch.from_numpy(np.asarray(cartesian_field))
    position = torch.from_numpy(np.asarray(surface_position_m)).to(field)
    spherical = cartesian_to_spherical(position, torch)
    return project_cartesian_to_spherical(field, spherical, torch).numpy()


def _component_statistics(
    model: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    selected = mask & np.isfinite(model) & np.isfinite(reference)
    if not np.any(selected):
        return {
            "count": 0,
            "bias_gauss": None,
            "mae_gauss": None,
            "rmse_gauss": None,
            "correlation": None,
        }
    residual = model[selected] - reference[selected]
    correlation = (
        float(np.corrcoef(model[selected], reference[selected])[0, 1])
        if selected.sum() > 1
        and np.std(model[selected]) > 0
        and np.std(reference[selected]) > 0
        else None
    )
    return {
        "count": int(selected.sum()),
        "bias_gauss": float(np.mean(residual)),
        "mae_gauss": float(np.mean(np.abs(residual))),
        "rmse_gauss": float(np.sqrt(np.mean(residual**2))),
        "correlation": correlation,
    }


def _mean_correlation(
    statistics: Mapping[str, Mapping[str, Any]], names
) -> float | None:
    values = [statistics[name]["correlation"] for name in names]
    finite = [
        float(value) for value in values if value is not None and np.isfinite(value)
    ]
    return float(np.mean(finite)) if finite else None


def _azimuth_convention_metrics(
    model_observer: np.ndarray,
    model_spherical: np.ndarray,
    field_gauss: np.ndarray,
    inclination_deg: np.ndarray,
    raw_azimuth_deg: np.ndarray,
    reference_flip: np.ndarray,
    stokes_basis: np.ndarray,
    surface_position_m: np.ndarray,
    valid: np.ndarray,
    strong: np.ndarray,
) -> dict[str, Any]:
    """Score every signed 90-degree HMI azimuth convention.

    The tested family is ``chi' = sign * chi + rotation`` with sign in
    ``{+1, -1}`` and rotation in ``{0, 90, 180, 270}``.  The selected HMI
    disambiguation bit is applied after this coordinate-convention transform.
    """

    component_names = ("b_r", "b_theta", "b_phi")
    model_azimuth = (
        np.rad2deg(np.arctan2(model_observer[..., 1], model_observer[..., 0])) % 360.0
    )
    configurations: dict[str, Any] = {}
    for reverse in (False, True):
        sign = -1.0 if reverse else 1.0
        prefix = "negative_chi" if reverse else "chi"
        for rotation in (0, 90, 180, 270):
            name = prefix if rotation == 0 else f"{prefix}_plus_{rotation}"
            transformed_azimuth = sign * raw_azimuth_deg + float(rotation)
            observer = _observer_components(
                field_gauss, inclination_deg, transformed_azimuth
            )
            observer[..., :2] *= np.where(reference_flip, -1.0, 1.0)[..., None]
            reference_azimuth = (transformed_azimuth + 180.0 * reference_flip) % 360.0
            director_residual = _angular_residual_deg(
                model_azimuth, transformed_azimuth, 180.0
            )
            signed_residual = _angular_residual_deg(
                model_azimuth, reference_azimuth, 360.0
            )
            cartesian = np.einsum("...ji,...j->...i", stokes_basis, observer)
            spherical = _spherical_components(cartesian, surface_position_m)
            all_statistics = {}
            strong_statistics = {}
            for index, component in enumerate(component_names):
                all_statistics[component] = _component_statistics(
                    model_spherical[..., index], spherical[..., index], valid
                )
                strong_statistics[component] = _component_statistics(
                    model_spherical[..., index], spherical[..., index], strong
                )
            configurations[name] = {
                "azimuth_transform": f"chi' = {'-' if reverse else ''}chi + {rotation} deg",
                "reverse_azimuth": reverse,
                "rotation_deg": rotation,
                "director_mae_deg_strong_transverse": float(
                    np.mean(np.abs(director_residual[strong]))
                )
                if np.any(strong)
                else None,
                "signed_azimuth_mae_deg_strong_transverse": float(
                    np.mean(np.abs(signed_residual[strong]))
                )
                if np.any(strong)
                else None,
                "mean_btheta_bphi_correlation_all_valid": _mean_correlation(
                    all_statistics, ("b_theta", "b_phi")
                ),
                "mean_btheta_bphi_correlation_strong_transverse": _mean_correlation(
                    strong_statistics, ("b_theta", "b_phi")
                ),
                "components_all_valid": all_statistics,
                "components_strong_transverse": strong_statistics,
            }

    ranking = sorted(
        (
            (name, values["mean_btheta_bphi_correlation_strong_transverse"])
            for name, values in configurations.items()
            if values["mean_btheta_bphi_correlation_strong_transverse"] is not None
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "family": "chi' = sign * chi + rotation; sign in {+1,-1}; rotation in {0,90,180,270} deg",
        "disambiguation_application": "selected DISAMBIG bit applied after transform",
        "ranking_metric": "mean Pearson correlation of Btheta and Bphi on strong-transverse pixels",
        "best_configuration": ranking[0][0] if ranking else None,
        "ranking": [
            {"configuration": name, "score": float(score)} for name, score in ranking
        ],
        "configurations": configurations,
    }


def _comparison_metrics(
    model_observer: np.ndarray,
    model_spherical: np.ndarray,
    reference_observer_raw: np.ndarray,
    reference_spherical: np.ndarray,
    reference_flip: np.ndarray,
    valid: np.ndarray,
    strong: np.ndarray,
) -> dict[str, Any]:
    model_azimuth = (
        np.rad2deg(np.arctan2(model_observer[..., 1], model_observer[..., 0])) % 360.0
    )
    raw_azimuth = (
        np.rad2deg(
            np.arctan2(reference_observer_raw[..., 1], reference_observer_raw[..., 0])
        )
        % 360.0
    )
    director_residual = _angular_residual_deg(model_azimuth, raw_azimuth, 180.0)
    reference_azimuth = (raw_azimuth + 180.0 * reference_flip) % 360.0
    signed_residual = _angular_residual_deg(model_azimuth, reference_azimuth, 360.0)
    model_strength = np.linalg.norm(model_observer, axis=-1)
    reference_strength = np.linalg.norm(reference_observer_raw, axis=-1)
    model_inclination = np.rad2deg(
        np.arccos(
            np.clip(
                model_observer[..., 2]
                / np.maximum(model_strength, np.finfo(float).tiny),
                -1.0,
                1.0,
            )
        )
    )
    reference_inclination = np.rad2deg(
        np.arccos(
            np.clip(
                reference_observer_raw[..., 2]
                / np.maximum(reference_strength, np.finfo(float).tiny),
                -1.0,
                1.0,
            )
        )
    )
    inferred_flip = (
        np.sum(model_observer[..., :2] * reference_observer_raw[..., :2], axis=-1) < 0.0
    )
    branch_mask = strong & np.isfinite(director_residual)
    branch_agreement = inferred_flip == reference_flip
    component_names = ("b_r", "b_theta", "b_phi")
    metrics: dict[str, Any] = {
        "valid_pixel_count": int(valid.sum()),
        "strong_transverse_pixel_count": int(strong.sum()),
        "director_mae_deg": float(np.mean(np.abs(director_residual[strong])))
        if np.any(strong)
        else None,
        "signed_azimuth_mae_deg": float(np.mean(np.abs(signed_residual[strong])))
        if np.any(strong)
        else None,
        "inclination_mae_deg": float(
            np.mean(np.abs(model_inclination[strong] - reference_inclination[strong]))
        )
        if np.any(strong)
        else None,
        "field_strength": _component_statistics(
            model_strength, reference_strength, strong
        ),
        "branch_agreement_fraction": float(np.mean(branch_agreement[branch_mask]))
        if np.any(branch_mask)
        else None,
        "branch_comparison_pixel_count": int(branch_mask.sum()),
        "components_all_valid": {},
        "components_strong_transverse": {},
    }
    for index, name in enumerate(component_names):
        metrics["components_all_valid"][name] = _component_statistics(
            model_spherical[..., index], reference_spherical[..., index], valid
        )
        metrics["components_strong_transverse"][name] = _component_statistics(
            model_spherical[..., index], reference_spherical[..., index], strong
        )
    return metrics


def _finite_symmetric_limit(*values: np.ndarray) -> float:
    finite = [np.abs(value[np.isfinite(value)]).ravel() for value in values]
    selected = np.concatenate(finite) if finite else np.empty(0)
    return max(float(np.percentile(selected, 98.0)), 1.0) if selected.size else 1.0


def _save_component_figure(
    path: Path,
    model: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    *,
    title: str,
    dpi: int,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    figure = Figure(figsize=(13.0, 10.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(3, 3, sharex=True, sharey=True))
    names = (r"$B_r$", r"$B_\theta$ (southward)", r"$B_\phi$ (westward)")
    for column, name in enumerate(names):
        component_limit = _finite_symmetric_limit(
            model[..., column][valid], reference[..., column][valid]
        )
        residual = model[..., column] - reference[..., column]
        residual_limit = _finite_symmetric_limit(residual[valid])
        rows = (
            (model[..., column], Normalize(-component_limit, component_limit)),
            (reference[..., column], Normalize(-component_limit, component_limit)),
            (residual, Normalize(-residual_limit, residual_limit)),
        )
        for row, (values, norm) in enumerate(rows):
            image = axes[row, column].imshow(
                np.where(valid, values, np.nan),
                origin="lower",
                cmap="RdBu_r",
                norm=norm,
                interpolation="nearest",
                rasterized=True,
            )
            figure.colorbar(image, ax=axes[row, column], shrink=0.78, label="G")
            axes[row, column].set_title(name)
    for row, label in enumerate(("PROM3THEUS", "HMI B_720s", "model − HMI")):
        axes[row, 0].set_ylabel(f"{label}\nCCD row")
    for axis in axes[-1]:
        axis.set_xlabel("CCD column in stored cutout")
    figure.suptitle(title)
    figure.savefig(path, dpi=dpi, facecolor="white")
    figure.clear()


def _save_azimuth_figure(
    path: Path,
    model_observer: np.ndarray,
    reference_observer_raw: np.ndarray,
    reference_flip: np.ndarray,
    valid: np.ndarray,
    strong: np.ndarray,
    *,
    title: str,
    dpi: int,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.figure import Figure

    model_azimuth = (
        np.rad2deg(np.arctan2(model_observer[..., 1], model_observer[..., 0])) % 360.0
    )
    raw_azimuth = (
        np.rad2deg(
            np.arctan2(reference_observer_raw[..., 1], reference_observer_raw[..., 0])
        )
        % 360.0
    )
    reference_azimuth = (raw_azimuth + 180.0 * reference_flip) % 360.0
    director_residual = _angular_residual_deg(model_azimuth, raw_azimuth, 180.0)
    inferred_flip = (
        np.sum(model_observer[..., :2] * reference_observer_raw[..., :2], axis=-1) < 0.0
    )
    mismatch = inferred_flip != reference_flip
    mask = valid & np.isfinite(model_azimuth) & np.isfinite(raw_azimuth)
    strong_mask = mask & strong

    figure = Figure(figsize=(13.0, 7.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(2, 3, sharex=True, sharey=True))
    panels = (
        (
            model_azimuth % 180.0,
            mask,
            "model azimuth modulo 180°",
            "twilight",
            Normalize(0, 180),
        ),
        (raw_azimuth % 180.0, mask, "HMI raw azimuth", "twilight", Normalize(0, 180)),
        (
            director_residual,
            strong_mask,
            "director residual",
            "twilight_shifted",
            Normalize(-90, 90),
        ),
        (
            model_azimuth,
            strong_mask,
            "model signed azimuth",
            "twilight",
            Normalize(0, 360),
        ),
        (
            reference_azimuth,
            strong_mask,
            "HMI disambiguated azimuth",
            "twilight",
            Normalize(0, 360),
        ),
        (
            mismatch.astype(float),
            strong_mask,
            "branch mismatch",
            ListedColormap(("#f7f7f7", "#d73027")),
            Normalize(0, 1),
        ),
    )
    for axis, (values, panel_mask, label, cmap, norm) in zip(
        axes.ravel(), panels, strict=True
    ):
        image = axis.imshow(
            np.where(panel_mask, values, np.nan),
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            rasterized=True,
        )
        colorbar = figure.colorbar(image, ax=axis, shrink=0.78)
        if label != "branch mismatch":
            colorbar.set_label("deg")
        else:
            colorbar.set_ticks((0, 1), labels=("agree", "mismatch"))
        axis.set_title(label)
        axis.set_xlabel("CCD column in stored cutout")
    axes[0, 0].set_ylabel("CCD row")
    axes[1, 0].set_ylabel("CCD row")
    figure.suptitle(title)
    figure.savefig(path, dpi=dpi, facecolor="white")
    figure.clear()


def _save_azimuth_convention_figure(
    path: Path,
    convention_metrics: Mapping[str, Any],
    *,
    title: str,
    dpi: int,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ranking = convention_metrics["ranking"]
    configurations = convention_metrics["configurations"]
    names = [entry["configuration"] for entry in ranking]
    labels = [
        configurations[name]["azimuth_transform"]
        .removeprefix("chi' = ")
        .replace(" deg", "°")
        for name in names
    ]
    theta = [
        configurations[name]["components_strong_transverse"]["b_theta"]["correlation"]
        for name in names
    ]
    phi = [
        configurations[name]["components_strong_transverse"]["b_phi"]["correlation"]
        for name in names
    ]
    score = [entry["score"] for entry in ranking]

    figure = Figure(figsize=(12.0, 6.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    positions = np.arange(len(names), dtype=float)
    width = 0.26
    axis.bar(positions - width, theta, width, label=r"$B_\theta$")
    axis.bar(positions, phi, width, label=r"$B_\phi$")
    axis.bar(positions + width, score, width, label="mean")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels=labels, rotation=30, ha="right")
    axis.set_ylim(-1.0, 1.0)
    axis.set_ylabel("Pearson correlation")
    axis.set_title("Strong-transverse pixels; ordered by mean correlation")
    axis.legend(loc="best")
    figure.suptitle(title)
    figure.savefig(path, dpi=dpi, facecolor="white")
    figure.clear()


def compare_hmi_save_state(
    save_state: str | Path,
    hmi_directory: str | Path,
    output_directory: str | Path,
    *,
    height_km: float = 0.0,
    disambig_bit: int = 0,
    minimum_transverse_gauss: float = 200.0,
    time_tolerance_seconds: float = 2.0,
    alignment_tolerance_pixels: float = 0.1,
    batch_size: int = 4096,
    device: str = "auto",
    dpi: int = 180,
) -> dict[str, Any]:
    """Compare one P3S model with a matching full-disk HMI B_720s record."""

    if type(disambig_bit) is not int or disambig_bit not in {0, 1, 2}:
        raise ValueError("disambig_bit must be 0, 1, or 2.")
    if not math.isfinite(minimum_transverse_gauss) or minimum_transverse_gauss < 0:
        raise ValueError("minimum_transverse_gauss must be finite and non-negative.")
    if not math.isfinite(alignment_tolerance_pixels) or alignment_tolerance_pixels < 0:
        raise ValueError("alignment_tolerance_pixels must be finite and non-negative.")
    if dpi < 50:
        raise ValueError("dpi must be at least 50.")

    paths = _discover_segments(hmi_directory)
    header, full_disk_shape = _reference_header(paths)
    loader = P3SLoader(save_state, device=device)
    if loader.observation.spec.observation_type != "hmi_stokes":
        raise ValueError("HMI comparison requires an hmi_stokes P3S save state.")
    target_time = parse_hmi_tai_time(str(header["T_OBS"]))
    raster_index, time_separation = loader.match_time(
        target_time,
        tolerance_seconds=time_tolerance_seconds,
    )
    if loader.observation_cache_path.is_dir():
        selection = loader.select_raster(index=raster_index)
        raster = selection.raster
        rows, columns, in_bounds, maximum_alignment_residual = _reference_pixel_indices(
            raster, header, full_disk_shape
        )
        if maximum_alignment_residual > alignment_tolerance_pixels:
            raise ValueError(
                "Stored Stokes raster and HMI B_720s WCS do not align to detector "
                "pixels: maximum nearest-pixel residual is "
                f"{maximum_alignment_residual:.4f} px."
            )
        model_cartesian = loader.raster_fields_at_height(
            float(height_km) * 1.0e3,
            batch_size=batch_size,
            raster_index=raster_index,
        )["magnetic_field_gauss"]
        raster_name = selection.name
    else:
        raster = _reference_evaluation_grid(loader, header, target_time)
        rows = raster.rows
        columns = raster.columns
        in_bounds = raster.valid_mask.detach().cpu().numpy()
        maximum_alignment_residual = 0.0
        model_cartesian = loader.fields_at_coordinates(
            raster.coordinates,
            raster.valid_mask,
            float(height_km) * 1.0e3,
            batch_size=batch_size,
        )["magnetic_field_gauss"]
        raster_name = loader.raster_names[raster_index]

    field = _sample_segment(paths["field"], rows, columns, in_bounds)
    inclination = _sample_segment(paths["inclination"], rows, columns, in_bounds)
    raw_azimuth = _sample_segment(paths["azimuth"], rows, columns, in_bounds)
    disambig = _sample_segment(paths["disambig"], rows, columns, in_bounds)
    reference_valid = (
        in_bounds
        & np.isfinite(field)
        & np.isfinite(inclination)
        & np.isfinite(raw_azimuth)
        & np.isfinite(disambig)
        & (field >= 0.0)
        & (inclination >= 0.0)
        & (inclination <= 180.0)
    )
    disambig_integer = np.where(np.isfinite(disambig), disambig, 0).astype(np.int16)
    reference_flip = ((disambig_integer >> disambig_bit) & 1).astype(bool)
    reference_observer_raw = _observer_components(field, inclination, raw_azimuth)
    reference_observer = reference_observer_raw.copy()
    reference_observer[..., :2] *= np.where(reference_flip, -1.0, 1.0)[..., None]

    basis = raster.stokes_basis.detach().cpu().numpy()
    model_observer = np.einsum("...ij,...j->...i", basis, model_cartesian)
    reference_cartesian = np.einsum("...ji,...j->...i", basis, reference_observer)
    surface = raster.surface_position_m.detach().cpu().numpy()
    model_spherical = _spherical_components(model_cartesian, surface)
    reference_spherical = _spherical_components(reference_cartesian, surface)
    model_valid = np.isfinite(model_cartesian).all(axis=-1)
    valid = raster.valid_mask.detach().cpu().numpy() & reference_valid & model_valid
    transverse = field * np.sin(np.deg2rad(inclination))
    strong = valid & (transverse >= minimum_transverse_gauss)

    metrics = _comparison_metrics(
        model_observer,
        model_spherical,
        reference_observer_raw,
        reference_spherical,
        reference_flip,
        valid,
        strong,
    )
    metrics["azimuth_convention_comparison"] = _azimuth_convention_metrics(
        model_observer,
        model_spherical,
        field,
        inclination,
        raw_azimuth,
        reference_flip,
        basis,
        surface,
        valid,
        strong,
    )
    metrics.update(
        {
            "save_state": str(loader.path),
            "hmi_record": paths["field"].name.removesuffix(".field.fits"),
            "hmi_t_obs": str(header["T_OBS"]),
            "hmi_t_rec": str(header["T_REC"]),
            "raster_name": raster_name,
            "raster_index": raster_index,
            "raster_count": loader.raster_count,
            "time_separation_seconds": time_separation,
            "height_km": float(height_km),
            "disambig_bit": disambig_bit,
            "minimum_transverse_gauss": float(minimum_transverse_gauss),
            "maximum_alignment_residual_pixels": maximum_alignment_residual,
            "component_convention": {
                "b_r": "outward",
                "b_theta": "southward",
                "b_phi": "westward/increasing Carrington longitude",
            },
        }
    )

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = str(header["T_REC"]).replace(".", "").replace(":", "")
    evaluation_tag = (
        f"h{height_km:g}km_bit{disambig_bit}_bt{minimum_transverse_gauss:g}G".replace(
            "-", "m"
        )
        .replace("+", "")
        .replace(".", "p")
    )
    label = (
        f"HMI B_720s comparison {header['T_REC']} — h={height_km:g} km, "
        f"DISAMBIG bit {disambig_bit}, Btrans≥{minimum_transverse_gauss:g} G"
    )
    component_path = output / f"hmi_{record}_{evaluation_tag}_spherical_components.png"
    azimuth_path = output / f"hmi_{record}_{evaluation_tag}_azimuth_disambiguation.png"
    convention_path = output / f"hmi_{record}_{evaluation_tag}_azimuth_conventions.png"
    metrics_path = output / f"hmi_{record}_{evaluation_tag}_metrics.json"
    _save_component_figure(
        component_path,
        model_spherical,
        reference_spherical,
        valid,
        title=label,
        dpi=dpi,
    )
    _save_azimuth_figure(
        azimuth_path,
        model_observer,
        reference_observer_raw,
        reference_flip,
        valid,
        strong,
        title=label,
        dpi=dpi,
    )
    _save_azimuth_convention_figure(
        convention_path,
        metrics["azimuth_convention_comparison"],
        title=label,
        dpi=dpi,
    )
    metrics["outputs"] = {
        "spherical_components": str(component_path),
        "azimuth_disambiguation": str(azimuth_path),
        "azimuth_conventions": str(convention_path),
        "metrics": str(metrics_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics


__all__ = ["compare_hmi_save_state"]
