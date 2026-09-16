"""SunPy pixel-to-ray geometry for the AIA training-store adapter."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any
import numpy as np
from prom3theus.observations.pixel_footprint import CenteredPixelCutout
from .preparation import AIA_SURFACE_FRAME, AIA_RAY_DIRECTION_CONVENTION, AIA_TIME_REPRESENTATION

AIA_REGISTERED_PLATE_SCALE_ARCSEC_PER_PIXEL = 0.6
AIA_NATIVE_PIXEL_SOLID_ANGLE_RELATIVE_TOLERANCE = 0.02
AIA_REFERENCE_NATIVE_PIXEL_SOLID_ANGLE_SR = 8.461580394691914e-12

@dataclass(frozen=True, slots=True)
class AIACarringtonGeometry:
    """Geometry evaluated at pixel centers of one native-grid AIA submap."""

    surface_position_m: np.ndarray
    ray_direction: np.ndarray
    on_disk_mask: np.ndarray
    footprint_mask: np.ndarray
    solar_radius_m: float
    absolute_tai_seconds: float
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def carrington_geometry(image_map, footprint, runtime) -> AIACarringtonGeometry:
    u = runtime.u
    hgc_frame = runtime.HeliographicCarrington(
        observer=image_map.observer_coordinate, obstime=image_map.reference_date,
    )
    hpc = runtime.all_coordinates_from_map(image_map)
    on_disk = np.asarray(runtime.coordinate_is_on_solar_disk(hpc), dtype=bool)
    surface_hgc = hpc.make_3d().transform_to(hgc_frame)
    surface = np.moveaxis(
        np.asarray(surface_hgc.cartesian.xyz.to_value(u.m), dtype=np.float64),
        0,
        -1,
    )
    observer = np.asarray(
        image_map.observer_coordinate.transform_to(
            hgc_frame
        ).cartesian.xyz.to_value(u.m),
        dtype=np.float64,
    )
    center = runtime.SkyCoord(
        lon=footprint.longitude_deg * u.deg,
        lat=footprint.latitude_deg * u.deg,
        radius=image_map.rsun_meters,
        frame=hgc_frame,
    )
    center_surface = np.asarray(
        center.cartesian.xyz.to_value(u.m), dtype=np.float64
    )
    center_ray = center_surface - observer
    center_ray /= np.linalg.norm(center_ray)

    finite_surface = np.isfinite(surface).all(axis=-1)
    on_disk &= finite_surface
    surface[~on_disk] = center_surface
    direction = surface - observer
    norm = np.linalg.norm(direction, axis=-1)
    finite_direction = np.isfinite(norm) & (norm > 0.0)
    on_disk &= finite_direction
    surface[~on_disk] = center_surface
    direction[on_disk] /= norm[on_disk, None]
    direction[~on_disk] = center_ray

    surface_norm = np.linalg.norm(surface, axis=-1)
    mu = np.sum((surface / surface_norm[..., None]) * -direction, axis=-1)
    on_disk &= np.isfinite(mu) & (mu > 0.0) & (mu <= 1.0 + 2.0e-12)
    surface[~on_disk] = center_surface
    direction[~on_disk] = center_ray

    longitude = np.asarray(surface_hgc.lon.to_value(u.deg), dtype=np.float64)
    latitude = np.asarray(surface_hgc.lat.to_value(u.deg), dtype=np.float64)
    delta_lon = (longitude - footprint.longitude_deg + 180.0) % 360.0 - 180.0
    footprint_mask = on_disk.copy()
    if not isinstance(footprint, CenteredPixelCutout):
        footprint_mask &= (
            (np.abs(delta_lon) <= footprint.width_deg / 2.0)
            & (np.abs(latitude - footprint.latitude_deg) <= footprint.height_deg / 2.0)
        )
    if not footprint_mask.any():
        raise ValueError(
            "Carrington footprint has no visible pixel centers in this AIA map."
        )
    if not isinstance(footprint, CenteredPixelCutout) and (
        footprint_mask[0].any()
        or footprint_mask[-1].any()
        or footprint_mask[:, 0].any()
        or footprint_mask[:, -1].any()
    ):
        raise ValueError(
            "Carrington footprint reaches a native cutout edge; increase "
            "boundary sampling or use a smaller visible footprint."
        )
    plate_scale = np.abs(
        np.asarray(
            u.Quantity(image_map.scale).to_value(u.arcsec / u.pix),
            dtype=np.float64,
        )
    )
    if plate_scale.shape != (2,) or not np.allclose(
        plate_scale,
        AIA_REGISTERED_PLATE_SCALE_ARCSEC_PER_PIXEL,
        rtol=AIA_NATIVE_PIXEL_SOLID_ANGLE_RELATIVE_TOLERANCE,
        atol=0.0,
    ):
        raise ValueError(
            "Prepared AIA cutout does not retain the 0.6 arcsec registered "
            "native pixel scale."
        )
    pixel_solid_angle_sr = float(np.prod(np.deg2rad(plate_scale / 3600.0)))
    if not math.isclose(
        pixel_solid_angle_sr,
        AIA_REFERENCE_NATIVE_PIXEL_SOLID_ANGLE_SR,
        rel_tol=AIA_NATIVE_PIXEL_SOLID_ANGLE_RELATIVE_TOLERANCE,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Prepared AIA cutout pixel solid angle is inconsistent with "
            "the response calibration."
        )
    return AIACarringtonGeometry(
        surface_position_m=surface,
        ray_direction=direction,
        on_disk_mask=on_disk,
        footprint_mask=footprint_mask,
        solar_radius_m=float(image_map.rsun_meters.to_value(u.m)),
        absolute_tai_seconds=float(image_map.reference_date.to_value("unix_tai")),
        provenance={
            "surface_intersection": "sunpy.Helioprojective.make_3d",
            "surface_frame": AIA_SURFACE_FRAME,
            "ray_construction": "unit(surface_hgc_cartesian-observer_hgc_cartesian)",
            "ray_direction": AIA_RAY_DIRECTION_CONVENTION,
            "time_representation": AIA_TIME_REPRESENTATION,
            "native_registered_grid_retained": True,
            "hmi_reprojection_performed": False,
            "crop_completeness_check": (
                "full_centered_pixel_rectangle" if isinstance(footprint, CenteredPixelCutout)
                else "footprint_mask_clear_of_all_native_submap_edges"
            ),
            "registered_plate_scale_arcsec_per_pixel": plate_scale.tolist(),
            "native_pixel_solid_angle_sr": pixel_solid_angle_sr,
            "footprint": footprint.metadata(),
        },
    )
