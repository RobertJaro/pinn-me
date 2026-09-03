"""Observer, detector, and Carrington geometry for HMI observations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import all_coordinates_from_map

from .acquisition import SEGMENT_KEYS
from .constants import (
    HMI_CCD_SIZE,
    HMI_OBSERVER_VELOCITY_KEYS,
    SPEED_OF_LIGHT_M_PER_S,
)


HMI_STOKES_REFERENCE_URL = "https://arxiv.org/abs/1404.1879"


def observer_velocity_rwn(headers) -> np.ndarray:
    """Validate one acquisition's radial, west, and north observer velocity."""

    if len(headers) != len(SEGMENT_KEYS):
        raise ValueError(
            f"HMI observer velocity requires exactly {len(SEGMENT_KEYS)} segment headers."
        )
    values = []
    for index, header in enumerate(headers):
        segment = (
            SEGMENT_KEYS[index] if index < len(SEGMENT_KEYS) else f"segment {index}"
        )
        missing = [key for key in HMI_OBSERVER_VELOCITY_KEYS if key not in header]
        if missing:
            raise KeyError(
                f"HMI segment {segment} is missing observer-velocity keyword(s) "
                f"{missing}; OBS_VR/OBS_VW/OBS_VN are required in every segment."
            )
        try:
            row = np.asarray(
                [float(header[key]) for key in HMI_OBSERVER_VELOCITY_KEYS],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"HMI segment {segment} has invalid OBS_VR/OBS_VW/OBS_VN values."
            ) from error
        if not np.isfinite(row).all():
            raise ValueError(
                f"HMI segment {segment} has non-finite OBS_VR/OBS_VW/OBS_VN values."
            )
        if np.linalg.norm(row) >= SPEED_OF_LIGHT_M_PER_S:
            raise ValueError(
                f"HMI segment {segment} has a non-physical observer velocity."
            )
        values.append(row)
    if not values:
        raise ValueError("HMI observer velocity requires at least one segment header.")
    stacked = np.stack(values)
    consistent = np.all(np.isclose(stacked, stacked[0], rtol=0.0, atol=1.0e-6), axis=1)
    if not np.all(consistent):
        first_inconsistent = int(np.flatnonzero(~consistent)[0])
        segment = (
            SEGMENT_KEYS[first_inconsistent]
            if first_inconsistent < len(SEGMENT_KEYS)
            else f"segment {first_inconsistent}"
        )
        raise ValueError(
            "All 24 HMI segments in one acquisition must share identical "
            "OBS_VR/OBS_VW/OBS_VN values; "
            f"I0={stacked[0].tolist()}, {segment}={stacked[first_inconsistent].tolist()}."
        )
    return stacked[0]


def project_observer_velocity_to_los(
    observer_xyz_m,
    toward_observer_los,
    observer_velocity_rwn_m_per_s,
) -> np.ndarray:
    """Project ``[radial-away, west, north]`` velocity onto toward-observer LOS."""

    observer = np.asarray(observer_xyz_m, dtype=np.float64)
    los = np.asarray(toward_observer_los, dtype=np.float64)
    velocity = np.asarray(observer_velocity_rwn_m_per_s, dtype=np.float64)
    if observer.shape != (3,) or not np.isfinite(observer).all():
        raise ValueError("HMI observer position must be one finite Cartesian vector.")
    if los.shape[-1:] != (3,) or not np.isfinite(los).all():
        raise ValueError(
            "HMI toward-observer LOS vectors must be finite three-vectors."
        )
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError(
            "HMI observer velocity must contain finite radial/west/north components."
        )
    if np.linalg.norm(velocity) >= SPEED_OF_LIGHT_M_PER_S:
        raise ValueError("HMI observer velocity must be subluminal.")
    observer_norm = np.linalg.norm(observer)
    los_norm = np.linalg.norm(los, axis=-1, keepdims=True)
    if observer_norm <= 0 or np.any(los_norm <= 0):
        raise ValueError("HMI observer position and LOS vectors must be non-zero.")
    radial = observer / observer_norm
    los = los / los_norm
    north = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    north -= np.dot(north, radial) * radial
    north_norm = np.linalg.norm(north)
    if north_norm <= 1.0e-12:
        raise ValueError(
            "HMI observer direction is degenerate with the solar rotation axis."
        )
    north /= north_norm
    west = np.cross(north, radial)
    observer_velocity_cartesian = (
        velocity[0] * radial + velocity[1] * west + velocity[2] * north
    )
    return np.einsum("...i,i->...", los, observer_velocity_cartesian)


@dataclass(frozen=True, slots=True)
class HMIDetectorGrid:
    """Compact regular CCD-coordinate grid used by HMI response phase maps."""

    spatial_shape: tuple[int, int]
    origin_xy_pixel: tuple[float, float]
    ccd_size: int = HMI_CCD_SIZE

    def __post_init__(self) -> None:
        height, width = map(int, self.spatial_shape)
        if height < 1 or width < 1:
            raise ValueError("HMI detector grids require a positive spatial shape.")
        if int(self.ccd_size) < 2:
            raise ValueError("HMI detector grids require ccd_size >= 2.")
        origin = tuple(map(float, self.origin_xy_pixel))
        if len(origin) != 2 or not all(map(math.isfinite, origin)):
            raise ValueError("HMI detector origin must be one finite (x, y) pair.")
        if (
            origin[0] < 0
            or origin[1] < 0
            or origin[0] + width > int(self.ccd_size)
            or origin[1] + height > int(self.ccd_size)
        ):
            raise ValueError("HMI detector grid must lie entirely inside the CCD.")
        object.__setattr__(self, "spatial_shape", (height, width))
        object.__setattr__(self, "origin_xy_pixel", origin)
        object.__setattr__(self, "ccd_size", int(self.ccd_size))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (*self.spatial_shape, 2)

    @property
    def origin_xy_normalized(self) -> tuple[float, float]:
        scale = 1.0 / (self.ccd_size - 1)
        return tuple(value * scale for value in self.origin_xy_pixel)

    @property
    def pixel_scale_normalized(self) -> float:
        return 1.0 / (self.ccd_size - 1)

    def at_indices(self, row_column) -> np.ndarray:
        """Return normalized phase-map coordinates for ``[..., row, column]``."""

        indices = np.asarray(row_column)
        if indices.shape[-1:] != (2,):
            raise ValueError("HMI detector indices must end in [row, column].")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("HMI detector indices must be integers.")
        if (
            np.any(indices[..., 0] < 0)
            or np.any(indices[..., 0] >= self.spatial_shape[0])
            or np.any(indices[..., 1] < 0)
            or np.any(indices[..., 1] >= self.spatial_shape[1])
        ):
            raise IndexError("HMI detector indices lie outside the image.")
        xy = indices[..., ::-1].astype(np.float64, copy=False)
        xy = xy + np.asarray(self.origin_xy_pixel, dtype=np.float64)
        return (xy / (self.ccd_size - 1)).astype(np.float32)

    def as_array(self) -> np.ndarray:
        rows, columns = np.indices(self.spatial_shape, dtype=np.int64)
        return self.at_indices(np.stack((rows, columns), axis=-1))

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self.as_array()
        if copy is True:
            values = values.copy()
        return values if dtype is None else values.astype(dtype, copy=False)


def detector_coordinates(s_map, ccd_size: int = HMI_CCD_SIZE) -> HMIDetectorGrid:
    """Create normalized full-detector coordinates for a full disk or cutout."""

    height, width = s_map.data.shape
    if (height, width) == (ccd_size, ccd_size):
        origin = (0.0, 0.0)
    elif "CCD_X0" in s_map.meta and "CCD_Y0" in s_map.meta:
        origin = (float(s_map.meta["CCD_X0"]), float(s_map.meta["CCD_Y0"]))
    else:
        raise ValueError("HMI cutouts require CCD_X0/CCD_Y0 detector-origin metadata.")
    return HMIDetectorGrid(
        spatial_shape=(height, width), origin_xy_pixel=origin, ccd_size=ccd_size
    )


def detector_stokes_basis(
    los: np.ndarray,
    hpc_x: np.ndarray,
    hpc_y: np.ndarray,
    pixel_scale_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return HMI's fixed ``[CCD-up, counter-clockwise]`` transverse axes."""

    matrix = np.asarray(pixel_scale_matrix, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("HMI WCS must provide a finite 2x2 pixel-scale matrix.")
    detector_x = matrix[:, 0]
    detector_y = matrix[:, 1]
    detector_x_norm = np.linalg.norm(detector_x)
    detector_y_norm = np.linalg.norm(detector_y)
    if detector_x_norm <= 0 or detector_y_norm <= 0:
        raise ValueError("HMI WCS detector axes must be non-zero.")
    detector_x = detector_x / detector_x_norm
    detector_y = detector_y / detector_y_norm
    if abs(float(np.dot(detector_x, detector_y))) > 1.0e-5:
        raise ValueError("HMI WCS detector axes must be orthogonal.")
    if float(np.linalg.det(np.stack((detector_x, detector_y), axis=1))) <= 0:
        raise ValueError("HMI WCS detector axes must preserve image orientation.")

    q_axis = detector_y[0] * hpc_x + detector_y[1] * hpc_y
    q_axis -= np.sum(q_axis * los, axis=-1, keepdims=True) * los
    q_axis /= np.linalg.norm(q_axis, axis=-1, keepdims=True)
    u_axis = np.cross(los, q_axis)
    u_axis /= np.linalg.norm(u_axis, axis=-1, keepdims=True)
    return q_axis, u_axis


def build_geometry(
    s_map,
    valid: np.ndarray,
    observer_velocity_rwn_m_per_s,
    *,
    scene_basis_rows=None,
):
    """Build rays, surface intersections, Stokes bases, and observer velocity."""

    map_coordinates = all_coordinates_from_map(s_map)
    obstime = s_map.date
    target_frame = frames.HeliographicCarrington(
        observer=s_map.observer_coordinate, obstime=obstime
    )
    surface_coordinate = map_coordinates.transform_to(target_frame)
    observer_coordinate = s_map.observer_coordinate.transform_to(target_frame)
    observer_xyz = np.moveaxis(
        observer_coordinate.cartesian.xyz.to_value(u.m), 0, -1
    ).astype(np.float64)
    surface_xyz = np.moveaxis(
        surface_coordinate.cartesian.xyz.to_value(u.m), 0, -1
    ).astype(np.float64)
    ray = surface_xyz - observer_xyz
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    los = -ray
    observer_los_velocity = project_observer_velocity_to_los(
        observer_xyz, los, observer_velocity_rwn_m_per_s
    )

    probes = (
        SkyCoord(
            Tx=np.asarray((0.0, 1.0, 0.0)) * u.arcsec,
            Ty=np.asarray((0.0, 0.0, 1.0)) * u.arcsec,
            frame=frames.Helioprojective,
            observer=s_map.observer_coordinate,
            obstime=obstime,
        )
        .make_3d()
        .transform_to(target_frame)
    )
    probe_xyz = np.moveaxis(probes.cartesian.xyz.to_value(u.m), 0, -1)
    probe_ray = probe_xyz - observer_xyz
    probe_ray /= np.linalg.norm(probe_ray, axis=-1, keepdims=True)
    camera_z = probe_ray[0]
    camera_x = probe_ray[1] - np.dot(probe_ray[1], camera_z) * camera_z
    camera_x /= np.linalg.norm(camera_x)
    image_x = camera_x - np.sum(camera_x * los, axis=-1, keepdims=True) * los
    image_x /= np.linalg.norm(image_x, axis=-1, keepdims=True)
    image_y = np.cross(los, image_x)
    image_y /= np.linalg.norm(image_y, axis=-1, keepdims=True)
    q_axis, u_axis = detector_stokes_basis(
        los, image_x, image_y, s_map.wcs.pixel_scale_matrix
    )
    stokes_basis = np.stack((q_axis, u_axis, los), axis=-2)

    surface_unit = surface_xyz / np.linalg.norm(surface_xyz, axis=-1, keepdims=True)
    geometry_valid = (
        valid
        & np.isfinite(surface_unit).all(axis=-1)
        & np.isfinite(ray).all(axis=-1)
        & np.isfinite(stokes_basis).all(axis=(-2, -1))
        & np.isfinite(observer_los_velocity)
    )
    if not np.any(geometry_valid):
        raise ValueError("HMI WCS contains no finite on-disk Carrington intersections.")
    if scene_basis_rows is None:
        centre = surface_unit[geometry_valid].mean(axis=0)
        centre /= np.linalg.norm(centre)
        chart_x = np.cross(np.asarray((0.0, 0.0, 1.0)), centre)
        if np.linalg.norm(chart_x) < 1.0e-8:
            chart_x = np.asarray((1.0, 0.0, 0.0))
        chart_x /= np.linalg.norm(chart_x)
        chart_y = np.cross(centre, chart_x)
        chart_y /= np.linalg.norm(chart_y)
        scene_basis = np.stack((chart_x, chart_y, centre))
    else:
        scene_basis = np.asarray(scene_basis_rows, dtype=np.float64)
        if (
            scene_basis.shape != (3, 3)
            or not np.allclose(
                scene_basis @ scene_basis.T, np.eye(3), rtol=0.0, atol=1.0e-10
            )
            or np.linalg.det(scene_basis) <= 0
        ):
            raise ValueError(
                "scene_basis_rows must be a right-handed orthonormal basis."
            )
    mu = np.sum(surface_unit * los, axis=-1)
    mu[~geometry_valid] = np.nan
    detector_y = np.asarray(s_map.wcs.pixel_scale_matrix, dtype=np.float64)[:, 1]
    solar_radius_m = float(s_map.rsun_meters.to_value(u.m))
    surface_radius_m = np.linalg.norm(surface_xyz[geometry_valid], axis=-1)
    if not np.allclose(
        surface_radius_m,
        solar_radius_m,
        rtol=0.0,
        atol=max(1.0, solar_radius_m * 1.0e-8),
    ):
        raise ValueError("HMI surface intersections do not match the WCS solar radius.")
    metadata = {
        "frame": "HeliographicCarrington Cartesian",
        "observer": "exact SDO observer coordinate at the nominal T_OBS instant",
        "solar_radius_m": solar_radius_m,
        "scene_basis_rows": scene_basis.tolist(),
        "scene_basis_order": ["chart_x", "chart_y", "chart_normal"],
        "stokes_basis_order": ["+Q", "+U", "toward_observer"],
        "stokes_reference": "CCD column-up; azimuth increases counter-clockwise",
        "stokes_reference_source": HMI_STOKES_REFERENCE_URL,
        "detector_y_hpc_coefficients": (
            detector_y / np.linalg.norm(detector_y)
        ).tolist(),
        "observer_velocity_basis": [
            "radial_away_from_sun",
            "solar_west",
            "solar_north",
        ],
    }
    return ray, surface_xyz, stokes_basis, mu, observer_los_velocity, metadata


__all__ = [
    "HMIDetectorGrid",
    "build_geometry",
    "detector_coordinates",
    "detector_stokes_basis",
    "observer_velocity_rwn",
    "project_observer_velocity_to_los",
]
