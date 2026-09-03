"""Carrington ray geometry and network coordinates for Hinode/SOT-SP."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import math

import numpy as np
from astropy import units as u
from astropy.constants import R_sun
from astropy.coordinates import SkyCoord, get_sun
from astropy.io import fits
from astropy.time import Time

from .constants import CALIB_SBSP_SOURCE_URL, HINODE_COORDINATE_NORMALIZATION_PIXELS


def _coordinate_xyz_m(coordinate) -> np.ndarray:
    return np.moveaxis(coordinate.cartesian.xyz.to_value(u.m), 0, -1)


def carrington_rays(
    solar_x_arcsec: np.ndarray,
    solar_y_arcsec: np.ndarray,
    times: Sequence[datetime],
    *,
    stokes_reference_angle_deg: float,
    valid_mask: np.ndarray,
    scene_basis_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build Earth-proxy rays and Stokes bases in a fixed Carrington frame."""

    from sunpy.coordinates import frames

    tx = np.asarray(solar_x_arcsec, dtype=np.float64)
    ty = np.asarray(solar_y_arcsec, dtype=np.float64)
    if tx.shape != ty.shape or tx.ndim != 2 or tx.shape[1] != len(times):
        raise ValueError("Hinode pointing arrays must be [slit, scan].")
    angle = float(stokes_reference_angle_deg)
    if not np.isfinite(angle):
        raise ValueError("stokes_reference_angle_deg must be finite.")
    angle_rad = np.deg2rad(angle)
    directions = np.empty((*tx.shape, 3), dtype=np.float64)
    surface_positions = np.empty_like(directions)
    stokes_bases = np.empty((*tx.shape, 3, 3), dtype=np.float64)

    for column, obstime in enumerate(times):
        target_frame = frames.HeliographicCarrington(observer="earth", obstime=obstime)
        centre_hpc = SkyCoord(
            Tx=0.0 * u.arcsec,
            Ty=0.0 * u.arcsec,
            frame=frames.Helioprojective,
            observer="earth",
            obstime=obstime,
        )
        observer = centre_hpc.observer.transform_to(target_frame)
        observer_xyz = np.asarray(_coordinate_xyz_m(observer), dtype=np.float64)

        probes = (
            SkyCoord(
                Tx=np.asarray((0.0, 1.0, 0.0)) * u.arcsec,
                Ty=np.asarray((0.0, 0.0, 1.0)) * u.arcsec,
                frame=frames.Helioprojective,
                observer="earth",
                obstime=obstime,
            )
            .make_3d()
            .transform_to(target_frame)
        )
        probe_ray = _coordinate_xyz_m(probes) - observer_xyz
        probe_ray /= np.linalg.norm(probe_ray, axis=-1, keepdims=True)
        camera_z = probe_ray[0]
        camera_x = probe_ray[1] - np.dot(probe_ray[1], camera_z) * camera_z
        camera_x /= np.linalg.norm(camera_x)
        camera_y = probe_ray[2] - np.dot(probe_ray[2], camera_z) * camera_z
        camera_y -= np.dot(camera_y, camera_x) * camera_x
        camera_y /= np.linalg.norm(camera_y)

        tx_rad = tx[:, column] * u.arcsec.to(u.rad)
        ty_rad = ty[:, column] * u.arcsec.to(u.rad)
        ray = (
            (np.cos(ty_rad) * np.sin(tx_rad))[:, None] * camera_x
            + np.sin(ty_rad)[:, None] * camera_y
            + (np.cos(ty_rad) * np.cos(tx_rad))[:, None] * camera_z
        )
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        projection = ray @ observer_xyz
        discriminant = projection**2 - (
            np.dot(observer_xyz, observer_xyz) - R_sun.to_value(u.m) ** 2
        )
        distance = -projection - np.sqrt(np.clip(discriminant, 0.0, None))
        surface_xyz = observer_xyz + distance[:, None] * ray

        los = -ray
        image_x = (
            camera_x[None, :]
            - np.sum(camera_x[None, :] * los, axis=-1, keepdims=True) * los
        )
        image_x /= np.linalg.norm(image_x, axis=-1, keepdims=True)
        image_y = np.cross(los, image_x)
        image_y /= np.linalg.norm(image_y, axis=-1, keepdims=True)
        q_axis = np.cos(angle_rad) * image_x + np.sin(angle_rad) * image_y
        u_axis = -np.sin(angle_rad) * image_x + np.cos(angle_rad) * image_y

        directions[:, column] = ray
        surface_positions[:, column] = surface_xyz
        stokes_bases[:, column] = np.stack((q_axis, u_axis, los), axis=-2)

    surface_unit = surface_positions / np.linalg.norm(
        surface_positions, axis=-1, keepdims=True
    )
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != tx.shape or not np.any(valid):
        raise ValueError("Carrington scene geometry requires valid on-disk pixels.")
    if scene_basis_rows is None:
        centre = np.mean(surface_unit[valid], axis=0)
        centre /= np.linalg.norm(centre)
        solar_north = np.asarray((0.0, 0.0, 1.0))
        chart_x = np.cross(solar_north, centre)
        if np.linalg.norm(chart_x) < 1.0e-8:
            chart_x = np.asarray((1.0, 0.0, 0.0))
        chart_x /= np.linalg.norm(chart_x)
        chart_y = np.cross(centre, chart_x)
        chart_y /= np.linalg.norm(chart_y)
        scene_basis = np.stack((chart_x, chart_y, centre), axis=0)
        scene_basis_source = "mean valid surface direction of this raster"
    else:
        scene_basis = np.asarray(scene_basis_rows, dtype=np.float64)
        if scene_basis.shape != (3, 3) or not np.isfinite(scene_basis).all():
            raise ValueError("scene_basis_rows must be a finite [3, 3] matrix.")
        if not np.allclose(scene_basis @ scene_basis.T, np.eye(3), atol=1.0e-10):
            raise ValueError("scene_basis_rows must be orthonormal.")
        if np.linalg.det(scene_basis) <= 0:
            raise ValueError("scene_basis_rows must be right-handed.")
        scene_basis_source = "shared multi-raster validation scene"
    metadata = {
        "frame": "HeliographicCarrington Cartesian",
        "observer": "SunPy Earth observer proxy at each DATE_OBS",
        "solar_radius_m": float(R_sun.to_value(u.m)),
        "scene_basis_rows": scene_basis.tolist(),
        "scene_basis_source": scene_basis_source,
        "scene_basis_order": ["chart_x", "chart_y", "chart_normal"],
        "stokes_basis_order": ["+Q", "+U", "toward_observer"],
        "stokes_reference_angle_deg_from_hpc_x_toward_hpc_y": angle,
        "stokes_reference_status": (
            "sp_prep Level-1 has already rotated Q/U with CROTA2 into the solar "
            "reference frame with +Q along solar east-west (HPC +X); zero degrees "
            "therefore applies no second CROTA2 rotation"
        ),
        "stokes_reference_provenance": {
            "processing_stage": "SolarSoft calib_sbsp operation 9 called by sp_prep",
            "input_frame": "FPP polarization reference frame",
            "stored_level1_frame": "solar reference frame; +Q along solar east-west",
            "crota2_application": (
                "already applied to Level-1 Q/U; spatial pointing only here"
            ),
            "source_url": CALIB_SBSP_SOURCE_URL,
        },
    }
    return directions, surface_positions, stokes_bases, metadata


def observer_distance_m(times: Sequence[datetime]) -> np.ndarray:
    """Return the geocentric Sun distance used by the Level-1 geometry proxy."""

    distances = np.asarray(
        [get_sun(Time(time)).distance.to_value(u.m) for time in times],
        dtype=np.float64,
    )
    if not np.isfinite(distances).all() or np.any(distances <= R_sun.to_value(u.m)):
        raise ValueError("Solar ephemeris returned an invalid Sun-observer distance.")
    return distances


def solar_radius_arcsec(
    times: Sequence[datetime], distances_m: np.ndarray
) -> np.ndarray:
    """Calculate the exact angular semidiameter for each observer distance."""

    distances = np.asarray(distances_m, dtype=np.float64)
    if distances.shape != (len(times),):
        raise ValueError("distances_m must contain one value per scan time.")
    ratio = R_sun.to_value(u.m) / distances
    if not np.isfinite(ratio).all() or np.any((ratio <= 0.0) | (ratio >= 1.0)):
        raise ValueError("Solar ephemeris returned an invalid Sun-observer distance.")
    return (np.arcsin(ratio) * u.rad).to_value(u.arcsec)


def coordinate_affine_metadata(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    valid: np.ndarray,
    *,
    headers: Sequence[fits.Header],
    observer_distances_m: np.ndarray,
    slit_indices: np.ndarray,
    scan_indices: np.ndarray,
) -> dict:
    """Build the saved isotropic affine shared by atmospheric networks."""

    valid_x = np.asarray(x_mm, dtype=np.float64)[valid]
    valid_y = np.asarray(y_mm, dtype=np.float64)[valid]
    if valid_x.size == 0:
        raise ValueError("Cannot define a coordinate affine without valid samples.")
    centre = np.asarray(
        (
            0.5 * (float(valid_x.min()) + float(valid_x.max())),
            0.5 * (float(valid_y.min()) + float(valid_y.max())),
        )
    )

    spacings: list[float] = []
    for axis in (0, 1):
        if x_mm.shape[axis] < 2:
            continue
        separation = np.hypot(np.diff(x_mm, axis=axis), np.diff(y_mm, axis=axis))
        detector_indices = slit_indices if axis == 0 else scan_indices
        detector_steps = np.abs(np.diff(np.asarray(detector_indices, dtype=np.float64)))
        if np.any(detector_steps == 0):
            raise ValueError("Hinode detector indices must be unique for scaling.")
        step_shape = [1, 1]
        step_shape[axis] = detector_steps.size
        separation = separation / detector_steps.reshape(step_shape)
        paired_valid = np.take(
            valid, range(valid.shape[axis] - 1), axis=axis
        ) & np.take(valid, range(1, valid.shape[axis]), axis=axis)
        selected = separation[paired_valid]
        spacings.extend(selected[np.isfinite(selected) & (selected > 0)].tolist())

    if not spacings:
        angular_scales = [
            abs(float(header["CDELT2"]))
            for header in headers
            if "CDELT2" in header
            and np.isfinite(float(header["CDELT2"]))
            and float(header["CDELT2"]) != 0.0
        ]
        if not angular_scales:
            raise ValueError(
                "A one-pixel Hinode crop requires CDELT2 to define its "
                "coordinate normalization."
            )
        distance = float(np.median(observer_distances_m))
        spacings = [
            distance * math.tan(value * u.arcsec.to(u.rad)) / 1.0e6
            for value in angular_scales
        ]

    native_spacing_mm = float(np.median(np.asarray(spacings)))
    if not np.isfinite(native_spacing_mm) or native_spacing_mm <= 0:
        raise ValueError("Hinode physical neighbour spacing must be positive.")
    scale_mm = HINODE_COORDINATE_NORMALIZATION_PIXELS * native_spacing_mm
    return {
        "center_mm": centre.tolist(),
        "scale_mm": [scale_mm, scale_mm],
        "formula": "normalized_xy=(solar_xy_mm-center_mm)/scale_mm",
        "isotropic": True,
        "native_neighbor_spacing_mm": native_spacing_mm,
        "normalization_pixels": HINODE_COORDINATE_NORMALIZATION_PIXELS,
        "spacing_estimator": (
            "median positive centre separation per detector-index step in scan/slit "
            "directions; native plate scale for a one-pixel crop"
        ),
    }


__all__ = [
    "carrington_rays",
    "coordinate_affine_metadata",
    "observer_distance_m",
    "solar_radius_arcsec",
]
