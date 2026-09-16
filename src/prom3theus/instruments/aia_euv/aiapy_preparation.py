"""Calibrate and crop AIA files, then stream them into the training image store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from prom3theus.observations.pixel_footprint import (
    CenteredPixelCutout, HMI_PIXEL_SCALE_ARCSEC, pixel_dimensions, centered_submap,
)

from prom3theus.resources import (
    AIA_RESOURCE_SET_ID,
    resolve_resource_reference,
    validate_resource_set,
)

from .download import _content_digest, _sha256_file, load_aia_acquisition_manifest
from .geometry import (
    AIACarringtonGeometry, carrington_geometry,
    AIA_REGISTERED_PLATE_SCALE_ARCSEC_PER_PIXEL,
    AIA_NATIVE_PIXEL_SOLID_ANGLE_RELATIVE_TOLERANCE,
    AIA_REFERENCE_NATIVE_PIXEL_SOLID_ANGLE_SR,
)
from .observation import AIA_INTENSITY_UNIT
from .preparation import (
    AIA_CHANNELS_ANGSTROM,
    AIA_RAY_DIRECTION_CONVENTION,
    AIA_SURFACE_FRAME,
    AIA_TIME_REPRESENTATION,
    RegisteredAIAChannelRaster,
    prepare_aia_observation_store,
)


AIAPY_PREPARATION_VERSION = "0.12.1"
AIAPY_PREPARATION_PYTHON = ">=3.12"
AIAPY_DEGRADATION_CALIBRATION_VERSION = 10
AIA_V10_CORRECTION_SOURCE_SHA256 = (
    "0a3f2db39d05c44185f6fdeec928089fb55d1ce1e0a805145050c6356cbc6e98"
)
AIA_V10_CORRECTION_SEMANTIC_SHA256 = (
    "7cbdb1e4387ba91196a19677d9f6755a25d8812140d5843cc7c35942babf73b0"
)
AIAPY_PIPELINE_ORDER = (
    "aiapy.update_pointing",
    "aiapy.register",
    "divide_by_exposure_seconds",
    "native_grid_carrington_cutout",
    "store_applies_degradation_once",
)

def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _explicit_ecsv_file(value: str | os.PathLike[str], name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be an explicit local ECSV path.")
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() != ".ecsv":
        raise ValueError(f"{name} must be an explicit .ecsv file.")
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}.")
    return path


def _calibration_convention_id() -> str:
    metadata = validate_resource_set(AIA_RESOURCE_SET_ID)
    contract = metadata.get("scientific_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("The verified AIA response has no scientific contract.")
    try:
        channels = tuple(int(value) for value in contract.get("channels", ()))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "The verified AIA response channel order is unsupported."
        ) from error
    if channels != AIA_CHANNELS_ANGSTROM:
        raise RuntimeError("The verified AIA response channel order is unsupported.")
    value = contract.get("calibration_convention_id")
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        raise RuntimeError("The verified AIA response has no calibration ID.")
    resource = resolve_resource_reference(f"{AIA_RESOURCE_SET_ID}:instrument_aia_euv")
    with resource.open("r", encoding="utf-8") as stream:
        instrument = json.load(stream)
    calibration = instrument.get("calibration_contract", {})
    degradation_contract = calibration.get("degradation", {})
    if (
        instrument.get("calibration_convention_id") != value
        or degradation_contract.get("calibration_version")
        != AIAPY_DEGRADATION_CALIBRATION_VERSION
        or degradation_contract.get("source_sha256") != AIA_V10_CORRECTION_SOURCE_SHA256
        or not math.isclose(
            float(calibration.get("native_pixel_solid_angle_sr", float("nan"))),
            AIA_REFERENCE_NATIVE_PIXEL_SOLID_ANGLE_SR,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError(
            "The AIA preparation calibration constants do not match the "
            "verified response resource."
        )
    return value


@dataclass(frozen=True, slots=True)
class CarringtonCutout:
    """Reusable pixel-center cutout in Heliographic Carrington degrees.

    The rectangle is centered at ``(longitude_deg, latitude_deg)``.  Width and
    height are full angular extents on the solar surface, not helioprojective
    arcseconds and not an output raster shape.  The production backend samples
    this rectangle to find a conservative native-grid bounding submap, then
    evaluates exact Carrington membership at every retained pixel center.
    """

    longitude_deg: float
    latitude_deg: float
    width_deg: float
    height_deg: float
    boundary_samples_per_axis: int = 33

    def __post_init__(self) -> None:
        longitude = _finite(self.longitude_deg, "longitude_deg") % 360.0
        latitude = _finite(self.latitude_deg, "latitude_deg")
        width = _positive(self.width_deg, "width_deg")
        height = _positive(self.height_deg, "height_deg")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("latitude_deg must lie in [-90, 90].")
        if width > 180.0:
            raise ValueError("width_deg must not exceed 180 degrees.")
        if height > 180.0 or (
            latitude - height / 2.0 < -90.0 or latitude + height / 2.0 > 90.0
        ):
            raise ValueError("height_deg must define a rectangle within the poles.")
        if (
            type(self.boundary_samples_per_axis) is not int
            or self.boundary_samples_per_axis < 5
        ):
            raise ValueError("boundary_samples_per_axis must be an integer >= 5.")
        object.__setattr__(self, "longitude_deg", longitude)
        object.__setattr__(self, "latitude_deg", latitude)
        object.__setattr__(self, "width_deg", width)
        object.__setattr__(self, "height_deg", height)

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "carrington_center_angular_rectangle",
            "longitude_deg": self.longitude_deg,
            "latitude_deg": self.latitude_deg,
            "width_deg": self.width_deg,
            "height_deg": self.height_deg,
            "boundary_samples_per_axis": self.boundary_samples_per_axis,
            "grid_policy": "native_aiapy_registered_submap_no_reprojection",
            "membership": "exact_pixel_center_carrington_rectangle",
        }


def _load_aiapy_runtime() -> SimpleNamespace:
    """Import the optional calibration stack only when preparation is requested."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.table import QTable
    from astropy.time import Time
    from aiapy.calibrate import degradation, register, update_pointing
    from sunpy.coordinates import HeliographicCarrington
    from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk

    return SimpleNamespace(
        u=u, SkyCoord=SkyCoord, fits=fits, QTable=QTable, Time=Time,
        degradation=degradation, register=register, update_pointing=update_pointing,
        HeliographicCarrington=HeliographicCarrington, Map=Map,
        all_coordinates_from_map=all_coordinates_from_map,
        coordinate_is_on_solar_disk=coordinate_is_on_solar_disk,
    )


class AiapyPreparationBackend:
    """SunPy/aiapy operations using shared, locally loaded calibration tables."""

    def __init__(self) -> None:
        self._runtime: SimpleNamespace | None = None

    @property
    def runtime(self) -> SimpleNamespace:
        if self._runtime is None:
            self._runtime = _load_aiapy_runtime()
        return self._runtime

    def dependency_metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "aiapy",
            **{name: importlib.metadata.version(name) for name in ("aiapy", "astropy", "sunpy")},
            "degradation_calibration_version": AIAPY_DEGRADATION_CALIBRATION_VERSION,
        }

    def _read_ecsv(self, path: Path):
        table = self.runtime.QTable.read(path, format="ascii.ecsv")
        for column in ("T_START", "T_STOP"):
            if not isinstance(table[column], self.runtime.Time):
                table[column] = self.runtime.Time(table[column], scale="utc")
        return table

    def read_pointing_table(self, path: Path) -> Any:
        return self._read_ecsv(path)

    def read_correction_table(self, path: Path) -> Any:
        table = self._read_ecsv(path)
        table = table[np.asarray(table["VER_NUM"]) == AIAPY_DEGRADATION_CALIBRATION_VERSION]
        if not len(table):
            raise ValueError("AIA correction table must contain response-compatible V10 calibration.")
        return table

    def load_level1_map(self, path: Path, record: Mapping[str, Any]) -> Any:
        # The header scan already selected and checked the image HDU.
        with self.runtime.fits.open(path, memmap=False) as hdul:
            hdu = hdul[record["image_hdu_index"]]
            data, header = hdu.data, hdu.header.copy()
        bunit = str(header.get("BUNIT", "DN")).lower().replace(" ", "")
        if any(unit in bunit for unit in ("/s", "s-1", "s^-1")):
            raise ValueError("AIA input is already exposure normalized.")
        return self.runtime.Map(data, header)

    def data(self, image_map: Any) -> np.ndarray:
        return np.asarray(image_map.data)

    def with_data(self, image_map: Any, data: np.ndarray) -> Any:
        return self.runtime.Map(np.asarray(data), image_map.meta.copy())

    def update_pointing(self, image_map: Any, pointing_table: Any) -> Any:
        return self.runtime.update_pointing(image_map, pointing_table=pointing_table)

    def register(self, image_map: Any, *, missing: float, order: int) -> Any:
        return self.runtime.register(
            image_map, missing=missing, order=order, method="scipy"
        )

    def _hgc_frame(self, image_map: Any) -> Any:
        return self.runtime.HeliographicCarrington(
            observer=image_map.observer_coordinate,
            obstime=image_map.reference_date,
        )

    def crop(self, image_map: Any, footprint: CarringtonCutout | CenteredPixelCutout) -> Any:
        u = self.runtime.u
        if isinstance(footprint, CenteredPixelCutout):
            width, height = pixel_dimensions(
                footprint.width_pixels, footprint.height_pixels, HMI_PIXEL_SCALE_ARCSEC,
                u.Quantity(image_map.scale).to_value(u.arcsec / u.pix),
            )
            return centered_submap(
                image_map, footprint.longitude_deg, footprint.latitude_deg, width, height,
            )
        corners = self.runtime.SkyCoord(
            lon=np.array([footprint.longitude_deg - footprint.width_deg / 2,
                          footprint.longitude_deg + footprint.width_deg / 2]) * u.deg,
            lat=np.array([footprint.latitude_deg - footprint.height_deg / 2,
                          footprint.latitude_deg + footprint.height_deg / 2]) * u.deg,
            radius=image_map.rsun_meters, frame=self._hgc_frame(image_map),
        )
        return image_map.submap(corners)

    def degradation_factor(
        self,
        image_map: Any,
        *,
        channel_angstrom: int,
        correction_table: Any,
    ) -> float:
        value = self.runtime.degradation(
            channel_angstrom * self.runtime.u.angstrom,
            image_map.reference_date,
            correction_table=correction_table,
        )
        array = np.asarray(value.to_value(self.runtime.u.one)).reshape(-1)
        if array.size != 1:
            raise ValueError("aiapy returned a non-scalar degradation factor.")
        return _positive(array[0], "AIA degradation factor")

    def carrington_geometry(self, image_map, footprint) -> AIACarringtonGeometry:
        return carrington_geometry(image_map, footprint, self.runtime)


def _validated_records(root: Path, manifest: Mapping[str, Any]):
    """Flatten the header scan into independent image jobs."""
    return tuple(
        (f"target-{index:04d}", root / record["file"], record)
        for index, target in enumerate(manifest["targets"])
        for record in target["records"]
    )


def _same_shape(values: Mapping[str, np.ndarray]) -> tuple[int, int]:
    shapes = {name: tuple(value.shape) for name, value in values.items()}
    unique = set(shapes.values())
    if len(unique) != 1:
        raise ValueError(f"Registered/cropped AIA products are misaligned: {shapes}.")
    shape = next(iter(unique))
    if len(shape) != 2 or 0 in shape:
        raise ValueError("Prepared AIA maps must be non-empty two-dimensional arrays.")
    return shape


def _prepare_record(
    *,
    backend: AiapyPreparationBackend,
    source_path: Path,
    record: Mapping[str, Any],
    exposure_group: str,
    footprint: CarringtonCutout | CenteredPixelCutout,
    pointing_table: Any,
    correction_table: Any,
    calibration_convention_id: str,
    table_provenance: Mapping[str, Any],
) -> RegisteredAIAChannelRaster:
    raw_map = backend.load_level1_map(source_path, record)
    raw_data = np.asarray(backend.data(raw_map), dtype=np.float64)
    if raw_data.ndim != 2 or not raw_data.size:
        raise ValueError("AIA backend returned an invalid Level-1 image array.")
    raw_valid = np.isfinite(raw_data)
    if not raw_valid.any():
        raise ValueError("AIA Level-1 image contains no finite detector samples.")

    # Pointing is updated before any interpolation.  Auxiliary maps inherit the
    # resulting WCS, so the mask cannot trigger a hidden
    # second pointing lookup.
    pointed = backend.update_pointing(raw_map, pointing_table)
    pointed_data = np.asarray(backend.data(pointed), dtype=np.float64)
    if pointed_data.shape != raw_data.shape:
        raise ValueError("AIA pointing update unexpectedly changed the image shape.")
    mask_map = backend.with_data(pointed, raw_valid.astype(np.float64))

    registered = backend.register(pointed, missing=float("nan"), order=3)
    registered_mask = backend.register(mask_map, missing=0.0, order=0)
    data_dn = np.asarray(backend.data(registered), dtype=np.float64)
    mask_values = np.asarray(backend.data(registered_mask), dtype=np.float64)
    spatial_shape = _same_shape(
        {"data": data_dn, "mask": mask_values}
    )
    exposure_seconds = _positive(record["exposure_seconds"], "AIA manifest EXPTIME")
    rate = data_dn / exposure_seconds

    # Exposure normalization precedes the geometry-only native-grid slice.
    # Slicing cannot mix pixels or alter per-native-pixel radiometry.
    registered_rate = backend.with_data(registered, rate)
    registered_rate = backend.crop(registered_rate, footprint)
    registered_mask = backend.crop(registered_mask, footprint)
    rate = np.asarray(backend.data(registered_rate), dtype=np.float64)
    mask_values = np.asarray(backend.data(registered_mask), dtype=np.float64)
    spatial_shape = _same_shape(
        {"rate": rate, "mask": mask_values}
    )

    geometry = backend.carrington_geometry(registered_rate, footprint)
    geometry_shapes = {
        "on_disk": np.asarray(geometry.on_disk_mask),
        "footprint": np.asarray(geometry.footprint_mask),
    }
    if any(tuple(value.shape) != spatial_shape for value in geometry_shapes.values()):
        raise ValueError("AIA Carrington masks do not match the cropped native grid.")
    if tuple(np.asarray(geometry.ray_direction).shape) != (*spatial_shape, 3) or (
        tuple(np.asarray(geometry.surface_position_m).shape) != (*spatial_shape, 3)
    ):
        raise ValueError("AIA Carrington vector geometry has an invalid shape.")

    on_disk = np.asarray(geometry.on_disk_mask, dtype=bool)
    footprint_mask = np.asarray(geometry.footprint_mask, dtype=bool)
    valid = (
        (mask_values >= 0.5)
        & np.isfinite(rate)
        & on_disk
        & footprint_mask
    )
    if not valid.any():
        raise ValueError("AIA preparation produced no valid on-disk footprint pixels.")

    # Finite detector negatives are retained.  Only samples excluded by the
    # explicit final validity mask receive placeholders.
    rate = np.where(valid, rate, 0.0)
    factor = backend.degradation_factor(
        registered_rate,
        channel_angstrom=record["channel_angstrom"],
        correction_table=correction_table,
    )
    provenance = {
        "source": {
            "record_id": record["record_id"],
            "relative_file": record["file"],
            "sha256": record["sha256"],
            "image_hdu_index": record["image_hdu_index"],
            "image_hdu_name": record["image_hdu_name"],
        },
        "operation_order": list(AIAPY_PIPELINE_ORDER),
        "exposure_seconds": exposure_seconds,
        "geometry": dict(geometry.provenance),
        "tables": dict(table_provenance),
        "degradation": {
            "factor": factor,
            "evaluated_by_backend": True,
            "applied_by_backend": False,
            "application_owner": "prepare_aia_observation_store",
            "input_correction_applications": 0,
        },
        "finite_negative_values_preserved": True,
        "invalid_value_policy": "finite_placeholders_only_where_valid_mask_is_false",
    }
    return RegisteredAIAChannelRaster(
        intensity=torch.from_numpy(np.ascontiguousarray(rate)),
        valid_mask=torch.from_numpy(np.ascontiguousarray(valid)),
        on_disk_mask=torch.from_numpy(np.ascontiguousarray(on_disk)),
        ray_direction=torch.from_numpy(np.ascontiguousarray(geometry.ray_direction)),
        surface_position_m=torch.from_numpy(
            np.ascontiguousarray(geometry.surface_position_m)
        ),
        absolute_tai_seconds=geometry.absolute_tai_seconds,
        exposure_group=exposure_group,
        channel_angstrom=record["channel_angstrom"],
        solar_radius_m=geometry.solar_radius_m,
        degradation_factor=factor,
        degradation_correction_applications=0,
        calibration_convention_id=calibration_convention_id,
        intensity_unit=AIA_INTENSITY_UNIT,
        time_representation=AIA_TIME_REPRESENTATION,
        surface_frame=AIA_SURFACE_FRAME,
        ray_direction_convention=AIA_RAY_DIRECTION_CONVENTION,
        raster_name=(f"{exposure_group}-aia-{record['channel_angstrom']}"),
        provenance=provenance,
    )


def prepare_aiapy_aia_observation_store(
    acquisition_directory: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    correction_table_path: str | os.PathLike[str],
    pointing_table_path: str | os.PathLike[str],
    footprint: CarringtonCutout | CenteredPixelCutout,
    observation_id: str = "aia_euv",
    backend: AiapyPreparationBackend | None = None,
    acquisition_manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Load calibration once; each worker reads, calibrates, crops, and writes one image."""

    output = Path(output_directory).expanduser().resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"AIA image-observation store already exists: {output}.")
    root = Path(acquisition_directory).expanduser().resolve()
    manifest = load_aia_acquisition_manifest(root) if acquisition_manifest is None else acquisition_manifest
    records = _validated_records(root, manifest)
    correction_path = _explicit_ecsv_file(
        correction_table_path, "correction_table_path"
    )
    pointing_path = _explicit_ecsv_file(pointing_table_path, "pointing_table_path")
    correction_sha256 = _sha256_file(correction_path)
    pointing_sha256 = _sha256_file(pointing_path)
    table_provenance = {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in (
            ("correction", correction_path, correction_sha256),
            ("pointing", pointing_path, pointing_sha256),
        )
    }
    table_dependencies = {
        name: {"ecsv_sha256": value["sha256"]} for name, value in table_provenance.items()
    }

    selected_backend = AiapyPreparationBackend() if backend is None else backend
    pointing_table = selected_backend.read_pointing_table(pointing_path)
    correction_table = selected_backend.read_correction_table(correction_path)
    convention_id = _calibration_convention_id()

    def prepare_record(item):
        exposure_group, source_path, record = item
        digest = _sha256_file(source_path)
        if record["sha256"] is not None and record["sha256"] != digest:
            raise ValueError(f"AIA checksum mismatch: {source_path}")
        record["sha256"] = digest
        return _prepare_record(
            backend=selected_backend,
            source_path=source_path,
            record=record,
            exposure_group=exposure_group,
            footprint=footprint,
            pointing_table=pointing_table,
            correction_table=correction_table,
            calibration_convention_id=convention_id,
            table_provenance=table_provenance,
        )

    def finalize_digest():
        if (_sha256_file(correction_path) != correction_sha256
                or _sha256_file(pointing_path) != pointing_sha256):
            raise RuntimeError("AIA calibration ECSV input changed during preparation.")
        return _content_digest([record for _, _, record in records])
    dependencies = {
        "implementation": dict(selected_backend.dependency_metadata()),
        "pipeline_order": list(AIAPY_PIPELINE_ORDER),
        "calibration_tables": table_dependencies,
        "cutout": footprint.metadata(),
        "degradation_application_owner": "prepare_aia_observation_store",
    }
    return prepare_aia_observation_store(
        output,
        source_files_sha256=finalize_digest,
        preparation_dependencies=dependencies,
        source_products=records,
        raster_backend=prepare_record,
        observation_id=observation_id,
    )


__all__ = [
    "AIA_NATIVE_PIXEL_SOLID_ANGLE_RELATIVE_TOLERANCE",
    "AIA_REFERENCE_NATIVE_PIXEL_SOLID_ANGLE_SR",
    "AIA_REGISTERED_PLATE_SCALE_ARCSEC_PER_PIXEL",
    "AIA_V10_CORRECTION_SEMANTIC_SHA256",
    "AIA_V10_CORRECTION_SOURCE_SHA256",
    "AIAPY_DEGRADATION_CALIBRATION_VERSION",
    "AIAPY_PIPELINE_ORDER",
    "AIAPY_PREPARATION_PYTHON",
    "AIAPY_PREPARATION_VERSION",
    "AIACarringtonGeometry",
    "AiapyPreparationBackend",
    "CarringtonCutout",
    "prepare_aiapy_aia_observation_store",
]
