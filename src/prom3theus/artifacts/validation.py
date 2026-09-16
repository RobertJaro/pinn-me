"""Scientific contracts for explicit observation selections."""
import math
import json
from typing import Any
from prom3theus.observations import (
    get_observation_adapter,
    resolve_velocity_synthesis_mode,
    CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
)
from collections.abc import Mapping
import numpy as np
import torch
from prom3theus.observations import ObservationRaster, ObservationSpec
from .errors import ArtifactExportError

_OBSERVATION_SPEC_FIELDS = frozenset(
    {
        "observation_id",
        "observation_type",
        "instrument_type",
        "wavelength_angstrom",
        "continuum_indices",
        "radiance_scale_w_m3_sr",
        "velocity_synthesis_mode",
        "velocity_synthesis",
        "required_line_ids",
        "line_support_angstrom",
        "instrument_options",
    }
)


def _json_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ArtifactExportError(f"{context} must be an object with string keys.")
    return dict(value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ArtifactExportError(f"{context} has {' and '.join(details)}.")


def _parse_observation_spec(value: Any) -> ObservationSpec:
    raw = _require_mapping(value, "manifest.observation.spec")
    _require_exact_fields(raw, _OBSERVATION_SPEC_FIELDS, "manifest.observation.spec")
    support = raw["line_support_angstrom"]
    try:
        spec = ObservationSpec(
            observation_id=raw["observation_id"],
            observation_type=raw["observation_type"],
            instrument_type=raw["instrument_type"],
            wavelength_angstrom=torch.as_tensor(raw["wavelength_angstrom"]),
            continuum_indices=tuple(raw["continuum_indices"]),
            radiance_scale_w_m3_sr=raw["radiance_scale_w_m3_sr"],
            velocity_synthesis_mode=raw["velocity_synthesis_mode"],
            required_line_ids=tuple(raw["required_line_ids"]),
            line_support_angstrom=(None if support is None else tuple(support)),
            instrument_options=_require_mapping(
                raw["instrument_options"],
                "manifest.observation.spec.instrument_options",
            ),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactExportError(
            f"manifest.observation.spec is invalid: {error}"
        ) from error
    canonical = spec.metadata()
    if _json_canonical(canonical) != _json_canonical(raw):
        raise ArtifactExportError(
            "manifest.observation.spec is not the exact canonical "
            "ObservationSpec.metadata() representation."
        )
    try:
        adapter = get_observation_adapter(spec.observation_type)
    except ValueError as error:
        raise ArtifactExportError(str(error)) from error
    if adapter.name != spec.observation_type:
        raise ArtifactExportError("Observation adapter identity is inconsistent.")
    return spec


def scene_basis(raster: ObservationRaster) -> np.ndarray:
    """Return the validated right-handed scene basis stored with a raster."""

    geometry = raster.metadata.get("ray_geometry")
    if not isinstance(geometry, Mapping):
        raise ArtifactExportError(
            "Observation metadata must contain ray_geometry.scene_basis_rows for export."
        )
    basis = np.asarray(geometry.get("scene_basis_rows"), dtype=np.float64)
    if basis.shape != (3, 3) or not np.isfinite(basis).all():
        raise ArtifactExportError(
            "Observation ray_geometry.scene_basis_rows must be a finite 3x3 matrix."
        )
    if (
        not np.allclose(basis @ basis.T, np.eye(3), rtol=0.0, atol=2.0e-5)
        or np.linalg.det(basis) <= 0
    ):
        raise ArtifactExportError(
            "Observation ray_geometry.scene_basis_rows must be right-handed and orthonormal."
        )
    return basis


def removed_solar_los_velocity(raster: ObservationRaster) -> Any | None:
    """Return the stored Hinode registration correction, when present."""

    correction = raster.metadata.get("observer_velocity_correction")
    if not isinstance(correction, Mapping):
        return None
    return correction.get("removed_solar_los_velocity_m_per_s")


def validate_raster_scientific_contract(
    raster: ObservationRaster,
    spec: ObservationSpec,
) -> None:
    """Validate Stokes ordering and radiometric normalization metadata."""

    if raster.metadata.get("stokes_order") != ["I", "Q", "U", "V"]:
        raise ArtifactExportError(
            "Canonical observation metadata must declare stokes_order=['I','Q','U','V']."
        )
    normalization = raster.metadata.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ArtifactExportError(
            "Canonical observation metadata must contain its normalization contract."
        )
    try:
        raw_indices = normalization["indices"]
        radiometry = normalization["radiometric_calibration"]
        raw_radiance_scale = radiometry["atlas_disk_center_continuum_radiance_w_m3_sr"]
    except (KeyError, TypeError) as error:
        raise ArtifactExportError(
            "Canonical observation normalization lacks continuum indices or its "
            "atlas continuum radiance."
        ) from error
    if not isinstance(raw_indices, (list, tuple)) or any(
        type(value) is not int for value in raw_indices
    ):
        raise ArtifactExportError(
            "Canonical observation normalization indices must be integers."
        )
    continuum_indices = tuple(raw_indices)
    if isinstance(raw_radiance_scale, bool) or not isinstance(
        raw_radiance_scale, (int, float)
    ):
        raise ArtifactExportError(
            "Canonical observation atlas continuum radiance must be numeric."
        )
    radiance_scale = float(raw_radiance_scale)
    if not math.isfinite(radiance_scale) or radiance_scale <= 0.0:
        raise ArtifactExportError(
            "Canonical observation atlas continuum radiance must be finite and positive."
        )
    if continuum_indices != spec.continuum_indices:
        raise ArtifactExportError(
            "Observation-store continuum indices differ from the observation spec."
        )
    if not math.isclose(
        radiance_scale,
        spec.radiance_scale_w_m3_sr,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ArtifactExportError(
            "Observation-store atlas continuum radiance differs from the observation spec."
        )


def validate_raster_velocity_contract(
    raster: ObservationRaster,
    velocity_synthesis_mode: str,
) -> None:
    """Validate the observation-specific velocity synthesis inputs."""

    mode = resolve_velocity_synthesis_mode(velocity_synthesis_mode)
    observer = raster.auxiliary.get("observer_los_velocity_m_per_s")
    removed = removed_solar_los_velocity(raster)
    if mode.value == CARRINGTON_OBSERVER_RELATIVE_VELOCITY:
        if observer is None:
            raise ArtifactExportError(
                "carrington_observer_relative export requires observation auxiliary "
                "'observer_los_velocity_m_per_s'."
            )
        if tuple(observer.shape) not in {
            raster.spatial_shape,
            (*raster.spatial_shape, 1),
        }:
            raise ArtifactExportError(
                "observer_los_velocity_m_per_s must match the raster spatial shape."
            )
        if not torch.isfinite(observer).all():
            raise ArtifactExportError(
                "observer_los_velocity_m_per_s must contain only finite values."
            )
        if removed is not None:
            raise ArtifactExportError(
                "Observer-relative observations must not contain a Hinode registration velocity."
            )
    elif mode.value == CARRINGTON_REGISTERED_RELATIVE_VELOCITY:
        values = np.asarray(removed)
        if values.shape != (raster.spatial_shape[1],) or not np.isfinite(values).all():
            raise ArtifactExportError(
                "carrington_registered_relative export requires one finite "
                "removed_solar_los_velocity_m_per_s value per scan column."
            )
        if observer is not None:
            raise ArtifactExportError(
                "Registered-relative observations must not contain an observer LOS array."
            )
    elif observer is not None or removed is not None:
        raise ArtifactExportError(
            "registered_relative export must not apply an observation-specific LOS correction."
        )
