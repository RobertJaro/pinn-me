"""HMI observation facade for the closed LTE adapter registry."""

from __future__ import annotations

from collections.abc import Mapping

from astropy import units as u

from prom3theus.observations import (
    CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
    ObservationSpec,
)

from .constants import HMI_LINE_ID, HMI_WAVELENGTH_CENTER
from .raster import load_raster
from .timeline import HMIDataModule


def _build_data(config: Mapping) -> HMIDataModule:
    """Translate the exact typed HMI section into runtime arguments."""

    raw = dict(config)
    expected = {"directory", "selection", "loader", "calibration"}
    if set(raw) != expected:
        raise TypeError(
            "HMI observation config must contain exactly directory, selection, "
            f"loader, and calibration; missing={sorted(expected - set(raw))}, "
            f"unknown={sorted(set(raw) - expected)}."
        )
    selection = dict(raw["selection"])
    loader = dict(raw["loader"])
    calibration = dict(raw["calibration"])
    expected_selection = {"acquisition_indices", "validation_raster"}
    expected_loader = {
        "batch_size",
        "validation_batch_size",
        "validation_stride",
        "preparation_workers",
        "workers",
        "pin_memory",
        "progress",
    }
    expected_calibration = {
        "transmission_profile_directory",
        "require_quality_zero",
        "quiet_sun_max_fractional_polarization",
        "quiet_sun_trim_quantiles",
        "minimum_quiet_sun_pixels",
        "calibration_sample_limit",
    }
    for name, value, fields in (
        ("selection", selection, expected_selection),
        ("loader", loader, expected_loader),
        ("calibration", calibration, expected_calibration),
    ):
        if set(value) != fields:
            raise TypeError(
                f"HMI observation {name} has missing={sorted(fields - set(value))}, "
                f"unknown={sorted(set(value) - fields)}."
            )
    return HMIDataModule(
        directory=raw["directory"],
        acquisition_indices=selection["acquisition_indices"],
        validation_raster=selection["validation_raster"],
        transmission_profile_directory=calibration.pop(
            "transmission_profile_directory"
        ),
        batch_size=loader["batch_size"],
        validation_batch_size=loader["validation_batch_size"],
        validation_stride=loader["validation_stride"],
        data_loading_workers=loader["preparation_workers"],
        num_workers=loader["workers"],
        pin_memory=loader["pin_memory"],
        progress=loader["progress"],
        **calibration,
    )


def _describe_data(data: HMIDataModule) -> ObservationSpec:
    normalization = data.normalization_metadata
    try:
        radiance_scale = normalization["radiometric_calibration"][
            "atlas_disk_center_continuum_radiance_w_m3_sr"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("HMI data lack radiometric normalization metadata.") from error
    center = float(HMI_WAVELENGTH_CENTER.to_value(u.AA))
    return ObservationSpec(
        observation_id="hmi_stokes",
        observation_type="hmi_stokes",
        instrument_type="hmi_filter_profiles",
        wavelength_angstrom=data.wavelength_angstrom,
        continuum_indices=tuple(normalization["indices"]),
        radiance_scale_w_m3_sr=float(radiance_scale),
        velocity_synthesis_mode=CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
        required_line_ids=(HMI_LINE_ID,),
        line_support_angstrom=(
            center - data.response_inner_half_width_angstrom,
            center + data.response_inner_half_width_angstrom,
        ),
        instrument_options={
            "quadrature_wavelength_angstrom": (
                data.response_quadrature_wavelength_angstrom.tolist()
            ),
            "inner_half_width_angstrom": data.response_inner_half_width_angstrom,
        },
    )


__all__ = [
    "HMIDataModule",
    "load_raster",
]
