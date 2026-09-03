"""Strict Hinode/SOT-SP observation adapter facade."""

from __future__ import annotations

from collections.abc import Mapping

from prom3theus.observations import (
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    ObservationSpec,
)

from .constants import HINODE_SP_FE_LINE_IDS
from .raster import load_raster
from .data import HinodeDataModule


def _build_data(config: Mapping) -> HinodeDataModule:
    """Translate the typed YAML observation section into runtime arguments."""

    raw = dict(config)
    expected = {"directory", "selection", "loader", "calibration"}
    if set(raw) != expected:
        raise TypeError(
            "Hinode observation config must contain exactly directory, selection, "
            f"loader, and calibration; missing={sorted(expected - set(raw))}, "
            f"unknown={sorted(set(raw) - expected)}."
        )
    selection = dict(raw.pop("selection"))
    loader = dict(raw.pop("loader"))
    calibration = dict(raw.pop("calibration"))
    expected_selection = {"slit_slice"}
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
        "continuum_edge_samples",
        "quiet_sun_max_fractional_polarization",
        "quiet_sun_continuum_trim_quantiles",
        "minimum_quiet_sun_pixels",
        "stokes_reference_angle_deg",
    }
    for name, value, fields in (
        ("selection", selection, expected_selection),
        ("loader", loader, expected_loader),
        ("calibration", calibration, expected_calibration),
    ):
        if set(value) != fields:
            raise TypeError(
                f"Hinode observation {name} has missing={sorted(fields - set(value))}, "
                f"unknown={sorted(set(value) - fields)}."
            )
    slit_slice = dict(selection.pop("slit_slice"))
    if set(slit_slice) != {"start", "stop"}:
        raise TypeError(
            "Hinode observation selection.slit_slice must contain exactly start and stop."
        )
    resolved = {
        "directory": raw["directory"],
        **calibration,
        **selection,
        "slit_slice": [slit_slice["start"], slit_slice["stop"]],
        "batch_size": loader["batch_size"],
        "validation_batch_size": loader["validation_batch_size"],
        "validation_stride": loader["validation_stride"],
        "data_loading_workers": loader["preparation_workers"],
        "num_workers": loader["workers"],
        "pin_memory": loader["pin_memory"],
        "progress": loader["progress"],
    }
    return HinodeDataModule(**resolved)


def _describe_data(data: HinodeDataModule) -> ObservationSpec:
    """Describe the exact scientific boundary consumed by LTE synthesis."""

    normalization = data.normalization_metadata
    try:
        continuum_indices = tuple(normalization["indices"])
        radiance_scale = normalization["radiometric_calibration"][
            "atlas_disk_center_continuum_radiance_w_m3_sr"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Hinode data lack radiometric normalization metadata."
        ) from error
    return ObservationSpec(
        observation_id="hinode_sp",
        observation_type="hinode_sp",
        instrument_type="hinode_sp",
        wavelength_angstrom=data.wavelength_angstrom,
        continuum_indices=continuum_indices,
        radiance_scale_w_m3_sr=float(radiance_scale),
        velocity_synthesis_mode=CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
        required_line_ids=HINODE_SP_FE_LINE_IDS,
        line_support_angstrom=(
            float(data.wavelength_angstrom.min()),
            float(data.wavelength_angstrom.max()),
        ),
    )


__all__ = ["HinodeDataModule", "load_raster"]
