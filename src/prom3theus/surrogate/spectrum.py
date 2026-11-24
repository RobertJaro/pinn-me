"""Spectral window selection and instrument response for held-out profiles."""

import math

import torch
import yaml

from prom3theus.instruments import build_instrument
from prom3theus.rt.wavelength import air_to_vacuum_angstrom


def read_window(config_path, window_id):
    source = yaml.safe_load(config_path.read_text())
    if not isinstance(source, dict):
        raise ValueError("Expected a mapping in the surrogate configuration")
    windows = source["synthesis"]["windows"] if "synthesis" in source else [source]
    selected = [w for w in windows if window_id is None or w.get("id") == window_id]
    if len(selected) != 1:
        raise ValueError("Select exactly one synthesis window with --window")
    return selected[0]


def resolve_window(window, window_id, instrument_path):
    """Read the optional detector and derive the window for detector-only runs."""
    detector = None
    if instrument_path:
        detector = yaml.safe_load(instrument_path.read_text())
    if window is None and detector is None:
        raise ValueError("Supply --config or --instrument-config")
    if window is None:
        grid = build_instrument(detector["instrument"]).synthesis_grid(
            torch.tensor(detector["observed_wavelength_angstrom"], dtype=torch.float64)
        )
        vacuum = air_to_vacuum_angstrom(grid)
        window = dict(
            id=window_id or "spectral_window",
            line_ids=detector["line_ids"],
            wavelength_bounds_vacuum_angstrom=[
                math.floor(vacuum[0]) - 1,
                math.ceil(vacuum[-1]) + 1,
            ],
        )
    if detector and window["line_ids"] != detector["line_ids"]:
        raise ValueError(
            "Detector validation and training windows must have matching line IDs"
        )
    return window, detector


def observing_grid(detector, fallback, device):
    if detector is None:
        return fallback, lambda stokes: stokes
    operator = build_instrument(detector["instrument"]).double()
    observed = torch.tensor(
        detector["observed_wavelength_angstrom"], dtype=torch.float64
    )
    wave = operator.synthesis_grid(observed)
    if hasattr(operator, "prepare"):
        operator.prepare(wave, observed)
    operator.to(device=device, dtype=torch.float32)
    wave, observed = wave.to(device), observed.to(device)
    response = {
        k: torch.tensor(v, device=device, dtype=torch.float32).unsqueeze(0)
        for k, v in (detector.get("response") or {}).items()
    }
    rotation = math.radians(-2 * operator.polarization_convention.offset_deg)

    def observe(stokes):
        i, q, u, v = stokes.unbind(-2)
        stokes = torch.stack(
            (
                i,
                math.cos(rotation) * q + math.sin(rotation) * u,
                -math.sin(rotation) * q + math.cos(rotation) * u,
                v,
            ),
            -2,
        )
        return operator(stokes.unsqueeze(0), wave, observed, **response).squeeze(0)

    return air_to_vacuum_angstrom(wave), observe
