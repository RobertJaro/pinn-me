"""Differentiable depth-stratified LTE polarized radiative transfer.

Public conveniences are resolved lazily so importing atomic data or the
standalone synthesizer does not import the Hinode/Lightning training stack.
Each implementation remains available from its focused submodule as well.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "AtomicDatabase": "pme.lte.atomic",
    "SpectralLine": "pme.lte.atomic",
    "GeometricHeightModel": "pme.lte.atmosphere",
    "StratifiedAtmosphere": "pme.lte.atmosphere",
    "StratifiedAtmosphereModel": "pme.lte.atmosphere",
    "HinodeLTEDataModule": "pme.lte.hinode",
    "HinodePixelDataset": "pme.lte.hinode",
    "HinodeRaster": "pme.lte.hinode",
    "load_hinode_raster": "pme.lte.hinode",
    "disk_center_continuum_radiance": "pme.lte.radiometry",
    "load_solar_reference": "pme.lte.radiometry",
    "neckel_continuum_limb_darkening": "pme.lte.radiometry",
    "HinodeSpectralPSF": "pme.lte.instrument",
    "ContinuumOpacity": "pme.lte.opacity",
    "PolarizedLineOpacity": "pme.lte.polarization",
    "VoigtFaraday": "pme.lte.profiles",
    "voigt_faraday": "pme.lte.profiles",
    "HINODE_SP_FE_LINE_IDS": "pme.lte.synthesis",
    "LTESynthesizer": "pme.lte.synthesis",
    "SynthesisDiagnostics": "pme.lte.synthesis",
    "PolarizedFormalSolver": "pme.lte.transfer",
    "scalar_formal_solution": "pme.lte.transfer",
    "air_to_vacuum_angstrom": "pme.lte.wavelength",
    "ZeemanComponent": "pme.lte.zeeman",
    "ZeemanPattern": "pme.lte.zeeman",
    "zeeman_components": "pme.lte.zeeman",
}

__all__ = tuple(_EXPORT_MODULES)


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
