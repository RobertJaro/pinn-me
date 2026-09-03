"""Hinode/SOT-SP runtime adapter and spectral operator."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "HinodeSpectralPSF":
        from .operator import HinodeSpectralPSF

        return HinodeSpectralPSF
    if name in {"HinodeDataModule", "load_raster"}:
        from .observation import HinodeDataModule, load_raster

        return {
            "HinodeDataModule": HinodeDataModule,
            "load_raster": load_raster,
        }[name]
    raise AttributeError(name)


__all__ = [
    "HinodeDataModule",
    "HinodeSpectralPSF",
    "load_raster",
]
