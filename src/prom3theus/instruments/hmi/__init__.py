"""SDO/HMI runtime adapter, response archive, operator, and preparation."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {
        "build_hmi_stokes_query",
        "download_hmi_stokes",
    }:
        from . import download as download_module

        return getattr(download_module, name)
    if name == "HMIFilterProfiles":
        from .operator import HMIFilterProfiles

        return HMIFilterProfiles
    if name in {
        "HMIResponseArchive",
        "load_response_manifest",
        "load_response_profile",
        "resolve_response_profile",
    }:
        from . import response as response_module

        return getattr(response_module, name)
    if name in {
        "HMIDataModule",
        "load_raster",
    }:
        from . import observation as observation_module

        return getattr(observation_module, name)
    if name == "prepare_hmi_response_directory":
        from .preparation import prepare_hmi_response_directory

        return prepare_hmi_response_directory
    if name == "prepare_hmi_subframes":
        from .subframe import prepare_hmi_subframes

        return prepare_hmi_subframes
    raise AttributeError(name)


__all__ = [
    "HMIDataModule",
    "HMIFilterProfiles",
    "HMIResponseArchive",
    "build_hmi_stokes_query",
    "download_hmi_stokes",
    "load_raster",
    "load_response_manifest",
    "load_response_profile",
    "prepare_hmi_response_directory",
    "prepare_hmi_subframes",
    "resolve_response_profile",
]
