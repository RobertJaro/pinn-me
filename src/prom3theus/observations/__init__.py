"""Public observation contracts, storage, datasets, and factories.

Exports are resolved lazily so importing a pure contract submodule does not
load the optional data-loader and Lightning runtime.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULE = {
    "CARRINGTON_OBSERVER_RELATIVE_VELOCITY": ".contracts",
    "CARRINGTON_REGISTERED_RELATIVE_VELOCITY": ".contracts",
    "COORDINATES": ".contracts",
    "INSTRUMENT_RESPONSE": ".contracts",
    "ObservationBatch": ".contracts",
    "ObservationRaster": ".contracts",
    "ObservationSample": ".contracts",
    "ObservationSpec": ".contracts",
    "PIXEL_INDEX": ".contracts",
    "RAY_DIRECTION": ".contracts",
    "STOKES": ".contracts",
    "STOKES_BASIS": ".contracts",
    "SURFACE_POSITION_M": ".contracts",
    "VALID_MASK": ".contracts",
    "VELOCITY_SYNTHESIS_MODES": ".contracts",
    "VelocitySynthesisMode": ".contracts",
    "resolve_velocity_synthesis_mode": ".contracts",
    "velocity_synthesis_contract": ".contracts",
    "ObservationDataModule": ".data",
    "StoredObservationDataModule": ".data",
    "ObservationBatchCollator": ".dataset",
    "ObservationPixelDataset": ".dataset",
    "ObservationResponseCollator": ".dataset",
    "collate_observation_samples": ".dataset",
    "ObservationAdapter": ".registry",
    "build_observation_data": ".registry",
    "describe_observation_data": ".registry",
    "get_observation_adapter": ".registry",
    "load_or_prepare_observation": ".registry",
    "OBSERVATION_STORE_FORMAT": ".store",
    "OBSERVATION_STORE_VERSION": ".store",
    "ObservationStore": ".store",
    "observation_store_signature": ".store",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULE[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORT_MODULE))


__all__ = list(_EXPORT_MODULE)
