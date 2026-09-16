"""Public observation contracts, storage, datasets, and factories.

Exports are resolved lazily so importing a pure contract submodule does not
load the optional data-loader and Lightning runtime.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULE = {
    "TensorDiskDataset": ".tensor_dataset",
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
    "ABSOLUTE_TAI_SECONDS": ".image_contracts",
    "CHANNEL_ANGSTROM": ".image_contracts",
    "CHANNEL_INDEX": ".image_contracts",
    "EXPOSURE_GROUP": ".image_contracts",
    "IMAGE_INDEX": ".image_contracts",
    "INTENSITY": ".image_contracts",
    "UNCERTAINTY": ".image_contracts",
    "ImageObservationBatch": ".image_contracts",
    "ImageObservationRaster": ".image_contracts",
    "ImageObservationSample": ".image_contracts",
    "ImageObservationSpec": ".image_contracts",
    "ImageObservationBatchCollator": ".image_dataset",
    "ImagePixelDataset": ".image_dataset",
    "collate_image_samples": ".image_dataset",
    "reconstruct_image": ".image_dataset",
    "StoredImageDataModule": ".image_data",
    "IMAGE_MANIFEST_FILENAME": ".image_store",
    "IMAGE_OBSERVATION_STORE_FORMAT": ".image_store",
    "IMAGE_OBSERVATION_STORE_VERSION": ".image_store",
    "ImageObservationStore": ".image_store",
    "image_observation_store_signature": ".image_store",
    "SceneContract": ".scene",
    "ObservationDescriptor": ".streams",
    "ObservationKind": ".streams",
    "ObservationStreamDataModule": ".streams",
    "PreparedObservationStream": ".streams",
    "PreparedObservationStreams": ".streams",
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
