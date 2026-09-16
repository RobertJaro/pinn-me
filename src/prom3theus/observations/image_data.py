"""Independent loader boundary for prepared scalar image observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from prom3theus.observations.loading import buffered_dataloader
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .image_contracts import ImageObservationRaster, ImageObservationSpec
from .image_dataset import (
    ImageObservationBatchCollator,
    ImagePixelDataset,
)


class StoredImageDataModule(LightningDataModule):
    """Memory-efficient data module for an immutable image-raster sequence.

    Unlike :class:`StoredObservationDataModule`, this class has no wavelength,
    Stokes-basis, or continuum-normalization assumptions.  Its validation unit
    is a complete exposure group containing every declared channel once.
    Optional fixed channel scales apply identically to all training, validation,
    and diagnostic samples, while the raster sequence retains physical units.
    """

    def __init__(
        self,
        rasters: Sequence[ImageObservationRaster],
        raster_names: Sequence[str],
        spec: ImageObservationSpec,
        *,
        validation_exposure_group: str,
        intensity_scales: Mapping[int, float] | None = None,
        store_metadata: Mapping | None = None,
        batch_size: int = 256,
        validation_batch_size: int | None = None,
        num_workers: int = 2,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if validation_batch_size is None:
            validation_batch_size = batch_size
        if type(validation_batch_size) is not int or validation_batch_size < 1:
            raise ValueError("validation_batch_size must be a positive integer.")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")
        if type(pin_memory) is not bool:
            raise TypeError("pin_memory must be boolean.")
        if not isinstance(validation_exposure_group, str) or not (
            validation_exposure_group := validation_exposure_group.strip()
        ):
            raise ValueError("validation_exposure_group must be a non-empty string.")

        self.rasters = list(rasters)
        self.raster_names = list(map(str, raster_names))
        if not self.rasters or len(self.rasters) != len(self.raster_names):
            raise ValueError(
                "Image raster names must match a non-empty raster sequence."
            )
        if any(not name for name in self.raster_names) or len(
            set(self.raster_names)
        ) != len(self.raster_names):
            raise ValueError("Image raster names must be non-empty and unique.")
        if not isinstance(spec, ImageObservationSpec):
            raise TypeError("spec must be an ImageObservationSpec.")
        self.spec = spec
        self.intensity_scales = (
            {channel: 1.0 for channel in spec.channels_angstrom}
            if intensity_scales is None
            else dict(intensity_scales)
        )
        if set(self.intensity_scales) != set(spec.channels_angstrom) or any(
            not math.isfinite(value) or value <= 0
            for value in self.intensity_scales.values()
        ):
            raise ValueError("intensity_scales must provide one finite positive scale per channel.")
        self.batch_intensity_unit = (
            spec.intensity_unit if intensity_scales is None else "dimensionless"
        )
        if validation_exposure_group not in spec.exposure_groups:
            raise ValueError(
                "validation_exposure_group is absent from the image specification."
            )
        self.validation_exposure_group = validation_exposure_group
        self.store_metadata = dict(store_metadata or {})
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collator = ImageObservationBatchCollator()

        self.dataset: Dataset | None = None
        self.image_datasets: list[ImagePixelDataset] = []
        self.validation_datasets: dict[int, ImagePixelDataset] = {}
        self.validation_raster_names: dict[int, str] = {}
        self._validate_sequence()

    def _validate_sequence(self) -> None:
        expected_channels = set(self.spec.channels_angstrom)
        expected_groups = set(self.spec.exposure_groups)
        by_identity: dict[tuple[str, int], int] = {}
        for index, raster in enumerate(self.rasters):
            if raster.channel_angstrom not in expected_channels:
                raise ValueError(
                    f"Raster channel {raster.channel_angstrom} is absent from the specification."
                )
            if raster.exposure_group not in expected_groups:
                raise ValueError(
                    f"Raster exposure group {raster.exposure_group!r} is absent from the specification."
                )
            unit = raster.metadata.get("intensity_unit")
            if unit != self.spec.intensity_unit:
                raise ValueError(
                    f"Raster {self.raster_names[index]!r} intensity unit {unit!r} "
                    f"does not match {self.spec.intensity_unit!r}."
                )
            identity = (raster.exposure_group, raster.channel_angstrom)
            if identity in by_identity:
                raise ValueError(
                    "Image rasters must contain one image per exposure group and channel."
                )
            by_identity[identity] = index
        missing = [
            (group, channel)
            for group in self.spec.exposure_groups
            for channel in self.spec.channels_angstrom
            if (group, channel) not in by_identity
        ]
        if missing:
            raise ValueError(
                "Every image exposure group must contain every configured channel; "
                f"missing={missing}."
            )
        if len(by_identity) != len(expected_groups) * len(expected_channels):
            raise ValueError("Image raster sequence contains unexpected group members.")
        self._raster_indices_by_identity = by_identity

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.dataset is not None:
            return
        channel_indices = {
            channel: index for index, channel in enumerate(self.spec.channels_angstrom)
        }
        self.image_datasets = [
            ImagePixelDataset(
                raster,
                image_index=image_index,
                channel_index=channel_indices[raster.channel_angstrom],
                intensity_scale=self.intensity_scales[raster.channel_angstrom],
                include_pixel_index=False,
            )
            for image_index, raster in enumerate(self.rasters)
        ]
        for dataset, name in zip(self.image_datasets, self.raster_names, strict=True):
            dataset.sequence_name = name
        self.dataset = (
            self.image_datasets[0]
            if len(self.image_datasets) == 1
            else ConcatDataset(self.image_datasets)
        )
        for channel in self.spec.channels_angstrom:
            raster_index = self._raster_indices_by_identity[
                (self.validation_exposure_group, channel)
            ]
            source_dataset = self.image_datasets[raster_index]
            self.validation_datasets[channel] = ImagePixelDataset(
                self.rasters[raster_index],
                image_index=raster_index,
                channel_index=channel_indices[channel],
                intensity_scale=self.intensity_scales[channel],
                include_pixel_index=True,
                pixel_indices=source_dataset._pixel_indices,
            )
            self.validation_raster_names[channel] = self.raster_names[raster_index]

    def _loader(
        self, dataset: Dataset, *, shuffle: bool, batch_size: int
    ) -> DataLoader:
        return buffered_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            collate_fn=self.collator,
        )

    def train_dataloader(self) -> DataLoader:
        if self.batch_size < len(self.spec.channels_angstrom):
            raise ValueError(
                "Training batch_size must include at least one pixel from every "
                "configured image channel."
            )
        if self.dataset is None:
            self.setup("fit")
        from .tensor_loader import PersistentTensorLoader
        channels = []
        for index in range(len(self.spec.channels_angstrom)):
            datasets = sorted(
                (d for d in self.image_datasets if d.channel_index == index),
                key=lambda d: (d.raster.absolute_tai_seconds, self.raster_names[d.image_index]),
            )
            channels.append(datasets[0] if len(datasets) == 1 else ConcatDataset(datasets))
        return PersistentTensorLoader(
            self.dataset, self.batch_size, channels=channels,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
        )

    def validation_dataloaders(self) -> dict[int, DataLoader]:
        """Return deterministic native-grid loaders for the complete group."""

        if not self.validation_datasets:
            self.setup("validate")
        return {
            channel: self._loader(
                self.validation_datasets[channel],
                shuffle=False,
                batch_size=self.validation_batch_size,
            )
            for channel in self.spec.channels_angstrom
        }

    def val_dataloader(self) -> list[DataLoader]:
        """Lightning-compatible ordered view of the complete validation group."""

        loaders = self.validation_dataloaders()
        return [loaders[channel] for channel in self.spec.channels_angstrom]

    def run_metadata(self) -> dict:
        if self.dataset is None:
            self.setup("fit")
        return {
            "observation": self.spec.metadata(),
            "raster_count": len(self.rasters),
            "raster_names": list(self.raster_names),
            "valid_pixel_count": sum(len(dataset) for dataset in self.image_datasets),
            "intensity_normalization": {
                "operation": "intensity / channel_scale",
                "batch_intensity_unit": self.batch_intensity_unit,
                "source_intensity_unit": self.spec.intensity_unit,
                "scale_by_channel_angstrom": {
                    str(channel): float(scale)
                    for channel, scale in self.intensity_scales.items()
                },
            },
            "training": {
                "sampling": "channel_balanced_sequential_cyclic",
                "batch_size": self.batch_size,
                "channels_angstrom": list(self.spec.channels_angstrom),
            },
            "validation": {
                "type": "complete exposure group on native channel grids",
                "exposure_group": self.validation_exposure_group,
                "raster_names_by_channel_angstrom": {
                    str(channel): self.validation_raster_names[channel]
                    for channel in self.spec.channels_angstrom
                },
                "valid_pixels_by_channel_angstrom": {
                    str(channel): len(self.validation_datasets[channel])
                    for channel in self.spec.channels_angstrom
                },
                "held_out": False,
            },
            "store": dict(self.store_metadata),
        }


__all__ = ["StoredImageDataModule"]
