"""Lightning-compatible loader boundary shared by independent adapters."""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .contracts import ObservationRaster
from .dataset import ObservationPixelDataset, ObservationResponseCollator


class ObservationDataModule(LightningDataModule):
    """Common loader mechanics; adapters remain responsible for ingestion.

    This base deliberately contains no instrument conditionals.  Hinode and
    HMI adapters subclass it directly and never subclass each other.
    """

    def __init__(
        self,
        files=None,
        directory=None,
        validation_raster: int | str = 0,
        batch_size: int = 4,
        validation_batch_size: int | None = None,
        validation_stride: int = 1,
        data_loading_workers: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        progress: bool = True,
    ) -> None:
        super().__init__()
        integer_options = {
            "batch_size": batch_size,
            "validation_stride": validation_stride,
            "data_loading_workers": data_loading_workers,
            "num_workers": num_workers,
        }
        if any(type(value) is not int for value in integer_options.values()):
            raise TypeError("Batch, stride, and worker options must be integers.")
        if validation_batch_size is not None and type(validation_batch_size) is not int:
            raise TypeError("validation_batch_size must be an integer or null.")
        if type(pin_memory) is not bool or type(progress) is not bool:
            raise TypeError("pin_memory and progress must be booleans.")
        if batch_size < 1 or (
            validation_batch_size is not None and validation_batch_size < 1
        ):
            raise ValueError("Batch sizes must be positive.")
        if validation_stride < 1:
            raise ValueError("validation_stride must be positive.")
        if data_loading_workers < 1 or num_workers < 0:
            raise ValueError(
                "data_loading_workers must be positive and num_workers non-negative."
            )
        self.files = files
        self.directory = directory
        self.validation_raster = validation_raster
        self.batch_size = int(batch_size)
        self.validation_batch_size = (
            self.batch_size
            if validation_batch_size is None
            else int(validation_batch_size)
        )
        self.validation_stride = int(validation_stride)
        self.data_loading_workers = int(data_loading_workers)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.progress = bool(progress)
        self.raster: ObservationRaster | None = None
        self.rasters: list[ObservationRaster] = []
        self.raster_names: list[str] = []
        self.dataset: Dataset | None = None
        self._evaluation_dataset: Dataset | None = None
        self.validation_dataset: Subset | None = None
        self.validation_raster_index: int | None = None
        self.validation_lattice_stride = (1, 1)
        self.fits_consistency_metadata: dict | None = None
        self.observation_sampling_bounds: dict | None = None

    def _prepare_validation_dataset(self, evaluation: ObservationPixelDataset) -> None:
        """Build a non-empty deterministic lattice over available valid pixels."""

        row_stride = max(1, int(math.floor(math.sqrt(self.validation_stride))))
        column_stride = max(1, int(math.ceil(self.validation_stride / row_stride)))
        pixels = evaluation.pixel_indices
        selected_indices: list[int] = []
        for row in torch.unique(pixels[:, 0], sorted=True)[::row_stride]:
            row_indices = torch.nonzero(pixels[:, 0] == row, as_tuple=False).squeeze(-1)
            selected_indices.extend(row_indices[::column_stride].tolist())
        if not selected_indices:
            raise RuntimeError(
                "Validation lattice selected no valid observation pixels."
            )
        self.validation_lattice_stride = (row_stride, column_stride)
        self.validation_dataset = Subset(evaluation, selected_indices)

    def _prepare_observation_sampling_bounds(self) -> None:
        if not self.rasters or self.raster is None:
            raise RuntimeError("Observation sampling bounds require prepared rasters.")
        sine_sum = torch.zeros((), dtype=torch.float64)
        cosine_sum = torch.zeros((), dtype=torch.float64)
        count = 0
        latitude_min, latitude_max = math.inf, -math.inf
        time_min, time_max = math.inf, -math.inf
        solar_radii_m: list[float] = []
        for raster in self.rasters:
            position = raster.surface_position_m[raster.valid_mask].to(torch.float64)
            if not position.numel():
                continue
            try:
                ray_geometry = raster.metadata["ray_geometry"]
                solar_radius_m = float(ray_geometry["solar_radius_m"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Every observation raster must declare ray_geometry.solar_radius_m."
                ) from error
            if not math.isfinite(solar_radius_m) or solar_radius_m <= 0:
                raise ValueError(
                    "Observation ray_geometry.solar_radius_m must be finite and positive."
                )
            surface_radius_m = torch.linalg.vector_norm(position, dim=-1)
            if not torch.allclose(
                surface_radius_m,
                torch.full_like(surface_radius_m, solar_radius_m),
                rtol=2e-5,
                atol=1.0,
            ):
                raise ValueError(
                    "Observation surface positions do not lie on the declared solar radius."
                )
            solar_radii_m.append(solar_radius_m)
            longitude = torch.atan2(position[:, 1], position[:, 0])
            latitude = torch.atan2(
                position[:, 2], torch.linalg.vector_norm(position[:, :2], dim=-1)
            )
            count += int(position.shape[0])
            sine_sum += torch.sin(longitude).sum()
            cosine_sum += torch.cos(longitude).sum()
            latitude_min = min(latitude_min, float(latitude.amin()))
            latitude_max = max(latitude_max, float(latitude.amax()))
            times = raster.coordinates[..., 2][raster.valid_mask]
            time_min = min(time_min, float(times.amin()))
            time_max = max(time_max, float(times.amax()))
        if count == 0:
            raise ValueError(
                "Physics sampling requires at least one valid observation pixel."
            )
        reference_radius_m = solar_radii_m[0]
        if any(
            not math.isclose(radius, reference_radius_m, rel_tol=1e-8, abs_tol=1.0)
            for radius in solar_radii_m[1:]
        ):
            raise ValueError("All observation rasters must use one solar radius.")
        center = torch.atan2(sine_sum, cosine_sum)
        longitude_min, longitude_max = math.inf, -math.inf
        for raster in self.rasters:
            position = raster.surface_position_m[raster.valid_mask].to(torch.float64)
            if not position.numel():
                continue
            longitude = torch.atan2(position[:, 1], position[:, 0])
            offset = torch.atan2(
                torch.sin(longitude - center), torch.cos(longitude - center)
            )
            longitude_min = min(longitude_min, float(offset.amin()))
            longitude_max = max(longitude_max, float(offset.amax()))
        self.observation_sampling_bounds = {
            "surface_longitude_center_rad": float(center),
            "surface_longitude_offset_rad": [longitude_min, longitude_max],
            "surface_latitude_rad": [latitude_min, latitude_max],
            "time_hours": [time_min, time_max],
            "solar_radius_m": reference_radius_m,
        }

    @property
    def wavelength_angstrom(self) -> torch.Tensor:
        if self.raster is None:
            raise RuntimeError("Call setup() before accessing wavelength_angstrom.")
        return self.raster.wavelength_angstrom

    @property
    def normalization_metadata(self) -> dict:
        if self.raster is None:
            raise RuntimeError("Call setup() before accessing normalization metadata.")
        return self.raster.metadata["normalization"]

    def run_metadata(self) -> dict:
        if self.raster is None:
            raise RuntimeError("Call setup() before requesting run metadata.")
        metadata = dict(self.raster.metadata)
        if len(self.rasters) > 1:
            metadata["multi_raster"] = {
                "raster_count": len(self.rasters),
                "raster_names": list(self.raster_names),
                "validation_raster": self.raster_names[self.validation_raster_index],
                "file_count": sum(item.metadata["file_count"] for item in self.rasters),
                "shared_reference_time": self.raster.metadata["ref_time"],
                "shared_scene_basis": True,
                "wavelength_grid_consistent": True,
                "fits_consistency": dict(self.fits_consistency_metadata or {}),
            }
        metadata["validation"] = {
            "type": "deterministic diagnostic subset of inversion pixels",
            "target_area_stride": self.validation_stride,
            "spatial_lattice_stride": list(self.validation_lattice_stride),
            "pixel_count": len(self.validation_dataset),
            "held_out": False,
        }
        metadata["training_sampling"] = {
            "type": "all valid observation pixels once per epoch",
            "samples_per_epoch": len(self.dataset),
            "valid_pixel_count": len(self.dataset),
        }
        return metadata

    def _data_loader(
        self, *, dataset, shuffle: bool, batch_size: int | None, collate_fn=None
    ) -> DataLoader:
        if self.dataset is None:
            self.setup()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and shuffle,
            collate_fn=collate_fn,
        )

    def _stokes_train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup()
        return self._data_loader(
            dataset=self.dataset, shuffle=True, batch_size=self.batch_size
        )

    def train_dataloader(self):
        return self._stokes_train_dataloader()

    def evaluation_dataset(self) -> Dataset:
        if self._evaluation_dataset is None:
            self.setup()
        return self._evaluation_dataset

    def evaluation_collate_fn(self):
        return None

    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is None:
            self.setup()
        return self._data_loader(
            dataset=self.validation_dataset,
            shuffle=False,
            batch_size=self.validation_batch_size,
        )


class StoredObservationDataModule(ObservationDataModule):
    """Ready-to-train view of a class-free, memory-mapped observation store."""

    def __init__(
        self,
        rasters,
        raster_names,
        *,
        validation_raster_index: int = 0,
        store_metadata: Mapping | None = None,
        batch_size: int = 4,
        validation_batch_size: int | None = None,
        validation_stride: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            validation_batch_size=validation_batch_size,
            validation_stride=validation_stride,
            num_workers=num_workers,
            pin_memory=pin_memory,
            progress=False,
        )
        self.rasters = list(rasters)
        self.raster_names = list(raster_names)
        if len(self.rasters) != len(self.raster_names) or not self.rasters:
            raise ValueError(
                "Stored raster names must match a non-empty raster sequence."
            )
        if any(
            not isinstance(name, str) or not name for name in self.raster_names
        ) or len(set(self.raster_names)) != len(self.raster_names):
            raise ValueError("Stored raster names must be non-empty unique strings.")
        if type(validation_raster_index) is not int:
            raise TypeError("validation_raster_index must be an integer.")
        self.validation_raster_index = validation_raster_index
        if not 0 <= self.validation_raster_index < len(self.rasters):
            raise IndexError("Stored validation raster index is outside the sequence.")
        self.raster = self.rasters[self.validation_raster_index]
        self.store_metadata = dict(store_metadata or {})
        self.response_collator = ObservationResponseCollator()

    def setup(self, stage=None) -> None:
        del stage
        if self.dataset is not None:
            return
        auxiliary = set(self.rasters[0].auxiliary)
        if any(set(raster.auxiliary) != auxiliary for raster in self.rasters[1:]):
            raise ValueError("Stored rasters must expose the same auxiliary fields.")
        wavelength = self.rasters[0].wavelength_angstrom
        if any(
            raster.wavelength_angstrom.shape != wavelength.shape
            or not torch.allclose(
                raster.wavelength_angstrom.to(torch.float64),
                wavelength.to(torch.float64),
                rtol=0.0,
                atol=1e-6,
            )
            for raster in self.rasters[1:]
        ):
            raise ValueError("Stored rasters must share one wavelength grid.")
        datasets = [
            ObservationPixelDataset(
                raster, auxiliary_fields=sorted(auxiliary), include_pixel_index=False
            )
            for raster in self.rasters
        ]
        self.dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        evaluation = ObservationPixelDataset(
            self.raster,
            auxiliary_fields=sorted(auxiliary),
            include_pixel_index=True,
            pixel_indices=datasets[self.validation_raster_index].pixel_indices,
        )
        self._evaluation_dataset = evaluation
        self._prepare_validation_dataset(evaluation)
        self.fits_consistency_metadata = dict(
            self.store_metadata.get("fits_consistency", {})
        )
        self._prepare_observation_sampling_bounds()

    def _stokes_train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup()
        return self._data_loader(
            dataset=self.dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.response_collator,
        )

    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is None:
            self.setup()
        return self._data_loader(
            dataset=self.validation_dataset,
            shuffle=False,
            batch_size=self.validation_batch_size,
            collate_fn=self.response_collator,
        )

    def evaluation_collate_fn(self):
        return self.response_collator


__all__ = ["ObservationDataModule", "StoredObservationDataModule"]
