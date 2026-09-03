"""Time-series assembly and data loading for HMI Stokes acquisitions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from astropy.time import Time
import torch
from torch.utils.data import ConcatDataset

from prom3theus.observations import (
    ObservationDataModule,
    ObservationPixelDataset,
    ObservationResponseCollator,
)
from prom3theus.observations.io import ordered_parallel_fits_map
from prom3theus.rt.radiometry import load_solar_reference

from .acquisition import read_acquisition_header, resolve_acquisition_groups
from .raster import HMI_SOLAR_REFERENCE_RESOURCE, load_raster
from .response import (
    HMIResponseArchive,
    _resolve_response_profile_for_acquisition,
    load_response_manifest,
)


@dataclass(frozen=True, slots=True)
class HMITimelineTask:
    """Immutable inputs required to load one complete acquisition."""

    paths: list[Path]
    transmission_profile_directory: str | Path
    reference_time: Time
    scene_basis_rows: object
    response_sampler: HMIResponseArchive
    solar_reference: object
    acquisition: dict
    loader_options: dict


def load_timeline_acquisition(task: HMITimelineTask):
    """Load one complete acquisition inside a timeline worker."""

    return load_raster(
        task.paths,
        task.transmission_profile_directory,
        reference_time=task.reference_time,
        scene_basis_rows=task.scene_basis_rows,
        response_sampler=task.response_sampler,
        solar_reference=task.solar_reference,
        acquisition=task.acquisition,
        **task.loader_options,
    )


def _validation_index(groups, selected: int | str) -> int:
    if isinstance(selected, str):
        matches = [index for index, (name, _) in enumerate(groups) if name == selected]
        if len(matches) != 1:
            raise ValueError(
                "validation_raster must identify exactly one HMI acquisition; "
                f"got {selected!r}."
            )
        return matches[0]
    if type(selected) is not int:
        raise TypeError(
            "validation_raster must be an integer index or acquisition name."
        )
    index = selected
    if not 0 <= index < len(groups):
        raise IndexError("validation_raster index is outside the HMI sequence.")
    return index


def _response_samplers(response_directory: Path, acquisitions):
    manifest = load_response_manifest(response_directory)
    archives: dict[Path, HMIResponseArchive] = {}
    acquisition_archives = []
    for acquisition in acquisitions:
        profile_file = _resolve_response_profile_for_acquisition(
            response_directory,
            manifest,
            acquisition,
        )
        archive = archives.get(profile_file)
        if archive is None:
            archive = HMIResponseArchive(profile_file)
            archives[profile_file] = archive
        acquisition_archives.append(archive)
    return archives, acquisition_archives


def _shared_response_grid(archives, wavelength):
    response_grids = [archive.quadrature_wavelength(wavelength) for archive in archives]
    if any(
        grid.shape != response_grids[0].shape
        or not torch.allclose(grid, response_grids[0], rtol=0.0, atol=1.0e-4)
        for grid in response_grids[1:]
    ):
        raise ValueError(
            "HMI timeline response profiles use different quadrature grids."
        )
    half_widths = {archive.inner_half_width_angstrom for archive in archives}
    if len(half_widths) != 1:
        raise ValueError("HMI timeline response profiles use different inner windows.")
    return response_grids[0], half_widths.pop()


class HMIDataModule(ObservationDataModule):
    """Data module for one or more complete HMI 720-second acquisitions."""

    def __init__(
        self,
        *,
        files=None,
        directory=None,
        transmission_profile_directory,
        validation_raster: int | str = 0,
        batch_size: int = 4,
        validation_batch_size: int | None = None,
        validation_stride: int = 1,
        data_loading_workers: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        progress: bool = True,
        require_quality_zero: bool = True,
        quiet_sun_max_fractional_polarization: float = 0.01,
        quiet_sun_trim_quantiles=(0.05, 0.95),
        minimum_quiet_sun_pixels: int = 32,
        calibration_sample_limit: int = 4096,
    ) -> None:
        self.transmission_profile_directory = transmission_profile_directory
        self.raster_options = {
            "require_quality_zero": require_quality_zero,
            "quiet_sun_max_fractional_polarization": (
                quiet_sun_max_fractional_polarization
            ),
            "quiet_sun_trim_quantiles": quiet_sun_trim_quantiles,
            "minimum_quiet_sun_pixels": minimum_quiet_sun_pixels,
            "calibration_sample_limit": calibration_sample_limit,
        }
        super().__init__(
            files=files,
            directory=directory,
            validation_raster=validation_raster,
            batch_size=batch_size,
            validation_batch_size=validation_batch_size,
            validation_stride=validation_stride,
            data_loading_workers=data_loading_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            progress=progress,
        )

    def _progress(self, message: str) -> None:
        if self.progress:
            print(f"[HMI LTE] {message}", flush=True)

    def setup(self, stage=None) -> None:
        del stage
        if self.raster is not None:
            return
        groups = resolve_acquisition_groups(self.files, self.directory)
        self._progress(
            f"discovered {len(groups)} HMI acquisitions containing "
            f"{sum(len(paths) for _, paths in groups):,} FITS segments; "
            "loading one complete acquisition per timeline worker"
        )
        validation_index = _validation_index(groups, self.validation_raster)

        reference_files = [
            next(path for path in paths if path.name.endswith(".I0.fits"))
            for _, paths in groups
        ]
        # Prime Astropy/SunPy process-wide state before concurrent WCS loading.
        acquisitions = ordered_parallel_fits_map(
            read_acquisition_header,
            reference_files,
            1,
            progress=self.progress,
            description="HMI acquisition headers",
            unit="acquisition",
        )
        dates = [item["date"] for item in acquisitions]
        if len({date.utc.isot for date in dates}) != len(dates):
            raise ValueError("HMI acquisitions must have unique T_OBS timestamps.")
        reference_time = min(dates)

        response_directory = Path(self.transmission_profile_directory).resolve()
        archives, acquisition_archives = _response_samplers(
            response_directory, acquisitions
        )
        solar_reference = load_solar_reference(
            resource_name=HMI_SOLAR_REFERENCE_RESOURCE,
        )
        reference_task = HMITimelineTask(
            paths=groups[validation_index][1],
            transmission_profile_directory=self.transmission_profile_directory,
            reference_time=reference_time,
            scene_basis_rows=None,
            response_sampler=acquisition_archives[validation_index],
            solar_reference=solar_reference,
            acquisition=acquisitions[validation_index],
            loader_options=self.raster_options,
        )
        reference_result = ordered_parallel_fits_map(
            load_timeline_acquisition, [reference_task], 1
        )[0]
        shared_basis = reference_result.metadata["ray_geometry"]["scene_basis_rows"]
        remaining_indices = [
            index for index in range(len(groups)) if index != validation_index
        ]
        remaining_tasks = [
            HMITimelineTask(
                paths=groups[index][1],
                transmission_profile_directory=self.transmission_profile_directory,
                reference_time=reference_time,
                scene_basis_rows=shared_basis,
                response_sampler=acquisition_archives[index],
                solar_reference=solar_reference,
                acquisition=acquisitions[index],
                loader_options=self.raster_options,
            )
            for index in remaining_indices
        ]
        remaining_results = ordered_parallel_fits_map(
            load_timeline_acquisition,
            remaining_tasks,
            self.data_loading_workers,
            progress=self.progress,
            description="HMI timeline datasets",
            unit="acquisition",
        )
        results = {validation_index: reference_result}
        results.update(zip(remaining_indices, remaining_results, strict=True))
        ordered_results = [results[index] for index in range(len(groups))]
        loaded = ordered_results

        archive_list = list(archives.values())
        (
            self.response_quadrature_wavelength_angstrom,
            self.response_inner_half_width_angstrom,
        ) = _shared_response_grid(
            archive_list, loaded[validation_index].wavelength_angstrom
        )
        self.response_collator = ObservationResponseCollator()
        datasets = [
            ObservationPixelDataset(
                raster,
                include_pixel_index=False,
            )
            for raster in loaded
        ]

        self.rasters = loaded
        self.raster_names = [raster.metadata["acquisition_key"] for raster in loaded]
        self.validation_raster_index = validation_index
        self.raster = loaded[validation_index]
        self.dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        self._evaluation_dataset = ObservationPixelDataset(
            self.raster,
            include_pixel_index=True,
            pixel_indices=datasets[validation_index].pixel_indices,
        )
        self._update_coordinate_normalization(reference_time)
        self.fits_consistency_metadata = {
            "complete_segment_set_per_acquisition": True,
            "segment_count_per_acquisition": 24,
            "observer_velocity_keywords_complete_and_consistent": True,
            "acquisition_count": len(loaded),
            "single_pass_segment_validation_and_data_read": True,
            "timeline_loading": {
                "unit": "complete HMI acquisition dataset",
                "data_loading_workers": self.data_loading_workers,
                "fits_readers_per_dataset": 1,
                "nested_fits_parallelism": False,
            },
            "unique_response_profiles_loaded": len(archives),
            "observation_time_range_tai": [
                str(min(dates).tai.isot),
                str(max(dates).tai.isot),
            ],
            "shared_scene_basis": True,
            "wavelength_grid_shared_across_rasters": True,
        }
        self._prepare_validation_dataset(self._evaluation_dataset)
        self._prepare_observation_sampling_bounds()
        self._progress(
            f"ready: {len(self.dataset):,} HMI training rays; "
            f"{len(self.validation_dataset):,} validation rays from "
            f"{self.raster_names[validation_index]}"
        )

    def _update_coordinate_normalization(self, reference_time: Time) -> None:
        xy_min = None
        xy_max = None
        time_min = math.inf
        time_max = -math.inf
        for raster in self.rasters:
            values = raster.coordinates[..., :2][raster.valid_mask]
            current_min = values.amin(dim=0)
            current_max = values.amax(dim=0)
            xy_min = (
                current_min if xy_min is None else torch.minimum(xy_min, current_min)
            )
            xy_max = (
                current_max if xy_max is None else torch.maximum(xy_max, current_max)
            )
            times = raster.coordinates[..., 2][raster.valid_mask]
            time_min = min(time_min, float(times.amin()))
            time_max = max(time_max, float(times.amax()))
        affine = self.raster.metadata["coordinates"]["network_affine"]
        affine["center_mm"] = (0.5 * (xy_min + xy_max)).tolist()
        affine["scale_mm"] = torch.maximum(
            0.5 * (xy_max - xy_min), torch.full_like(xy_min, 1.0e-6)
        ).tolist()
        self.raster.metadata["coordinates"]["time_origin"] = str(reference_time.isot)
        self.raster.metadata["coordinates"]["time_affine"] = {
            "center_hours": 0.5 * (time_min + time_max),
            "scale_hours": max(0.5 * (time_max - time_min), 1.0 / 3600.0),
            "range_hours": [time_min, time_max],
            "formula": "normalized_time=(time_hours-center_hours)/scale_hours",
        }

    def _stokes_train_dataloader(self):
        if self.dataset is None:
            self.setup()
        return self._data_loader(
            dataset=self.dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.response_collator,
        )

    def val_dataloader(self):
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

    def run_metadata(self) -> dict:
        metadata = super().run_metadata()
        if len(self.rasters) > 1:
            metadata["multi_raster"]["acquisitions"] = [
                {
                    "acquisition_key": raster.metadata["acquisition_key"],
                    "times": list(raster.metadata["times"]),
                    "files": list(raster.metadata["files"]),
                    "spectral_response": dict(raster.metadata["spectral_response"]),
                    "radiometric_calibration": dict(
                        raster.metadata["normalization"]["radiometric_calibration"]
                    ),
                    "observer_velocity_correction": dict(
                        raster.metadata["observer_velocity_correction"]
                    ),
                    "quality_mask": dict(raster.metadata["quality_mask"]),
                    "data_fingerprints": dict(raster.metadata["data_fingerprints"]),
                }
                for raster in self.rasters
            ]
        return metadata


__all__ = ["HMIDataModule", "HMITimelineTask", "load_timeline_acquisition"]
