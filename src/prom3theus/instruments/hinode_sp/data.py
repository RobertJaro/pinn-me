"""Single-raster data loading for static Hinode/SOT-SP LTE inversions."""

from __future__ import annotations

from pathlib import Path

from prom3theus.observations import ObservationDataModule, ObservationPixelDataset
from prom3theus.rt.radiometry import load_solar_reference

from .constants import HINODE_COORDINATE_NORMALIZATION_PIXELS
from .fits_io import resolve_files
from .raster import load_raster


def resolve_single_raster_files(files=None, directory=None) -> list[Path]:
    """Resolve one flat prepared Hinode raster and reject sequence layouts."""

    if files is not None and directory is not None:
        raise ValueError("Specify either Hinode files or directory, not both.")
    if directory is not None:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Hinode raster directory does not exist: {root}.")
        paths = sorted(root.glob("*.fits"))
        nested = sorted(path for path in root.rglob("*.fits") if path.parent != root)
        if nested:
            raise ValueError(
                "Hinode LTE accepts one flat raster directory; nested raster "
                "sequences are unsupported."
            )
        if not paths:
            raise FileNotFoundError(
                f"Hinode raster directory contains no FITS files: {root}."
            )
        return resolve_files(paths)
    if files is None:
        raise ValueError("Hinode data require files or directory.")
    return resolve_files(files)


class HinodeDataModule(ObservationDataModule):
    """Prepare exactly one Hinode/SP raster for a static LTE inversion."""

    def __init__(
        self,
        *,
        files=None,
        directory=None,
        batch_size: int = 4,
        validation_batch_size: int | None = None,
        validation_stride: int = 1,
        data_loading_workers: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        progress: bool = True,
        slit_slice=None,
        continuum_edge_samples: int = 8,
        quiet_sun_max_fractional_polarization: float = 0.01,
        quiet_sun_continuum_trim_quantiles=(0.05, 0.95),
        minimum_quiet_sun_pixels: int = 1,
        stokes_reference_angle_deg: float = 0.0,
    ) -> None:
        self.raster_options = {
            "slit_slice": slit_slice,
            "continuum_edge_samples": continuum_edge_samples,
            "quiet_sun_max_fractional_polarization": (
                quiet_sun_max_fractional_polarization
            ),
            "quiet_sun_continuum_trim_quantiles": (quiet_sun_continuum_trim_quantiles),
            "minimum_quiet_sun_pixels": minimum_quiet_sun_pixels,
            "stokes_reference_angle_deg": stokes_reference_angle_deg,
        }
        super().__init__(
            files=files,
            directory=directory,
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
            print(f"[Hinode LTE] {message}", flush=True)

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.raster is not None:
            return
        paths = resolve_single_raster_files(self.files, self.directory)
        self._progress(f"loading one raster containing {len(paths):,} FITS files")
        raster = load_raster(
            paths,
            solar_reference=load_solar_reference(),
            fits_read_workers=self.data_loading_workers,
            progress=self.progress,
            **self.raster_options,
        )
        spatial = raster.coordinates[..., :2][raster.valid_mask]
        spacing = float(
            raster.metadata["coordinates"]["network_affine"][
                "native_neighbor_spacing_mm"
            ]
        )
        affine = raster.metadata["coordinates"]["network_affine"]
        affine["center_mm"] = (
            0.5 * (spatial.amin(dim=0) + spatial.amax(dim=0))
        ).tolist()
        affine["scale_mm"] = [HINODE_COORDINATE_NORMALIZATION_PIXELS * spacing] * 2

        self.raster = raster
        self.rasters = [raster]
        self.raster_names = [Path(paths[0]).parent.name]
        self.validation_raster_index = 0
        times = raster.metadata["times"]
        wavelength = raster.wavelength_angstrom
        self.fits_consistency_metadata = {
            "required_keywords_validated_per_file": [
                "DATE_OBS",
                "SLITINDX",
                "CRVAL1",
                "CRPIX1",
                "CDELT1",
                "CUNIT1",
                "DOP_RCV",
                "SPWLSHFT",
                "SPWLSFT0",
                "XCEN",
                "YCEN",
                "CRPIX2",
                "CDELT2",
                "CROTA2",
            ],
            "strictly_increasing_date_obs_and_slitindx": True,
            "stokes_and_wavelength_shape": [4, int(wavelength.numel())],
            "slit_length": raster.spatial_shape[0],
            "date_obs_range": [min(times), max(times)],
            "file_timestamp_count": len(times),
            "unique_file_timestamp_count": len(set(times)),
            "time_source": "per-scan FITS DATE_OBS; never inferred from SLITINDX",
            "raster_loading": {
                "unit": "one complete raster",
                "data_loading_workers": self.data_loading_workers,
                "fits_readers": self.data_loading_workers,
                "nested_fits_parallelism": False,
                "solar_reference_loaded_once": True,
            },
        }
        dataset = ObservationPixelDataset(raster, include_pixel_index=False)
        self.dataset = dataset
        evaluation = ObservationPixelDataset(
            raster,
            include_pixel_index=True,
            pixel_indices=dataset.pixel_indices,
        )
        self._evaluation_dataset = evaluation
        self._prepare_validation_dataset(evaluation)
        self._prepare_observation_sampling_bounds()
        self._progress(
            f"ready: {len(dataset):,} training rays; "
            f"{len(self.validation_dataset):,} validation rays"
        )


__all__ = ["HinodeDataModule", "resolve_single_raster_files"]
