"""Synthetic SDO FITS products with an explicit, lossless training contract.

These are radiance/reference-response products, not raw detector exposures.
The primary images follow SDO ordering and WCS; binary-table extensions carry
exact sampling geometry and response metadata for the flat MURaM ray bundle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.time import Time, TimeDelta

from prom3theus.instruments.aia_euv.preparation import (
    AIA_INTENSITY_UNIT,
    AIA_RAY_DIRECTION_CONVENTION,
    AIA_SURFACE_FRAME,
    AIA_TIME_REPRESENTATION,
    RegisteredAIAChannelRaster,
    _verified_response_contract,
    prepare_aia_observation_store,
)
from prom3theus.instruments.hmi.acquisition import format_jsoc_time
from prom3theus.instruments.hmi.calibration import hmi_continuum_radiance
from prom3theus.observations import ObservationRaster
from prom3theus.rt.radiometry import load_solar_reference, reference_summary

from .geometry import SOLAR_RADIUS_M
from .provenance import check_stokes_forward_provenance
from .synthesize import AIA_RESPONSE_RESOURCE, MURAM_HMI_RADIANCE_SCALE_W_M3_SR
from .reporting import progress_bar, progress_enabled, timed_stage

logger = logging.getLogger(__name__)

SCHEMA = "PROM3THEUS-MURAM-1"
EPOCH = "2010-03-24T00:00:00"


def observation_time(time_s):
    return Time(EPOCH, scale="utc") + TimeDelta(float(time_s), format="sec")


def array_hdu(name, value):
    array = np.asarray(value)
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="VALUES", format=f"{array.size}D", array=[array.ravel()])],
        name=name,
    )
    hdu.header["SHAPE"] = json.dumps(list(array.shape))
    return hdu


def read_array(hdus, name, *, dtype=torch.float32):
    hdu = hdus[name]
    value = np.array(hdu.data["VALUES"][0], dtype=np.float64, copy=True)
    return torch.as_tensor(value.reshape(json.loads(hdu.header["SHAPE"])), dtype=dtype)


def metadata_hdu(metadata):
    text = json.dumps(metadata, sort_keys=True, allow_nan=False)
    return fits.BinTableHDU.from_columns(
        [fits.Column(name="JSON", format=f"{len(text)}A", array=[text])],
        name="CONTRACT",
    )


def read_metadata(hdus):
    return json.loads(hdus["CONTRACT"].data["JSON"][0])


def write_fits(path, hdus):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".fits", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        fits.HDUList(hdus).writeto(temporary, overwrite=True, checksum=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def image_header(geometry, dx_m, dy_m, time_s, iteration, instrument):
    time = observation_time(time_s)
    distance = 149597870700.0
    x0, y0 = geometry.coordinates[0, 0, :2].double().numpy() * 1e6
    toward_observer = geometry.stokes_basis[0, 0, 2].double().numpy()
    observer_longitude = float(np.rad2deg(np.arctan2(toward_observer[1], toward_observer[0])))
    observer_latitude = float(np.rad2deg(np.arcsin(toward_observer[2])))
    header = fits.Header(
        {
            "TELESCOP": f"SDO/{instrument}",
            "INSTRUME": instrument,
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CDELT1": float(np.rad2deg(dx_m / (distance - SOLAR_RADIUS_M)) * 3600),
            "CDELT2": float(np.rad2deg(dy_m / (distance - SOLAR_RADIUS_M)) * 3600),
            "CRPIX1": float(1 - x0 / dx_m),
            "CRPIX2": float(1 - y0 / dy_m),
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CROTA2": 0.0,
            "DSUN_OBS": distance,
            "RSUN_REF": SOLAR_RADIUS_M,
            "RSUN_OBS": float(np.rad2deg(np.arcsin(SOLAR_RADIUS_M / distance)) * 3600),
            # Match the actual scene orientation, including old polar products.
            "CRLN_OBS": observer_longitude,
            "CRLT_OBS": observer_latitude,
            "HGLN_OBS": 0.0,
            "HGLT_OBS": observer_latitude,
            "DATE-OBS": time.utc.isot,
            "TIMESYS": "UTC",
            "EXPTIME": 1.0,
            "QUALITY": 0,
            "SYNTHET": True,
            "SYNSCHEM": SCHEMA,
            "SIMTIME": float(time_s),
            "MURAMIT": int(iteration),
        }
    )
    header.add_history(
        "Synthetic MURaM flat, parallel disk-center rays; exact geometry in tables."
    )
    header.add_history(
        "Not an observed SDO exposure. WCS is a visualization embedding."
    )
    return header


def write_hmi_fits(directory, result, *, time_s, iteration, progress=True):
    time = observation_time(time_s)
    stamp = time.tai.to_datetime().strftime("%Y%m%d_%H%M%S")
    prefix = f"hmi.S_720s.{stamp}_TAI.3"
    header = image_header(
        result.geometry, result.dx_m, result.dy_m, time_s, iteration, "HMI"
    )
    header.update(
        {
            "T_OBS": format_jsoc_time(time.tai.to_datetime()),
            "T_REC": format_jsoc_time(time.tai.to_datetime()),
            "CAMERA": 3,
            "HCAMID": 3,
            "OBS_VR": 0.0,
            "OBS_VW": 0.0,
            "OBS_VN": 0.0,
            "BUNIT": "W m-3 sr-1",
            "RADSCALE": MURAM_HMI_RADIANCE_SCALE_W_M3_SR,
        }
    )
    metadata = result.observation_metadata()
    metadata["synthetic"] = {
        "schema": SCHEMA,
        "iteration": iteration,
        "response": "idealized Gaussian; no empirical phase maps",
        "spatial_sampling": "native column synthesis followed by block averaging; no additional detector PSF",
        "spatial_stride": result.spatial_stride,
        "spatial_binning": result.spatial_binning,
    }
    paths = []
    with timed_stage(logger, "Iteration %s: writing 24 HMI FITS segments", iteration), progress_bar(
        total=24, description=f"Iteration {iteration}: HMI FITS", unit="file", enabled=progress,
    ) as bar:
        for component, label in enumerate("IQUV"):
            for index in range(6):
                local = header.copy()
                local["STOKES"] = label
                local["WAVELNTH"] = float(result.observed_wavelength_angstrom[5 - index])
                # HMI FITS is red-to-blue, whereas synthesis is blue-to-red.
                image = (
                    result.stokes[..., component, 5 - index].numpy()
                    * MURAM_HMI_RADIANCE_SCALE_W_M3_SR
                )
                hdus = [fits.PrimaryHDU(image.astype(np.float32), local)]
                if component == 0 and index == 0:
                    for name, value in (
                        ("RAYS", result.geometry.ray_direction),
                        ("SURFACE", result.geometry.surface_position_m),
                        ("BASIS", result.geometry.stokes_basis),
                        ("COORDS", result.geometry.coordinates),
                        ("QUAD", result.raw_wavelength_angstrom[1:-1]),
                        ("WEIGHTS", result.response["spectral_weights"]),
                        ("CONTIN", result.response["continuum_weights"]),
                    ):
                        hdus.append(array_hdu(name, value.numpy()))
                    hdus.append(metadata_hdu(metadata))
                path = Path(directory) / f"{prefix}.{label}{index}.fits"
                write_fits(path, hdus)
                paths.append(path)
                bar.update()
    return paths


def synthetic_response_path(directory, reference_file):
    """Resolve explicit profiles; retain embedded responses for older runs."""
    directory = Path(directory).resolve()
    reference_file = Path(reference_file).resolve()
    if directory == reference_file.parent:
        return reference_file
    path = directory / f"{reference_file.name.removesuffix('.I0.fits')}.response.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic HMI transmission profile not found: {path}")
    return path


def write_hmi_transmission_profiles(directory, reference_file):
    """Export the exact response from an I0 FITS, including existing syntheses.

    Arrays are in increasing wavelength order. These are synthetic response
    archives, with no invented observational phase-map or detune provenance.
    """
    reference_file = Path(reference_file)
    response = SyntheticHMIResponse(reference_file)
    with fits.open(reference_file) as hdus:
        contract = read_metadata(hdus)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{reference_file.name.removesuffix('.I0.fits')}.response.npz"
    if path.exists():
        raise FileExistsError(f"Synthetic transmission profile already exists: {path}")
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(
                handle,
                quadrature_wavelength_angstrom=response.quadrature.numpy(),
                spectral_weights=response.weights.numpy(),
                continuum_weights=response.continuum_weights.numpy(),
                metadata=np.array(json.dumps(contract, allow_nan=False)),
            )
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


class SyntheticHMIResponse:
    """The response that actually generated a marked synthetic acquisition."""

    def __init__(self, path):
        self.profile_file = Path(path)
        if self.profile_file.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                self.quadrature = torch.tensor(archive["quadrature_wavelength_angstrom"], dtype=torch.float64)
                self.weights = torch.tensor(archive["spectral_weights"], dtype=torch.float32)
                self.continuum_weights = torch.tensor(archive["continuum_weights"], dtype=torch.float32)
                contract = json.loads(str(archive["metadata"].item()))
            if contract.get("synthetic", {}).get("schema") != SCHEMA:
                raise ValueError("Unknown synthetic HMI transmission contract")
        else:
            with fits.open(path) as hdus:
                if hdus[0].header.get("SYNSCHEM") != SCHEMA:
                    raise ValueError("Unknown synthetic HMI FITS contract")
                self.quadrature = read_array(hdus, "QUAD", dtype=torch.float64)
                self.weights = read_array(hdus, "WEIGHTS")
                self.continuum_weights = read_array(hdus, "CONTIN")
                contract = read_metadata(hdus)
        self.inner_half_width_angstrom = contract["muram_synthesis"]["instrument"][
            "inner_half_width_angstrom"
        ]
        self.metadata = contract["synthetic"]
        if self.weights.shape != (
            6,
            self.quadrature.numel(),
        ) or self.continuum_weights.shape != (6,):
            raise ValueError("Invalid synthetic HMI response shape")
        if any(
            not torch.isfinite(value).all() or torch.any(value < 0)
            for value in (self.weights, self.continuum_weights)
        ):
            raise ValueError("Invalid synthetic HMI response weights")
        if not torch.allclose(
            self.weights.sum(-1) + self.continuum_weights,
            torch.ones(6),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Synthetic HMI response weights must sum to one")
        if (
            self.quadrature.ndim != 1
            or not torch.isfinite(self.quadrature).all()
            or not torch.all(self.quadrature[1:] > self.quadrature[:-1])
        ):
            raise ValueError("Synthetic HMI quadrature must be finite and increasing")

    def quadrature_wavelength(self, observed):
        return self.quadrature

    def sample(self, detector_xy):
        shape = torch.as_tensor(detector_xy).shape[:-1]
        return {
            "spectral_weights": self.weights.expand(*shape, *self.weights.shape),
            "continuum_weights": self.continuum_weights.expand(*shape, 6),
        }


def load_hmi_raster(
    segments, headers, stokes, *, reference_time=None, scene_basis_rows=None,
    response_sampler=None, solar_reference=None, require_quality_zero=True,
):
    if any(header.get("SYNSCHEM") != SCHEMA for header in headers):
        raise ValueError("Cannot mix synthetic and observed HMI segments")
    if (
        any(
            header.get(key) != headers[0].get(key)
            for header in headers
            for key in ("SIMTIME", "MURAMIT", "BUNIT")
        )
        or headers[0].get("BUNIT") != "W m-3 sr-1"
    ):
        raise ValueError("Synthetic HMI segments have inconsistent physical metadata")
    with fits.open(segments["I0"]) as hdus:
        metadata = read_metadata(hdus)
        coords = read_array(hdus, "COORDS")
        ray = read_array(hdus, "RAYS")
        surface = read_array(hdus, "SURFACE")
        basis = read_array(hdus, "BASIS")
        truth = read_array(hdus, "BZTAU1") if "BZTAU1" in hdus else None
    metadata["synthesis_provenance_verification"] = check_stokes_forward_provenance(
        metadata.get("muram_synthesis", {}).get("forward_model_provenance"),
        "hmi_filter_profiles",
    )
    if scene_basis_rows is not None and not torch.allclose(
        torch.as_tensor(scene_basis_rows, dtype=torch.float64),
        torch.as_tensor(metadata["ray_geometry"]["scene_basis_rows"], dtype=torch.float64),
    ):
        raise ValueError(
            "Synthetic HMI scene basis must match the embedded MURaM geometry"
        )
    time = observation_time(headers[0]["SIMTIME"])
    origin = time if reference_time is None else Time(reference_time)
    coords[..., 2] = float((time - origin).to_value("hour"))
    synthesis_scale = float(headers[0]["RADSCALE"])
    if (
        not np.isfinite(synthesis_scale)
        or synthesis_scale <= 0
        or any(h["RADSCALE"] != synthesis_scale for h in headers)
    ):
        raise ValueError("Invalid synthetic radiance scale")
    # RADSCALE is the internal transfer unit, not the training data unit.
    # FITS already holds SI radiance; use precisely the observed HMI atlas unit
    # for targets and forward predictions so MSE/physics weights are comparable.
    if solar_reference is None:
        solar_reference = load_solar_reference("hmi_stokes/solar_reference_617nm.json")
    scale = hmi_continuum_radiance(solar_reference)
    response = SyntheticHMIResponse(segments["I0"])
    if response_sampler is not None:
        if (
            any(
                not torch.equal(getattr(response, name), getattr(response_sampler, name))
                for name in ("quadrature", "weights", "continuum_weights")
            )
            or response.inner_half_width_angstrom != response_sampler.inner_half_width_angstrom
        ):
            raise ValueError("Transmission profile does not match the response used for synthesis")
        response = response_sampler
    metadata["coordinates"].update(
        time_origin=origin.isot, time_scale=origin.scale.upper()
    )
    metadata.update(
        files=[str(path) for path in segments.values()],
        times=[time.isot],
        ref_time=origin.isot,
        acquisition_key=f"{headers[0]['T_REC']}|HCAMID=3",
        spectral_response={"file": str(response.profile_file), "metadata": response.metadata},
        quality_mask={
            "require_quality_zero": bool(require_quality_zero),
            "quality_values": [int(header["QUALITY"]) for header in headers],
        },
    )
    metadata["normalization"]["radiometric_calibration"].update(
        type="known synthetic physical radiance; no fitted detector gain",
        atlas_disk_center_continuum_radiance_w_m3_sr=scale,
        stored_stokes_unit="I_c,atlas(mu=1)",
        solar_reference=reference_summary(solar_reference),
        synthesis_radiance_scale_w_m3_sr=synthesis_scale,
    )
    metadata["observer_velocity_correction"] = {
        "source_keywords": ["OBS_VR", "OBS_VW", "OBS_VN"],
        "drms_components_m_per_s": [0.0, 0.0, 0.0],
    }
    values = torch.from_numpy(np.asarray(stokes / scale, dtype=np.float32))
    return ObservationRaster(
        stokes=values,
        wavelength_angstrom=torch.tensor(
            [headers[5 - i]["WAVELNTH"] for i in range(6)], dtype=torch.float64
        ),
        coordinates=coords,
        ray_direction=ray,
        surface_position_m=surface,
        stokes_basis=basis,
        valid_mask=torch.isfinite(values).all(dim=(-1, -2)),
        metadata=metadata,
        auxiliary={
            **({"ground_truth_bz_gauss": truth} if truth is not None else {}),
            "observer_los_velocity_m_per_s": torch.zeros(values.shape[:2]),
            **{
                "instrument_response:" + key: value[0, 0].clone().expand_as(value)
                for key, value in response.sample(coords[..., :2]).items()
            },
        },
    )


def write_aia_fits(directory, result, geometry, *, time_s, iteration, progress=True):
    if result.response_resource != AIA_RESPONSE_RESOURCE:
        raise ValueError("Standard AIA FITS require the standard response resource; use NPZ for custom responses.")
    time = observation_time(time_s)
    paths = []
    with timed_stage(logger, "Iteration %s: writing %d AIA FITS files", iteration, len(result.channels_angstrom)), progress_bar(
        total=len(result.channels_angstrom), description=f"Iteration {iteration}: AIA FITS", unit="file", enabled=progress,
    ) as bar:
        for index, channel in enumerate(result.channels_angstrom):
            header = image_header(
                geometry, result.dx_m, result.dy_m, time_s, iteration, "AIA"
            )
            header.update(
                {
                    "WAVELNTH": channel,
                    "WAVEUNIT": "angstrom",
                    "BUNIT": "DN / s",
                    "LVL_NUM": 1.5,
                    "DEGFACT": 1.0,
                    "EXPNORM": True,
                }
            )
            header.add_history(
                "Reference-epoch response, unity degradation, count rate per reference AIA pixel."
            )
            path = (
                Path(directory)
                / f"aia.lev1_euv_12s.{time.utc.to_datetime().strftime('%Y-%m-%dT%H%M%S')}.{channel}.fits"
            )
            write_fits(
                path,
                [
                    fits.PrimaryHDU(result.intensity[..., index].numpy(), header),
                    array_hdu("RAYS", geometry.ray_direction.numpy()),
                    array_hdu("SURFACE", geometry.surface_position_m.numpy()),
                    metadata_hdu({"native_reference": result.native_reference,
                                  "response_resource": result.response_resource}),
                ],
            )
            paths.append(path)
            bar.update()
    return paths


def load_aia_fits(path):
    with fits.open(path) as hdus:
        header = hdus[0].header
        if (
            header.get("SYNSCHEM") != SCHEMA
            or header.get("BUNIT") != "DN / s"
            or header.get("DEGFACT") != 1.0
        ):
            raise ValueError("Invalid synthetic AIA reference-response FITS")
        intensity = torch.from_numpy(
            np.array(hdus[0].data, dtype=np.float32, copy=True)
        )
        ray, surface = read_array(hdus, "RAYS"), read_array(hdus, "SURFACE")
        synthesis_metadata = read_metadata(hdus) if "CONTRACT" in hdus else {}
        if synthesis_metadata.get("response_resource", AIA_RESPONSE_RESOURCE) != AIA_RESPONSE_RESOURCE:
            raise ValueError("Synthetic AIA FITS response differs from the standard calibration contract.")
        return RegisteredAIAChannelRaster(
            intensity=intensity,
            valid_mask=torch.isfinite(intensity),
            on_disk_mask=torch.ones_like(intensity, dtype=torch.bool),
            ray_direction=ray,
            surface_position_m=surface,
            absolute_tai_seconds=float(
                observation_time(header["SIMTIME"]).tai.unix_tai
            ),
            exposure_group=f"muram_{header['MURAMIT']:06d}",
            channel_angstrom=int(header["WAVELNTH"]),
            solar_radius_m=float(header["RSUN_REF"]),
            degradation_factor=1.0,
            degradation_correction_applications=0,
            calibration_convention_id=_verified_response_contract()[0],
            intensity_unit=AIA_INTENSITY_UNIT,
            time_representation=AIA_TIME_REPRESENTATION,
            surface_frame=AIA_SURFACE_FRAME,
            ray_direction_convention=AIA_RAY_DIRECTION_CONVENTION,
            provenance={
                "synthetic": True,
                **synthesis_metadata,
                "file": str(path),
                "response": "reference epoch; no detector aging",
                "dx_m": float(np.deg2rad(header["CDELT1"] / 3600) * (header["DSUN_OBS"] - header["RSUN_REF"])),
                "dy_m": float(np.deg2rad(header["CDELT2"] / 3600) * (header["DSUN_OBS"] - header["RSUN_REF"])),
                "geometry": "parallel MURaM columns; use embedded geometry",
                "degradation": "unity by construction",
            },
        )


def prepare_aia_fits(paths, output, *, progress=True):
    def digest():
        value = hashlib.sha256()
        with timed_stage(logger, "Computing saved AIA FITS checksums"), progress_bar(
            paths, description="AIA FITS checksums", unit="file", enabled=progress,
        ) as files:
            for path in files:
                with Path(path).open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        value.update(block)
        return value.hexdigest()

    return prepare_aia_observation_store(
        output,
        source_products=paths,
        raster_backend=load_aia_fits,
        source_files_sha256=digest,
        preparation_dependencies={"synthetic_fits_contract": SCHEMA},
        progress=progress_enabled(progress),
    )
