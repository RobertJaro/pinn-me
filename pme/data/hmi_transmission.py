"""Prepare HMI spectral transmission profiles for offline inversions.

The network-facing part of this module is deliberately separated from the
training code.  It resolves the phase-map record used by the definitive HMI
Milne--Eddington pipeline, downloads that record with ``drms``, and writes a
small, self-contained ``.npz`` file.  Training only reads the resulting file.

The phase-map product contains phases for the three tunable elements (the
narrow- and wide-band Michelsons and Lyot E1), followed by fitted solar line
width and depth maps.  HMI does not publish a ready-made six-profile segment.
We therefore reconstruct the profiles from the published element model,

    T(lambda) = (1 + B cos(2 pi lambda / FSR + Phi + 4 phi)) / 2,

using the record-specific phase maps, motor positions, and free spectral ranges.
The archive does not provide time-dependent contrast maps or the measured
front-window response in this series. The reconstruction consequently uses
published mean ground-calibration contrasts, blocking-filter width, and
untuned-stack center; these assumptions are recorded in the output metadata.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import drms
import numpy as np
from astropy.io import fits
from dateutil.parser import parse

from pme.instrument import HMI_WAVELENGTH_GRID


HMI_ME_SERIES = "hmi.ME_720s_fd10"
HMI_PHASE_SERIES = "hmi.phasemaps_extended"

PHASE_CHANNELS = ("NB", "WB", "E1")
FSR_KEYS = ("FSRNB", "FSRWB", "FSRE1", "FSRE2", "FSRE3", "FSRE4", "FSRE5")
HCM_KEYS = ("HCMNB", "HCMWB", "HCME1")

# One HCM step rotates a tuning half-wave plate by 1.5 degrees.  The optical
# phase in the element transmission is four times the plate angle.
HCM_OPTICAL_DEGREES_PER_STEP = 6.0

# HCM increments between adjacent 68.8 mA observing positions.  The E1 motor
# is installed with the opposite angular orientation to the Michelson motors.
HCM_STEPS_PER_SAMPLE = np.array([24.0, 12.0, 6.0], dtype=np.float64)
HCM_ORIENTATION = np.array([1.0, 1.0, -1.0], dtype=np.float64)

# Average observing-mode element contrasts measured during ground calibration
# (NB, WB, E1, E2, E3, E4, E5; Couvidat et al. 2012). The phase-map series
# does not provide spatially or temporally resolved contrast maps.
HMI_NOMINAL_CONTRASTS = np.array([0.969, 0.997, 0.962, 0.985, 0.965, 0.988, 1.000], dtype=np.float64)

DEFAULT_PROFILE_HALF_WIDTH = 0.65  # Angstrom about the Fe I line center
DEFAULT_PROFILE_SAMPLES = 81
DEFAULT_BLOCKER_TAIL_SIGMA = 7.0
DEFAULT_CONTINUUM_PANEL_WIDTH = 0.1  # Angstrom
DEFAULT_CONTINUUM_PANEL_NODES = 8
CONTINUUM_CHUNK_PANELS = 8
HMI_CCD_SIZE = 4096

# Mean ground-calibration values from Couvidat et al. (2012). The phase-map
# product does not contain the measured non-tunable response itself.
DEFAULT_BLOCKER_FWHM = 8.43  # Angstrom
HMI_UNTUNED_CENTER = 0.016  # Angstrom relative to the Fe I target wavelength

PHASE_QUERY_KEYS = (
    "T_REC,FSN_REC,HCAMID,NX,T_START,T_STOP,HCMNB,HCMWB,HCMPOL,HCME1,"
    "FSRNB,FSRWB,FSRE1,FSRE2,FSRE3,FSRE4,FSRE5,CBLOCKER,"
    "PHASENBM,PHASEWBM,PHASELYO,COMMENT"
)
MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "pme.hmi_transmission_manifest.v4"


def format_jsoc_time(value: str | datetime) -> str:
    """Return a JSOC record time, treating an unqualified input as TAI."""
    if isinstance(value, str) and value.endswith("_TAI"):
        return value
    parsed = parse(value) if isinstance(value, str) else value
    return parsed.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def hmi_acquisition_key(observation_time: str | datetime, hcamid: int) -> str:
    return f"{format_jsoc_time(observation_time)}|HCAMID={int(hcamid)}"


def read_hmi_fits_acquisition(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the acquisition identity from any HMI Stokes FITS segment."""
    path = Path(path)
    with fits.open(path, memmap=False) as hdus:
        header = next((hdu.header for hdu in hdus if 'T_REC' in hdu.header), None)
    if header is None:
        raise ValueError(f"No T_REC keyword found in any FITS extension of {path}.")
    try:
        observation_time = format_jsoc_time(str(header['T_REC']))
        hcamid = int(header['HCAMID'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid T_REC/HCAMID in {path}.") from error
    return {
        'path': str(path.resolve()),
        'observation_time': observation_time,
        'hcamid': hcamid,
        'acquisition_key': hmi_acquisition_key(observation_time, hcamid),
    }


def discover_hmi_acquisitions(inputs: Iterable[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    """Discover unique HMI acquisitions from files, directories, or globs."""
    inputs = list(inputs)
    candidates = []
    for value in inputs:
        value = os.fspath(value)
        if os.path.isdir(value):
            candidates.extend(glob.glob(os.path.join(value, '*.fits')))
        elif os.path.isfile(value):
            candidates.append(value)
        else:
            candidates.extend(glob.glob(value))
    candidates = sorted(set(candidates))
    if not candidates:
        raise ValueError(f"No FITS files matched HMI inputs: {list(map(os.fspath, inputs))!r}.")

    # I0 is sufficient to identify a complete acquisition and is required by
    # the inversion loader's strict I0...V5 segment convention.
    representatives = [path for path in candidates if path.endswith('.I0.fits')]
    if not representatives:
        raise ValueError(
            "No HMI I0 FITS segments found. Expected filenames ending in '.I0.fits'."
        )
    acquisitions = {}
    for path in representatives:
        acquisition = read_hmi_fits_acquisition(path)
        acquisitions.setdefault(acquisition['acquisition_key'], acquisition)
    return [acquisitions[key] for key in sorted(acquisitions)]


def load_hmi_transmission_manifest(directory: str | os.PathLike[str]) -> dict[str, Any]:
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"HMI transmission manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != MANIFEST_FORMAT:
        raise ValueError(
            f"Unsupported HMI transmission manifest format in {manifest_path}. "
            "Regenerate the calibration directory with --overwrite."
        )
    profiles = manifest.get('profiles')
    acquisitions = manifest.get('acquisitions')
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"HMI transmission manifest contains no profiles: {manifest_path}")
    if not isinstance(acquisitions, dict) or not acquisitions:
        raise ValueError(f"HMI transmission manifest contains no acquisitions: {manifest_path}")
    for profile_key, profile in profiles.items():
        if not isinstance(profile, dict) or not isinstance(profile.get('file'), str):
            raise ValueError(f"Invalid HMI transmission profile {profile_key!r} in {manifest_path}.")
        profile_path = directory / profile['file']
        if not profile_path.is_file():
            raise FileNotFoundError(f"HMI transmission profile listed in manifest is missing: {profile_path}")
    for acquisition_key, acquisition in acquisitions.items():
        if not isinstance(acquisition, dict) or acquisition.get('profile') not in profiles:
            raise ValueError(f"Invalid HMI acquisition {acquisition_key!r} in {manifest_path}.")
    return manifest


def resolve_hmi_transmission_profile(
    directory: str | os.PathLike[str],
    reference_file: str | os.PathLike[str],
) -> Path:
    """Return the downloaded response matching one HMI FITS acquisition."""
    directory = Path(directory).resolve()
    acquisition = read_hmi_fits_acquisition(reference_file)
    manifest = load_hmi_transmission_manifest(directory)
    try:
        profile_key = manifest['acquisitions'][acquisition['acquisition_key']]['profile']
        profile_name = manifest['profiles'][profile_key]['file']
    except KeyError as error:
        raise KeyError(
            f"No downloaded HMI transmission profile for {acquisition['acquisition_key']}. "
            f"Run pme.data.hmi_transmission over the folder containing {reference_file}."
        ) from error
    return directory / profile_name


def load_hmi_transmission_profile(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Load and validate one spatial HMI response map for dataset-side sampling."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HMI transmission profile not found: {path}")
    with np.load(path, allow_pickle=False) as profile_data:
        required = ('offsets', 'weights', 'continuum_weights')
        missing = [key for key in required if key not in profile_data]
        if missing:
            raise ValueError(
                f"HMI transmission profile {path} is missing arrays: {missing}. "
                "Regenerate it with the current downloader."
            )
        offsets = np.array(profile_data['offsets'], dtype=np.float32, copy=True)
        weights = np.array(profile_data['weights'], dtype=np.float32, copy=True)
        continuum_weights = np.array(profile_data['continuum_weights'], dtype=np.float32, copy=True)

    if offsets.ndim != 2 or offsets.shape[0] != 6 or offsets.shape[1] < 1:
        raise ValueError(f"Invalid HMI transmission offsets in {path}: {offsets.shape}.")
    # A spatially constant response is represented as a one-pixel response map.
    if weights.ndim == 2:
        weights = weights[None, None]
        continuum_weights = continuum_weights[None, None]
    if weights.ndim != 4 or weights.shape[-2:] != offsets.shape:
        raise ValueError(
            f"HMI transmission weights in {path} must have shape [y,x,6,sample]; got {weights.shape}."
        )
    if continuum_weights.shape != weights.shape[:-1]:
        raise ValueError(
            f"HMI continuum weights in {path} must have shape {weights.shape[:-1]}; "
            f"got {continuum_weights.shape}."
        )
    if (not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(continuum_weights))):
        raise ValueError(f"HMI transmission profile {path} contains non-finite values.")
    if np.any(weights < 0) or np.any(continuum_weights < 0):
        raise ValueError(f"HMI transmission profile {path} contains negative weights.")
    total_weight = weights.sum(axis=-1) + continuum_weights
    if not np.allclose(total_weight, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError(f"HMI transmission weights in {path} do not sum to one.")

    return {
        'offsets': offsets,
        'weights': weights,
        'continuum_weights': continuum_weights,
    }


def _single_row(frame, description: str):
    if frame is None or len(frame) != 1:
        count = 0 if frame is None else len(frame)
        raise RuntimeError(f"Expected one {description} record, found {count}.")
    return frame.iloc[0]


def resolve_hmi_phase_map(
    client: drms.Client,
    observation_time: str | datetime,
    hcamid: int,
    phase_map_fsn: Optional[int] = None,
) -> dict[str, Any]:
    """Resolve the exact phase-map record used for an HMI Stokes record."""
    jsoc_time = format_jsoc_time(observation_time)

    if phase_map_fsn is None:
        phase_map_fsn, hcamid = resolve_hmi_phase_map_identity(client, jsoc_time, hcamid=hcamid)
    else:
        phase_map_fsn, hcamid = int(phase_map_fsn), int(hcamid)

    phase_query = f"{HMI_PHASE_SERIES}[{int(phase_map_fsn)}]"
    phase_keys, phase_segments = client.query(phase_query, key=PHASE_QUERY_KEYS, seg="phases")
    if phase_keys is None or phase_segments is None:
        raise RuntimeError(f"No phase-map data returned for {phase_query}.")

    camera_mask = phase_keys["HCAMID"].astype(int) == int(hcamid)
    camera_keys = phase_keys.loc[camera_mask]
    camera_segments = phase_segments.loc[camera_mask]
    key_row = _single_row(camera_keys, f"phase map {phase_map_fsn}, HCAMID={hcamid}")
    segment_row = _single_row(camera_segments, f"phase-map segment {phase_map_fsn}, HCAMID={hcamid}")

    metadata = {key: key_row[key] for key in key_row.index}
    metadata.update({
        "observation_time": jsoc_time,
        "phase_map_fsn": int(phase_map_fsn),
        "HCAMID": int(hcamid),
        "NX": int(key_row["NX"]),
        "segment": str(segment_row["phases"]),
    })
    metadata["record"] = (
        f"{HMI_PHASE_SERIES}[{int(phase_map_fsn)}][][{int(hcamid)}][{metadata['NX']}]"
    )
    return metadata


def resolve_hmi_phase_map_identity(
    client: drms.Client,
    observation_time: str | datetime,
    hcamid: int,
) -> tuple[int, int]:
    """Resolve only the phase-map and camera IDs, avoiding profile downloads."""
    jsoc_time = format_jsoc_time(observation_time)

    me_record = f"{HMI_ME_SERIES}[{jsoc_time}]"
    me_keys = client.query(me_record, key="T_REC,INVPHMAP")
    me_row = _single_row(me_keys, f"{HMI_ME_SERIES} at {jsoc_time}")
    try:
        phase_map_fsn = int(me_row["INVPHMAP"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid INVPHMAP in {me_record}: {me_row.get('INVPHMAP')!r}") from error
    return int(phase_map_fsn), int(hcamid)


def download_phase_map(client: drms.Client, record: str, directory: str | os.PathLike[str]) -> Path:
    """Download the ``phases`` segment through the DRMS export interface."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    request = client.export(f"{record}{{phases}}", method="url_quick", protocol="as-is")
    result = request.download(str(directory), fname_from_rec=False)
    if result is None or len(result) != 1:
        count = 0 if result is None else len(result)
        raise RuntimeError(f"Expected one downloaded phase-map FITS file, found {count}.")
    downloaded = result.iloc[0]["download"]
    if downloaded is None or not Path(downloaded).is_file():
        raise RuntimeError(f"DRMS did not download the phase-map segment for {record}.")
    return Path(downloaded)


def _float_metadata(metadata: Mapping[str, Any], key: str) -> float:
    try:
        value = float(metadata[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid phase-map keyword {key!r}.") from error
    if not np.isfinite(value):
        raise ValueError(f"Phase-map keyword {key!r} is not finite.")
    return value


def build_hmi_transmission_profiles(
    phases_deg: np.ndarray,
    metadata: Mapping[str, Any],
    samples: int = DEFAULT_PROFILE_SAMPLES,
    half_width: float = DEFAULT_PROFILE_HALF_WIDTH,
    blocker_fwhm: float = DEFAULT_BLOCKER_FWHM,
    blocker_tail_sigma: float = DEFAULT_BLOCKER_TAIL_SIGMA,
    continuum_panel_width: float = DEFAULT_CONTINUUM_PANEL_WIDTH,
    continuum_panel_nodes: int = DEFAULT_CONTINUUM_PANEL_NODES,
    contrasts: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Construct six quadrature-ready HMI transmission profiles.

    Returned offsets are relative to the nominal six HMI wavelength positions
    and are expressed in Angstrom. All filters are integrated over the same
    global interval around the Fe I line center. ``weights`` include
    Gauss--Legendre quadrature weights for that explicitly synthesized region;
    ``continuum_weights`` contain all remaining modeled passband throughput.
    The outer integral is bounded by the Gaussian blocking-filter tail and is
    evaluated with chunked composite Gauss--Legendre quadrature.
    """
    phases_deg = np.asarray(phases_deg, dtype=np.float64)
    if phases_deg.ndim < 1 or phases_deg.shape[-1] != 3 or not np.all(np.isfinite(phases_deg)):
        raise ValueError("phases_deg must end in three finite values for NB, WB, and E1.")
    if samples < 9:
        raise ValueError("samples must be at least 9.")
    if half_width <= 0 or blocker_fwhm <= 0:
        raise ValueError("half_width and blocker_fwhm must be positive.")
    if blocker_tail_sigma < 3:
        raise ValueError("blocker_tail_sigma must be at least 3.")
    if continuum_panel_width <= 0 or continuum_panel_nodes < 2:
        raise ValueError("continuum_panel_width must be positive and continuum_panel_nodes at least 2.")

    fsr = np.array([_float_metadata(metadata, key) for key in FSR_KEYS], dtype=np.float64)
    if np.any(fsr <= 0):
        raise ValueError("All HMI free spectral ranges must be positive.")
    hcm = np.array([_float_metadata(metadata, key) for key in HCM_KEYS], dtype=np.float64)
    contrasts = HMI_NOMINAL_CONTRASTS.copy() if contrasts is None else np.asarray(contrasts, dtype=np.float64)
    if contrasts.shape != (7,) or np.any((contrasts < 0) | (contrasts > 1)):
        raise ValueError("contrasts must contain seven values between zero and one.")

    # Gauss--Legendre nodes provide accurate passband integration with far
    # fewer monochromatic synthesis points than a dense uniform wavelength grid.
    nodes, quadrature = np.polynomial.legendre.leggauss(int(samples))
    line_wavelength_1d = nodes * float(half_width)
    quadrature = quadrature * float(half_width)

    nominal = HMI_WAVELENGTH_GRID.to_value("Angstrom").astype(np.float64)
    wavelength = np.broadcast_to(line_wavelength_1d, (nominal.size, samples)).copy()
    offsets = wavelength - nominal[:, None]

    # Segment I0 is the redmost sample and I5 is the bluemost sample. Derive
    # the cotune index from wavelength rather than array position so the motor
    # tuning and data-segment convention cannot silently diverge.
    cotune_index = nominal / 0.0688

    # Absolute optical phase at the target HCM positions.  Phase-cube values
    # are Phi in degrees; 4*phi contributes 6 degrees per motor step.
    target_phase_deg = phases_deg + HCM_ORIENTATION * HCM_OPTICAL_DEGREES_PER_STEP * hcm

    blocker_center = _float_metadata(metadata, "CBLOCKER")

    def evaluate_transmission(evaluation_wavelength):
        result = np.ones((*phases_deg.shape[:-1], *evaluation_wavelength.shape), dtype=np.float64)
        for element in range(3):
            sample_phase_deg = (
                target_phase_deg[..., element, None]
                - HCM_OPTICAL_DEGREES_PER_STEP * HCM_STEPS_PER_SAMPLE[element] * cotune_index
            )
            argument = (2 * np.pi * evaluation_wavelength / fsr[element]
                        + np.deg2rad(sample_phase_deg[..., None]))
            result *= 0.5 * (1 + contrasts[element] * np.cos(argument))
        for element in range(3, 7):
            argument = 2 * np.pi * (evaluation_wavelength - HMI_UNTUNED_CENTER) / fsr[element]
            result *= 0.5 * (1 + contrasts[element] * np.cos(argument))
        result *= np.exp(
            -4 * np.log(2) * ((evaluation_wavelength - blocker_center) / blocker_fwhm) ** 2
        )
        return np.clip(result, 0, None)

    transmission = evaluate_transmission(wavelength)
    inner_integral_weights = transmission * quadrature

    inner_integral = inner_integral_weights.sum(axis=-1)

    # Integrate the complete modeled throughput outside the line region. The
    # blocker supplies a finite tail bound, while composite panels resolve the
    # narrow periodic orders. Panel chunks keep 128x128 phase maps memory-safe.
    blocker_sigma = blocker_fwhm / (2 * np.sqrt(2 * np.log(2)))
    lower_bound = min(-half_width, blocker_center - blocker_tail_sigma * blocker_sigma)
    upper_bound = max(half_width, blocker_center + blocker_tail_sigma * blocker_sigma)
    panel_nodes, panel_weights = np.polynomial.legendre.leggauss(int(continuum_panel_nodes))

    def integrate_interval(start, stop):
        integral = np.zeros((*phases_deg.shape[:-1], nominal.size), dtype=np.float64)
        if stop <= start:
            return integral
        n_panels = int(np.ceil((stop - start) / continuum_panel_width))
        edges = np.linspace(start, stop, n_panels + 1, dtype=np.float64)
        for panel_start in range(0, n_panels, CONTINUUM_CHUNK_PANELS):
            panel_stop = min(panel_start + CONTINUUM_CHUNK_PANELS, n_panels)
            left = edges[panel_start:panel_stop]
            right = edges[panel_start + 1:panel_stop + 1]
            centers = 0.5 * (left + right)
            panel_half_widths = 0.5 * (right - left)
            chunk_wavelength = (
                centers[:, None] + panel_half_widths[:, None] * panel_nodes
            ).reshape(-1)
            chunk_weights = (panel_half_widths[:, None] * panel_weights).reshape(-1)
            evaluation_wavelength = np.broadcast_to(
                chunk_wavelength, (nominal.size, chunk_wavelength.size)
            )
            integral += (evaluate_transmission(evaluation_wavelength) * chunk_weights).sum(axis=-1)
        return integral

    continuum_integral = (
        integrate_interval(lower_bound, -half_width)
        + integrate_interval(half_width, upper_bound)
    )
    full_integral = inner_integral + continuum_integral

    if not np.all(np.isfinite(inner_integral_weights)) or np.any(full_integral <= 0):
        raise RuntimeError("Constructed HMI transmission profiles are invalid.")
    weights = inner_integral_weights / full_integral[..., None]
    continuum_weights = continuum_integral / full_integral

    return {
        "offsets": offsets.astype(np.float32),
        "weights": weights.astype(np.float32),
        "continuum_weights": continuum_weights.astype(np.float32),
        "transmission": transmission.astype(np.float32),
        "quadrature_weights": quadrature.astype(np.float32),
        "wavelength_grid": nominal.astype(np.float32),
        "phases_deg": phases_deg.astype(np.float32),
        "target_phase_deg": target_phase_deg.astype(np.float32),
        "fsr": fsr.astype(np.float32),
        "contrasts": contrasts.astype(np.float32),
        "continuum_bounds": np.array([lower_bound, upper_bound], dtype=np.float32),
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_transmission_file(
    output: str | os.PathLike[str],
    profiles: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Transmission file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized_metadata = json.dumps(
        {key: _json_compatible(value) for key, value in metadata.items()},
        sort_keys=True,
    )
    np.savez_compressed(output, **profiles, metadata=np.array(serialized_metadata))
    return output


def _write_hmi_transmission_profile(
    archive_metadata: Mapping[str, Any],
    output: str | os.PathLike[str],
    client: drms.Client,
    samples: int = DEFAULT_PROFILE_SAMPLES,
    half_width: float = DEFAULT_PROFILE_HALF_WIDTH,
    blocker_fwhm: float = DEFAULT_BLOCKER_FWHM,
    blocker_tail_sigma: float = DEFAULT_BLOCKER_TAIL_SIGMA,
    continuum_panel_width: float = DEFAULT_CONTINUUM_PANEL_WIDTH,
    continuum_panel_nodes: int = DEFAULT_CONTINUUM_PANEL_NODES,
    overwrite: bool = False,
) -> Path:
    """Download one resolved phase map and write its spatial profile field."""
    archive_metadata = dict(archive_metadata)

    with tempfile.TemporaryDirectory(prefix="pme_hmi_phase_") as temporary_directory:
        phase_path = download_phase_map(client, archive_metadata["record"], temporary_directory)
        phase_cube = fits.getdata(phase_path)
        phase_cube = np.asarray(phase_cube, dtype=np.float64)
        if phase_cube.ndim != 3:
            raise ValueError(f"Expected a three-dimensional HMI phase map, got {phase_cube.shape}.")
        channel_axes = [axis for axis, size in enumerate(phase_cube.shape) if 3 <= size <= 8]
        if len(channel_axes) != 1:
            raise ValueError(f"HMI phase map does not contain NB, WB, and E1 planes: {phase_cube.shape}.")
        phase_cube = np.moveaxis(phase_cube, channel_axes[0], -1)
        phases_deg = phase_cube[..., :3]
        if not np.all(np.isfinite(phases_deg)):
            raise ValueError("HMI phase map contains non-finite tunable-element phases.")
        digest = hashlib.sha256(phase_path.read_bytes()).hexdigest()

    profiles = build_hmi_transmission_profiles(
        phases_deg,
        archive_metadata,
        samples=samples,
        half_width=half_width,
        blocker_fwhm=blocker_fwhm,
        blocker_tail_sigma=blocker_tail_sigma,
        continuum_panel_width=continuum_panel_width,
        continuum_panel_nodes=continuum_panel_nodes,
    )
    output_metadata = dict(archive_metadata)
    output_metadata.update({
        "format": "pme.hmi_transmission.v5",
        "model": (
            "HMI element-product model with record-specific tunable phases and "
            "VFISV-style far-wing continuum integration"
        ),
        "wavelength_unit": "Angstrom",
        "phase_channels": list(PHASE_CHANNELS),
        "ccd_size": HMI_CCD_SIZE,
        "phase_map_shape": list(phases_deg.shape[:-1]),
        "samples": int(samples),
        "half_width": float(half_width),
        "blocker_fwhm": float(blocker_fwhm),
        "blocker_tail_sigma": float(blocker_tail_sigma),
        "continuum_bounds": profiles["continuum_bounds"].tolist(),
        "continuum_panel_width": float(continuum_panel_width),
        "continuum_panel_nodes": int(continuum_panel_nodes),
        "untuned_stack_center": HMI_UNTUNED_CENTER,
        "element_contrasts": HMI_NOMINAL_CONTRASTS.tolist(),
        "phase_fits_sha256": digest,
        "assumptions": (
            "Mean ground-calibration element contrasts, untuned-stack center, and Gaussian 8.43 A blocking filter; "
            "individual fixed-element phases and measured untuned-stack shape omitted; "
            "front-window fine structure and I-ripple omitted."
        ),
    })
    return write_transmission_file(output, profiles, output_metadata, overwrite=overwrite)


def prepare_hmi_transmission_directory(
    inputs: Iterable[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    email: str,
    samples: int = DEFAULT_PROFILE_SAMPLES,
    half_width: float = DEFAULT_PROFILE_HALF_WIDTH,
    blocker_fwhm: float = DEFAULT_BLOCKER_FWHM,
    blocker_tail_sigma: float = DEFAULT_BLOCKER_TAIL_SIGMA,
    continuum_panel_width: float = DEFAULT_CONTINUUM_PANEL_WIDTH,
    continuum_panel_nodes: int = DEFAULT_CONTINUUM_PANEL_NODES,
    overwrite: bool = False,
    client: Optional[drms.Client] = None,
) -> Path:
    """Prepare every unique calibration used by a FITS series and a manifest."""
    if not email:
        raise ValueError("A registered JSOC email address is required for DRMS export.")
    client = drms.Client(email=email) if client is None else client
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    acquisitions = discover_hmi_acquisitions(inputs)

    identities = {}
    for acquisition in acquisitions:
        identity = resolve_hmi_phase_map_identity(
            client,
            acquisition['observation_time'],
            hcamid=acquisition['hcamid'],
        )
        identities[acquisition['acquisition_key']] = identity

    profiles = {}
    for phase_map_fsn, hcamid in sorted(set(identities.values())):
        profile_key = f"INVPHMAP={phase_map_fsn}|HCAMID={hcamid}"
        profile_name = f"hmi_transmission_INVPHMAP{phase_map_fsn}_HCAMID{hcamid}.npz"
        profile_path = output_directory / profile_name
        representative_time = next(
            acquisition['observation_time'] for acquisition in acquisitions
            if identities[acquisition['acquisition_key']] == (phase_map_fsn, hcamid)
        )
        archive_metadata = resolve_hmi_phase_map(
            client,
            representative_time,
            hcamid=hcamid,
            phase_map_fsn=phase_map_fsn,
        )
        if overwrite or not profile_path.is_file():
            _write_hmi_transmission_profile(
                archive_metadata=archive_metadata,
                output=profile_path,
                client=client,
                samples=samples,
                half_width=half_width,
                blocker_fwhm=blocker_fwhm,
                blocker_tail_sigma=blocker_tail_sigma,
                continuum_panel_width=continuum_panel_width,
                continuum_panel_nodes=continuum_panel_nodes,
                overwrite=overwrite,
            )
        profiles[profile_key] = {
            'file': profile_name,
            'phase_map_fsn': phase_map_fsn,
            'hcamid': hcamid,
            'record': archive_metadata['record'],
        }

    acquisition_entries = {}
    for acquisition in acquisitions:
        identity = identities[acquisition['acquisition_key']]
        profile_key = f"INVPHMAP={identity[0]}|HCAMID={identity[1]}"
        acquisition_entries[acquisition['acquisition_key']] = {
            'observation_time': acquisition['observation_time'],
            'hcamid': acquisition['hcamid'],
            'profile': profile_key,
        }

    manifest = {
        'format': MANIFEST_FORMAT,
        'profiles': profiles,
        'acquisitions': acquisition_entries,
    }
    manifest_path = output_directory / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        existing = load_hmi_transmission_manifest(output_directory)
        existing['profiles'].update(manifest['profiles'])
        existing['acquisitions'].update(manifest['acquisitions'])
        manifest = existing
    manifest_path.write_text(json.dumps(_json_compatible(manifest), indent=2, sort_keys=True) + '\n')
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an HMI phase map with DRMS and prepare offline transmission profiles."
    )
    parser.add_argument("--input", nargs='+', required=True, help="HMI FITS files, directories, or glob patterns")
    parser.add_argument("--output", required=True, help="Output calibration directory")
    parser.add_argument("--email", default=os.getenv("JSOC_EMAIL"), help="Registered JSOC email (or JSOC_EMAIL)")
    parser.add_argument("--samples", type=int, default=DEFAULT_PROFILE_SAMPLES)
    parser.add_argument("--half-width", type=float, default=DEFAULT_PROFILE_HALF_WIDTH)
    parser.add_argument("--blocker-fwhm", type=float, default=DEFAULT_BLOCKER_FWHM)
    parser.add_argument("--blocker-tail-sigma", type=float, default=DEFAULT_BLOCKER_TAIL_SIGMA)
    parser.add_argument("--continuum-panel-width", type=float, default=DEFAULT_CONTINUUM_PANEL_WIDTH)
    parser.add_argument("--continuum-panel-nodes", type=int, default=DEFAULT_CONTINUUM_PANEL_NODES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.email:
        parser.error("--email or JSOC_EMAIL is required for the DRMS export")

    output = prepare_hmi_transmission_directory(
        inputs=args.input,
        output_directory=args.output,
        email=args.email,
        samples=args.samples,
        half_width=args.half_width,
        blocker_fwhm=args.blocker_fwhm,
        blocker_tail_sigma=args.blocker_tail_sigma,
        continuum_panel_width=args.continuum_panel_width,
        continuum_panel_nodes=args.continuum_panel_nodes,
        overwrite=args.overwrite,
    )
    print(f"Wrote HMI transmission manifest to {output}")


if __name__ == "__main__":
    main()
