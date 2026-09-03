"""Optional network preparation of offline HMI response archives.

Importing this module never imports DRMS. The dependency is loaded only when
the preparation function must construct its own network client.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from astropy.io import fits

from prom3theus.core import sha256_file

from .acquisition import (
    _resolve_input_paths,
    discover_hmi_acquisitions,
    format_jsoc_time,
    parse_hmi_tai_time,
)
from .constants import HMI_CCD_SIZE, HMI_WAVELENGTH_GRID
from .response import (
    DEFAULT_PROFILE_HALF_WIDTH,
    MANIFEST_FORMAT,
    MANIFEST_NAME,
    load_response_manifest,
)

if TYPE_CHECKING:
    import drms

HMI_PHASE_SERIES = "hmi.phasemaps_extended"
PHASE_CHANNELS = ("NB", "WB", "E1")
FSR_KEYS = ("FSRNB", "FSRWB", "FSRE1", "FSRE2", "FSRE3", "FSRE4", "FSRE5")
HCM_KEYS = ("HCMNB", "HCMWB", "HCME1")
HCM_OPTICAL_DEGREES_PER_STEP = 6.0
HCM_STEPS_PER_SAMPLE = np.array([24.0, 12.0, 6.0], dtype=np.float64)
HCM_ORIENTATION = np.array([1.0, 1.0, -1.0], dtype=np.float64)
HMI_NOMINAL_CONTRASTS = np.array(
    [0.969, 0.997, 0.962, 0.985, 0.965, 0.988, 1.000], dtype=np.float64
)
DEFAULT_PROFILE_SAMPLES = 81
DEFAULT_BLOCKER_TAIL_SIGMA = 7.0
DEFAULT_CONTINUUM_PANEL_WIDTH = 0.1
DEFAULT_CONTINUUM_PANEL_NODES = 8
CONTINUUM_CHUNK_PANELS = 8
DEFAULT_BLOCKER_FWHM = 8.43
HMI_UNTUNED_CENTER = 0.016
PHASE_QUERY_KEYS = (
    "T_REC,FSN_REC,HCAMID,NX,T_START,T_STOP,HCMNB,HCMWB,HCMPOL,HCME1,"
    "FSRNB,FSRWB,FSRE1,FSRE2,FSRE3,FSRE4,FSRE5,CBLOCKER,"
    "PHASENBM,PHASEWBM,PHASELYO,COMMENT"
)


def _single_row(frame, description: str):
    if frame is None or len(frame) != 1:
        count = 0 if frame is None else len(frame)
        raise RuntimeError(f"Expected one {description} record, found {count}.")
    return frame.iloc[0]


def _validate_phase_map_time(metadata: Mapping[str, Any]) -> None:
    """Validate the phase map's calibration-sequence timestamps.

    ``T_START`` and ``T_STOP`` delimit the short detune sequence used to
    construct the phase map.  They are provenance, not an observation-time
    validity interval for the resulting calibration record.
    """
    try:
        start = parse_hmi_tai_time(str(metadata["T_START"]))
        record = parse_hmi_tai_time(str(metadata["T_REC"]))
        stop = parse_hmi_tai_time(str(metadata["T_STOP"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "HMI phase-map metadata require explicit TAI T_START/T_REC/T_STOP "
            "detune-sequence timestamps."
        ) from error
    if not start <= record <= stop or stop <= start:
        raise ValueError(
            "HMI phase-map detune timestamps must satisfy "
            "T_START <= T_REC <= T_STOP with T_START < T_STOP."
        )


def resolve_hmi_phase_map(
    client: drms.Client,
    observation_time: str | datetime,
    hcamid: int,
    phase_map_fsn: int,
) -> dict[str, Any]:
    """Resolve the exact phase-map record used for an HMI Stokes record."""
    jsoc_time = format_jsoc_time(observation_time)
    if type(phase_map_fsn) is not int or phase_map_fsn < 0:
        raise ValueError("phase_map_fsn must be a non-negative integer.")
    if type(hcamid) is not int or hcamid not in {2, 3}:
        raise ValueError("hcamid must be integer camera 2 or 3.")

    phase_query = f"{HMI_PHASE_SERIES}[{int(phase_map_fsn)}]"
    phase_keys, phase_segments = client.query(
        phase_query, key=PHASE_QUERY_KEYS, seg="phases"
    )
    if phase_keys is None or phase_segments is None:
        raise RuntimeError(f"No phase-map data returned for {phase_query}.")

    camera_mask = phase_keys["HCAMID"].astype(int) == int(hcamid)
    camera_keys = phase_keys.loc[camera_mask]
    camera_segments = phase_segments.loc[camera_mask]
    key_row = _single_row(camera_keys, f"phase map {phase_map_fsn}, HCAMID={hcamid}")
    segment_row = _single_row(
        camera_segments, f"phase-map segment {phase_map_fsn}, HCAMID={hcamid}"
    )

    metadata = {key: key_row[key] for key in key_row.index}
    metadata.update(
        {
            "observation_time": jsoc_time,
            "phase_map_fsn": int(phase_map_fsn),
            "HCAMID": int(hcamid),
            "NX": int(key_row["NX"]),
            "segment": str(segment_row["phases"]),
        }
    )
    metadata["record"] = (
        f"{HMI_PHASE_SERIES}[{int(phase_map_fsn)}][][{int(hcamid)}][{metadata['NX']}]"
    )
    _validate_phase_map_time(metadata)
    return metadata


def download_phase_map(
    client: drms.Client, record: str, directory: str | os.PathLike[str]
) -> Path:
    """Download the ``phases`` segment through the DRMS export interface."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    request = client.export(f"{record}{{phases}}", method="url_quick", protocol="as-is")
    result = request.download(str(directory), fname_from_rec=False)
    if result is None or len(result) != 1:
        count = 0 if result is None else len(result)
        raise RuntimeError(
            f"Expected one downloaded phase-map FITS file, found {count}."
        )
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
    if (
        phases_deg.ndim < 1
        or phases_deg.shape[-1] != 3
        or not np.all(np.isfinite(phases_deg))
    ):
        raise ValueError(
            "phases_deg must end in three finite values for NB, WB, and E1."
        )
    if samples < 9:
        raise ValueError("samples must be at least 9.")
    if half_width <= 0 or blocker_fwhm <= 0:
        raise ValueError("half_width and blocker_fwhm must be positive.")
    if blocker_tail_sigma < 3:
        raise ValueError("blocker_tail_sigma must be at least 3.")
    if continuum_panel_width <= 0 or continuum_panel_nodes < 2:
        raise ValueError(
            "continuum_panel_width must be positive and continuum_panel_nodes at least 2."
        )

    fsr = np.array(
        [_float_metadata(metadata, key) for key in FSR_KEYS], dtype=np.float64
    )
    if np.any(fsr <= 0):
        raise ValueError("All HMI free spectral ranges must be positive.")
    hcm = np.array(
        [_float_metadata(metadata, key) for key in HCM_KEYS], dtype=np.float64
    )
    contrasts = (
        HMI_NOMINAL_CONTRASTS.copy()
        if contrasts is None
        else np.asarray(contrasts, dtype=np.float64)
    )
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
        result = np.ones(
            (*phases_deg.shape[:-1], *evaluation_wavelength.shape), dtype=np.float64
        )
        for element in range(3):
            sample_phase_deg = (
                target_phase_deg[..., element, None]
                - HCM_OPTICAL_DEGREES_PER_STEP
                * HCM_STEPS_PER_SAMPLE[element]
                * cotune_index
            )
            argument = 2 * np.pi * evaluation_wavelength / fsr[element] + np.deg2rad(
                sample_phase_deg[..., None]
            )
            result *= 0.5 * (1 + contrasts[element] * np.cos(argument))
        for element in range(3, 7):
            argument = (
                2 * np.pi * (evaluation_wavelength - HMI_UNTUNED_CENTER) / fsr[element]
            )
            result *= 0.5 * (1 + contrasts[element] * np.cos(argument))
        result *= np.exp(
            -4
            * np.log(2)
            * ((evaluation_wavelength - blocker_center) / blocker_fwhm) ** 2
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
    panel_nodes, panel_weights = np.polynomial.legendre.leggauss(
        int(continuum_panel_nodes)
    )

    def integrate_interval(start, stop):
        integral = np.zeros((*phases_deg.shape[:-1], nominal.size), dtype=np.float64)
        if stop <= start:
            return integral
        n_panels = int(np.ceil((stop - start) / continuum_panel_width))
        edges = np.linspace(start, stop, n_panels + 1, dtype=np.float64)
        for panel_start in range(0, n_panels, CONTINUUM_CHUNK_PANELS):
            panel_stop = min(panel_start + CONTINUUM_CHUNK_PANELS, n_panels)
            left = edges[panel_start:panel_stop]
            right = edges[panel_start + 1 : panel_stop + 1]
            centers = 0.5 * (left + right)
            panel_half_widths = 0.5 * (right - left)
            chunk_wavelength = (
                centers[:, None] + panel_half_widths[:, None] * panel_nodes
            ).reshape(-1)
            chunk_weights = (panel_half_widths[:, None] * panel_weights).reshape(-1)
            evaluation_wavelength = np.broadcast_to(
                chunk_wavelength, (nominal.size, chunk_wavelength.size)
            )
            integral += (
                evaluate_transmission(evaluation_wavelength) * chunk_weights
            ).sum(axis=-1)
        return integral

    continuum_integral = integrate_interval(
        lower_bound, -half_width
    ) + integrate_interval(half_width, upper_bound)
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
        return _json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("HMI metadata mappings require string keys.")
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("HMI metadata floating-point values must be finite.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported HMI metadata value type: {type(value).__name__}.")


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
        _json_compatible(metadata), sort_keys=True, allow_nan=False
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

    with tempfile.TemporaryDirectory(
        prefix="prom3theus_hmi_phase_"
    ) as temporary_directory:
        phase_path = download_phase_map(
            client, archive_metadata["record"], temporary_directory
        )
        phase_cube = fits.getdata(phase_path)
        phase_cube = np.asarray(phase_cube, dtype=np.float64)
        if phase_cube.ndim != 3:
            raise ValueError(
                f"Expected a three-dimensional HMI phase map, got {phase_cube.shape}."
            )
        channel_axes = [
            axis for axis, size in enumerate(phase_cube.shape) if 3 <= size <= 8
        ]
        if len(channel_axes) != 1:
            raise ValueError(
                f"HMI phase map does not contain NB, WB, and E1 planes: {phase_cube.shape}."
            )
        phase_cube = np.moveaxis(phase_cube, channel_axes[0], -1)
        phases_deg = phase_cube[..., :3]
        if not np.all(np.isfinite(phases_deg)):
            raise ValueError(
                "HMI phase map contains non-finite tunable-element phases."
            )
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
    output_metadata.update(
        {
            "format": "prom3theus.hmi_response.v1",
            "model": (
                "HMI element-product model with record-specific tunable phases and "
                "far-wing continuum integration"
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
        }
    )
    # Runtime needs only the common quadrature offsets and the spatial inner/
    # outer response weights.  The phase cubes, reconstructed transmission,
    # and intermediate quadrature arrays are preparation diagnostics already
    # summarized by the provenance metadata; storing them duplicated the bulk
    # of every response archive without serving inversion or evaluation.
    runtime_profiles = {
        key: profiles[key] for key in ("offsets", "weights", "continuum_weights")
    }
    return write_transmission_file(
        output,
        runtime_profiles,
        output_metadata,
        overwrite=overwrite,
    )


def _publish_response_directory(
    staged: Path,
    output: Path,
    *,
    overwrite: bool,
) -> None:
    """Publish a staged directory while retaining any unrestorable backup."""

    if not output.exists():
        os.replace(staged, output)
        return
    if not overwrite:
        raise FileExistsError(
            f"HMI response output appeared during preparation: {output}."
        )

    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.response-backup-",
            dir=output.parent,
        )
    )
    backup.rmdir()
    os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException as publish_error:
        if output.exists():
            raise RuntimeError(
                "Failed to publish the HMI response directory; the previous "
                f"directory was retained at {backup}."
            ) from publish_error
        try:
            os.replace(backup, output)
        except BaseException as restore_error:
            raise RuntimeError(
                "Failed to publish or restore the HMI response directory; the "
                f"previous directory was retained at {backup}."
            ) from restore_error
        raise
    shutil.rmtree(backup)


def prepare_hmi_response_directory(
    inputs: Iterable[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    email: str,
    phase_map_fsn: int,
    samples: int = DEFAULT_PROFILE_SAMPLES,
    half_width: float = DEFAULT_PROFILE_HALF_WIDTH,
    blocker_fwhm: float = DEFAULT_BLOCKER_FWHM,
    blocker_tail_sigma: float = DEFAULT_BLOCKER_TAIL_SIGMA,
    continuum_panel_width: float = DEFAULT_CONTINUUM_PANEL_WIDTH,
    continuum_panel_nodes: int = DEFAULT_CONTINUUM_PANEL_NODES,
    overwrite: bool = False,
    client: Optional[drms.Client] = None,
) -> Path:
    """Atomically prepare one exact response directory for the selected inputs."""
    if not email:
        raise ValueError("A registered JSOC email address is required for DRMS export.")
    if type(phase_map_fsn) is not int or phase_map_fsn < 0:
        raise ValueError("phase_map_fsn must be a non-negative integer.")
    output_directory = Path(output_directory).expanduser().resolve(strict=False)
    if os.path.lexists(output_directory) and not output_directory.is_dir():
        raise NotADirectoryError(
            f"HMI response output is not a directory: {output_directory}."
        )
    if output_directory.exists() and not overwrite:
        raise FileExistsError(
            f"HMI response output already exists: {output_directory}. "
            "Pass overwrite=True to replace the complete prepared directory."
        )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    input_selection = (
        inputs if isinstance(inputs, (str, os.PathLike)) else tuple(inputs)
    )
    source_paths = {
        path.resolve()
        for path in _resolve_input_paths(input_selection)
        if path.suffix.lower() == ".fits"
    }
    acquisitions = discover_hmi_acquisitions(input_selection)
    source_paths.update(
        Path(acquisition["path"]).resolve()
        for acquisition in acquisitions
        if "path" in acquisition
    )
    contained_sources = sorted(
        path for path in source_paths if path.is_relative_to(output_directory)
    )
    if contained_sources:
        raise ValueError(
            "HMI response output must not equal or contain source FITS files, "
            "even with overwrite=True: "
            + ", ".join(str(path) for path in contained_sources)
        )
    if client is None:
        import drms

        client = drms.Client(email=email)

    identities = {
        acquisition["acquisition_key"]: (phase_map_fsn, int(acquisition["hcamid"]))
        for acquisition in acquisitions
    }

    with tempfile.TemporaryDirectory(
        prefix=f".{output_directory.name}.prepare-",
        dir=output_directory.parent,
    ) as workspace_name:
        workspace = Path(workspace_name)
        staged_directory = workspace / "responses"
        staged_directory.mkdir()
        profiles = {}
        for phase_map_fsn, hcamid in sorted(set(identities.values())):
            profile_key = f"INVPHMAP={phase_map_fsn}|HCAMID={hcamid}"
            profile_name = (
                f"hmi_transmission_INVPHMAP{phase_map_fsn}_HCAMID{hcamid}.npz"
            )
            profile_path = staged_directory / profile_name
            matching_acquisitions = [
                acquisition
                for acquisition in acquisitions
                if identities[acquisition["acquisition_key"]] == (phase_map_fsn, hcamid)
            ]
            representative_time = matching_acquisitions[0]["observation_time"]
            archive_metadata = resolve_hmi_phase_map(
                client,
                representative_time,
                hcamid=hcamid,
                phase_map_fsn=phase_map_fsn,
            )
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
                overwrite=False,
            )
            profiles[profile_key] = {
                "file": profile_name,
                "phase_map_fsn": phase_map_fsn,
                "hcamid": hcamid,
                "record": archive_metadata["record"],
                "sha256": sha256_file(profile_path),
            }

        acquisition_entries = {}
        for acquisition in acquisitions:
            identity = identities[acquisition["acquisition_key"]]
            profile_key = f"INVPHMAP={identity[0]}|HCAMID={identity[1]}"
            acquisition_entries[acquisition["acquisition_key"]] = {
                "record_time": acquisition["record_time"],
                "observation_time": acquisition["observation_time"],
                "hcamid": acquisition["hcamid"],
                "profile": profile_key,
            }

        manifest = {
            "format": MANIFEST_FORMAT,
            "profiles": profiles,
            "acquisitions": acquisition_entries,
        }
        staged_manifest = staged_directory / MANIFEST_NAME
        staged_manifest.write_text(
            json.dumps(
                _json_compatible(manifest),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        load_response_manifest(staged_directory)

        _publish_response_directory(
            staged_directory,
            output_directory,
            overwrite=overwrite,
        )
    return output_directory / MANIFEST_NAME


__all__ = [
    "build_hmi_transmission_profiles",
    "download_phase_map",
    "prepare_hmi_response_directory",
    "resolve_hmi_phase_map",
    "write_transmission_file",
]
