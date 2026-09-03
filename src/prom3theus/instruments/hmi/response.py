"""Offline runtime access to prepared HMI transmission archives.

This module intentionally has no DRMS or network import.  Runtime inversion
only resolves acquisition identities and samples immutable local response
arrays; archive construction lives in :mod:`.preparation`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prom3theus.core import sha256_file

from .acquisition import (
    hmi_acquisition_key,
    parse_hmi_tai_time,
    read_acquisition_header,
)


DEFAULT_PROFILE_HALF_WIDTH = 0.65
MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "prom3theus.hmi_response_manifest.v3"


def _read_profile_metadata(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"offsets", "weights", "continuum_weights", "metadata"}
            if set(archive.files) != required:
                raise ValueError(
                    f"HMI response profile {path} must contain exactly {sorted(required)}."
                )
            raw_metadata = archive["metadata"]
            if raw_metadata.shape != ():
                raise ValueError(
                    f"HMI response metadata in {path} must be scalar JSON."
                )
            metadata = json.loads(str(raw_metadata.item()))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"HMI response metadata in {path} is not valid JSON."
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError(f"HMI response metadata in {path} must be an object.")
    return metadata


def _validate_detune_sequence_times(metadata: dict[str, Any], context: str) -> None:
    """Validate phase-map provenance without treating it as a validity window."""

    try:
        start = parse_hmi_tai_time(str(metadata["T_START"]))
        record = parse_hmi_tai_time(str(metadata["T_REC"]))
        stop = parse_hmi_tai_time(str(metadata["T_STOP"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{context} has invalid phase-map detune-sequence timestamps."
        ) from error
    if not start <= record <= stop or stop <= start:
        raise ValueError(
            f"{context} phase-map timestamps must satisfy "
            "T_START <= T_REC <= T_STOP with T_START < T_STOP."
        )


def load_response_manifest(directory: str | os.PathLike[str]) -> dict[str, Any]:
    directory = Path(directory).expanduser().resolve()
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"HMI response manifest not found: {path}.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "format",
        "profiles",
        "acquisitions",
    }:
        raise ValueError(f"Invalid HMI response manifest schema in {path}.")
    if manifest["format"] != MANIFEST_FORMAT:
        raise ValueError(
            f"Unsupported HMI response manifest in {path}; reprepare it with the current framework."
        )
    profiles, acquisitions = manifest["profiles"], manifest["acquisitions"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"HMI response manifest contains no profiles: {path}.")
    if not isinstance(acquisitions, dict) or not acquisitions:
        raise ValueError(f"HMI response manifest contains no acquisitions: {path}.")
    profile_files: set[str] = set()
    for key, profile in profiles.items():
        expected = {"file", "phase_map_fsn", "hcamid", "record", "sha256"}
        if not isinstance(profile, dict) or set(profile) != expected:
            raise ValueError(f"Invalid HMI response profile {key!r} in {path}.")
        if (
            not isinstance(key, str)
            or key != f"INVPHMAP={profile['phase_map_fsn']}|HCAMID={profile['hcamid']}"
            or type(profile["phase_map_fsn"]) is not int
            or profile["phase_map_fsn"] < 0
            or type(profile["hcamid"]) is not int
            or profile["hcamid"] not in {2, 3}
            or not isinstance(profile["record"], str)
            or not profile["record"].strip()
        ):
            raise ValueError(
                f"Invalid HMI response profile identity {key!r} in {path}."
            )
        if (
            not isinstance(profile["file"], str)
            or not profile["file"].endswith(".npz")
            or Path(profile["file"]).name != profile["file"]
            or profile["file"] in profile_files
        ):
            raise ValueError(
                f"Invalid HMI response profile filename {key!r} in {path}."
            )
        profile_files.add(profile["file"])
        candidate = (directory / profile["file"]).resolve()
        if candidate.parent != directory or not candidate.is_file():
            raise FileNotFoundError(
                f"Unsafe or missing HMI response profile: {candidate}."
            )
        expected_digest = profile["sha256"]
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError(f"Invalid HMI response checksum {key!r} in {path}.")
        if sha256_file(candidate) != expected_digest:
            raise ValueError(f"HMI response checksum mismatch for {candidate}.")
        metadata = _read_profile_metadata(candidate)
        if (
            type(metadata.get("phase_map_fsn")) is not int
            or metadata["phase_map_fsn"] != profile["phase_map_fsn"]
            or type(metadata.get("HCAMID")) is not int
            or metadata["HCAMID"] != profile["hcamid"]
            or metadata.get("record") != profile["record"]
        ):
            raise ValueError(
                f"HMI response profile identity {key!r} does not match {candidate}."
            )
        _validate_detune_sequence_times(metadata, f"HMI response profile {key!r}")
    for key, acquisition in acquisitions.items():
        if (
            not isinstance(key, str)
            or not isinstance(acquisition, dict)
            or set(acquisition)
            != {"record_time", "observation_time", "hcamid", "profile"}
            or acquisition["profile"] not in profiles
        ):
            raise ValueError(f"Invalid HMI acquisition {key!r} in {path}.")
        try:
            expected_key = hmi_acquisition_key(
                acquisition["record_time"], acquisition["hcamid"]
            )
            parse_hmi_tai_time(acquisition["observation_time"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid HMI acquisition {key!r} in {path}.") from error
        profile = profiles[acquisition["profile"]]
        if key != expected_key or acquisition["hcamid"] != profile["hcamid"]:
            raise ValueError(f"Inconsistent HMI acquisition {key!r} in {path}.")
    return manifest


def resolve_response_profile_for_key(
    directory: str | os.PathLike[str], acquisition_key: str
) -> Path:
    root = Path(directory).expanduser().resolve()
    manifest = load_response_manifest(root)
    try:
        profile_key = manifest["acquisitions"][acquisition_key]["profile"]
        filename = manifest["profiles"][profile_key]["file"]
    except KeyError as error:
        raise KeyError(
            f"No prepared HMI response profile for {acquisition_key}. Run the HMI preparation command."
        ) from error
    return root / filename


def _resolve_response_profile_for_acquisition(
    directory: str | os.PathLike[str],
    manifest: dict[str, Any],
    acquisition: dict[str, Any],
) -> Path:
    """Resolve one profile only when manifest and current FITS identity agree."""

    root = Path(directory).expanduser().resolve()
    try:
        acquisition_key = acquisition["acquisition_key"]
        entry = manifest["acquisitions"][acquisition_key]
    except KeyError as error:
        raise KeyError(
            "No HMI response profile for acquisition "
            f"{acquisition.get('acquisition_key')!r}. Run the HMI preparation command."
        ) from error
    identity_fields = ("record_time", "observation_time", "hcamid")
    mismatches = [
        field for field in identity_fields if acquisition.get(field) != entry.get(field)
    ]
    if mismatches:
        raise ValueError(
            f"HMI response acquisition {acquisition_key!r} does not match current "
            "FITS metadata for "
            + ", ".join(mismatches)
            + ". Reprepare the HMI response directory for these inputs."
        )
    profile_key = entry["profile"]
    return root / manifest["profiles"][profile_key]["file"]


def resolve_response_profile(
    directory: str | os.PathLike[str], reference_file: str | os.PathLike[str]
) -> Path:
    root = Path(directory).expanduser().resolve()
    manifest = load_response_manifest(root)
    acquisition = read_acquisition_header(Path(reference_file))
    return _resolve_response_profile_for_acquisition(root, manifest, acquisition)


def load_response_profile(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HMI response profile not found: {path}.")
    metadata = _read_profile_metadata(path)
    with np.load(path, allow_pickle=False) as archive:
        offsets = np.array(archive["offsets"], dtype=np.float32, copy=True)
        weights = np.array(archive["weights"], dtype=np.float32, copy=True)
        continuum = np.array(archive["continuum_weights"], dtype=np.float32, copy=True)
    if offsets.ndim != 2 or offsets.shape[0] != 6 or offsets.shape[1] < 2:
        raise ValueError(f"Invalid HMI response offsets in {path}: {offsets.shape}.")
    if (
        weights.ndim != 4
        or min(weights.shape[:2]) < 2
        or weights.shape[-2:] != offsets.shape
    ):
        raise ValueError(
            f"HMI weights must have shape [y,x,6,sample]; got {weights.shape}."
        )
    if continuum.shape != weights.shape[:-1]:
        raise ValueError(
            f"HMI continuum weights must have shape {weights.shape[:-1]}; got {continuum.shape}."
        )
    if not all(np.isfinite(value).all() for value in (offsets, weights, continuum)):
        raise ValueError(f"HMI response profile {path} contains non-finite values.")
    if np.any(weights < 0) or np.any(continuum < 0):
        raise ValueError(f"HMI response profile {path} contains negative weights.")
    if not np.allclose(weights.sum(axis=-1) + continuum, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError(f"HMI response weights in {path} do not sum to one.")
    if metadata.get("format") != "prom3theus.hmi_response.v1":
        raise ValueError(f"Unsupported HMI response profile format in {path}.")
    if (
        metadata.get("wavelength_unit") != "Angstrom"
        or type(metadata.get("phase_map_fsn")) is not int
        or metadata["phase_map_fsn"] < 0
        or type(metadata.get("HCAMID")) is not int
        or metadata["HCAMID"] not in {2, 3}
        or not isinstance(metadata.get("record"), str)
        or not metadata["record"].strip()
        or type(metadata.get("ccd_size")) is not int
        or metadata["ccd_size"] < 2
        or metadata.get("phase_map_shape") != list(weights.shape[:2])
        or metadata.get("samples") != offsets.shape[1]
    ):
        raise ValueError(f"Invalid HMI response metadata contract in {path}.")
    try:
        half_width = float(metadata["half_width"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"HMI response metadata in {path} requires a numeric half_width."
        ) from error
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError(f"HMI response half_width in {path} must be positive.")
    _validate_detune_sequence_times(metadata, f"HMI response profile {path}")
    return {
        "offsets": offsets,
        "weights": weights,
        "continuum_weights": continuum,
        "metadata": metadata,
    }


class HMIResponseArchive:
    """Vectorized CPU sampler for one immutable spatial response profile."""

    def __init__(self, profile_file: str | os.PathLike[str]):
        self.profile_file = Path(profile_file).expanduser().resolve()
        arrays = load_response_profile(self.profile_file)
        # FITS I0..I5 and response archives are red-to-blue.  Runtime spectra
        # use strictly increasing wavelength, hence the joint reversal here.
        self.offsets = torch.from_numpy(np.ascontiguousarray(arrays["offsets"][::-1]))
        self.weights = torch.from_numpy(
            np.ascontiguousarray(arrays["weights"][..., ::-1, :])
        )
        self.continuum_weights = torch.from_numpy(
            np.ascontiguousarray(arrays["continuum_weights"][..., ::-1])
        )
        self.metadata = dict(arrays["metadata"])
        self.inner_half_width_angstrom = float(self.metadata["half_width"])

    def quadrature_wavelength(self, observed_wavelength_angstrom) -> torch.Tensor:
        observed = torch.from_numpy(
            np.ascontiguousarray(observed_wavelength_angstrom, dtype=np.float64)
        )
        if observed.shape != (self.offsets.shape[0],):
            raise ValueError(
                "HMI observed wavelengths do not match the response filters."
            )
        if not torch.isfinite(observed).all() or not torch.all(
            observed[1:] > observed[:-1]
        ):
            raise ValueError("HMI observed wavelengths must be finite and increasing.")
        targets = observed[:, None] + self.offsets.to(torch.float64)
        reference = targets.mean(dim=0)
        if not torch.allclose(targets, reference[None], rtol=0.0, atol=1e-3):
            raise ValueError(
                "HMI response filters do not share one global quadrature grid."
            )
        if not torch.all(reference[1:] > reference[:-1]):
            raise ValueError("HMI response quadrature wavelengths must increase.")
        return reference.to(torch.float32)

    def sample(self, detector_xy) -> dict[str, torch.Tensor]:
        coordinates = torch.as_tensor(detector_xy, dtype=torch.float32, device="cpu")
        if coordinates.shape[-1:] != (2,):
            raise ValueError("detector_xy must end in [x, y].")
        if (
            not torch.isfinite(coordinates).all()
            or torch.any(coordinates < 0)
            or torch.any(coordinates > 1)
        ):
            raise ValueError("detector_xy must be finite and lie inside [0, 1]^2.")
        original_shape = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2)
        height, width = self.weights.shape[:2]
        x = flat[:, 0] * (width - 1)
        y = flat[:, 1] * (height - 1)
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
        fx, fy = x - x0, y - y0

        def interpolate(array: torch.Tensor) -> torch.Tensor:
            extra = array.ndim - 2
            wx = fx.reshape(len(flat), *([1] * extra))
            wy = fy.reshape(len(flat), *([1] * extra))
            return (
                array[y0, x0] * (1 - wx) * (1 - wy)
                + array[y0, x1] * wx * (1 - wy)
                + array[y1, x0] * (1 - wx) * wy
                + array[y1, x1] * wx * wy
            )

        weights = interpolate(self.weights)
        continuum = interpolate(self.continuum_weights)
        return {
            "spectral_weights": weights.reshape(*original_shape, *self.offsets.shape),
            "continuum_weights": continuum.reshape(
                *original_shape, self.offsets.shape[0]
            ),
        }


__all__ = [
    "DEFAULT_PROFILE_HALF_WIDTH",
    "HMIResponseArchive",
    "MANIFEST_FORMAT",
    "MANIFEST_NAME",
    "load_response_manifest",
    "load_response_profile",
    "resolve_response_profile",
    "resolve_response_profile_for_key",
]
