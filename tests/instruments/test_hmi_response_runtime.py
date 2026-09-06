"""Runtime HMI response sampling is offline and DRMS-free."""

from __future__ import annotations

import importlib
import json
import sys

import numpy as np
import pytest
import torch
from astropy.io import fits

from prom3theus.core import sha256_file


def _profile(path, **metadata_updates):
    observed = np.linspace(6173.0, 6173.5, 6, dtype=np.float32)
    quadrature = np.array([6173.2, 6173.3, 6173.4], dtype=np.float32)
    # On disk, HMI filter order is red-to-blue; runtime reverses it to its
    # increasing-wavelength training order before adding filter offsets.
    offsets = (quadrature[None] - observed[:, None])[::-1].copy()
    weights = np.full((2, 2, 6, 3), 0.2, dtype=np.float32)
    continuum = np.full((2, 2, 6), 0.4, dtype=np.float32)
    metadata = {
        "format": "prom3theus.hmi_response.v1",
        "half_width": 0.5,
        "wavelength_unit": "Angstrom",
        "ccd_size": 2,
        "phase_map_shape": [2, 2],
        "samples": 3,
        "phase_map_fsn": 1,
        "HCAMID": 3,
        "record": "test[1]",
        "T_START": "2023.10.10_19:25:31_TAI",
        "T_REC": "2023.10.10_19:26:26_TAI",
        "T_STOP": "2023.10.10_19:27:22_TAI",
    }
    metadata.update(metadata_updates)
    np.savez(
        path,
        offsets=offsets,
        weights=weights,
        continuum_weights=continuum,
        metadata=np.array(json.dumps(metadata)),
    )


def test_response_runtime_imports_without_drms_and_samples_a_local_archive(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "drms", None)
    sys.modules.pop("prom3theus.instruments.hmi.response", None)
    response = importlib.import_module("prom3theus.instruments.hmi.response")
    assert sys.modules.get("drms") is None

    profile = tmp_path / "response.npz"
    _profile(profile)
    archive = response.HMIResponseArchive(profile)
    observed = torch.linspace(6173.0, 6173.5, 6)
    quadrature = archive.quadrature_wavelength(observed)
    assert quadrature.shape == (3,)
    descending_storage = observed.numpy()[::-1].copy()
    increasing_negative_stride_view = descending_storage[::-1]
    assert increasing_negative_stride_view.strides[0] < 0
    torch.testing.assert_close(
        archive.quadrature_wavelength(increasing_negative_stride_view), quadrature
    )
    sampled = archive.sample(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert sampled["spectral_weights"].shape == (2, 6, 3)
    torch.testing.assert_close(
        sampled["spectral_weights"].sum(dim=-1) + sampled["continuum_weights"],
        torch.ones(2, 6),
    )
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        archive.sample(torch.tensor([[1.01, 0.5]]))


@pytest.mark.parametrize(
    "metadata_updates",
    [
        {"T_START": "2023-10-10T19:25:31Z"},
        {"T_REC": "2023.10.10_19:28:00_TAI"},
        {"T_STOP": "2023.10.10_19:24:00_TAI"},
    ],
)
def test_response_profile_validates_detune_sequence_tai_order(
    tmp_path, metadata_updates
):
    from prom3theus.instruments.hmi import response

    profile = tmp_path / "response.npz"
    _profile(profile, **metadata_updates)

    with pytest.raises(ValueError, match="phase-map.*timestamps"):
        response.load_response_profile(profile)


def test_response_manifest_binds_profile_bytes(tmp_path):
    from prom3theus.instruments.hmi import response

    profile = tmp_path / "response.npz"
    _profile(profile)
    (tmp_path / response.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format": response.MANIFEST_FORMAT,
                "phase_map_assignment_series": response.PHASE_MAP_ASSIGNMENT_SERIES,
                "profiles": {
                    "INVPHMAP=1|HCAMID=3": {
                        "file": profile.name,
                        "phase_map_fsn": 1,
                        "hcamid": 3,
                        "record": "test[1]",
                        "sha256": sha256_file(profile),
                    }
                },
                "acquisitions": {
                    "2024.03.24_01:00:00_TAI|HCAMID=3": {
                        "record_time": "2024.03.24_01:00:00_TAI",
                        "observation_time": "2024.03.24_01:00:00_TAI",
                        "hcamid": 3,
                        "phase_map_fsn": 1,
                        "assignment_record": "hmi.B_720s[2024.03.24_01:00:00_TAI]",
                        "profile": "INVPHMAP=1|HCAMID=3",
                    }
                },
            }
        )
    )
    response.load_response_manifest(tmp_path)
    profile.write_bytes(profile.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        response.load_response_manifest(tmp_path)


def test_response_resolution_binds_current_fits_observation_time(tmp_path):
    from prom3theus.instruments.hmi import response
    from prom3theus.instruments.hmi.timeline import _response_samplers

    profile = tmp_path / "response.npz"
    _profile(profile)
    acquisition_key = "2024.03.24_01:00:00_TAI|HCAMID=3"
    (tmp_path / response.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format": response.MANIFEST_FORMAT,
                "phase_map_assignment_series": response.PHASE_MAP_ASSIGNMENT_SERIES,
                "profiles": {
                    "INVPHMAP=1|HCAMID=3": {
                        "file": profile.name,
                        "phase_map_fsn": 1,
                        "hcamid": 3,
                        "record": "test[1]",
                        "sha256": sha256_file(profile),
                    }
                },
                "acquisitions": {
                    acquisition_key: {
                        "record_time": "2024.03.24_01:00:00_TAI",
                        "observation_time": "2024.03.24_00:59:58_TAI",
                        "hcamid": 3,
                        "phase_map_fsn": 1,
                        "assignment_record": "hmi.B_720s[2024.03.24_01:00:00_TAI]",
                        "profile": "INVPHMAP=1|HCAMID=3",
                    }
                },
            }
        )
    )
    reference = tmp_path / "current.I0.fits"
    fits.writeto(
        reference,
        np.zeros((1, 1), dtype=np.float32),
        fits.Header(
            {
                "T_REC": "2024.03.24_01:00:00_TAI",
                "T_OBS": "2024.03.24_00:59:59_TAI",
                "CAMERA": 3,
                "HCAMID": 3,
            }
        ),
    )

    with pytest.raises(
        ValueError, match="does not match current FITS.*observation_time"
    ):
        response.resolve_response_profile(tmp_path, reference)
    current = {
        "acquisition_key": acquisition_key,
        "record_time": "2024.03.24_01:00:00_TAI",
        "observation_time": "2024.03.24_00:59:59_TAI",
        "hcamid": 3,
    }
    with pytest.raises(
        ValueError, match="does not match current FITS.*observation_time"
    ):
        _response_samplers(tmp_path, [current])


def test_response_manifest_binds_profile_identity(tmp_path):
    from prom3theus.instruments.hmi import response

    profile = tmp_path / "response.npz"
    _profile(profile)
    manifest = {
        "format": response.MANIFEST_FORMAT,
        "phase_map_assignment_series": response.PHASE_MAP_ASSIGNMENT_SERIES,
        "profiles": {
            "INVPHMAP=2|HCAMID=3": {
                "file": profile.name,
                "phase_map_fsn": 2,
                "hcamid": 3,
                "record": "test[2]",
                "sha256": sha256_file(profile),
            }
        },
        "acquisitions": {
            "2024.03.24_01:00:00_TAI|HCAMID=3": {
                "record_time": "2024.03.24_01:00:00_TAI",
                "observation_time": "2024.03.24_01:00:00_TAI",
                "hcamid": 3,
                "phase_map_fsn": 2,
                "assignment_record": "hmi.B_720s[2024.03.24_01:00:00_TAI]",
                "profile": "INVPHMAP=2|HCAMID=3",
            }
        },
    }
    (tmp_path / response.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        response.load_response_manifest(tmp_path)


def test_response_manifest_rejects_fields_outside_the_current_schema(tmp_path):
    from prom3theus.instruments.hmi import response

    (tmp_path / response.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format": response.MANIFEST_FORMAT,
                "profiles": {},
                "acquisitions": {},
                "unknown": True,
            }
        )
    )
    with pytest.raises(ValueError, match="manifest schema"):
        response.load_response_manifest(tmp_path)
