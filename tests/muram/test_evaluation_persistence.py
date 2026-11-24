"""Evaluation products retain native identity and bounded response storage."""

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from prom3theus.observations.arrays import ArrayRef, write_array


@pytest.mark.parametrize("layout", ["broadcast", "transpose"])
def test_noncontiguous_arrays_write_c_order_without_full_tensor_copy(tmp_path, monkeypatch, layout):
    value = (torch.arange(6.).reshape(2, 3).expand(4, 5, 2, 3) if layout == "broadcast"
             else torch.arange(30.).reshape(5, 6).T)
    expected = value.numpy().copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Writer must not materialize a full contiguous tensor")

    monkeypatch.setattr(torch.Tensor, "contiguous", forbidden)
    path = tmp_path / "array.npy"
    write_array(path, value)
    restored = ArrayRef.from_path(path)
    np.testing.assert_array_equal(restored.read().numpy(), expected)


def test_export_metadata_retains_native_reference_and_loading_binning(tmp_path):
    from prom3theus.artifacts.archive import build_save_state_export_metadata, write_export_archive

    reference = {"schema": "muram_native_reference_v1", "iteration": 42,
                 "native_geometry": {"height_origin": "bottom native cell center"}}
    contract = {"native_reference": reference, "spatial_stride": 2, "spatial_binning": 1}
    mode = "carrington_observer_relative"
    state = SimpleNamespace(
        observation=SimpleNamespace(spec=SimpleNamespace(metadata=lambda: {}, velocity_synthesis_mode=mode),
                                    source_signature="a" * 64, raster_names=["muram_000042"],
                                    validation_raster_index=0, times=[], bounds={}),
        resources={}, epoch=0, global_step=1, config=SimpleNamespace(to_dict=lambda: {}),
    )
    selection = SimpleNamespace(name="muram_000042", index=0, raster_count=1,
                                raster=SimpleNamespace(metadata={"muram_synthesis": contract,
                                                                  "spatial_binning": {"factor": 4}}))
    arrays = {"valid_mask": np.ones((1, 1), dtype=bool)}
    metadata = build_save_state_export_metadata(
        state, arrays, selection, save_state_sha256="b" * 64, depth_samples=2,
        include_stokes=False, full_shell=None, storage_dtype="float32", device="cpu",
    )
    path = write_export_archive(tmp_path / "evaluation.npz", arrays, metadata)
    with np.load(path) as archive:
        restored = json.loads(str(archive["metadata_json"].item()))
    assert restored["observation"]["muram_synthesis"]["native_reference"] == reference
    assert restored["observation"]["spatial_binning"] == {"factor": 4}


@pytest.mark.parametrize("fault", ["batch", "option", "missing_response", "missing_header", "binning"])
def test_invalid_replacement_preserves_existing_output(tmp_path, fault):
    import yaml
    from prom3theus.muram.cli import main
    from tests.muram.test_column_loading import _write_snapshot
    from prom3theus.muram.hmi_filter import build_idealized_hmi_filter

    source, output = tmp_path / "source", tmp_path / "output"
    _write_snapshot(source, (2, 2, 15), 2, physical=True)
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing valid output")
    args = ["synthesize", "--source-path", str(source), "--out", str(output),
            "--iterations", "10", "--output-format", "store", "--overwrite", "--device", "cpu"]
    if fault == "batch":
        args += ["--batch-size", "0"]
    elif fault == "binning":
        args += ["--binning", "3"]
    elif fault == "missing_header":
        (source / "Header.000010").unlink()
    else:
        if fault == "option":
            options = {"unrecognized_option": True}
        else:
            operator, observed, _, _ = build_idealized_hmi_filter()
            options = {"instrument": {"type": "hmi_filter_profiles",
                                      "quadrature_wavelength_angstrom": operator.quadrature_wavelength_angstrom.tolist(),
                                      "inner_half_width_angstrom": operator.inner_half_width_angstrom},
                       "observed_wavelength_angstrom": observed.tolist(), "line_ids": ["FeI_6173.3352"]}
        config = tmp_path / "instrument.yaml"
        config.write_text(yaml.safe_dump(options))
        args += ["--instrument-config", str(config)]
    with pytest.raises(SystemExit):
        main(args)
    assert marker.read_text() == "existing valid output"


def test_nonstandard_aia_response_is_not_written_as_standard_fits(tmp_path):
    from dataclasses import replace
    from prom3theus.muram.fits import write_aia_fits
    from prom3theus.muram.synthesize import synthesize_sdo
    from tests.muram.test_sdo_fits import snapshot

    hmi, aia = synthesize_sdo(snapshot(), progress=False)
    with pytest.raises(ValueError, match="standard response resource"):
        write_aia_fits(tmp_path, replace(aia, response_resource="custom_v1:response"),
                       hmi.geometry, time_s=0, iteration=1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("invalid", ["device", "dtype"])
def test_direct_sdo_invalid_execution_preserves_output(tmp_path, monkeypatch, invalid):
    from prom3theus.muram.sdo import synthesize_sdo_sequence

    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    args = SimpleNamespace(out=output, overwrite=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises((RuntimeError, ValueError), match="CUDA|dtype"):
        synthesize_sdo_sequence(args, {}, device="cuda" if invalid == "device" else "cpu",
                                dtype=torch.float16 if invalid == "dtype" else torch.float32)
    assert marker.read_text() == "keep"


def test_optional_hinode_response_accepts_null():
    import yaml
    from pathlib import Path
    from prom3theus.muram.configuration import validate_options
    from prom3theus.muram.synthesize import synthesize_stokes

    config = Path(__file__).resolve().parents[2] / "configs/muram/hinode/synthesis.yaml"
    options = yaml.safe_load(config.read_text())
    options["response"] = None
    validate_options(synthesize_stokes, options, kind="stokes")
