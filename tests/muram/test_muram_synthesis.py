from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prom3theus.muram.synthesize import synthesize_hmi, synthesize_stokes
from prom3theus.muram.atmosphere_builder import build_native_atmosphere, build_stratified_atmosphere
from prom3theus.muram.observation import MuramStoredObservationData, _describe_data
from prom3theus.observations.store import ObservationStore


def snapshot():
    shape = (2, 2, 15)
    z = np.linspace(2, -6, shape[-1])
    q = {name: np.zeros(shape) for name in ("bx", "by", "bz", "vx", "vy", "vz")}
    q.update(
        temperature=np.broadcast_to(np.linspace(6500, 4500, 15), shape).copy(),
        # Include a genuinely opaque lower atmosphere in this synthetic box.
        pressure=np.broadcast_to(np.geomspace(2e6, 10, 15), shape).copy(),
        tau=np.broadcast_to(10.0**z, shape).copy(),
        ds_m=(1e5, 1e5, 1e4),
    )
    q["bx"][:] = 300
    q["bz"][:] = 100
    q["temperature"][1] += 500
    return SimpleNamespace(time_s=7200, synthesis_ready_quantities=lambda: q)


def test_hmi_striding_batching_and_store_contract(tmp_path):
    snap = snapshot()
    full = synthesize_hmi(snap, batch_size=3)
    sampled = synthesize_hmi(snap, stride=2, batch_size=1)
    torch.testing.assert_close(sampled.stokes, full.stokes[::2, ::2])
    assert torch.all(full.geometry.mu == 1)
    assert torch.all(full.geometry.coordinates[..., 2] == 2)
    raster = full.to_observation_raster()
    assert "instrument_response:spectral_weights" in raster.auxiliary
    ObservationStore.save_sequence(
        tmp_path / "store", [raster], source_signature="a" * 64
    )
    data = MuramStoredObservationData(tmp_path / "store", 0)
    data.setup()
    spec = _describe_data(data)
    from prom3theus.artifacts.validation import (
        validate_raster_scientific_contract,
        validate_raster_velocity_contract,
    )

    validate_raster_scientific_contract(data.raster, spec)
    validate_raster_velocity_contract(data.raster, spec.velocity_synthesis_mode)
    assert spec.instrument_type == "hmi_filter_profiles"


def test_packaged_hinode():
    result = synthesize_stokes(
        snapshot(),
        instrument={
            "type": "hinode_sp",
            "spectral_psf": {
                "type": "gaussian",
                "fwhm_angstrom": 0.03,
                "oversample": 2,
                "truncate_sigma": 4,
            },
        },
        observed_wavelength_angstrom=np.linspace(6301.1, 6302.8, 41),
        line_ids=["FeI_6301.5008", "FeI_6302.4932"],
        batch_size=2,
    )
    assert result.stokes.shape == (2, 2, 4, 41)
    assert torch.isfinite(result.stokes).all()


def test_invalid_optical_depth_and_stride():
    q = snapshot().synthesis_ready_quantities()
    q["tau"][0, 0, 3] = 1e20
    with pytest.raises(ValueError, match="Optical depth"):
        build_stratified_atmosphere(q, [-5, 0, 1])
    with pytest.raises(ValueError, match="covered"):
        build_stratified_atmosphere(snapshot().synthesis_ready_quantities(), [-9, 0, 1])
    with pytest.raises(ValueError, match="stride"):
        synthesize_hmi(snapshot(), stride=0)


@pytest.mark.parametrize("instrument_type", ["hmi", "hinode"])
def test_legacy_store_runtime(tmp_path, instrument_type):
    from prom3theus.muram import config as _registration  # noqa: F401
    from prom3theus.config import load_config, parse_config
    from prom3theus.application.joint_assembly import build_joint_runtime
    from prom3theus.application.joint_evaluation import evaluate_joint_runtime

    if instrument_type == "hinode":
        import yaml
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        options = yaml.safe_load((root / "configs/muram/hinode/synthesis.yaml").read_text())
        result = synthesize_stokes(snapshot(), stride=1, batch_size=2, **options)
    else:
        result = synthesize_hmi(snapshot(), stride=1, batch_size=2)
    store = tmp_path / "store"
    ObservationStore.save_sequence(
        store, [result.to_observation_raster()], source_signature="b" * 64
    )
    preset = "configs/muram/hinode/mhs.yaml" if instrument_type == "hinode" else "configs/muram/sdo/joint.yaml"
    raw = load_config(preset).to_dict()
    # Legacy stores remain supported, but the shipped preset now uses HMI+AIA FITS.
    raw["streams"] = raw["streams"][:1]
    raw["streams"][0]["id"] = "muram_synthetic_stokes"
    raw["scene"]["reference_stream"] = "muram_synthetic_stokes"
    raw["training"]["reference_stream"] = "muram_synthetic_stokes"
    raw["solver"].update(
        output_directory=str(tmp_path / "output"), work_directory=str(tmp_path / "work")
    )
    obs = {
        "type": "muram_stored", "directory": str(store),
        "validation_raster_index": 0,
        "loader": raw["streams"][0]["observation"]["loader"],
    }
    raw["streams"][0]["observation"] = obs
    obs["loader"].update(
        workers=0,
        preparation_workers=1,
        validation_stride=1,
        batch_size=2,
        validation_batch_size=2,
    )
    raw["atmosphere"]["network"].update(hidden_dimension=8, hidden_layers=1)
    raw["diagnostics"]["visualization"]["enabled"] = False
    raw["physics"]["potential_boundary"]["enabled"] = False
    raw["physics"]["potential_boundary"]["photosphere"]["enabled"] = False
    raw["dry_run"].update(
        device="cpu",
        evaluate_gradients=False,
        gradient_sample_count=2,
        max_samples_per_stream=2,
        quadrature_samples=[4, 8],
    )
    runtime = build_joint_runtime(parse_config(raw, base_directory=tmp_path))
    runtime.model.set_step(5000)
    evaluated = evaluate_joint_runtime(runtime)
    assert evaluated.report
    term = runtime.model.terms["muram_synthetic_stokes"]
    loss = term.evaluate_batch(
        runtime.validation_batches["muram_synthetic_stokes"]
    ).likelihood_loss
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in runtime.model.parameters()
    )


@pytest.mark.parametrize("vertical", [0, 1, 2])
@pytest.mark.parametrize("descending", [False, True])
def test_reader_vector_axes_units_and_handedness(tmp_path, vertical, descending):
    from prom3theus.muram.reader import MURaMSnapshot

    shape = (4, 5, 6)
    tau_profile = np.geomspace(1e2, 1e-6, shape[vertical])
    if descending:
        tau_profile = tau_profile[::-1]
    view = [1, 1, 1]
    view[vertical] = shape[vertical]
    fields = {
        "tau": np.broadcast_to(tau_profile.reshape(view), shape),
        "eosP": np.full(shape, 1e5),
        "eosT": np.full(shape, 6000),
    }
    for axis in range(3):
        fields[f"result_prim_{axis + 1}"] = np.full(shape, (axis + 1) * 100)
        fields[f"result_prim_{axis + 5}"] = np.full(
            shape, (axis + 1) / np.sqrt(4 * np.pi)
        )
    spacing = [1e7, 2e7, 3e7]
    spacing[vertical] = 1e6
    np.savetxt(tmp_path / "Header.000010", [*shape, *spacing, 1234])
    for name, value in fields.items():
        value.astype("float32").ravel(order="F").tofile(tmp_path / f"{name}.000010")
    result = MURaMSnapshot(str(tmp_path), 10).synthesis_ready_quantities()
    order = tuple(a for a in range(3) if a != vertical) + (vertical,)
    parity = round(np.linalg.det(np.eye(3)[list(order)]))
    signs = (1, parity * (-1 if descending else 1), -1 if descending else 1)
    native_components = tuple(i + 1 for i in order)
    for j, axis in enumerate("xyz"):
        np.testing.assert_allclose(
            result["b" + axis], signs[j] * native_components[j], rtol=1e-6
        )
        np.testing.assert_allclose(
            result["v" + axis], signs[j] * native_components[j], rtol=1e-6
        )
    assert np.all(np.diff(result["tau"], axis=-1) < 0)
    assert result["ds_m"] == tuple(spacing[a] * .01 for a in order)


def test_aia_density_square_full_cell_length_and_striding():
    from prom3theus.muram.synthesize import synthesize_aia, AIA_RESPONSE_RESOURCE
    from prom3theus.instruments.aia_euv.operator import AIAEmissionOperator
    from prom3theus.rt import HybridSolarEOS

    snap = snapshot()
    q = snap.synthesis_ready_quantities()
    q["temperature"][:] = 1e6
    q["pressure"][:] = 0.01
    q["pressure"][1] = 0.02
    q["tau"][:] = 1e-10
    full = synthesize_aia(snap, dtype=torch.float64)
    sampled = synthesize_aia(snap, stride=2, batch_size=1, dtype=torch.float64)
    torch.testing.assert_close(sampled.intensity, full.intensity[::2, ::2])
    torch.testing.assert_close(full.intensity[:, 1], 4 * full.intensity[:, 0])
    temperature = torch.tensor(1e6, dtype=torch.float64)
    ne = (
        HybridSolarEOS()
        .double()
        .electron_density(temperature, temperature.new_tensor(0.01))
    )
    response = (
        AIAEmissionOperator(
            full.channels_angstrom, response_resource=AIA_RESPONSE_RESOURCE
        )
        .double()
        .local_response(temperature)
    )
    expected = (
        response * ne.square() * q["temperature"].shape[-1] * q["ds_m"][2] * 1e-10
    )
    torch.testing.assert_close(full.intensity[0, 0], expected)


def test_hmi_azimuth_and_geometric_path():
    from prom3theus.validation.reference_synthesis import LTESynthesizer
    from prom3theus.validation.reference_transfer import GeometricHeightPath
    from prom3theus.muram.synthesize import (
        MURAM_HMI_RADIANCE_SCALE_W_M3_SR,
    )

    snap = snapshot()
    result = synthesize_hmi(snap, stride=1, dtype=torch.float64)
    atmosphere = build_native_atmosphere(snap.synthesis_ready_quantities())
    # At the equator the toward-observer component of Omega x r is -Omega*y.
    # Include this inertial shift before comparing the independent formal solve.
    from prom3theus.core.solar_velocity import CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S
    atmosphere.velocity_field[..., 2] -= (
        CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S
        * result.geometry.surface_position_m[..., 1].T.reshape(-1, 1)
    )
    # A 90-degree magnetic azimuth rotation reverses Q and U, preserving I,V.
    synth = LTESynthesizer(line_ids=["FeI_6173.3352"]).double()
    raw = synth(
        atmosphere,
        result.raw_wavelength_angstrom,
        path=GeometricHeightPath(atmosphere.geometric_height_m),
        radiance_scale=MURAM_HMI_RADIANCE_SCALE_W_M3_SR,
    )
    raw[:, 1:3] *= -1
    torch.testing.assert_close(
        result.raw_stokes, raw.reshape(2, 2, 4, -1).transpose(0, 1)
    )


@pytest.mark.parametrize("instrument_type", ["hmi", "hinode_sp", "aia_euv"])
@pytest.mark.parametrize("fail_second", [False, True])
def test_synthesis_cli(tmp_path, monkeypatch, instrument_type, fail_second):
    import json
    import weakref
    import yaml
    import prom3theus.muram.synthesize as synthesis_module
    from prom3theus.muram.cli import main
    from prom3theus.muram.reader import _FIELD_IDS, _UNIT_SCALE

    q = snapshot().synthesis_ready_quantities()
    source = tmp_path / "source"
    source.mkdir()
    for iteration, time in [(10, 3600), (20, 7200)]:
        np.savetxt(source / f"Header.{iteration:06d}", [2, 2, 15, 1e7, 1e7, 1e6, time])
        for name, values in q.items():
            if name == "ds_m":
                continue
            (values / _UNIT_SCALE[name]).astype("float32").ravel(order="F").tofile(
                source / f"{_FIELD_IDS[name]}.{iteration:06d}"
            )
    out = tmp_path / "output"
    args = [
        "synthesize", "--output-format", "store",
        "--source-path",
        str(source),
        "--out",
        str(out),
        "--iterations",
        "10",
        "20",
        "--batch-size",
        "2",
    ]
    if instrument_type != "hmi":
        options = {"instrument": {"type": instrument_type}}
        if instrument_type == "aia_euv":
            options["instrument"].update(
                channels_angstrom=[171],
                response_resource="aia_euv_v1:aia_temperature_response",
            )
        else:
            options["instrument"]["spectral_psf"] = {
                "type": "gaussian",
                "fwhm_angstrom": 0.03,
                "oversample": 2,
                "truncate_sigma": 4,
            }
            options.update(
                line_ids=["FeI_6301.5008", "FeI_6302.4932"],
                observed_wavelength_angstrom=np.linspace(6301.0, 6302.9, 41).tolist(),
            )
        path = tmp_path / "instrument.yaml"
        path.write_text(yaml.safe_dump(options))
        args += ["--instrument-config", str(path)]
    function_name = {
        "hmi": "synthesize_hmi",
        "hinode_sp": "synthesize_stokes",
        "aia_euv": "synthesize_aia",
    }[instrument_type]
    original_synthesize = getattr(synthesis_module, function_name)
    references = []
    previous_workers = []

    def checked_synthesis(current_snapshot, **kwargs):
        assert current_snapshot.synthesis_columns()._owner.loader is None
        assert all(not worker.is_alive() for worker in previous_workers)
        if references:
            # The previous native data and complete result must be released
            # before the next synthesis starts, not held for a final save.
            assert all(reference() is None for reference in references)
            if instrument_type == "aia_euv":
                assert (out / "muram_000010.npz").is_file()
            else:
                manifest = ObservationStore.manifest(out)
                assert len(manifest["rasters"]) == 1
                assert manifest["metadata"]["complete"] is False
                ObservationStore.load_sequence(out, verify=True)
            if fail_second:
                raise RuntimeError("injected second snapshot failure")
        result = original_synthesize(current_snapshot, **kwargs)
        previous_workers[:] = current_snapshot.synthesis_columns()._owner.loader._iterator._workers
        references[:] = [weakref.ref(current_snapshot), weakref.ref(result)]
        return result

    monkeypatch.setattr(synthesis_module, function_name, checked_synthesis)
    if fail_second:
        with pytest.raises(RuntimeError, match="injected second"):
            main(args)
        status = json.loads((out / "synthesis_status.json").read_text())
        assert status["state"] == "failed"
        assert status["completed_iterations"] == [10]
        if instrument_type != "aia_euv":
            rasters, _, _ = ObservationStore.load_sequence(out, verify=True)
            assert len(rasters) == 1
        saved = {path.name: path.read_bytes() for path in out.glob("muram_000010*")}
        resumed = []

        def resume_synthesis(current_snapshot, **kwargs):
            resumed.append(current_snapshot.iteration)
            return original_synthesize(current_snapshot, **kwargs)

        monkeypatch.setattr(synthesis_module, function_name, resume_synthesis)
        # Existing runs predate the dedicated resume contract file.
        (out / "synthesis_run.json").unlink()
        # Batch size is an execution setting and may change after an OOM.
        main([*args, "--batch-size", "1"])
        assert resumed == [20]
        assert saved == {path.name: path.read_bytes() for path in out.glob("muram_000010*")}
        status = json.loads((out / "synthesis_status.json").read_text())
        assert status["state"] == "complete"
        assert status["completed_iterations"] == [10, 20]
        return
    main(args)
    assert all(reference() is None for reference in references)
    assert not list(out.rglob("*.jpg"))
    status = json.loads((out / "synthesis_status.json").read_text())
    assert status["state"] == "complete"
    assert status["completed_iterations"] == [10, 20]
    if instrument_type == "aia_euv":
        with np.load(out / "muram_000020.npz") as result:
            assert result["time_s"] == 7200
            assert result["intensity"].shape == (2, 2, 1)
    else:
        data = MuramStoredObservationData(out, 1)
        data.setup()
        spec = _describe_data(data)
        assert spec.instrument_type == (
            "hmi_filter_profiles" if instrument_type == "hmi" else instrument_type
        )
        assert torch.all(data.rasters[0].coordinates[..., 2] == 1)
        assert torch.all(data.rasters[1].coordinates[..., 2] == 2)

    def unexpected_synthesis(*args, **kwargs):
        pytest.fail("Completed snapshots must not be synthesized again")

    monkeypatch.setattr(synthesis_module, function_name, unexpected_synthesis)
    main(args)
    with pytest.raises(ValueError, match="settings changed"):
        main([*args, "--stride", "2"])
    replaced = []

    def overwrite_synthesis(current_snapshot, **kwargs):
        replaced.append(current_snapshot.iteration)
        return original_synthesize(current_snapshot, **kwargs)

    monkeypatch.setattr(synthesis_module, function_name, overwrite_synthesis)
    (out / "obsolete.txt").write_text("old output")
    main([*args, "--overwrite"])
    assert replaced == [10, 20]
    assert not (out / "obsolete.txt").exists()


def test_device_selection_and_unavailable_cuda(monkeypatch):
    from prom3theus.muram.synthesize import resolve_synthesis_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_synthesis_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        resolve_synthesis_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_synthesis_device("auto") == torch.device("cuda:1")
    assert resolve_synthesis_device("cuda:0") == torch.device("cuda:0")
    assert resolve_synthesis_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="unavailable"):
        resolve_synthesis_device("cuda:2")


def test_interpolation_matches_independent_numpy_reference():
    from prom3theus.muram.atmosphere_builder import _interp_along_last_axis

    rng = np.random.default_rng(42)
    x = np.cumsum(rng.uniform(0.01, 1, (3, 2, 30)), axis=-1)
    y = rng.normal(size=x.shape)
    target = np.linspace(-1, 30, 61)
    actual = _interp_along_last_axis(target, x, y)
    expected = np.stack(
        [
            np.interp(target, row_x, row_y)
            for row_x, row_y in zip(x.reshape(-1, 30), y.reshape(-1, 30))
        ]
    )
    np.testing.assert_allclose(actual, expected.reshape(3, 2, -1), atol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("kind", ["hmi", "hinode", "aia"])
def test_cuda_synthesis_matches_cpu_and_returns_host_products(kind, monkeypatch):
    from prom3theus.muram.synthesize import synthesize_aia
    from prom3theus.validation.reference_synthesis import LTESynthesizer
    from prom3theus.instruments.aia_euv.operator import AIAEmissionOperator

    seen = []
    cls = AIAEmissionOperator if kind == "aia" else LTESynthesizer
    method = "local_emission_measure_integrand" if kind == "aia" else "forward"
    original = getattr(cls, method)

    def record(self, value, *args, **kwargs):
        tensor = value if kind == "aia" else value.temperature
        seen.append(tensor.device.type)
        assert next(self.buffers()).device.type == tensor.device.type
        return original(self, value, *args, **kwargs)

    monkeypatch.setattr(cls, method, record)
    options = {"batch_size": 2, "progress": False}
    if kind == "aia":
        fn = synthesize_aia
    elif kind == "hinode":
        fn = synthesize_stokes
        options.update(
            instrument={
                "type": "hinode_sp",
                "spectral_psf": {
                    "type": "gaussian",
                    "fwhm_angstrom": 0.03,
                    "oversample": 2,
                    "truncate_sigma": 4,
                },
            },
            observed_wavelength_angstrom=np.linspace(6301, 6302.9, 41),
            line_ids=["FeI_6301.5008", "FeI_6302.4932"],
        )
    else:
        fn = synthesize_hmi
    cpu = fn(snapshot(), device="cpu", **options)
    gpu = fn(snapshot(), device="cuda", **options)
    field = "intensity" if kind == "aia" else "stokes"
    assert getattr(gpu, field).device.type == "cpu"
    assert "cpu" in seen and "cuda" in seen
    torch.testing.assert_close(
        getattr(cpu, field), getattr(gpu, field), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("instrument_type", ["hmi", "hinode"])
def test_float32_preserves_calibrated_profiles(instrument_type):
    outputs = []
    snap = snapshot()
    q = snap.synthesis_ready_quantities()
    q["vx"][:] = 173.2
    q["vz"][:] = 1234.5
    q["by"][:] = 450.0
    q["bz"][:] = 975.0
    for dtype in (torch.float64, torch.float32):
        if instrument_type == "hmi":
            result = synthesize_hmi(snap, dtype=dtype, progress=False)
        else:
            result = synthesize_stokes(
                snap,
                dtype=dtype,
                progress=False,
                instrument={
                    "type": "hinode_sp",
                    "spectral_psf": {
                        "type": "gaussian",
                        "fwhm_angstrom": 0.03,
                        "oversample": 2,
                        "truncate_sigma": 4,
                    },
                },
                observed_wavelength_angstrom=np.linspace(6301.1, 6302.8, 61),
                line_ids=["FeI_6301.5008", "FeI_6302.4932"],
            )
        outputs.append(result)
    reference, actual = outputs
    assert actual.stokes.dtype == torch.float32
    # Calibration labels must survive serialization without optical-wavelength
    # rounding; they are not part of the differentiable float32 runtime.
    assert actual.raw_wavelength_angstrom.dtype == torch.float64
    assert actual.observed_wavelength_angstrom.dtype == torch.float64
    torch.testing.assert_close(
        actual.raw_wavelength_angstrom, reference.raw_wavelength_angstrom
    )
    for name in ("stokes", "raw_stokes"):
        ref = getattr(reference, name)
        assert (
            (getattr(actual, name).double() - ref).abs() / ref[..., :1, :].abs()
        ).max() < 1e-4


def test_aia_float32_matches_double_across_temperature_response():
    from prom3theus.muram.synthesize import synthesize_aia

    snap = snapshot()
    q = snap.synthesis_ready_quantities()
    q["temperature"][:] = np.geomspace(1e5, 1e7, q["temperature"].shape[-1])
    q["pressure"][:] = np.geomspace(0.0001, 0.1, q["pressure"].shape[-1])
    reference = synthesize_aia(snap, dtype=torch.float64, progress=False)
    actual = synthesize_aia(snap, dtype=torch.float32, progress=False)
    torch.testing.assert_close(
        actual.intensity.double(), reference.intensity, rtol=2e-5, atol=1e-10
    )


@pytest.mark.parametrize("kind", ["hmi", "hinode", "aia"])
@pytest.mark.parametrize("stride", [2, 8])
def test_stride_synthesizes_only_selected_native_columns(monkeypatch, kind, stride):
    import prom3theus.muram.synthesize as module

    original = snapshot().synthesis_ready_quantities()
    quantities = {
        name: np.tile(value, (2, 3, 1))[:3, :5].copy()
        if isinstance(value, np.ndarray) and value.ndim == 3
        else value
        for name, value in original.items()
    }
    quantities["bz"] += np.arange(15).reshape(3, 5, 1) * 50
    if kind == "aia":
        quantities["temperature"][:] = 1e6
        quantities["pressure"][:] = np.arange(1, 16).reshape(3, 5, 1) * 0.01
        quantities["tau"][:] = 1e-10
    snap = SimpleNamespace(time_s=7200, synthesis_ready_quantities=lambda: quantities)
    options = {"progress": False, "batch_size": 2, "device": "cpu"}
    if kind == "aia":
        synthesize = module.synthesize_aia
    else:
        synthesize = module.synthesize_hmi
        if kind == "hinode":
            synthesize = module.synthesize_stokes
            options.update(
                instrument={
                    "type": "hinode_sp",
                    "spectral_psf": {
                        "type": "gaussian",
                        "fwhm_angstrom": 0.03,
                        "oversample": 2,
                        "truncate_sigma": 4,
                    },
                },
                observed_wavelength_angstrom=np.linspace(6301.1, 6302.8, 31),
                line_ids=["FeI_6301.5008", "FeI_6302.4932"],
            )
    full = synthesize(snap, **options)
    batches = module.column_batches
    processed = []

    def counted_batches(quantities, *args, **kwargs):
        for ix, iy, columns in batches(quantities, *args, **kwargs):
            processed.append(len(ix))
            yield ix, iy, columns

    monkeypatch.setattr(module, "column_batches", counted_batches)
    sampled = synthesize(snap, stride=stride, **options)
    assert sum(processed) == len(range(0, 3, stride)) * len(range(0, 5, stride))
    assert sampled.dx_m == full.dx_m * stride
    assert sampled.dy_m == full.dy_m * stride
    for name in ("intensity",) if kind == "aia" else ("stokes", "raw_stokes"):
        torch.testing.assert_close(
            getattr(sampled, name), getattr(full, name)[::stride, ::stride]
        )
    if kind != "aia":
        for name in (
            "coordinates",
            "surface_position_m",
            "ray_direction",
            "stokes_basis",
            "mu",
        ):
            torch.testing.assert_close(
                getattr(sampled.geometry, name),
                getattr(full.geometry, name)[::stride, ::stride],
            )
        assert (
            sampled.to_observation_raster().metadata["muram_synthesis"][
                "spatial_stride"
            ]
            == stride
        )


@pytest.mark.parametrize("boundary", ["top", "bottom"])
def test_tau_reversals_outside_sampling_brackets_do_not_change_atmosphere(boundary):
    quantities = snapshot().synthesis_ready_quantities()
    target = np.linspace(-5, 1, 25)
    reference = build_stratified_atmosphere(quantities, target)
    # Boundary tau artifacts remain wholly outside the formation interval.
    extra_tau = (
        [1e-8, 8e-8, 1e-9, 0.0] if boundary == "top" else [200.0, 500.0, 300.0, 1000.0]
    )
    extended = dict(quantities)
    for name, value in quantities.items():
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            continue
        extra = np.repeat(
            value[..., -1:] if boundary == "top" else value[..., :1], 4, axis=-1
        )
        if name == "tau":
            extra[:] = extra_tau
        extended[name] = np.concatenate(
            (value, extra) if boundary == "top" else (extra, value), axis=-1
        )
    actual = build_stratified_atmosphere(extended, target)
    for name in ("temperature", "gas_pressure", "magnetic_field", "velocity_field"):
        torch.testing.assert_close(getattr(actual, name), getattr(reference, name))
    offset = 0 if boundary == "top" else 4 * quantities["ds_m"][2]
    torch.testing.assert_close(
        actual.geometric_height_m - offset, reference.geometric_height_m
    )


def test_native_synthesis_does_not_use_tau_labels():
    snap = snapshot()
    reference = synthesize_hmi(snap, progress=False, batch_size=1)
    snap.synthesis_ready_quantities()["tau"][1, 0, 3] = 1e20
    actual = synthesize_hmi(snap, progress=False, batch_size=1)
    torch.testing.assert_close(actual.raw_stokes, reference.raw_stokes)


def test_reversed_tau_axis_is_not_silently_sorted():
    quantities = snapshot().synthesis_ready_quantities()
    quantities["tau"] = quantities["tau"][..., ::-1]
    with pytest.raises(ValueError, match="not covered"):
        build_stratified_atmosphere(quantities, [-5, 0, 1])


@pytest.mark.parametrize("kind", ["hmi", "hinode", "aia"])
@pytest.mark.parametrize("stride", [1, 2])
def test_binning_averages_after_synthesis(monkeypatch, kind, stride):
    import prom3theus.muram.synthesize as module

    quantities = snapshot().synthesis_ready_quantities()
    quantities = {name: np.tile(value, (2, 2, 1))
                  if isinstance(value, np.ndarray) else value
                  for name, value in quantities.items()}
    quantities["bz"] += np.arange(16).reshape(4, 4, 1) * 50
    if kind == "aia":
        quantities["temperature"][:] = 1e6
        quantities["pressure"][:] = np.arange(1, 17).reshape(4, 4, 1) * 0.01
        quantities["tau"][:] = 1e-10
    snap = SimpleNamespace(time_s=7200, synthesis_ready_quantities=lambda: quantities)
    synthesize = module.synthesize_hmi if kind == "hmi" else module.synthesize_aia
    options = dict(stride=stride, batch_size=3, progress=False, device="cpu")
    if kind == "hinode":
        synthesize = module.synthesize_stokes
        options.update(
            instrument={"type": "hinode_sp", "spectral_psf": {
                "type": "gaussian", "fwhm_angstrom": 0.03,
                "oversample": 2, "truncate_sigma": 4,
            }},
            observed_wavelength_angstrom=np.linspace(6301.1, 6302.8, 31),
            line_ids=["FeI_6301.5008", "FeI_6302.4932"],
        )
    full = synthesize(snap, **options)
    processed = []
    batches = module.column_batches

    def counted_batches(quantities, *args, **kwargs):
        for ix, iy, columns in batches(quantities, *args, **kwargs):
            processed.append(len(ix))
            yield ix, iy, columns

    monkeypatch.setattr(module, "column_batches", counted_batches)
    binned = synthesize(snap, binning=2, **options)
    assert sum(processed) == (4 // stride) ** 2
    assert binned.dx_m == full.dx_m * 2
    assert binned.dy_m == full.dy_m * 2
    for name in (("stokes", "raw_stokes") if kind != "aia" else ("intensity",)):
        source, actual = getattr(full, name), getattr(binned, name)
        for y in range(actual.shape[0]):
            for x in range(actual.shape[1]):
                torch.testing.assert_close(actual[y, x], source[2*y:2*y+2, 2*x:2*x+2].mean((0, 1)))
    if kind != "aia":
        from prom3theus.muram.binning import bin_stokes_raster
        loaded = bin_stokes_raster(full.to_observation_raster(), 2)
        torch.testing.assert_close(loaded.stokes, binned.stokes)
        torch.testing.assert_close(loaded.coordinates, binned.geometry.coordinates)
        torch.testing.assert_close(loaded.surface_position_m, binned.geometry.surface_position_m)
        assert binned.to_observation_raster().metadata["muram_synthesis"]["spatial_binning"] == 2
        torch.testing.assert_close(binned.geometry.coordinates, module._bin_image(full.geometry.coordinates, 2))
        torch.testing.assert_close(torch.linalg.vector_norm(binned.geometry.surface_position_m, dim=-1),
                                   torch.full_like(binned.geometry.mu, module.SOLAR_RADIUS_M))


@pytest.mark.parametrize("binning", [0, -1, True, 1.5, 3])
def test_invalid_binning(binning):
    from prom3theus.muram.synthesize import synthesize_aia

    for synthesize in (synthesize_hmi, synthesize_aia):
        with pytest.raises(ValueError, match="binning"):
            synthesize(snapshot(), binning=binning, progress=False)


@pytest.mark.parametrize("flag", ["--binning", "-binning"])
def test_cli_passes_independent_stride_and_binning(monkeypatch, tmp_path, flag):
    from prom3theus.muram.cli import main
    import prom3theus.muram.sdo as sdo

    captured = []
    monkeypatch.setattr(sdo, "synthesize_sdo_sequence", lambda args, *a, **kw: captured.append(args))
    main(["synthesize", "--source-path", str(tmp_path), "--out", str(tmp_path / "out"),
          "--iterations", "0", "--stride", "2", flag, "4", "--device", "cpu"])
    assert captured[0].stride == 2
    assert captured[0].binning == 4
