"""Run the bounded local HMI LTE -> spherical potential-field baseline.

Run with PYTHONPATH=src and a Python environment with the observations extras.
Calibration preparation is a separate, explicit operation; this script is offline.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def fit_report(initial, prediction, target):
    """Quantify a fixed sample's fit, rejecting a zero-polarization solution.

    Arrays use [pixel, IQUV, wavelength] in atlas-continuum units. These are
    explicit baseline acceptance limits, not observational uncertainty estimates.
    The sample may also occur in training; it is not a held-out generalization test.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in (initial, prediction, target)]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("Stokes comparison arrays must have matching shapes")
    if arrays[0].ndim != 3 or arrays[0].shape[1] != 4 or min(arrays[0].shape) < 1:
        raise ValueError("Stokes comparisons require nonempty [pixel,4,wavelength] arrays")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Non-finite Stokes comparisons cannot validate an inversion")
    initial, prediction, target = arrays
    rms = lambda value: np.sqrt(np.mean(value**2, axis=(0, 2)))
    before, after, signal = rms(initial - target), rms(prediction - target), rms(target)
    # Absolute I limit is 5% of atlas continuum. Weak Q/U/V need an absolute
    # floor, while stronger polarization must be fit better than zero by 25%.
    limits = np.maximum(0.002, 0.75 * signal)
    limits[0] = 0.05
    weights = np.array([10000.0, 390625.0, 390625.0, 390625.0])
    initial_loss, final_loss = float(weights @ before**2), float(weights @ after**2)
    checks = {
        "stokes_rms_within_limits": bool(np.all(after <= limits)),
        "weighted_loss_halved": final_loss <= 0.5 * initial_loss,
        "measurable_circular_polarization": bool(signal[3] > 0.002),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": len(target),
        "sample_role": "fixed diagnostic sample, also eligible for training",
        "units": "disk-center atlas continuum",
        "comparison_stokes_weights": dict(zip("IQUV", weights.tolist())),
        "initial_weighted_mse": initial_loss,
        "final_weighted_mse": final_loss,
        "components": {
            component: {
                "initial_rms": float(before[index]),
                "final_rms": float(after[index]),
                "observed_rms": float(signal[index]),
                "acceptance_rms": float(limits[index]),
            }
            for index, component in enumerate("IQUV")
        },
    }


def representative_pixel_indices(spatial_shape):
    """Sample this 256-square crop on a 32-square lattice offset by four pixels.

    Coordinates are [row, column]. The lattice has no overlap with the primary
    16-pixel validation lattice and remains eligible for training.
    """
    if tuple(spatial_shape) != (256, 256):
        raise ValueError("The local representative sample requires a 256 x 256 crop")
    rows, columns = np.meshgrid(np.arange(4, 256, 8), np.arange(4, 256, 8), indexing="ij")
    return np.stack((rows, columns), axis=-1).reshape(-1, 2)


def representative_batch(runtime, stream_id):
    """Read the offset lattice from the existing calibrated observation raster."""
    import torch
    from prom3theus.application.runtime import _move_value
    from prom3theus.observations.dataset import ObservationPixelDataset, ObservationResponseCollator

    evaluation = runtime.streams[stream_id].data_module.evaluation_dataset()
    indices = representative_pixel_indices(evaluation.raster.spatial_shape)
    dataset = ObservationPixelDataset(
        evaluation.raster,
        auxiliary_fields=evaluation.auxiliary_fields,
        include_surface_position=evaluation.include_surface_position,
        include_pixel_index=True,
        pixel_indices=torch.from_numpy(indices),
    )
    batch = ObservationResponseCollator()([dataset[index] for index in range(len(dataset))])
    return _move_value(batch, runtime.device)


def predict_stokes_sample(model, stream_id, batch, *, batch_size=128):
    """Predict bounded batches in deterministic eval mode, restoring prior mode."""
    import torch
    from prom3theus.application.batches import _batch_length, _indexed_batch

    count = _batch_length(batch)
    if batch_size < 1 or count < 1:
        raise ValueError("Stokes prediction requires positive sample and batch sizes")
    was_training = model.training
    model.eval()
    try:
        predictions = []
        with torch.no_grad():
            for start in range(0, count, batch_size):
                indices = torch.arange(start, min(start + batch_size, count),
                                       device=batch["stokes"].device)
                prediction = model.terms[stream_id].predict(_indexed_batch(batch, indices))
                predictions.append(prediction.detach().cpu().numpy())
        return np.concatenate(predictions)
    finally:
        model.train(was_training)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hmi_local_constant.yaml",
                        help="configuration for this same local HMI crop")
    parser.add_argument("--initialize-from-state", type=Path,
                        help="copy compatible P3S network weights into a new run with a fresh optimizer")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="total steps, including a resumed run")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.steps is not None and args.steps < 1):
        parser.error("threads and steps must be positive")

    import torch
    from prom3theus.config import load_config
    from prom3theus.preprocess.hmi import preprocess_hmi_time_series
    from prom3theus.instruments.hmi.response import load_response_profile, resolve_response_profile
    from prom3theus.instruments.hmi.acquisition import resolve_acquisition_groups

    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    if args.steps is not None:
        config = replace(config, training=replace(config.training, max_steps=args.steps))
    factories = None
    initialization = {"type": "seeded_unfitted_reference", "seed": 0}
    if args.initialize_from_state is not None and not args.prepare_only:
        from prom3theus.application.hmi_warm_start import build_hmi_warm_start_factories
        factories, initialization = build_hmi_warm_start_factories(config, args.initialize_from_state)
    output = config.solver.output_directory
    output.mkdir(parents=True, exist_ok=True)
    observation = config.streams[0].observation
    magnetic_reference = config.atmosphere.parameters.magnetic_field.reference_height_megameter
    report = {
        "format": "prom3theus.local_hmi_baseline.v1",
        "status": "preparing",
        "configuration_path": str(args.config.resolve()),
        "source_directory": str(ROOT / "data/hmi"),
        "photospheric_model": "LTE Stokes inversion",
        "magnetic_reference_height_megameter": magnetic_reference,
        "magnetic_height_dependence": "unrestricted" if magnetic_reference is None else "constant along radial columns",
        "initialization": initialization,
        "coronal_model": "spherical Neumann potential field, alpha=0",
        "coronal_top_megameter": 20.0,
        "limitations": [
            "The potential corona is a separate derived magnetic model; the LTE network ends at 1.5 Mm.",
            "Only inferred photospheric Br constrains the potential corona; tangential fields can differ at the interface.",
            "Source cells span the observation's angular bounding rectangle; the network fills unobserved corners.",
            "Br outside that rectangle is assumed zero; this is not a measured global solar map.",
            "Acceptance checks establish a numerical baseline, not agreement with an observed coronal field.",
        ],
    }
    # A calibration check must not replace a completed scientific result.
    report_path = output / ("preparation.json" if args.prepare_only else "report.json")
    _write_json(report_path, report)
    print("Preparing the local 256 x 256 bipolar HMI region", flush=True)
    report["preparation"] = preprocess_hmi_time_series(
        input_directory=ROOT / "data/hmi",
        output_directory=observation.directory,
        longitude_deg=213.2877,
        latitude_deg=-12.9481,
        width_pixels=256,
        height_pixels=256,
    )
    try:
        # Validate the actual acquisition assignment, file checksums, and local
        # profiles before assembling a forward model. No nominal substitute.
        for _, files in resolve_acquisition_groups(files=list(observation.directory.glob("*.fits"))):
            profile = resolve_response_profile(
                observation.calibration.transmission_profile_directory, files[0],
            )
            load_response_profile(profile)
    except (FileNotFoundError, KeyError, ValueError) as error:
        report.update(status="missing_or_invalid_calibration", calibration_error=str(error))
        _write_json(report_path, report)
        print(f"Calibration required: {error}\nStatus written to {report_path}", flush=True)
        return 2
    if args.prepare_only:
        report["status"] = "prepared"
        _write_json(report_path, report)
        print(f"prepared: {report_path}", flush=True)
        return 0

    from prom3theus.application.joint_assembly import build_joint_runtime
    from prom3theus.application.joint_evaluation import evaluate_joint_runtime
    from prom3theus.application.joint_training import run_joint_inversion
    from prom3theus.application.runtime import _json_value
    from prom3theus.diagnostics.coronal_potential import export_p3s_potential_corona
    from prom3theus.diagnostics.hmi_baseline_fit import export_hmi_baseline_fit

    if magnetic_reference is not None:
        report["limitations"].append(
            "Photospheric magnetism is assumed constant along radial columns; HMI does not uniquely determine its height profile."
        )
    torch.manual_seed(0)
    runtime = build_joint_runtime(config, factories=factories)
    setup = evaluate_joint_runtime(runtime, evaluate_quadrature=False)
    _write_json(output / "setup.json", _json_value(setup.report))
    name = config.scene.reference_stream
    initial = setup.evaluation.streams[name].diagnostics["prediction"].detach().cpu().numpy()
    target = runtime.validation_batches[name]["stokes"].detach().cpu().numpy()
    representative = representative_batch(runtime, name)
    representative_initial = predict_stokes_sample(runtime.model, name, representative)
    representative_target = representative["stokes"].detach().cpu().numpy()
    from prom3theus.diagnostics.initial_reference import preserve_initial_reference
    reference_path = output / "initial_reference.npz"
    reference, report["initialization"] = preserve_initial_reference(
        reference_path,
        {
            "initial": initial, "target": target,
            "pixel_index": runtime.validation_batches[name]["pixel_index"].cpu().numpy(),
            "representative_initial": representative_initial,
            "representative_target": representative_target,
            "representative_pixel_index": representative["pixel_index"].cpu().numpy(),
        },
        report["initialization"],
        resume=(output / "last.ckpt").exists() or config.training.resume_from_checkpoint is not None,
    )
    initial, representative_initial = reference["initial"], reference["representative_initial"]
    report["comparison_initial_reference"] = str(reference_path)
    report["status"] = "training"
    _write_json(report_path, report)
    torch.manual_seed(0)
    run = run_joint_inversion(config, factories=factories)
    run.module.model.eval()
    with torch.no_grad():
        prediction = run.module.model.terms[name].predict(runtime.validation_batches[name]).cpu().numpy()
    report["fit"] = fit_report(initial, prediction, target)
    representative_prediction = predict_stokes_sample(run.module.model, name, representative)
    report["representative_fit"] = {
        **fit_report(representative_initial, representative_prediction, representative_target),
        "sample_role": "spatially stratified diagnostic sample, eligible for training; not held out",
        "sampling": {
            "spatial_shape": [256, 256],
            "lattice_shape": [32, 32],
            "pixel_offset_yx": [4, 4],
            "pixel_step_yx": [8, 8],
        },
    }
    report["global_step"] = int(run.trainer.global_step)
    report["save_state"] = str(run.save_state_path)
    if magnetic_reference is not None:
        from prom3theus.diagnostics.magnetic_height import check_magnetic_height
        report["magnetic_height"] = check_magnetic_height(run.save_state_path, output / "magnetic_height")
    np.savez_compressed(output / "stokes_comparison.npz", initial=initial,
                        prediction=prediction, target=target,
                        pixel_index=runtime.validation_batches[name]["pixel_index"].cpu().numpy())
    np.savez_compressed(output / "representative_stokes_comparison.npz",
                        initial=representative_initial, prediction=representative_prediction,
                        target=representative_target,
                        pixel_index=representative["pixel_index"].cpu().numpy())
    report["status"] = "extrapolating"
    _write_json(report_path, report)
    report["fit_figures"] = export_hmi_baseline_fit(output / "stokes_comparison.npz", report_path)
    report["corona"] = export_p3s_potential_corona(
        run.save_state_path, output / "corona",
        source_grid_size=48, source_quadrature_order=4, horizontal_points=32,
        heights_megameter=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
        trace_count=16,
    )
    # The exporter records its physical/numerical acceptance checks explicitly.
    corona_passed = bool(report["corona"]["report"]["valid"])
    height_passed = report.get("magnetic_height", {"report": {"passed": True}})["report"]["passed"]
    passed = report["fit"]["passed"] and report["representative_fit"]["passed"] and corona_passed and height_passed
    report["status"] = "passed" if passed else "failed_validation"
    _write_json(report_path, report)
    print(f"{report['status']}: {report_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
