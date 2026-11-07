"""Measure the production LTE forward/backward on fixed stored synthetic pixels.

Use a Stokes-only joint configuration. No optimizer updates or checkpoints are
written; this measures the full atmosphere/opacity/transfer/instrument graph.
Run actual fitting with run_local_hinode.py using a separate output directory.
"""

import argparse
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import torch

from prom3theus.muram import config as _register_muram_schema
from prom3theus.application.joint_assembly import build_joint_runtime
from prom3theus.config import load_config
from prom3theus.inversion.data_terms.stokes import StokesObservationTerm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--profile", action="store_true",
                        help="Record one additional step's operator timings; adds profiling overhead")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or (args.chunk_size is not None and args.chunk_size < 1):
        parser.error("repeats, threads, and chunk-size must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    runtime = build_joint_runtime(load_config(args.configuration))
    model = runtime.model
    if len(model.terms) != 1 or model.shared_objectives:
        parser.error("Use one Stokes stream with shared physics/regularization disabled")
    stream_id, term = next(iter(model.terms.items()))
    if not isinstance(term, StokesObservationTerm):
        parser.error("The configured stream must synthesize LTE Stokes")
    window = term.synthesis_window
    if args.chunk_size is not None:
        window.wavelength_chunk_size = args.chunk_size
    data = runtime.streams[stream_id].data_module
    dataset = data.evaluation_dataset()
    if any(index < 0 or index >= len(dataset) for index in args.pixels):
        parser.error("Pixel indices must lie inside the evaluation dataset")
    batch = data.evaluation_collate_fn()([dataset[index] for index in args.pixels])

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(runtime.device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    batch = move(batch)

    def synchronize():
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)

    counts = dict(network=0, plasma=0, plasma_points=0, line_preparations=0)
    def count_network(*unused):
        counts["network"] += 1
    handle = model.atmosphere_model.network.register_forward_hook(count_network)
    provider = window.coefficients.plasma_provider
    original = provider.prepare_plasma_state
    def prepare(temperature, pressure):
        counts["plasma"] += 1
        counts["plasma_points"] += temperature.numel()
        return original(temperature, pressure)
    provider.prepare_plasma_state = prepare
    line_provider = window.coefficients.line_opacity
    original_line_prepare = line_provider.prepare
    def prepare_lines(*args, **kwargs):
        counts["line_preparations"] += 1
        return original_line_prepare(*args, **kwargs)
    line_provider.prepare = prepare_lines
    samples = []
    try:
        for step in range(args.repeats + 1):
            model.zero_grad(set_to_none=True)
            counts.update(network=0, plasma=0, plasma_points=0, line_preparations=0)
            if step == 1 and runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runtime.device)
            synchronize()
            start = time.perf_counter()
            result = model.evaluate_batches({stream_id: batch}, diagnostics=False)
            synchronize()
            forward_end = time.perf_counter()
            result.loss.backward()
            synchronize()
            end = time.perf_counter()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite inversion gradients")
            if step:
                samples.append((forward_end - start, end - forward_end, end - start))
    finally:
        handle.remove()
        provider.prepare_plasma_state = original
        line_provider.prepare = original_line_prepare
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "configuration": str(args.configuration.resolve()),
        "scope": "Training forward/backward; excludes optimizer, loading and detailed fit diagnostics",
        "gpu": torch.cuda.get_device_name(runtime.device) if runtime.device.type == "cuda" else None,
        "seed": args.seed, "device": str(runtime.device), "torch": torch.__version__,
        "threads": args.threads, "pixels": args.pixels, "depth_count": window.depth_count,
        "jitter_fraction": window.jitter_fraction,
        "wavelength_count": term.synthesis_wavelength_angstrom.numel(),
        "chunk_size": window.wavelength_chunk_size, "repeats": args.repeats,
        "calls_per_forward": counts, "loss": float(result.loss.detach()),
        "median_seconds": dict(zip(("forward", "backward", "total"),
            (statistics.median(sample[index] for sample in samples) for index in range(3)))),
        "source_signature": runtime.streams[stream_id].prepared.source_signature,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated(runtime.device)
            if runtime.device.type == "cuda" else None),
    }
    torch.save(result.streams[stream_id].diagnostics["prediction"].detach().cpu(),
               args.output / "prediction.pt")
    if args.profile:
        model.zero_grad(set_to_none=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if runtime.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profiler:
            model.evaluate_batches({stream_id: batch}, diagnostics=False).loss.backward()
            synchronize()
        profiler.export_chrome_trace(str(args.output / "trace.json"))
        key = "self_device_time_total" if runtime.device.type == "cuda" else "self_cpu_time_total"
        (args.output / "operators.txt").write_text(
            profiler.key_averages().table(sort_by=key, row_limit=30))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
