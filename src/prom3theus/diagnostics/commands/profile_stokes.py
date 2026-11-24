"""Profile the numerical reference composition on in-memory HMI rays.

This isolates computation, not file preparation, loading, Lightning or plotting.
This profiles the independent sequential reference. Use benchmark_formal_transfer.py
for the production parallel integrator.
Uses idealized HMI responses and an initialized atmosphere, not a fitted spot.
Example: PYTHONPATH=src python -m prom3theus.diagnostics.commands.profile_stokes --device cuda
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
import json
import platform
import statistics
import time

import torch

from prom3theus.inversion.depth_sampling import DepthRefinement
from prom3theus.validation.reference_forward import ForwardRuntime, LTEForwardComposition, LTESynthesisBackend
from prom3theus.muram.hmi_filter import build_idealized_hmi_filter
from prom3theus.validation.reference_synthesis import LTESynthesizer
from prom3theus.rt import StratifiedAtmosphereModel


@contextmanager
def scopes(targets, timings, sync):
    originals = []
    for owner, attribute, label in targets:
        original = getattr(owner, attribute)
        originals.append((owner, attribute, original))
        def wrapped(*args, _original=original, _label=label, **kwargs):
            sync()
            start = time.perf_counter()
            with torch.profiler.record_function(_label):
                result = _original(*args, **kwargs)
            sync()
            timings[_label] = timings.get(_label, 0.) + time.perf_counter()-start
            return result
        setattr(owner, attribute, wrapped)
    try:
        yield
    finally:
        for owner, attribute, original in originals:
            setattr(owner, attribute, original)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/stokes_performance/profile'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--coarse', type=int, default=128)
    parser.add_argument('--fine', type=int, default=64)
    parser.add_argument('--spectral', type=int, default=161)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--representation', choices=['direct', 'vector_potential'], default='vector_potential')
    parser.add_argument('--profile', action='store_true', help='Also record one separate instrumented step.')
    parser.add_argument('--legacy-indexing', action='store_true', help='Benchmark the old recurrence for an accuracy-preserving before/after comparison.')
    parser.add_argument('--split-backward', action='store_true', help='Separately time RT and atmospheric-network backward using the physical field interface.')
    args = parser.parse_args()
    if min(args.batch, args.width, args.layers, args.threads, args.repeats) < 1 or args.coarse < 2 or args.fine < 0:
        parser.error('Positive sizes required; coarse >= 2 and fine >= 0.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if args.legacy_indexing:
        import prom3theus.validation.reference_transfer as transfer
        def indexed(attenuation, source, stokes):
            for i in range(source.shape[-3]-1, -1, -1):
                layer = source[..., i, :, :]
                stokes = layer + (attenuation[..., i, :, :, :] @ (stokes-layer).unsqueeze(-1)).squeeze(-1)
            return stokes
        transfer._propagate_layers = indexed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = StratifiedAtmosphereModel(
        shell_height_bounds_megameter=(1.5, -1.), time_dependent=True,
        height_input_scale_m=1e7, uniform_spatial_scaling=True,
        temperature_log_scale=.1, gas_pressure_log_scale=.5,
        microturbulence_log_scale=.2, magnetic_scale_gauss=100.,
        magnetic_representation=args.representation,
        scene_geometry_config={'solar_radius_m': 695700000., 'scene_basis': torch.eye(3)},
        model_config={'type': 'siren', 'dim': args.width, 'n_layers': args.layers,
                      'first_omega_0': 5., 'hidden_omega_0': 1., 'radial_weighting_config': None},
    ).to(device)
    instrument, observed, weights, continuum = build_idealized_hmi_filter(n_quadrature=args.spectral, dtype=torch.float32)
    synth = LTESynthesizer(line_ids=['FeI_6173.3352'])
    composition = LTEForwardComposition(
        atmosphere_model=model, backend=LTESynthesisBackend(synth), instrument=instrument,
        velocity_synthesis_mode='carrington_observer_relative', height_sampling='uniform',
        depth_refinement=DepthRefinement(args.fine > 0, max(args.fine, 1), .05),
    )
    wave = composition.prepare_wavelength_grid(observed)
    # Prepare spectral differences before casting absolute wavelengths.
    synth.to(device=device, dtype=torch.float32)
    instrument.to(device=device, dtype=torch.float32)
    runtime = ForwardRuntime(torch.linspace(-5, 1, args.coarse, device=device),
                             observed.to(device), wave.to(device), 1e13, 0., 0.)
    coords = torch.zeros(args.batch, 3, device=device)
    coords[:, 0] = torch.linspace(-20, 20, args.batch, device=device)
    rays = torch.tensor([0., 0., -1.], device=device).expand(args.batch, -1)
    basis = torch.eye(3, device=device).expand(args.batch, -1, -1)
    response = {'spectral_weights': weights.to(device).expand(args.batch, *weights.shape),
                'continuum_weights': continuum.to(device).expand(args.batch, *continuum.shape)}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    component_weights = torch.tensor([1., 200., 200., 200.], device=device)
    last_prediction = None
    formal_inputs = {}
    if args.split_backward:
        original_formal = synth.formal_solver.forward
        def capture_formal(matrix, source, *args, **kwargs):
            formal_inputs.update(matrix=matrix, source=source)
            return original_formal(matrix, source, *args, **kwargs)
        synth.formal_solver.forward = capture_formal

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def step():
        nonlocal last_prediction
        optimizer.zero_grad(set_to_none=True)
        formal_inputs.clear()
        sync(); start = time.perf_counter()
        with torch.profiler.record_function('step.forward'):
            result = composition.synthesize(coords, runtime=runtime, ray_direction=rays,
                stokes_basis=basis, observer_los_velocity_m_per_s=torch.zeros(args.batch, device=device),
                instrument_response=response, return_atmosphere=args.split_backward)
            prediction = result['stokes']
            # A fixed residual drives all Stokes channels without fitting a dataset.
            loss = ((prediction - .01).square() * component_weights[None, :, None]).mean()
            last_prediction = prediction.detach()
        sync(); forward_done = time.perf_counter()
        split_times = {}
        with torch.profiler.record_function('step.backward'):
            if args.split_backward:
                atmosphere = result['atmosphere']
                fields = tuple(getattr(atmosphere, name) for name in (
                    'temperature', 'gas_pressure', 'microturbulence', 'velocity_field', 'magnetic_field'))
                transfer_inputs = (formal_inputs['matrix'], formal_inputs['source'])
                transfer_gradients = torch.autograd.grad(loss, transfer_inputs)
                sync(); formal_done = time.perf_counter()
                gradients = torch.autograd.grad(transfer_inputs, fields, grad_outputs=transfer_gradients)
                sync(); transfer_done = time.perf_counter()
                torch.autograd.backward(fields, gradients)
                sync(); atmosphere_done = time.perf_counter()
                split_times = {'backward_formal_s': formal_done-forward_done,
                               'backward_opacity_source_s': transfer_done-formal_done,
                               'backward_transfer_s': transfer_done-forward_done,
                               'backward_atmosphere_s': atmosphere_done-transfer_done}
            else:
                loss.backward()
        sync(); backward_done = time.perf_counter()
        # Time optimizer work but hold the atmosphere fixed across repetitions.
        # lr=0 preserves parameters while still exercising Adam state operations.
        with torch.profiler.record_function('step.optimizer'):
            optimizer.step()
        sync(); done = time.perf_counter()
        return {**split_times, 'forward_s': forward_done-start, 'backward_s': backward_done-forward_done,
                'optimizer_s': done-backward_done, 'total_s': done-start,
                'loss': float(loss.detach()),
                'parameter_gradient_l2': float(torch.stack([p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt()),
                'finite_gradients': all(
                    p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())}

    optimizer.param_groups[0]['lr'] = 0.
    step()  # Untimed warmup: populate caches and Adam state.
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    rows = [step() for _ in range(args.repeats)]
    result = {'environment': {'platform': platform.platform(), 'torch': str(torch.__version__),
              'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor()},
              'settings': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'synthesis_wavelength_count': wave.numel(), 'rows': rows,
              'control_profiles': last_prediction.cpu().tolist(),
              'median': {key: statistics.median(row[key] for row in rows) for key in rows[0] if key.endswith('_s')},
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    # Persist timings before optional profiler aggregation, which can be expensive.
    Path(str(args.output)+'.json').write_text(json.dumps(result, indent=2)+'\n')
    targets = [(model, 'trace_rays', 'forward.coarse_atmosphere'),
                   (torch, 'matrix_exp', 'forward.matrix_exp'),
                   (composition, '_refine_ray_sampling', 'forward.refinement_including_fine_atmosphere'),
                   (synth, 'refinement_extinction', 'forward.proposal_opacity'),
                   (synth, 'forward', 'forward.synthesis'),
                   (synth.continuum_opacity, 'forward', 'forward.continuum_opacity'),
                   (synth.line_opacity, 'forward', 'forward.line_opacity'),
                   (synth.formal_solver, 'forward', 'forward.formal_solver'),
               (instrument, 'forward', 'forward.instrument')]
    stage_timings = {}
    with scopes(targets, stage_timings, sync):
        step()
    result['instrumented_forward_regions_s'] = stage_timings
    Path(str(args.output)+'.json').write_text(json.dumps(result, indent=2)+'\n')
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profiler:
            step()
        events = profiler.key_averages()
        Path(str(args.output)+'.operators.txt').write_text(events.table(sort_by='self_cpu_time_total', row_limit=40))
        if device.type == 'cuda':
            Path(str(args.output)+'.cuda_operators.txt').write_text(events.table(sort_by='self_device_time_total', row_limit=40))
        result['profile_cpu_regions_ms'] = {e.key: e.cpu_time_total/1000 for e in events if e.key.startswith(('forward.', 'step.'))}
        result['profile_cpu_scalar_extractions'] = sum(e.count for e in events if e.key == 'aten::_local_scalar_dense')
    Path(str(args.output)+'.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
