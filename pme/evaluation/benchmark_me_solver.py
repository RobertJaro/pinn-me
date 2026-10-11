"""Compare eager/compiled real ME algebra and complete monochromatic synthesis.

Example: python -m pme.evaluation.benchmark_me_solver --device cuda --batch-size 32768
Compilation and forward/backward warm-up are excluded from steady-state timings.
"""
import argparse
import json
import statistics
import time

import torch

from pme.instrument import HMI_WAVELENGTH_CENTER
from pme.train.me_atmosphere import HMIMEAtmosphere, MEAtmosphere, compiled_profile_solver


def physical_inputs(batch_size, wavelengths, device, dtype=torch.float32):
    # Deterministic, varied atmospheres including disk-center/limb and weak fields.
    column = torch.linspace(0, 1, batch_size, device=device, dtype=dtype)[:, None]
    parameters = dict(b_field=10 + 1990 * column, cos_inc=.1 + .8 * column,
                      vmac=1500 + 1000 * column, damping=.01 + .99 * column,
                      b0=.8 - .3 * column, b1=.2 + .3 * column,
                      mu=.05 + .95 * column, vdop=-3000 + 6000 * column,
                      kl=5 + 20 * column)
    parameters['sin_inc2'] = 1 - parameters['cos_inc'].square()
    parameters['sin_inc2_cos2azi'] = parameters['sin_inc2'] * torch.cos(column * 4)
    parameters['sin_inc2_sin2azi'] = parameters['sin_inc2'] * torch.sin(column * 4)
    parameters = {key: value.detach().requires_grad_() for key, value in parameters.items()}
    grid = torch.linspace(-.2, .2, wavelengths, device=device, dtype=dtype)[None] * 1e-10
    # Match the per-profile grid layout in training.
    return parameters, grid.expand(batch_size, -1).contiguous()


def solver_inputs(atmosphere, parameters, grid):
    width = atmosphere.d_lambda(**parameters)
    profiles = atmosphere.calculate_voigt_faraday_profiles(
        nu=grid / width, nu_m=atmosphere.nu_m(**parameters, d_lambda=width),
        damping=parameters['damping'], lambda_dop=atmosphere.lambda_dop(**parameters, d_lambda=width))
    keys = ('sin_inc2', 'cos_inc', 'sin_inc2_cos2azi', 'sin_inc2_sin2azi', 'kl', 'b0', 'b1', 'mu')
    return {key: value.detach().requires_grad_()
            for key, value in {**profiles, **{k: parameters[k] for k in keys}}.items()}


def benchmark(device='cpu', batch_size=1024, wavelengths=54, iterations=20, mode='default'):
    device = torch.device(device)
    atmosphere = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER).to(device)
    parameters, grid = physical_inputs(batch_size, wavelengths, device)
    inputs = solver_inputs(atmosphere, parameters, grid)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def evaluate(function, values):
        result = function(**values)
        gradients = torch.autograd.grad(sum(x.square().mean() for x in result), tuple(values.values()))
        return tuple(result) + gradients

    def full(**values):
        return atmosphere(wavelength_grid=grid, **values)

    results = {'device': str(device), 'torch': torch.__version__, 'mode': mode,
               'shape': [batch_size, wavelengths], 'timings': {}}
    for region, eager, compiled, values in (
            ('solver', MEAtmosphere._solve_profiles, compiled_profile_solver(mode), inputs),
            ('synthesis', full, full, parameters)):
        reference = None
        for variant, function in (('eager', eager), ('compiled', compiled)):
            atmosphere._compile_solver_mode = mode if variant == 'compiled' else None
            sync()
            start = time.perf_counter()
            actual = evaluate(function, values)
            sync()
            first_call = time.perf_counter() - start
            if reference is None:
                reference = tuple(x.detach().clone() for x in actual)
            else:
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, rtol=5e-4, atol=1e-7)
            del actual
            for _ in range(3):
                evaluate(function, values)
            sync()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            times = []
            for _ in range(iterations):
                sync()
                start = time.perf_counter()
                evaluate(function, values)
                sync()
                times.append((time.perf_counter() - start) * 1000)
            timing = dict(forward_backward_ms=statistics.median(times), first_call_seconds=first_call,
                          outputs_and_gradients_match=True)
            if device.type == 'cuda':
                timing['peak_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 2**20
            results['timings'][f'{region}_{variant}'] = timing
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--wavelengths', type=int, default=54)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--mode', default='default', choices=(
        'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'))
    args = parser.parse_args()
    if min(args.batch_size, args.wavelengths, args.iterations, args.threads) < 1:
        parser.error('Sizes, iterations and threads must be positive.')
    torch.set_num_threads(args.threads)
    print(json.dumps(benchmark(args.device, args.batch_size, args.wavelengths, args.iterations, args.mode), indent=2))
