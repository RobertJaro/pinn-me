"""Benchmark shared line profiles, optionally with targeted torch.compile.

Run on the training GPU with, for example:
python -m pme.evaluation.benchmark_profiles --device cuda --batch-size 32768 --compile
This measures one profile pair, not full training. Compilation time is separate.
"""

import argparse
import json
import statistics
import time

import torch

from pme.train.profile_functions import FaradayVoigt, Voigt


def benchmark(device='cpu', batch_size=1024, wavelengths=6, iterations=20, compile_pair=False):
    device = torch.device(device)
    absorption, dispersion = Voigt().to(device), FaradayVoigt().to(device)
    x = torch.linspace(-8, 8, batch_size * wavelengths, device=device).reshape(batch_size, wavelengths)
    inputs = (x.requires_grad_(), torch.ones(batch_size, 1, device=device).requires_grad_(),
              torch.full((batch_size, 1), .2, device=device, requires_grad=True))

    def separate(*args):
        return absorption(*args), dispersion(*args)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def evaluate(function, function_inputs):
        values = function(*function_inputs)
        gradients = torch.autograd.grad(sum(v.square().mean() for v in values), function_inputs)
        return values, gradients

    reference = evaluate(separate, inputs)
    unit_inputs = (inputs[0], inputs[2])
    unit_reference = evaluate(lambda x, gamma: separate(x, torch.ones_like(x), gamma), unit_inputs)
    functions = {'separate': (separate, inputs, reference),
                 'shared': (absorption.forward_pair, inputs, reference),
                 'physical_unit': (absorption.forward_pair_unit, unit_inputs, unit_reference)}
    if compile_pair:
        functions['compiled_shared'] = (torch.compile(absorption.forward_pair, fullgraph=True), inputs, reference)
    results = {'device': str(device), 'torch': torch.__version__, 'shape': list(x.shape), 'timings': {}}
    for name, (function, function_inputs, expected_result) in functions.items():
        try:
            synchronize()
            start = time.perf_counter()
            values, gradients = evaluate(function, function_inputs)
            synchronize()
            setup = time.perf_counter() - start
            for actual, expected in zip((*values, *gradients), (*expected_result[0], *expected_result[1])):
                torch.testing.assert_close(actual, expected, rtol=3e-4, atol=1e-7)
            for _ in range(3):
                evaluate(function, function_inputs)
            synchronize()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            durations = []
            for _ in range(iterations):
                synchronize()
                start = time.perf_counter()
                evaluate(function, function_inputs)
                synchronize()
                durations.append((time.perf_counter() - start) * 1000)
            results['timings'][name] = {'forward_backward_ms': statistics.median(durations),
                                       'first_call_seconds': setup,
                                       'outputs_and_gradients_match': True}
            if device.type == 'cuda':
                results['timings'][name]['peak_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 2**20
        except Exception as error:
            if name != 'compiled_shared':
                raise
            results['timings'][name] = {'error': f'{type(error).__name__}: {error}'}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--wavelengths', type=int, default=6,
                        help='Use the response-integrated wavelength count for realistic memory sizing.')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.wavelengths, args.iterations, args.threads) < 1:
        parser.error('Sizes, iterations, and threads must be positive.')
    torch.set_num_threads(args.threads)
    print(json.dumps(benchmark(args.device, args.batch_size, args.wavelengths,
                               args.iterations, args.compile), indent=2))
