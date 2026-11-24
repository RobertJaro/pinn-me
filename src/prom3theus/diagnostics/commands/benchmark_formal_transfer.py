"""Compare parallel and sequential transfer with identical frozen LTE layers.

Run with PYTHONPATH=src. CUDA timings synchronize both passes and report peak
allocated memory. This isolates transfer, not atmosphere/opacity/instrument cost.
"""
import argparse
import json
import statistics
import time

import torch

from prom3theus.core.validation import skip_tensor_validation
from prom3theus.rt.formal_transfer import formal_stokes, affine_transfer_reduce
from prom3theus.rt.polarized_exponential import polarized_matrix_exp, propagation_matrix
from prom3theus.validation.reference_transfer import PolarizedFormalSolver, RayDistancePath, _propagate_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rays", type=int, default=4)
    parser.add_argument("--depth", type=int, default=257)
    parser.add_argument("--wavelengths", type=int, default=163)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(41)
    device, dtype = torch.device(args.device), getattr(torch, args.precision)
    shape = (args.rays, args.depth, args.wavelengths)
    coefficients = .03 * torch.randn(*shape, 7, device=device, dtype=dtype)
    coefficients[..., 0] = 1 + coefficients[..., 1:4].norm(dim=-1)
    coefficients.requires_grad_()
    source = torch.randn(*shape, 4, device=device, dtype=dtype).requires_grad_()
    distance = torch.linspace(0, 10, args.depth, device=device, dtype=dtype)
    index = torch.arange(args.depth, device=device, dtype=dtype)
    reference = PolarizedFormalSolver()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(fn, inputs):
        samples = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(args.repeats + 2):
            synchronize()
            start = time.perf_counter()
            with skip_tensor_validation():
                result = fn()
            synchronize()
            forward = time.perf_counter()
            torch.autograd.grad(result.square().mean(), inputs)
            synchronize()
            end = time.perf_counter()
            if step >= 2:
                samples.append((forward-start, end-forward, end-start))
        return dict(zip(("forward_ms", "backward_ms", "total_ms"),
                        [1000*statistics.median(s[i] for s in samples) for i in range(3)]),
                    peak_allocated_bytes=(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None))

    def serial():
        return reference(propagation_matrix(coefficients), source, index, path=RayDistancePath(distance))

    def prefix():
        dense = propagation_matrix(coefficients)
        midpoint = .5 * (dense[:, :-1] + dense[:, 1:])
        width = distance.diff()[None, :, None, None, None]
        attenuation, increment = polarized_matrix_exp(-midpoint * width, return_increment=True)
        return affine_transfer_reduce(attenuation, .5 * (source[:, :-1] + source[:, 1:]),
                                      source[:, -1], increment=increment)

    def sweep():
        return formal_stokes(coefficients, source, distance, precision=args.precision, polarized=True)

    with torch.no_grad():
        expected, actual = serial(), sweep()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, prefix())
    error = float((actual-expected).detach().abs().max())
    del actual, expected
    result = {"settings": vars(args), "torch": torch.__version__, "max_absolute_error": error,
              "serial_full": measure(serial, (coefficients, source)),
              "prefix_full": measure(prefix, (coefficients, source)),
              "vector_sweep_full": measure(sweep, (coefficients, source))}
    k = propagation_matrix(coefficients.detach())
    a = torch.matrix_exp(-.5*(k[:, :-1]+k[:, 1:])*(distance[1:]-distance[:-1])[None, :, None, None, None]).detach().requires_grad_()
    s = (.5*(source[:, :-1]+source[:, 1:])).detach().requires_grad_()
    boundary = source[:, -1].detach().requires_grad_()
    result["serial_reduction"] = measure(lambda: _propagate_layers(a, s, boundary), (a, s, boundary))
    result["prefix_integration"] = measure(lambda: affine_transfer_reduce(a, s, boundary), (a, s, boundary))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
