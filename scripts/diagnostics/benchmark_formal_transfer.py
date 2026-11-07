"""Compare parallel and sequential transfer with identical frozen LTE layers.

Run with PYTHONPATH=src. CUDA timings synchronize both passes and report peak
allocated memory. This isolates transfer, not atmosphere/opacity/instrument cost.
"""
import argparse
import json
import statistics
import time

import torch

from prom3theus.rt.formal_transfer import formal_stokes, affine_transfer_reduce
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
    coefficients = (.03*torch.randn(*shape, 6, device=device, dtype=dtype)).requires_grad_()
    def propagation():
        a, b, c, x, y, z = coefficients.unbind(-1)
        diagonal = 1 + coefficients[..., :3].norm(dim=-1)
        return torch.stack((diagonal,a,b,c,a,diagonal,z,-y,b,-z,diagonal,x,
                            c,y,-x,diagonal),-1).reshape(*shape,4,4)
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
        return reference(propagation(), source, index, path=RayDistancePath(distance))

    def parallel():
        return formal_stokes(propagation(), source, distance, precision=args.precision, polarized=True)

    expected, actual = serial(), parallel()
    torch.testing.assert_close(actual, expected)
    error = float((actual-expected).detach().abs().max())
    del actual, expected
    result = {"settings": vars(args), "torch": torch.__version__, "max_absolute_error": error,
              "serial_full": measure(serial, (coefficients, source)),
              "tree_full": measure(parallel, (coefficients, source))}
    k = propagation()
    a = torch.matrix_exp(-.5*(k[:, :-1]+k[:, 1:])*(distance[1:]-distance[:-1])[None, :, None, None, None]).detach().requires_grad_()
    s = (.5*(source[:, :-1]+source[:, 1:])).detach().requires_grad_()
    boundary = source[:, -1].detach().requires_grad_()
    result["serial_reduction"] = measure(lambda: _propagate_layers(a, s, boundary), (a, s, boundary))
    result["tree_reduction"] = measure(lambda: affine_transfer_reduce(a, s, boundary), (a, s, boundary))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
