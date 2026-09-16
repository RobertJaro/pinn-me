import pytest
import torch
from prom3theus.training.transfer import prefetch_cuda


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for copy-stream validation"
)
def test_cuda_lookahead_matches_values_and_gradients():
    device = torch.device("cuda")
    batches = [
        {
            "x": torch.arange(n, dtype=torch.float32).pin_memory(),
            "nested": {"y": torch.ones(n, dtype=torch.float64).pin_memory()},
        }
        for n in (3, 8, 2, 9)
    ]

    def source():
        yield from batches

    parameter = torch.tensor(2.0, device=device, requires_grad=True)
    outputs = []
    for cpu, gpu in zip(batches, prefetch_cuda(source(), device), strict=True):
        outputs.append((gpu["x"] * parameter).sum())
        assert gpu["nested"]["y"].dtype == torch.float64
        torch.testing.assert_close(gpu["x"].cpu(), cpu["x"])
    sum(outputs).backward()
    assert parameter.grad.item() == sum(float(b["x"].sum()) for b in batches)
