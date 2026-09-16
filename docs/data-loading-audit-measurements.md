# Data-loading audit measurements

Measured 2026-09-11. See the [complete refactoring plan](data-loading-refactor-plan.md).
These are reproducible microbenchmarks and diagnostic probes, not production throughput
or a claim about total training speed.

## Environment and method

- macOS 14.6.1, arm64; Python 3.11.15; PyTorch 2.12.0; NumPy 2.4.6.
- One PyTorch CPU thread. CUDA unavailable.
- Synthetic NPY pool: 65,536 samples × six fields × 16 float32 values = 24 MiB payload.
- Read all fixed slices in seeded random batch order, three repeats per batch size.
- Current path: `TensorDiskDataset.read`, opening and closing six NPY mappings per batch.
- Comparison: six process-owned read-only mappings, reused for the pass. Each read still
  makes an owned NumPy copy and returns a Torch tensor in the same dictionary structure.
- First-batch values are checked for equality. The probe is not a numerical test of an
  integrated replacement loader. Mapping setup is outside the timed prototype loop.
- Files have just been written, so data is warm in the OS cache. This intentionally
  isolates Python/header/mapping overhead. It does not test cold reads, network storage,
  LRU eviction, concurrent ranks, full prefetch, pinned memory, or GPU transfers.
- Timings are observational, with three repeats rather than a statistical confidence
  interval. See all individual measurements in the JSON; benchmark order is fixed.

## Warm-cache read cost

Times below are median milliseconds for one complete pool pass, not one batch.

| Batch size | Batches/pass | Current path (ms) | Reused mappings (ms) | Current / prototype |
| --- | --- | --- | --- | --- |
| 64 | 1,024 | 1426.83 | 13.92 | 102.5× |
| 256 | 256 | 288.79 | 4.15 | 69.5× |
| 4,096 | 16 | 17.34 | 0.85 | 20.3× |

The large ratio is specific to repeated mapping of many small slices. It supports
prioritizing a reader that owns mappings. It does not imply an equivalent optimization
speedup when GPU physics or cold storage dominates a training step.

## Reproduced behavior

| Probe | Observed result | Interpretation |
| --- | --- | --- |
| Save a tensor already mapping an NPY | One new 65,664-byte NPY, equal to the native file size | Snapshot persistence duplicates existing data |
| Shape/dtype/numel on a restored tensor | Zero mapping calls | Basic metadata inspection is cheap |
| 100 slice copies on a restored tensor | 100 mapping calls | Tensor operations repeatedly reopen the mapping |
| 100 native slab cache hits | 400 mapping calls; only one bulk payload read including initial fill | Row-size inspection maps fields even when the packed slab is cached |
| In-place update followed by reread | Update is lost | The tensor wrapper does not provide ordinary in-place tensor semantics |
| Restore module after deleting its payload | Restore succeeds, first slice raises `FileNotFoundError` | Load does not validate dependency completeness |
| Nonzero rank with rebuild and existing file, no process group | Returns the old generation immediately | Worker path does not wait for rebuild publication |
| One training read under an instrumented read semaphore | Zero semaphore entries | `reader_workers` does not control this path |
| Two independently restored training loaders | Identical first batch | There is no rank partition in the loader schedule |
| Prepare, close, prepare a persistent loader | `TypeError: object of type 'NoneType' has no len()` | Closing destroys both possible sources for reopening |

No runtime code was changed to obtain these results. Some probes deliberately exercise
unsupported or unsafe lifecycle cases to expose missing contracts; they are not claims
that each failure already occurs in every normal training run.

## Test runs

The first command completed with 149 passed, four failed, one CUDA skip. The second
completed with 74 passed. Both used the existing `nf2` environment. Worker tests required
execution outside the desktop filesystem sandbox for PyTorch shared-memory support.

```bash
PYTHONPATH=src conda run -n nf2 pytest \
  tests/observations tests/application tests/training \
  tests/artifacts/test_stream_state.py -q --disable-warnings

PYTHONPATH=src conda run -n nf2 pytest \
  tests/instruments/test_hmi_observation.py \
  tests/instruments/test_hmi_header_scan.py \
  tests/instruments/test_aia_observation.py \
  tests/instruments/test_registry_and_operators.py \
  tests/inversion/test_shell_sampling.py tests/test_architecture.py \
  -q --disable-warnings

PYTHONPATH=src conda run -n nf2 python \
  docs/assets/data-loading-audit/probe.py
```

[Per-test results](assets/data-loading-audit/test-results.json),
[probe source](assets/data-loading-audit/probe.py), and
[raw timing/probe results](assets/data-loading-audit/probe-results.json)
are saved with this report. The main plan classifies all four failures and includes
follow-up acceptance criteria. Real GPU validation and representative production-data
measurements remain necessary before implementation performance can be signed off.
