# Investigating iteration time

Similar V100 and A100 iteration times can indicate host overhead, but timing
alone does not identify the limiting operation. The current training path has
several explicit tensor-to-Python decisions that synchronize CUDA execution.

The synchronization fixes batch existing validation flags without changing the
forward equations, tensor precision, or gradient checks:

- LTE atmospheric-field validation: nine scalar decisions become one.
- Polarized opacity input validation: thirteen scalar decisions on valid inputs
  become one small flag-vector transfer. Invalid inputs still identify the
  failed constraint.
- Post-backward gradient validation: one scalar decision per parameter becomes
  one flag-vector transfer per device. Errors still identify the parameter.

A local CPU profiler check with 40 gradient tensors recorded 40
`aten::_local_scalar_dense` calls before the gradient change and zero afterward;
the new code explicitly transfers the flag vector once on CUDA. This confirms
the removed scalar reads, not a measured GPU speedup. No CUDA device was
available for the investigation.

## Record the actual training workload

Run the same configuration on each GPU, adding a separate trace directory:

```bash
prom3theus invert configs/hmi_aia_dynamic.yaml --profile /tmp/p3s-a100
```

Use your actual configuration if different. This runs normal training, including
its usual checkpoint/resume behavior. The profiler skips five training steps,
warms up for two, and records three steps once. Allow at least ten steps before
the run ends. It writes a `*.pt.trace.json` trace and Lightning's profiling
summary. CUDA activity is included when CUDA is available. Open the trace in
a compatible trace viewer, such as Perfetto.

Compare the same batch sizes, depth/ray sample counts, active streams, precision,
physics objectives, and software versions. Measure steady iteration time with
profiling disabled; shape and memory tracing add overhead. Exclude startup,
validation rendering, and checkpoint steps from that comparison.

In the trace, inspect:

- Batch-fetch regions and GPU idle gaps: native observation loaders now read
  contiguous slabs with bounded host queues and CUDA lookahead. See the global-shuffle
  loader controls below. The scheduler saves completed batches and resumes
  through direct cursors; older shuffled states require explicit migration.
- Scalar reads, device-to-host copies, and synchronization: additional runtime
  geometry, plasma, and transfer validation checks remain.
- Forward/backward kernel duration and launch gaps: the polarized solver uses a
  batched matrix exponential followed by a sequential Python depth recurrence.
  The recurrence and higher-order physics derivatives need GPU profiling before
  choosing compilation or a different algorithm.
- Logging and callbacks: progress metrics are read on the host, and validation
  rendering/checkpointing add periodic work unrelated to steady training compute.
- Zero-weight streams: these still execute their diagnostic forward pass in the
  joint evaluator, without gradients. Check whether the measured configuration
  includes an expensive diagnostic-only stream.

These remaining paths were not disabled to obtain a lower timing. Further
changes should follow the actual GPU trace and retain the scientific validation
and sampling contracts.

## Global tensor shuffle and random batch order

The complete prepared data module is saved with `torch.save` to
`<work_directory>/data_module.pt`. Global rank zero prepares it only when absent
(or when observation rebuilding is explicitly requested). Other ranks wait for
publication and restore it with `torch.load(..., weights_only=False)`; rank zero
also restores the saved module. Use a work directory shared by all workers.
Existing modules are reused without source scans. A changed data selection, scene,
resource contract, or validation stride raises an explicit rebuild error. Batch size,
reader/prefetch budgets, pinning, device, and batch-order seed bind at runtime.
An evaluation-only module is upgraded by rank zero with missing training pools from
its saved references, without re-ingesting the observations.

The archive contains portable dataset state, scene/domain summaries, and training
pool specifications. Unchanged canonical NPY arrays are referenced directly;
derived payloads live in `data-arrays-*` and shuffled pools in `training-tensors`.
`ArrayRef` objects carry only file/schema metadata. Process-local `ReadSession`
instances own bounded read-only mappings; returned batches own their memory and
remain valid after readers close. Channel slices fill the final batch directly,
including pinned allocations when CUDA is enabled. Keep the referenced canonical
stores and work-directory arrays available. This is a trusted local Python snapshot,
separate from model checkpoints. Legacy incompatible snapshots require an explicit
rebuild; failed loads never silently regenerate data.

For distributed training, launch one process per device, for example:

```bash
torchrun --standalone --nproc-per-node=2 -m prom3theus.cli.main invert configs/hmi_aia_dynamic.yaml
```

The application initializes the process group before preparing data. Global rank zero
publishes a generation or broadcasts its preparation error, and all ranks restore
the same generation. Use shared storage across nodes. Each rank receives different
batch IDs; incomplete final rank groups use zero-weight data placeholders so optimizer
steps stay synchronized. Likelihoods use global sample counts, including per-channel
AIA counts. Shared physics priors remain active on every rank. Checkpoint cursors count
committed optimizer steps; changing world size requires explicit data-order migration.
Model snapshots, online logging, and diagnostics are owned by rank zero.

The CPU distributed training/resume tests cover this path. Actual multi-GPU throughput,
pinned-memory limits, and NCCL behavior still require validation on the deployment GPUs.
See [implementation evidence](data-loading-implementation.md).

Each configured stream completes loading, tensor preparation, shuffling, and persistence
before the next stream loads. The reference stream runs first to establish the scene.
Adapter source-reading parallelism remains enabled within the current stream; finished
streams retain file references, so their raw tensors can be released immediately.

Native Stokes and AIA training use `PersistentTensorLoader`. At startup,
contiguous native row slabs are read, compacted and normalized. One global
permutation assigns every valid sample to a position in persisted tensor files;
all nested fields use the same permutation. The loader scatters sequentially
read slabs into this layout without making a full RAM copy of the payload.
A matching cache reopens the existing files without any sample shuffle or rewrite.

The prepared tensors are divided into fixed batch-sized slices. Each pass
randomizes batch IDs without replacement. Training reads contiguous slices from these files;
it never gathers random pixels from disk or from the prepared tensors. Spatial
locations and observation times are therefore mixed from the first batch.
The global pixel layout remains fixed throughout training. HMI and Hinode use
one pool across all selected rasters. AIA uses one globally shuffled pool per
channel across all selected exposures, maintaining every channel in each batch.

AIA assigns `batch_size // channel_count` samples per channel, with the remainder
assigned to the first channels in configured order. Each channel independently
permutes its fixed chunks on every complete channel pass. Short final chunks
are retained; smaller channels cycle independently and total batch size can
therefore be smaller than the configured maximum. No sample is repeated before
its channel's complete pass is consumed.

Diagnostics use `SequentialBulkLoader` and preserve native image order.

| Key under `training` | Default | Meaning |
| --- | --- | --- |
| `loader_seed` | 0 | Private seed for global layout and per-pass batch permutations |
| `reader_workers` | 2 | Global concurrent native slab reads |
| `loader_block_bytes` | 33554432 | Contiguous read/scatter block target |
| `loader_cache_bytes` | 134217728 | Startup and diagnostic slab cache budget |
| `loader_prefetch_batches` | 2 | Ready host batches per stream |
| `loader_max_batch_bytes` | 67108864 | Batch admission limit |
| `gpu_prefetch` | true | One future joint batch copied on a dedicated CUDA stream |
| `migrate_data_order` | false | Explicitly restart sampling with a changed order policy |

**Memory:** training keeps the complete packed, globally shuffled payload in CPU
RAM, including per-pixel response arrays. The slab cache budget does not limit
this allocation. Startup additionally needs one int64 permutation for the pool
being prepared and bounded read/scatter slabs. Account for pinned batches,
queued batches, diagnostics, and model memory separately. Payload preparation
occurs once per loader lifetime, including once when restarting a process.

CUDA training pins only ready batches, not the full corpus. Source allocations
remain alive until asynchronous copies finish. The compute stream waits for its
batch's readiness event and records ownership of device tensors. Turning off
`gpu_prefetch` retains the ordinary transfer path for comparison.

Checkpoint sampling contracts include the seed, stream identity, batch size,
and dataset identity. A completed-batch cursor reconstructs both global tensor
layout and batch selection, independent of read budgets or prefetch depth.
Restart rebuilds shuffled tensors by sequentially rereading the native corpus;
it does not replay optimization or previously consumed minibatches. The first
batch-end callback commits the cursor before checkpoint writers run.

Old sampling orders cannot resume identically under this policy. Use a fresh
output directory for a clean convergence comparison, or explicitly set
`migrate_data_order: true` to retain model/optimizer state and restart data
coverage. The migration is recorded in the checkpoint. Changes to source data
or batch sizes are still rejected.

Assembly throughput can be inspected with:

```bash
PYTHONPATH=src python -m prom3theus.diagnostics.benchmark_loading
```

The first global-shuffle pass includes preparation; subsequent passes measure
batch slicing and selection. This CPU benchmark does not establish disk or GPU
throughput. Validate convergence on the same observations and optimizer budget.
