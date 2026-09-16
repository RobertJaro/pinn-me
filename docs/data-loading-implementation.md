# Data loading implementation

Implemented against the [audit and plan](data-loading-refactor-plan.md).
The audit measurements describe the previous implementation; they are retained as
historical evidence rather than overwritten with new results.

## Structure and lifecycle

Cold preparation completes **one configured stream at a time**:

1. Load its source data using the adapter's configured parallel workers.
2. Set up its tensors and rebase Stokes coordinates to the shared scene.
3. Shuffle and persist its training pools.
4. Persist any remaining native/validation arrays and keep only file references.
5. Release the stream's raw tensors and loaders, then load the next stream.

The reference stream goes first because it establishes the shared chart and time
convention. Scene builders now receive that reference stream alone; later image
streams are validated against the established scene as they arrive. Final stream
ordering still follows configuration order. Global pixel mixing within each stream
(and each AIA channel pool) is preserved. Acquisition loading within a stream remains
parallel; acquisitions are not independently shuffled into a different sampling policy.

The final `torch.save` publishes the already prepared descriptors. It no longer
performs a deferred all-stream payload-persistence phase. Domain summaries are computed
from the persisted references after preparation. An existing module still bypasses
source loading, and evaluation-only upgrades prepare missing pools from disk.

- `application/data.py` owns preparation, scene rebasing, reusable domain summaries,
  data-contract checking, and upgrading an evaluation-only module for training.
- `observations/arrays.py` provides plain `ArrayRef` descriptors and `ReadSession`.
  Each process has one reader with a bounded mapping cache and read semaphore.
  Native slab reads and packed reads share the concurrency limit.
- `observations/snapshot.py` converts prepared state into importable records, file
  references, and small metadata constants. Native datasets preserve compact catalogs;
  extension providers cross this boundary as finite disk-backed batch specifications.
  Provider closures and local runtime classes are not serialized.
- `observations/persistence.py` publishes `data_module.pt` atomically with `torch.save`,
  restores using `torch.load`, validates array stat identities, and coordinates ranks.
- `observations/tensor_dataset.py` stores aligned sample fields in NPY files.
  `PackedLoaderSpec` binds runtime loaders independently of preparation.
- `training/streams.py` owns batch order, rank partitioning, and committed cursors.
  `core/distributed.py` handles global likelihood normalization.

Rank zero prepares only an absent module or an explicitly requested rebuild. Existing
modules restore without provider setup, source verification scans, raster construction,
geometry scans, or pixel permutation construction. Other ranks receive the published
generation and load the same module. A file lock serializes concurrent rank-zero calls;
failed publication preserves the previous module. Process-group startup propagates
preparation failures instead of leaving workers waiting for a nonexistent result.
Filesystem-only coordination requires a launch-specific run token.

An evaluation-only module gains missing training pools from its existing references.
Runtime batch sizes and worker/prefetch/pinning settings are rebound after loading.
The data/scene/resource contract remains checked, including validation selection.
The global pixel permutation stays fixed in the pool; changing the runtime seed changes
batch order and remains visible to checkpoint-order validation.

Saved data contain no payload tensors or open mappings in datasets. Coordinate bounds,
wavelength metadata, and other small scientific constants may remain ordinary tensors.
Random physics samples are generated on the model device from bounds. Unchanged
canonical arrays are not copied into the snapshot. Temporary training pools are linked
into durable snapshot storage before the temporary loader can be destroyed. Copies
across filesystems are bounded file copies. Moving a run directory rebases internal
references; external canonical stores must remain available.

## Computational changes

Packed datasets initialize lazy read-only mappings with `np.load(..., mmap_mode="r",
allow_pickle=False)`, including after deserialization and runtime binding. Mapping
initialization reads headers, not array payloads. Worker configuration preserves existing
mappings. Reads reuse them rather than opening and closing files for every query.
The mapping cache has a default maximum of 64 per process; explicit close, process
changes, or cache eviction can require a mapping to reopen. Mappings
are read-only and leased while in use; only idle mappings can be evicted. Descriptor
shape/dtype/length access performs no file reads. Every returned batch owns its memory,
so eviction or reader shutdown cannot invalidate a consumer's tensors.

Packed channel slices copy directly into one final batch allocation. The byte limit is
checked before allocation. CUDA uses a final pinned allocation rather than concatenating
intermediate channel batches and then copying them again for pinning. `close()` stops
producers while retaining prepared file descriptors, allowing a loader to reopen.

Cold HMI acquisition results are spilled to files as workers finish, bounding retained
heap payload by acquisition concurrency instead of timeline length. Canonical writers
reuse unchanged files. Coordinate normalization and Stokes scene rebasing use bounded
row slabs. AIA computes the exact lower median with a disk-backed scratch vector and
computes maxima and geometry checks in slabs. These preserve the existing objective
and geometry definitions.

Canonical AIA ingestion still verifies the full prepared store and its published
normalization before applying a requested time window. This is required by the current
full-store scale provenance. Warm module restores skip this work entirely; switching to
selection-only verification would require a separately validated store-statistics contract.

Global shuffle still uses one int64 permutation (8 bytes per sample) and disk-backed
scattered output. Payload slabs are bounded; permutation memory is not constant in
corpus size. External shuffle/sharding remains the plan's measurement-gated follow-up,
not an implicit change to spatial mixing or reproducible data order.

## Distributed behavior

An externally launched process group is initialized before data/model setup. Each rank
uses its local CUDA device, with Lightning DDP and explicit batch ownership. The
reference stream defines the global epoch. Its real batch IDs occur once across ranks;
other streams continue cyclically. If the final group has fewer batches than ranks,
extra ranks evaluate zero-weight data placeholders, maintaining matching collectives.
Shared physics priors remain active on all ranks.

Likelihood gradients are normalized by global sample counts, compensating for DDP's
average gradient reduction. AIA reduces counts per channel. Training metrics are
reduced together across ranks. Checkpoint state records world size and committed steps;
uncommitted prefetched batches replay. World-size changes require explicit migration.
Rank zero owns diagnostics, online logging, and model snapshot publication.

## Compatibility and remaining validation

Canonical observation-store formats remain readable. Old incompatible Python data-module
snapshots raise an explicit rebuild error; they are never silently replaced. This change
does not supply an automatic converter for historical tensor-subclass/provider pickles.
The obsolete startup-cache implementation, provider cache-source callbacks, and unused
channel sampler were removed.

No CUDA hardware is available in this environment. CPU tests exercise two-process
publication, Lightning DDP optimization, global weighting, and checkpoint resume. Real
multi-GPU/NCCL tests, pinned-memory measurements, deployment filesystem behavior, and
GPU wait-time/steps-per-second profiling remain required before claiming a GPU speedup.

See `assets/data-loading-implementation/benchmark.py` and `results.json` for the
repeatable CPU comparison. Both paths return owned tensors and read the same randomized
batch slices; the baseline reopens NPY mappings for each field/batch. The new reader
opens six mappings for the entire fixture. These are warm-cache local CPU measurements,
not end-to-end training or disk-throughput measurements.

## Verification results

Affected suites: **360 passed, 1 CUDA-only skipped**. Final focused rerun: **50 passed**.
These focused checks cover
publication, transient provider memory, native and custom-provider assembly, metrics,
and real two-process Lightning training/resume. `ruff` checks for the new persistence,
reader, coordinator, and distributed modules pass; `git diff --check` passes.

Full project suite: **872 passed, 5 failed, 2 skipped**. The five failures are
outside this refactor:

- CLI fixture expects AIA weight `0.1`; the existing example config uses `0.0`.
- Side-view logging fixture expects `validation/coronal`; the existing logging fallback
  is `Observation comparison`.
- Boundary fixture expects upper/side no-inflow constraints enabled; the existing
  mixed HMI/AIA config disables them.
- Two Stokes reduction cases use a `SimpleNamespace` harness missing the existing
  `_qu_weight_factor()` method.

The second skip requires local CHIANTI/fiasco resources. No configuration or scientific
behavior was changed to satisfy these unrelated expectations.

Isolated local warm-cache benchmark, median of three passes (65,536 samples, six
float32 fields with 16 values/sample, identical owned-copy semantics):

| Batch size | Reopen per batch | Reused mappings | Ratio |
| --- | ---: | ---: | ---: |
| 64 | 937.12 ms | 32.82 ms | 28.6× |
| 256 | 228.72 ms | 8.82 ms | 25.9× |
| 4096 | 17.17 ms | 1.62 ms | 10.6× |

Each time covers a complete pass, not one batch. These ratios isolate CPU file-opening
and batch assembly overhead; they are not expected whole-training speedups. The raw
measurements include every repetition and the environment versions.
