# Data loading: global shuffle, contiguous tensor batches

The training sampling policy is a single global permutation of valid samples at
startup, followed by random selection of fixed batch-sized tensor slices.
The former local-shuffle policy has been removed because it restricted early
training to neighboring pixels and observation times.

## Preparation

1. Build compact valid-pixel catalogs and read native stores in contiguous slabs.
2. Compact masks and normalize data while each slab is resident.
3. Allocate owned CPU tensors for the complete training pool and create one
   private, seeded global permutation across all selected rasters.
4. Scatter every field, including coordinates, Stokes measurements, and nested
   instrument responses, with the same permutation. Release the permutation and
   startup slabs after preparation.

HMI and Hinode each use one stream-wide pool. AIA uses one pool per channel,
with global mixing across spatial locations and all exposure times. Native
stores are never accessed through random individual-pixel reads.

## Training

Partition each prepared pool into fixed slices according to its batch quota.
Randomly permute slice IDs without replacement on each pool pass. Keep the pixel
layout fixed. Training uses tensor slicing, plus channel concatenation for AIA;
it performs no random pixel gathers. Retain partial final slices and cycle
smaller AIA channels independently, preserving all channels in each batch.

Readers and GPU transfers use bounded queues. CUDA pins only ready batches,
retains source allocations until copy completion, and tracks device ownership.
Diagnostics retain native order through the sequential bulk reader.

The complete packed training corpus must fit in host RAM. Reader cache settings
limit startup slabs and diagnostics, not the prepared global tensors.

## Resume and verification

A private seed, stream identity, pool layout, and completed batch cursor fully
specify the sampling order. Rebuild global tensors once on process restart and
seek directly to the next batch ID. Prefetch and discarded lookahead do not
advance committed progress. Old ordering policies require explicit
`migrate_data_order: true`; incompatible data or batch sizes remain errors.

Tests cover global spatial/time mixing from the first batch; aligned nested
fields; sequential source reads; no per-pixel fetch; no training-time disk
reads or pixel gathers; complete coverage and final slices; independent AIA
channel passes; model RNG isolation; and exact resume across reader budgets.
CUDA ownership/overlap tests require a CUDA host. Compare reconstruction quality
at equal optimizer budgets with the formerly working globally mixed sampler.

See [training-performance.md](training-performance.md) for configuration and
memory details.
