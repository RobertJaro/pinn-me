# Stokes computational efficiency audit — 23 September 2026

The dominant measured cost is differentiating the polarized formal transfer
solution. Improving that path and validating smaller spectral quadrature are
higher priorities than replacing SIREN. Keep the depth coverage established
by the [ray audit](hmi-ray-sampling-audit.md).

## Scope and method

Measurements use macOS ARM, PyTorch 2.13.0, CPU float32, one CPU thread unless
specified, four fixed rays, a 256-wide/eight-layer SIREN, vector-potential B,
128 uniform-height coarse plus 64 adaptive depths, and idealized HMI filters
with 161 inner quadrature wavelengths plus two continuum endpoints. The actual
production ray tracer, refinement, EOS, polarized synthesis and HMI operator
are used. Parameters stay fixed; Adam runs with zero learning rate so its work
is included without changing the atmosphere between repetitions. Timings are
medians of three steps after warmup, with no detailed profiler active.

This is an initialized-atmosphere computational control, not a failed checkpoint
or complete Lightning run. It excludes data preparation, disk loading, transfer
from host to device, logging, plotting, validation and checkpoint writing. No
CUDA or MPS device was available. GPU performance and peak GPU memory remain
unmeasured; the provided script supports CUDA synchronization and memory reporting.

## Confirmed inefficiency and implemented fix

The formal-solver loop previously indexed `attenuation[..., depth, :, :, :]`
and the source at every layer. Each indexed backward operation scatters its
gradient into a full-depth tensor. This repeats work across depth, creating
quadratic depth-dependent gradient assembly around an otherwise linear
recurrence.

The solver now calls `unbind` once for each depth sequence and iterates over
the resulting views. Autograd assembles those layer gradients once. The
mathematics, recurrence order, depth resolution and spectral resolution are
unchanged. The small recurrence kernel is isolated as `_propagate_layers` for
future compilation or an explicitly tested fused implementation.

| Implementation | Forward | Backward | Total step |
|---|---:|---:|---:|
| Old indexed recurrence | 0.190 s | 1.327 s | 1.511 s |
| Unbound recurrence | 0.171 s | 1.124 s | 1.296 s |

This is **14% less step time, or 1.17× throughput**, in this CPU control.
Some forward difference is timing variation: the intended improvement is
gradient assembly. A separate repeat measured 1.328 s for the optimized step.
Do not interpret these small samples as a GPU speedup prediction.

Regression tests compare float32/float64 spectra, first derivatives and second
derivatives against the old indexed recurrence, including scalar and multiple
batch dimensions. Spectra agree exactly; gradients agree within dtype tolerance.
The transfer/synthesis/forward/precision suite passed **44 tests**, with six
CUDA-only skips. Existing independent Milne–Eddington and path-consistency tests
also pass.

## Remaining cost

An additional diagnostic splits the same backward graph at the formal solver's
matrix/source inputs and at the atmosphere's physical fields:

| Operation | Time |
|---|---:|
| Complete forward | 0.177 s |
| Formal transfer backward, including instrument-to-transfer path | 1.040 s |
| Opacity/source backward | 0.084 s |
| Atmosphere/network backward, including curl | 0.028 s |
| Adam | 0.0009 s |

The split and ordinary backward produce the same total parameter-gradient norm
(33.491718 in this control), and essentially the same step time. Formal
transfer backward alone accounts for approximately **78%** of the step.

Separately instrumented forward calls put the formal solver at 0.103 s,
including 0.091 s in `torch.matrix_exp`; line opacity takes 0.044 s and the
instrument integration about 0.0004 s. These nested timings should not be
summed with each other or substituted for the uninstrumented step median.

At the configured batch size 64, 192 depths and 163 wavelengths imply about
**1.99 million 4×4 layer matrix exponentials per step**. The attenuation tensor
alone is about 128 MB in float32; the propagation matrix adds another roughly
128 MB, before gradients, matrix-exponential intermediates, line/EOS tensors
and neural activations. These are array-size calculations, not measured peak
memory. Disabling physics residuals does not remove this work.

## Measured upgrade options

| Control variant | CPU step time | Interpretation |
|---|---:|---|
| Optimized, 161 spectral points | 1.296 s | Current comparison baseline |
| 121 spectral points | 1.012 s | About 22% less time; promising first spectral control |
| 81 spectral points | 0.715 s | About 45% less time; stronger accuracy gate needed |
| Direct B instead of vector potential | 1.278 s | Little difference here; changes magnetic representation and initialization |
| 192 uniform depths, no refinement | 1.410 s | No improvement in this fixture; retain refinement |
| Eight CPU threads | 1.106 s | Hardware-dependent tuning, not a recommendation for GPU worker counts |

Direct B has a different initial field, so that row is not a clean isolated
measurement of curl cost. The split backward is stronger evidence that the
network is a small fraction of this particular computation. A smaller network
or MLP should not be the first performance intervention on this evidence.

### Spectral accuracy gate

Eight native columns (four strong, four quiet) were evaluated at identical 513
physical-depth points and idealized HMI filters, varying only spectral
quadrature. Relative component L2 errors against 321 spectral points are:

| Inner spectral points | I | Q | U | V |
|---|---:|---:|---:|---:|
| 161 | 0.000009% | 0.000028% | 0.000095% | 0.000155% |
| 121 | 0.000387% | 0.00191% | 0.00341% | 0.00371% |
| 81 | 0.0162% | 0.0741% | 0.126% | 0.152% |
| 61 | 0.119% | 0.570% | 0.761% | 0.581% |

These are wavelength-integration comparisons at fixed depth resolution, not
total forward errors. For 81 points, the worst selected pixel's Q/U/V RMS
errors are 0.0050%/0.0097%/0.0074% of its floored reference continuum. This
supports **121 points as a conservative synthetic-data option**, with 81 as a
faster experiment. The current production defaults were not changed.

`build_idealized_hmi_filter(n_quadrature=...)` already supports this choice;
exposing it in synthetic generation/export is a natural next change. Exported
responses must remain consistent with their wavelength grid and prepared
caches. Do not simply slice stored response weights. Real calibrated HMI
filters have their own transmission structure: repeat convergence tests using
those responses, large Doppler shifts and strong fields before reducing their
quadrature. Parameter-gradient convergence also needs checking for an adopted
lower spectral resolution.

## Engineering options in priority order

1. **Keep the recurrence fix and measure on the training GPU.** Sweep batch
   sizes 16, 32 and 64; compare rays/second and peak allocated memory. CPU
   seconds per batch are insufficient to choose GPU batch size or hardware.
2. **Validate and expose the 121-point synthetic spectral option.** It reduces
   both matrix-exponential count and spectral intermediates without sacrificing
   the newly established depth coverage. Move to 81 only if the application's
   profile and gradient tolerances permit it.
3. **Prototype a purpose-built formal solver.** Compare an efficient
   differentiable DELO-family solver with the current frozen-layer exponential
   reference, or optimize the structured 4×4 exponential and its adjoint.
   DELO–Bézier methods were developed for efficient polarized transfer with
   gradients in atmospheric quantities ([de la Cruz Rodríguez and Piskunov](https://arxiv.org/abs/1212.2737)).
   This is a substantial numerical change: verify weak/zero fields, strong
   polarization, thick/thin intervals, velocity gradients, source gradients,
   and parameter derivatives. No speedup factor is established yet.
4. **Compile/fuse small numerical kernels after GPU profiling.** Start with
   `_propagate_layers` or opacity kernels, rather than compiling the entire
   application. Many current finite-value checks use tensor-dependent Python
   branches; these can synchronize CUDA and break compilation. PyTorch documents
   these [graph-break causes](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html).
   Consolidate validation at trusted boundaries while retaining errors; do not
   indiscriminately remove checks. A fused recurrence alone may have limited
   benefit while generic matrix-exponential backward dominates.
5. **Separate validation-only streams from training I/O.** Zero-weight AIA
   synthesis is already skipped during training, but the scheduler still fetches
   and transfers its batches. Avoiding that work requires changing the scheduler
   and exact-stream-set contract, while preserving validation and checkpoint
   behavior. Its wall-time impact on the cluster has not been measured.
6. **Reduce diagnostic frequency/size during fitting experiments.** Full maps,
   height slices and AIA validation still run even when their training penalties
   are disabled. Keep selected core profiles and scalar residuals frequent;
   render larger products less often. Measure rendering separately from steps.

The current training entry point already requests pinned memory on CUDA and
uses asynchronous GPU prefetch. Its runtime settings override the observation
loader's `pin_memory: false`, so flipping that YAML value is not a missing
optimization. Cluster filesystem behavior and loader wait times still need
measurement in the actual training run.

Do not cut back to 25+16 depths for speed: that grid failed the native accuracy
tests. Also distinguish per-step and per-epoch cost. Moving from batch 256 to
64 makes approximately four times as many steps per epoch; increasing depths
from 41 to 192 means roughly 4.7 times the atmospheric samples per pixel.
Keeping sample count per batch similar does not keep epoch cost similar.
The current preset has a 500-epoch limit and no step cap; set a short explicit
step budget for controlled performance/recovery comparisons.

## Reproduction

From the repository root, using the intended training environment:

```bash
PYTHONPATH=src python scripts/diagnostics/profile_stokes.py \
  --device cuda --batch 64 --repeats 5 --output data/stokes_performance/gpu

PYTHONPATH=src python scripts/diagnostics/profile_stokes.py \
  --device cuda --batch 64 --repeats 5 --legacy-indexing \
  --output data/stokes_performance/gpu_legacy

PYTHONPATH=src python scripts/diagnostics/profile_stokes.py \
  --device cuda --batch 64 --repeats 5 --split-backward \
  --output data/stokes_performance/gpu_split
```

Use separate sequential runs to avoid device contention. `--spectral 121`,
`--spectral 81`, `--representation direct`, and `--threads` provide the bounded
comparisons above. The script saves settings, timings, fixed profiles, gradient
norms, finite-gradient checks and CUDA peak allocation. It also times a separate
instrumented forward pass. `--profile` additionally produces operator tables;
use a small batch first because detailed profiling itself can be expensive.
See the [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
for interpreting CPU dispatch time versus device execution time.

Local results are under `data/stokes_performance/`; `probe_spectral.py` and
`spectral_convergence.json` contain the native spectral comparison. No production
run, CUDA timing, rendering benchmark or cluster-I/O benchmark was performed.

## Reduced depth budget

The Stokes-only preset used for this historical benchmark had **96 coarse + 48 fine** depths,
uniform physical-height placement over the same -1 to +1.5 megameter shell, coarse
training jitter, deterministic fine quantiles, and batch 64. Validation has
no jitter. Fresh `muram_stokes_only_uniform144_jitter_v04` directories avoid
resuming an incompatible sampling contract. Fine positions still respond to
the coarse opacity proposal; deterministic quantiles do not freeze positions.

Eight native columns (including both strong-field cores) were compared to
1537 native-reference depths, using 121 inner HMI quadrature wavelengths,
float64, and three coarse-jitter seeds (19, 37, 71). Errors below are relative
L2 per Stokes channel aggregated over those columns and filter samples;
“worst jitter” means the largest channel/seed error among Q, U, V, not the
largest individual-pixel error. These are synthesis controls, not training
convergence measurements.

| Coarse + fine | Fixed I error | Fixed Q | Fixed U | Fixed V | Worst jitter Q/U/V error |
|---|---:|---:|---:|---:|---:|
| 32 + 32 | 0.59% | 1.18% | 2.04% | 5.04% | 17.35% |
| 64 + 32 | 0.90% | 2.94% | 1.51% | 1.60% | 15.23% |
| 64 + 64 | 0.26% | 3.94% | 3.84% | 5.50% | 13.45% |
| 96 + 32 | 0.49% | 4.90% | 3.74% | 2.07% | 9.60% |
| **96 + 48** | **0.31%** | **1.21%** | **0.64%** | **1.56%** | **2.86%** |
| 96 + 64 | 0.20% | 1.21% | 0.95% | 2.12% | 3.46% |
| 128 + 64 | 0.13% | 0.61% | 0.38% | 1.06% | 1.26% |

96+48 removes 25% of sample points. The same initialized-atmosphere CPU
benchmark took **1.037 s/step**, versus **1.296 s/step** for 128+64: about
20% less time (1.25x throughput). This timing uses fixed coarse placement to
isolate the point-count change, not jittered native columns or GPU training.
The record is `data/stokes_performance/depth144.json`.

Its I jitter variability is 0.65% of
observed continuum proxy (RMS across columns/filter samples, population
variance across three draws); 64+32 gives 1.09%, and 128+64 gives 0.065%.
With an intentionally mismatched initial-FALC proposal evaluated on the
native atmosphere, 96+48 errors are I/Q/U/V = 0.84/3.20/2.22/5.10%.
This stress test illustrates why coarse coverage matters before the opacity
is fitted; it is not an actual initialized-network forward spectrum.

The choice is a speed/accuracy compromise. Three seeds and eight columns do
not establish a universal bound; adaptive grids produce nonmonotonic errors.
Jitter does not remove quadrature bias. Compare fitted strong-field pixels
against 128+64 and a doubled dense grid before interpreting small residuals.
Results and reproduction probes are in `data/depth_budget_audit/` locally.

Further options, in priority order:

- Reduce spectral quadrature to 121 inner wavelengths, rebuilding matching
  response weights. The earlier native-column test and timing above support
  this as a separate experiment; it is not enabled by this depth change.
- Use sparse training with occasional dense convergence diagnostics. Separate
  training/validation depth budgets would need configuration support; current
  validation uses the same count, with jitter disabled.
- Test nonuniform physical-height grids concentrating coarse points around
  the formation region while retaining the full shell. Avoid relying solely
  on the initially inaccurate learned opacity to discover depressed layers.
- A specialized polarized transfer solver/adjoint addresses the dominant
  backward cost, but requires spectrum and gradient parity tests before use.
