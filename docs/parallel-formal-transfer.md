# Parallel formal LTE transfer

The production forward model is the atmosphere network followed by local LTE
coefficients, formal Stokes integration and the instrument response. The learned
radiation network, RT collocation dataset, PDE losses, monitoring policy and
associated scripts have been removed. Non-LTE transport is future work.

For each layer the polarized solver averages the seven independent propagation
coefficients and the source at the two nodes. With `A = exp(-K ds)`, the exact
constant-layer update is `outgoing = A @ incoming + (Id - A) @ source`.
Each ray and wavelength sweeps from the lower boundary toward the observer.
Working relative to the bottom Stokes vector preserves thermal equilibrium.
A separately evaluated exponential increment retains emission in optically thin layers.

On Triton-capable CUDA devices the solver applies the cubic polynomial for
`A` directly to four-component vectors in registers. The backward sweep traverses
the layers in reverse using the saved incident vectors. Other devices and
higher derivatives use differentiable PyTorch matrix-vector operations.
Both paths use the analytic polarized exponential, with stable scalar weights
for weak fields and opaque layers. Float32 is the default, with explicit
float64 transfer available for reference work. Arbitrary matrices passed to
`formal_stokes` use a general matrix exponential and ordered transmission
prefixes. The analytic constant-layer solution is described in
[Mościbrodzka & Gammie (2018)](https://arxiv.org/abs/1712.03057).

The atmosphere, plasma state, continuum-table interpolation, line populations,
magnetic orientation, Doppler factors and widths, damping, Zeeman splitting and
integrated line strengths are prepared once per sampling stage. Spectral coordinates
are prepared once for the full ray batch and shared across stages. These are
live differentiable tensors scoped to one forward call, not a persistent cache
across optimizer updates. Per-ray wavelengths broadcast over the existing point
state; continuum gathers allocate only requested spectral values, not wavelength
copies of the full tables. Optimizer batches skip numerical validation predicates. Diagnostics check their
flags together at the prediction boundary, and standalone numerical APIs check immediately. See [the staged code flow](architecture.md#one-inversion-update). The fixed Voigt polynomial saves its analytic
derivative for ordinary backward; higher derivatives remain supported.

`inversion.lte.LTEWindow` evaluates true spherical-shell characteristics in
the model dtype, intersecting the observer ray with the actual outer and inner
spheres. Grazing rays that miss the lower boundary are rejected. Samples lie on
this straight ray, stratified in physical distance rather than radial height.
With `jitter_fraction: 1.0`, training draws each interior point independently
between the neighboring uniform-grid midpoints, leaving an epsilon margin at
both sides. The margin scales with dtype precision and the number of nodes,
keeping neighboring float32 distances distinct without sorting or extra sampling
checks. This follows the
stratified sampling idea in [NeRF, section 4](https://arxiv.org/html/2003.08934v2#S4),
adapted to keep both shell endpoints fixed for the LTE boundary condition. Points
stay ordered; formal transfer uses their actual unequal interval lengths.
For ray length `L`, node count `N`, and dimensionless jitter offsets `u_i`,
`s_i = L (i + u_i)/(N-1)` and `ds_i = L (1 + u_(i+1) - u_i)/(N-1)`.
The endpoint offsets are zero. Computing `ds` from these local offsets avoids
subtracting large float32 distances and preserves the true oblique line elements.
One device-local random draw covers all coarse rays and depths. Optional
`coarse_to_fine` refinement reuses the repository's inverse-CDF sampler. The
Hinode Stokes-only preset uses `depth_count: 16` with `fine_sample_count: 16`
and `uniform_weight_floor: 0.05`. Coarse local coefficients provide interval
masses `exp(-tau_before) * (1 - exp(-delta_tau))` from scalar Stokes-I extinction.
Each wavelength's masses are normalized separately and combined, covering
continuum and Doppler/Zeeman-shifted line formation. This detached proposal is
a sampling proxy; final polarized transfer still uses the complete matrix K.

Only the 16 new fine points receive a second batched atmosphere/point-physics
evaluation. Coarse coefficients and their gradients are reused. The merged 32
points are sorted by physical ray distance. The 31 final interval differences
supply the formal integration. No second coarse integral or importance-probability
weight is added. Fine-point spacing reserves an epsilon gap to both coarse
endpoints and other fine points; coarse jitter reserves enough room even when
all fine samples concentrate in one interval. All geometry remains float32.
Disabling refinement preserves the one-pass fixed-depth path. Jitter and
refinement do not replace depth-convergence checks for polarized quadrature.

Atmospheric height uses rationalized local offsets rather than subtracting two
solar-scale radii. Chart coordinates retain their local scene reference instead
of projecting a rounded global position back into the scene. `RaySamples` carries
these chart/height values directly into the atmosphere network; global positions
are used only for smooth rotation velocities.

Spectral queries use `CenteredWavelength`: a fixed origin plus small offsets.
Observer and calibration shifts use a rationalized inverse-Doppler increment;
line profiles subtract their centers from those offsets directly. Absolute
float32 wavelengths are used only for smooth continuum and Planck calculations.
Fixed calibration grids are prepared once on CPU at higher precision, then
converted before runtime arithmetic. The default Stokes forward/backward graph
has no float64-producing operations, including vector-potential atmospheres.

All inversion presets enable full training jitter. Evaluation via the owning
model's `.eval()` uses a uniform coarse grid and deterministic midpoint fine
quantiles when refinement is enabled; it consumes no sampling
RNG. A zero fraction disables jitter; omitted values retain the older uniform
behavior for existing configurations and artifacts. This setting lives under
`synthesis.windows`, independently of physics collocation or AIA sampling.

Local atmospheric velocities include Carrington rotation.
Observer/registration motion and calibration shift the vacuum wavelength query;
the instrument adapter retains detector-basis and normalization conventions.
The bottom boundary uses the modeled local Planck source by default; a supplied
`incoming_temperature_k` selects a fixed thermal boundary. Lower-boundary optical
thickness and depth convergence must be assessed for the scientific setup.

Schema v5 uses `synthesis.windows` and Stokes terms of type `lte_stokes` referencing
`synthesis_window`. Numerical controls are `depth_count` (default 257),
`jitter_fraction` (default 0, presets 1 during training) and `transfer_precision`
(`float32`). The solver integrates every wavelength for the whole ray batch
together. Reduce the observation loader batch size to lower peak memory. The evaluation artifact is P3S v3;
checkpoints identify formal-transfer format v1. Previous learned-RT artifacts
require a new run rather than silently restoring a different forward model.
Changing training jitter or transfer precision is allowed
on resume; the archived settings remain recorded. Depth count, wavelength bounds
and physical shell bounds remain part of the checkpoint's scientific contract.
Existing formal-LTE training checkpoints can therefore resume with the updated
float32 presets without rebuilding the observed profiles.

The tests compare spectra and first/second derivatives to the independent serial
solver with noncommuting matrices, odd/even layer counts and optical intervals
from 1e-8 to about 32. Analytic slabs test thin, thick and equilibrium limits.
Integration tests exercise atmosphere gradients, calibration gradients, per-ray
wavelengths, full-spectrum batching and detector basis rotations.

Measure forward time, backward time and peak memory on the target GPU.

```bash
PYTHONPATH=src python -m prom3theus.diagnostics.commands.benchmark_formal_transfer --device cuda
```

The benchmark compares complete vector-sweep, matrix-prefix and serial transfer,
including synchronized GPU timings and peak allocated memory. It excludes the
atmosphere, local coefficient calculation and instrument response.

Measure the complete inversion graph on fixed stored pixels with a Stokes-only
configuration (the command does not update model weights):

```bash
PYTHONPATH=src python -m prom3theus.diagnostics.commands.benchmark_lte_inversion CONFIG.yaml \
  --pixels 0 1 --repeats 5 --output /tmp/lte-benchmark
```

The report separates forward/backward time and counts network/plasma calls.
Vary the number of `--pixels` to compare ray batch sizes.
`--profile` adds a separate operator profile after the unprofiled measurements.
