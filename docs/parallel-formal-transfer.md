# Parallel formal LTE transfer

The production forward model is the atmosphere network followed by local LTE
coefficients, formal Stokes integration and the instrument response. The learned
radiation network, RT collocation dataset, PDE losses, monitoring policy and
associated scripts have been removed. Non-LTE transport is future work.

For a frozen layer, `I_out = S + A (I_in - S)`, with
`A = exp(-K * ds)`. Adjacent maps `(A,b)` compose as
`(A_near @ A_far, b_near + A_near @ b_far)`. Composition is associative but not
commutative. The balanced tree preserves ray ordering and combines all pairs
in each stage simultaneously. For N layers it needs ceil(log2 N) stages and
O(N) total work; it does not compute all intermediate intensities. Ordinary
PyTorch operations differentiate through the tree, including second derivatives.

This is the attenuated emission integral, not a dependency on previously solved
intensities. With layer 0 nearest the observer, the outgoing intensity equals
`(A_0 ... A_N-1) I_bottom + sum_i (A_0 ... A_i-1) b_i`, where
`b_i = (identity - A_i) S_i` is emission integrated through the local layer.
Local LTE emissivity is `j_i = K_i S_i`; no incoming intensity is needed to
calculate it. The tree evaluates this sum by composing transmission/emission
pairs, without explicitly forming every prefix product.

`rt.formal_transfer` uses midpoint coefficients and sources. The affine emission
terms use a stable exponential increment `exp(-K ds)-I` relative to the bottom
boundary, preserving thin emission and a uniform thermal equilibrium. The
polarized increment is analytic; general matrices use a masked thin-layer series.
No inverse attenuation matrix is formed. Float32 transfer is the default, and
outputs retain the local coefficient dtype. Explicit float64 transfer remains
available for numerical reference comparisons.

Production LTE matrices use an analytic seven-coefficient matrix exponential
(`rt.polarized_exponential`), retaining the same frozen-layer solution. Its
cubic matrix polynomial uses a small-invariant series at zero field and
degenerate generators, and combines scalar attenuation with hyperbolic terms
before exponentiation to avoid overflow in opaque layers. First and second
derivatives are checked against the general matrix exponential. Arbitrary
matrices passed to `formal_stokes` retain the general implementation unless
`polarized=True` is explicitly requested. This constant-layer analytic approach
is described in [Mościbrodzka & Gammie (2018)](https://arxiv.org/abs/1712.03057).

The atmosphere, plasma state, continuum-table interpolation, line populations,
magnetic orientation, Doppler factors and widths, damping, Zeeman splitting and
integrated line strengths are prepared once per ray batch. Spectral coordinates
are prepared once for the full query and sliced into blocks. These are
live differentiable tensors scoped to one forward call, not a persistent cache
across optimizer updates. Per-ray wavelengths broadcast over the existing point
state; continuum gathers allocate only requested spectral values, not wavelength
copies of the full tables. Input validity flags are checked together at the
prediction boundary, with immediate checking retained for standalone numerical
APIs. See [the staged code flow](architecture.md#one-inversion-update). The fixed Voigt polynomial saves its analytic
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
Hinode Stokes-only preset uses `depth_count: 32` with `fine_sample_count: 32`
and `uniform_weight_floor: 0.05`. Coarse local coefficients provide interval
masses `exp(-tau_before) * (1 - exp(-delta_tau))` from scalar Stokes-I extinction.
Each wavelength's masses are normalized separately and combined, covering
continuum and Doppler/Zeeman-shifted line formation. This detached proposal is
a sampling proxy; final polarized transfer still uses the complete matrix K.

Only the 32 new fine points receive a second batched atmosphere/point-physics
evaluation. Coarse coefficients and their gradients are reused. The merged 64
points are sorted by physical ray distance; the 63 final interval differences
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
`jitter_fraction` (default 0, presets 1; training only),
`wavelength_chunk_size` (16) and `transfer_precision` (`float32`). Wavelength
chunking does not alter instrument sampling. The evaluation artifact is P3S v3;
checkpoints identify formal-transfer format v1. Previous learned-RT artifacts
require a new run rather than silently restoring a different forward model.
Changing wavelength chunk size, training jitter or transfer precision is allowed
on resume; the archived settings remain recorded. Depth count, wavelength bounds
and physical shell bounds remain part of the checkpoint's scientific contract.
Existing formal-LTE training checkpoints can therefore resume with the updated
float32 presets without rebuilding the observed profiles.

The tests compare spectra and first/second derivatives to the independent serial
solver with noncommuting matrices, odd/even layer counts and optical intervals
from 1e-8 to about 32. Analytic slabs test thin, thick and equilibrium limits.
Integration tests exercise atmosphere gradients, calibration gradients, per-ray
wavelengths, chunking and detector basis rotations.

The tree trades sequential matrix-vector operations for batched matrix-matrix
operations. It removes the depth dependency chain but does not remove opacity
evaluation or matrix exponentials. Measure both forward and backward on the
target GPU; CPU timing is not a GPU speedup estimate:

```bash
PYTHONPATH=src python scripts/diagnostics/benchmark_formal_transfer.py --device cuda
```

The benchmark reports the reduction separately from complete formal transfer,
including synchronized GPU timings and peak allocated memory. It excludes the
atmosphere, local coefficient calculation and instrument response.

Measure the complete inversion graph on fixed stored pixels with a Stokes-only
configuration (the command does not update model weights):

```bash
PYTHONPATH=src python scripts/diagnostics/benchmark_lte_inversion.py CONFIG.yaml \
  --pixels 0 1 --repeats 5 --output /tmp/lte-benchmark
```

The report separates forward/backward time and counts network/plasma calls.
`--chunk-size` compares spectral block sizes without changing the sampling;
`--profile` adds a separate operator profile after the unprofiled measurements.
