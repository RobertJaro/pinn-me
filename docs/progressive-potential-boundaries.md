# Progressive potential boundaries

The HMI dynamic configurations fit the side and top magnetic vectors to a
potential extrapolation of detached model Br at **r = 1 solar radius**. The
Stokes objective constrains the photospheric and subphotospheric atmosphere.
Potential references do not normalize volume physics losses.

## Spherical reconstruction

`geometry: spherical_neumann` solves the exterior spherical Neumann problem:

- B = −∇Φ and ∇²Φ = 0 outside the photosphere.
- −∂Φ/∂r = Br at r = R_sun, in the boundary limit from above.
- The field decays at infinity. Its net source flux is retained.

The implementation integrates the spherical exterior Neumann Green kernel
([Hammer & Finlay, equations 3–6](https://doi.org/10.1093/gji/ggy515)) over source
cells. It does not project the source onto a plane, bury a reference surface,
zero-pad an FFT, or extrapolate below the photosphere. The old Cartesian FFT
utilities remain available separately, but this objective does not call them.

The Stokes angular bounding footprint is divided into `grid_size`² equal-solid-
angle cells in longitude and sin(latitude). At each refresh, model Br is sampled
at the photosphere using `source_supersampling`-point Gauss quadrature per cell
axis, then averaged within each cell. These piecewise constant cell values are
the discretized Neumann boundary data. Finer cells resolve more source structure;
there is no additional smoothing caused by a buried plane. Source geometry is
independent of the locations or density of boundary/normalization queries.

**Exterior assumption:** Br is zero outside the Stokes bounding footprint.
Unobserved gaps inside the bounding footprint are filled by the inferred model.
This local-map extension retains net flux (the spherical monopole mode, decaying
as r⁻²), rather than imposing a global flux-balanced solar map. It is an explicit
local extrapolation assumption; a measured global magnetic map would provide a
better specification of the unobserved exterior.

Point-source quadrature is inaccurate near the photosphere. Instead, source
cells near each query are subdivided geometrically and their kernels integrated
with sixth-order Gauss quadrature until the cell diameter is at most 0.8 times
the source-query distance. Br stays constant within each original source cell.
The CPU double-precision integration creates a geometry-only matrix once, which
is reused on the training device. Refreshes multiply that matrix by new detached
cell values. No gradient crosses source sampling or target reconstruction.

The exterior evaluator accepts only r > 1. A separate surface-limit evaluator
constructs photospheric targets at source-cell centres, where the piecewise-
constant boundary data are continuous. Its radial operator is exactly the
identity: target Br equals the source-cell Br. Tangential components use two
exterior evaluations at numerical heights of 1e-4 and 5e-5 times the smallest
angular cell width, linearly extrapolated to zero height. For a 3 Mm cell these
are approximately 300 m and 150 m. These are quadrature regularization lengths;
**the model fitting coordinates and the source remain exactly at r=1**. Nothing
is reconstructed below the photosphere. Tests check radial identity, convergence
of this tangential limit, the full-sphere monopole, and dipole convergence.

## Surface sampling and normalization

There are no potential volume samples and no separate normalization grid.
The configured dynamic HMI presets use:

| Surface | Grid | Fitting samples per step |
| --- | --- | ---: |
| Top at the domain's outer height | 16 × 16 horizontal cell centres | 256 |
| Each of four angular sides | 16 heights × 32 horizontal cell centres | 64 |
| Photosphere at r=1, over the Stokes source footprint | 64 × 64 source-cell centres | 256 until step 8000 |

Side height cells run from zero to the outer height, with queries at their
centres; angular side coordinates lie exactly on each face. Top and photosphere
use longitude/sin(latitude) cell centres, so their means are equal-solid-angle
horizontal averages. Sides are averaged along their horizontal grid direction
at each height independently. A source footprint smaller than the joint domain
therefore constrains only the observed footprint on the photosphere.

For every reference time, **after source blending**, compute:

- Top and photosphere: one mean of |B_pot| over both horizontal axes, separately
  for each surface.
- Each side: mean |B_pot| over its horizontal edge direction, retaining height.
  The four sides have independent profiles.

On the top and sides, the full vector residual is divided by
`sqrt(mean(|B_pot|)^2 + field_floor_gauss^2)`, then squared and averaged over
points and components. Signed vectors are never averaged to obtain the scale.
Top and combined sides each use `potential_boundary.weight`; the temporary
photosphere term has its independent `photosphere.weight`.
Volume MHS/divergence collocation and normalization remain independent.

The photospheric loss matches **only the signed horizontal orientation**.
Both fields are projected onto the tangent plane at the actual jittered query
position. Let `angle = atan2(rhat · (Bpot_horizontal × B_horizontal),
Bpot_horizontal · B_horizontal)`. Its residual is
`max(0, (abs(angle) - pi/2) / (pi/2))**2`, averaged over sampled points
(masked points contribute zero). Deviations through 90 degrees have zero loss
and zero gradient; 135 degrees gives 0.25 and reversal gives 1. Neither Br nor
the horizontal magnitude is matched, and the cached mean-field normalization
is not used by this angular loss. A 180-degree reversal has a corrective angular
gradient, unlike a cosine loss at its stationary maximum. Potential horizontal
fields below `photosphere.minimum_horizontal_field_gauss` (default 50 G) are
skipped. This threshold uses the detached reference horizontal magnitude only;
large Br does not qualify a point, and weakening the prediction cannot disable
the angular penalty. At numerically zero predicted
horizontal field (less than `1e-6 * field_floor_gauss`), a directional seed gives
a finite gradient toward the reference instead of an undefined angle. The seed
does not match the reference magnitude. The existing photospheric schedule,
jitter, targets, and hard cutoff are unchanged; cached references remain reusable.

Each surface's cells share one seeded permutation. Training selects a contiguous
chunk of cells, then draws a **fresh position inside each selected cell on every
iteration**. `jitter_fraction: 1.0` covers the whole cell (up to half a cell width
in each surface coordinate); zero restores fixed-centre sampling. Tiny numerical
margins exclude exact cell edges. Top/photosphere jitter in longitude and
sin(latitude); sides jitter in height and their horizontal direction. Fixed
surface coordinates never move, so photospheric radius stays exactly r=1.
The newly constructed Cartesian coordinates are passed directly to the model
and the loss; cached grid-centre coordinates are not used for those predictions.

Potential vectors are bilinearly interpolated from the cached Cartesian surface
grid at the perturbed location, then interpolated in time. Outer half cells use
linear extrapolation from the nearest two centres. At the photosphere the
interpolated field is projected onto the local tangent plane and the containing
source cell's exact, time-interpolated Br is restored along the new radial unit
vector. This is a spatial approximation to the potential field, not a fresh
Green-kernel solve; tangential subcell detail is limited by the reference grid.
There are no per-batch Green matrices or geometry rebuilds.

Top/photospheric normalization stays constant across its horizontal surface.
Each side's mean-field profile is interpolated at the jittered height, with
nearest-endpoint values in the outer half height cells. Reference normalization
is still computed only from the fixed surface grids. A short final chunk is
weighted to preserve the uniform cell-average objective. Validation visits the
fixed, unjittered grids at every reference time, so its spatial support is stable.

## Schedule and configuration

Settings live under `physics.potential_boundary`:

| Setting | Dynamic preset | Meaning |
| --- | ---: | --- |
| `geometry` | `spherical_neumann` | Exterior spherical reconstruction |
| `source_height_megameter` | 0 | Photosphere, required |
| `start_step` / `ramp_steps` | 500 / 500 | First refresh / boundary-weight ramp |
| `update_every_n_steps` / `freeze_step` | 500 / 10000 | Refresh cadence / final refresh |
| `blend` / `weight` | 0.5 / 1 | Source blending / top and combined side weights |
| `grid_size` / `source_supersampling` | 64 / 2 | Source cells per axis / Gauss nodes per cell axis |
| `top_grid_size` | 16 | Top horizontal resolution per axis |
| `side_horizontal_points` / `side_height_points` | 32 / 16 | Resolution of each side |
| `batch_size` | 256 | Top sample budget, combined side budget, source inference batch |
| `seed` | 0 | Once-per-surface spatial shuffle |
| `jitter_fraction` | 1 | Fraction of each surface cell covered by fresh jitter |
| `field_floor_gauss` | 1 | Normalization floor |
| `photosphere.enabled` / `photosphere.batch_size` | true / 256 | Temporary photospheric fitting |
| `photosphere.start_step` / `photosphere.ramp_steps` | 500 / 500 | Photospheric weight ramp |
| `photosphere.weight` / `photosphere.end_step` | 0.1 / 8000 | Weight / hard cutoff |

The first refresh uses inferred source Br directly. Later refreshes blend old
and new source coefficients before evaluating the same cached surface operator.
Every distinct valid Stokes time gets a reference; joint interval endpoints are
added when needed to cover AIA times. Training draws continuous times and
linearly interpolates targets and their surface scales between adjacent anchors.
Its sample count is independent of the number of time anchors. Validation
averages equally over all anchors. Static atmospheres use one midpoint reference.

At step 8000 the photospheric loss switches off abruptly, including its fitting
forward passes and validation loss. The side/top constraints continue. Source
Br must still be inferred at r=1 during reference refreshes to define those
outer constraints; disabling photospheric fitting does not disable that source
inference. `photosphere_weight` and the `photosphere` loss component replace the
former interior metrics. Q/U warmup remains independently configured.

Lightning checkpoints preserve geometry, surface targets/scales, shuffle and
update state (reference state version 8). The geometry matrix is derived and
not serialized; it is rebuilt only if another refresh is needed after resume.
The preset matrix has 6400 surface queries × 3 components × 4096 source cells,
or **300 MiB in float32**. Surface-limit temporary allocations are batched.
There is no geometry reconstruction when selecting training chunks or during
ordinary reference updates. Changing surface resolution, sampling seed, or the old volume prior rebuilds
only the derived reference buffers on resume, retaining learned model weights,
optimizer state and global step. The new references are inferred from the resumed
model. A resume after step 8000 therefore keeps photospheric fitting off unless
its configured cutoff is extended. Source reconstruction geometry (e.g. source
grid resolution) remains part of the compatibility contract. Schedule, weight and jitter-amplitude changes preserve the existing targets;
jitter uses the checkpointed training RNG sequence.

Potential boundary diagnostics upload to W&B after each reference creation or
refresh, and once on resume for a restored reference. They display the first
time anchor directly from the cached surface targets, with no additional model
or potential-field evaluations. The three rows show Br, Btheta and Bphi; the
six columns show the photosphere, top and four sides. Each column has its own
symmetric value range, shared across its components, and one horizontal
colorbar below the column.

## Neural coordinate scaling in the dynamic configurations

Both HMI dynamic presets use `uniform_spatial_scaling: true` with
`height_input_scale_m: 1.0e7`. The network receives
`[(x_mm-center_x_mm)/10, (y_mm-center_y_mm)/10, height_m/1.0e7, (time_hours-center_hours)/scale_hours]`.
All three spatial coordinates therefore use a common 10 Mm normalization.
The scene's chart geometry and observation/ray coordinate transforms are unchanged.
`radial_weighting: null` removes the SIREN coordinate warp entirely, including
its previous height-dependent multiplication of time. Time retains its independent
scene-derived center and scale. Legacy models retain their previous scaling unless
uniform scaling is explicitly selected. A fresh run is required when changing
the representation; old weights were fitted to different input coordinates.


With `data_term.objective.qu_warmup_steps: 8000` (both dynamic presets),
training omits Q/U contributions for optimizer steps 0–7,999. At step 8,000
configured Q/U weights return immediately, without a ramp. This schedule is
independent of whether the potential prior is enabled and when it ends.
I/V coefficients are unchanged; weights are not renormalized during warmup.
Validation retains the full Stokes objective and Q/U diagnostics throughout.
The training metric `streams.<id>.qu_weight_factor` reports 0 or 1. The
optimizer global step supplies the schedule, including on checkpoint resume.
The default `qu_warmup_steps: 0` enables Q/U from the first step.
