# HMI ray sampling audit — 23 September 2026

HMI rays reach the intended image region in the checked data. The stronger
problem is **insufficient depth resolution in depressed umbral formation
layers**, compounded by refinement that depends on the current opacity guess.
The existing -1 to +1.5 megameter shell includes the relevant layers in these native
controls. Increasing the shell extent is not the first remedy.

This follows the [spectropolarimetric audit](spectropolarimetric-audit.md).
The failing remote checkpoint was unavailable. Experiments use the local real
HMI cutout and 128 sparse MURaM columns from iteration 200000. Native height zero
is one shared median native tau=1 height, 7,457,449.9 m; individual columns are
not recentered. Absolute formation heights are conditional on that registration.

## Where the rays go

The actual real-HMI preparation path was checked against independent WCS,
helioprojective, Carrington and sphere-intersection calculations. In the local
256 by 256 cutout, mu ranges from 0.9800 to 0.99924.

- Independent and prepared ray directions agree to 5.1e-14 per component.
- Surface/chart reconstruction agrees within 9 millimeters in float64 and 109 m in
  float32, compared with approximately 363–371 km per HMI pixel.
- No pixel-axis swap, handedness flip or large lateral offset was detected.
- The network receives each traced point's chart coordinates, actual physical
  height and acquisition time. There is no hidden height clamp or fixed-height
  magnetic query in the current vector-potential configuration.

An inclined ray naturally changes horizontal chart position with height. The
maximum displacement is 102 km at +0.5 megameter and 307 km at +1.5 megameter in this real
cutout. This is perspective geometry. Edge rays may leave the rectangle of
zero-height pixel centers; the atmosphere still evaluates these points and
receives their data gradients.

Synthetic MURaM generation has a separate approximation: it integrates flat
vertical columns but exports parallel rays anchored on a sphere. These rays
drift through the inversion chart by up to 37 km at +0.5 megameter and 110 km at
+1.5 megameter. This is small compared with the sunspot, but prevents an exact spatial
closure claim. The earlier same-column transfer agreement bypassed the
production spatial tracer. An exact benchmark needs either an explicit
Cartesian embedding of the native atmosphere or a flat-column inversion mode.

## What the rays need to resolve

With the package's actual continuum opacity, the 6173 Å continuum tau=1 surface
lies between -634 and +101 km across the native columns. Twelve strong-field
columns have median -352 km, compared with +15.5 km among 54 quiet columns.
The old 25-point coarse grid had only four nodes below zero:
**-303, -552, -778 and -1000 km**.

Two approximately 3200 G cores have central 90% continuum source-contribution
intervals of -690 to -548 km and -651 to -554 km. These contain only one and
zero old coarse nodes, respectively. Fine sampling based on initial FALC
opacity adds **zero** nodes to either interval. This continuum calculation is
a positive scalar contribution proxy, not a polarized response function.

The result was checked with the actual polarized solver: differentiate all
12 emergent HMI Q/U samples with respect to all three magnetic components at
each of 1025 uniform depth nodes. The norm of these derivatives at each depth
defines the magnetic-response strength used below.

| Core | Central 90% Q/U magnetic-response interval | Old 25+16 nodes | FALC-shaped 128+64 nodes | Uniform-height 128+64 nodes |
|---|---:|---:|---:|---:|
| 99 | -634 to -373 km | 1 | 6 | 14 |
| 100 | -611 to -399 km | 1 | 4 | 12 |

All node counts in this table use the initial FALC opacity for the fine-sample
proposal. The response median is about -540 km in both cores. The 128-point
uniform grid has 19.7 km spacing throughout the shell; the 128-point
FALC-shaped grid still has 42–57 km gaps below zero. Refinement can populate
these layers once the opacity is correct, but it cannot guarantee coverage
while the atmosphere is wrong.

![Native core response regions and sampling nodes](../data/deep_audit_rt/ray_sampling_coverage.png)

A wider -2 to +3 megameter calculation places at least 99.46% of the continuum
contribution proxy and 99.74% of the line proxy inside the existing shell
across all 128 columns. Actual Q/U magnetic response outside the shell is zero
to computed precision in three tested strong columns. These are numerical
support checks on these atmospheres, not a universal formation-height bound.

## Accuracy at the same sample budget

These errors compare identical fixed native atmospheres in eight selected
columns, identical shell endpoints and identical 121-point HMI spectral
quadrature. The reference uses 1537 depth nodes. Each error is the residual
L2 norm divided by the reference component norm, aggregated over the columns
and six filters; it is not a percentage of continuum intensity.

| 128 coarse + 64 fine | I | Q | U | V |
|---|---:|---:|---:|---:|
| FALC-shaped coarse, native-opacity proposal | 0.397% | 2.43% | 1.60% | 1.96% |
| Uniform-height coarse, native-opacity proposal | 0.126% | 0.607% | 0.381% | 1.055% |
| FALC-shaped coarse, initial-FALC proposal | 2.52% | 10.3% | 32.9% | 22.8% |
| Uniform-height coarse, initial-FALC proposal | 0.560% | 0.759% | 1.71% | 2.73% |

The last two rows deliberately propose points with initial FALC opacity and
evaluate native thermodynamics there. They are a stress test of proposal
dependence, not a self-consistent trained atmosphere or a replay of the failed
run. The first two rows establish an improvement even when the correct native
opacity guides refinement. Uniform sampling at 25+16 remains inaccurate;
placement does not remove the need for sufficient resolution.

Uniform placement is not better everywhere. A separate quiet-column check
finds I RMS / reference continuum errors changing from 1.030% to 0.134%,
0.106% to 0.0826%, 0.0638% to 0.0517%, and 0.0801% to 0.1347% for the four
quiet columns. A smooth FALC fixture changes from 0.0182% to 0.0304% in those
units. This is a modest quiet-region tradeoff for more reliable deep coverage,
and reinforces the need to check convergence by region on the fitted model.

## Implemented changes

1. Added `depth_sampling.height_sampling: uniform`, wired through schema,
   runtime configuration and the production Stokes forward composition.
   The default remains `reference` for existing configurations. The sampling
   coordinate called `reference_log_tau500` labels placement; it is not the
   learned atmosphere's optical depth. Uniform mode maps it linearly to height.
2. Initially enabled uniform placement in the former MURaM/SDO Stokes-only preset, keeping
   128 coarse + 64 fine points, disabled depth jitter, batch 64, SIREN and
   existing Stokes weights. Fresh `muram_stokes_only_uniform192_v03`
   directories prevent automatic reuse of the preceding control checkpoint.
3. Fixed `StratifiedAtmosphereModel.trace_rays`: it now intersects every
   requested spherical height, rather than interpolating ray distance between
   the two endpoint intersections. Previously intermediate heights shifted by
   up to 3.36 km at mu=0.5, 11.3 km at mu=0.3 and 107 km at mu=0.1. This defect
   is small near disk center and does not explain the central native result.
   Rays that cannot reach the bottom retain a monotonic height distribution
   above their tangent limit, without duplicate or reversed points.
   On 81 actual WCS rays across the real cutout, the corrected production
   tracer visits requested heights within 0.5 m in float32 and 7e-10 m in
   float64, with ordered path distances and finite atmospheric outputs.
4. Fixed a separate HMI preparation crash: observer-velocity projection now
   excludes nonfinite off-disk LOS vectors before applying the final mask.
   The central cutout has no off-disk pixels, so this is not its Q/U cause.
5. Canonicalized omitted legacy sampling defaults in checkpoint contracts.
   An unchanged old configuration remains resumable; an explicit switch to
   uniform heights or deterministic sampling remains a scientific-policy
   change. Also restored required `max_steps: -1` in the control preset,
   preserving its existing 500-epoch limit without a step cap.

Regression checks cover requested heights, oblique and tangent rays, float32
and float64, gradients, configuration roundtrips and production forward
placement. The ray/atmosphere/forward suite passed 52 tests with 6 CUDA-only
skips; the sampling/configuration/resume suite passed 89 tests; the HMI
geometry/observation/subframe suite passed 63 tests. These suites overlap.
The final resume/trace/sampling/forward review passed 40 tests, including
legacy default compatibility and rejection of changed sampling policies.
These checks do not substitute for a full training run.

## Inversion interpretation and next experiment

Optical-depth stratification and geometric-height stratification are different
inference problems. SIR-derived optical-depth atmospheres require additional
constraints to construct a common geometric scale and Wilson depression
([Puschmann, Ruiz Cobo and Martínez Pillet, 2010](https://arxiv.org/abs/1007.2779)).
The [SIR manual](https://github.com/BasilioRuiz/SIR-code/blob/master/manual/manual.tex)
also motivates inspecting response functions when deciding what the spectra
constrain. In this implementation, using a reference optical-depth label for
a fixed geometric grid does not make that grid track the learned tau surfaces.

A plausible failure mechanism is that an initially quiet-Sun-like atmosphere
guides refinement toward shallow layers, while a depressed sunspot needs
strong thermodynamic and magnetic changes deeper down. Sparse coarse sampling
then supplies inaccurate spectra and derivatives for those changes. This
mechanism is consistent with the controls, but has not been demonstrated on
the failed checkpoint. Raising Q/U weights cannot repair inaccurate numerical
integration.

The next comparison should use corrected, regenerated synthetic targets and
the uniform-height Stokes-only preset, holding the architecture and weights
fixed. First fit a small set of quiet, penumbral and umbral pixels. Compare
128+64 with 256+128 on the same fitted atmosphere, then inspect individual
signed profiles and raw component residuals. For observations, perform the
same quadrature comparison on the real fitted atmosphere; native formation
heights should not be assumed to transfer unchanged.

After this control, compare inferred fields on actual tau surfaces or weighted
by the line response as well as at fixed geometric heights. A weak field on
one geometric slice alone does not show what field produced the spectra.
Before expanding to joint 3-D recovery, establish a spatial-geometry closure
test and a low-dimensional per-pixel inversion baseline. Add physics only
after the Stokes fit and its numerical convergence are demonstrated.

Local evidence and reproduction scripts are in `data/deep_audit_rt/`
(`probe_formation_coverage.py`, `probe_polarized_response_coverage.py`,
`probe_uniform_depth.py`), `data/ray_region_audit/` (`loader_geometry.py`,
`expected_displacement.py`) and `data/deep_audit_rays/model_queries.py`.
These artifacts require the sparse native data and local HMI cutout; they
are local audit inputs rather than bundled unit-test fixtures.

The subsequent faster preset uses 96 coarse + 48 fine samples with coarse-only
jitter, retaining this physical shell and uniform-height coverage. See the
[depth budget comparison](stokes-performance-audit.md#reduced-depth-budget).
