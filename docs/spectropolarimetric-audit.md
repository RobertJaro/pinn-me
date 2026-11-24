# Spectropolarimetric inversion audit — 23 September 2026

The strongest findings are an insufficiently opaque lower boundary in the
MURaM synthetic targets, under-resolved depth integration in the inversion,
and a substantially underconstrained atmospheric parameterization. The tests
do **not** support a generally broken SIREN, curl derivative, Q/U synthesis,
or common observed-continuum division. Increasing polarization weights alone
does not address the first two problems.

The [HMI ray-sampling follow-up](hmi-ray-sampling-audit.md) localizes the depth
problem to poorly sampled, depressed umbral formation layers. It adds uniform
physical-height coarse sampling and checks the actual WCS-to-atmosphere path.

This investigation used the working checkout, native MURaM iteration 200000,
controlled atmospheres and the supplied validation image. The failing remote
checkpoint and its original prepared observations were not available. Thus
these are reproducible code/physics findings, not a replay demonstrating that
one patch repairs that exact run. Earlier changes already present in the
checkout, including spectral/geometric precision fixes, are not counted as
new discoveries here.

## 1. Synthetic lower boundary: confirmed physical inconsistency

Synthetic generation previously selected geometric samples using native
MURaM `log(tau)` bounds of -5 to +1. Those labels are not the optical depth
computed by this package's EOS and continuum opacity. In eight selected native
columns, the resulting continuum optical depth at 6173 Å at the lower boundary
was **0.80–10.16**. An unpolarized Planck boundary is not a reliable deep-atmosphere
approximation when the boundary remains visible.

Holding the native atmosphere and upper boundary fixed, extending only the
lower boundary changed one quiet-column I profile by **123% of its original
continuum proxy** (wavelength RMS). In a strong-field column the changes were
**32.5% in I, 9.9% in Q, and 12.3% in U** in the same continuum units. Increasing
the sample count from 193 to 1537 with the old endpoints changed I by at most
0.034%: this particular discrepancy is a boundary problem, not quadrature.
Extending only the upper boundary had a smaller, but measurable, effect
(at most 0.735% in I in this sample).

This makes a synthetic recovery experiment inconsistent: generation and
inversion can give different spectra for the same native atmosphere when the
inversion includes a deeper shell. It does not prove that no alternative
atmosphere can fit the old targets, but it invalidates physical recovery claims
from those targets without further checks.

Generation now checks the actual continuum optical depth over the synthesis
wavelengths and extends the sampling downward until every selected column is
at least 20 in continuum optical depth. This bounds direct lower-boundary
transmission by `exp(-20)` within the discretized transfer calculation; it
does **not** certify quadrature convergence, top-boundary completeness, or the
absolute opacity model. The resolved grid and boundary criterion are recorded. Native
data that cannot support the required boundary fail explicitly. Regenerate
synthetic targets and rebuild their preparation caches after this change;
changing only the inversion cannot repair already stored spectra.

The corrected generator was exercised on all 128 sparse native columns. The
worst initial continuum optical depth across columns and synthesis wavelengths
was **0.230**. It extended the common native grid to +3 (257 samples), giving
a minimum depth of **112.8**. Extending further to +3.5 at the same spacing
left all profiles unchanged. Doubling 257 to 513 samples still changes I by
0.041% RMS / 0.750% maximum of the corrected per-pixel far-wing continuum;
Q/U/V RMS differences are 0.0078%/0.0128%/0.0129%. This separates the fixed
boundary defect from the remaining quadrature error.

Evidence: `data/deep_audit_rt/native_{endpoint,boundary}_results.json` and
`data/deep_audit_data/`. These local audit artifacts use sparse native columns,
not the entire snapshot.

## 2. Depth integration: confirmed numerical problem

The old inversion used 25 coarse plus 16 adaptive samples across a -1 to
+1.5 megameter photospheric shell. Controlled FALC tests at these settings show
1.5% continuum-normalized I error; a smooth field/velocity-gradient fixture
reaches 6.6%. Repeating only the sampling jitter gives about **5.4% I variability**
for FALC. These intensity errors are wavelength RMS divided by the reference
outer-filter intensity mean; the variability is RMS sample standard deviation
across eight seeded jitter draws in those same units. The reference is a dense
integration of the same atmosphere.

Eight native columns (four strongest, four quietest) are more demanding.
The following values are residual L2 norm divided by reference component L2
norm, aggregated across those columns and six HMI filters. Both sides use the
same geometric shell and physical atmosphere; this isolates depth integration
from the preceding boundary problem. The common height zero is explicitly
chosen as the median native tau=1 height across the 128-column sparse sample;
individual columns retain their Wilson depression. This is an assumed
registration, not a recovered height zero from the unavailable run. All
resolutions use float64 and the same 121-point HMI spectral quadrature plus
its continuum endpoints.

| Coarse + fine samples | I error | Q error | U error | V error |
|---|---:|---:|---:|---:|
| 25 + 16 | 19.3% | 171% | 225% | 566% |
| 64 + 32 | 1.56% | 35.5% | 37.5% | 33.3% |
| 128 + 64 | 0.40% | 2.43% | 1.60% | 1.96% |
| 3073 + 0 versus 1537 reference | 0.048% | 0.050% | 0.141% | 0.102% |

These numbers are not errors measured on the failing network. They establish
that the old quadrature can make errors larger than the polarized signal even
with known, fixed atmospheres. Jitter during training is a plausible contributor
to unstable intensity fitting. Validation itself is deterministic, so the
speckled validation image cannot be attributed directly to fresh validation
jitter; the learned spatial field can inherit unstable training updates.

The adaptive positions are deliberately detached from autograd. This is
reasonable for a converged quadrature of a continuous physical integral.
At 25+16, however, derivatives with respect to temperature/pressure scaling
disagree with finite differences of the re-adapted calculation by 19%/42% in
one FALC fixture. Frozen-node finite differences agree with autograd to about
1e-9. This distinguishes discretization/adaptation error from a broken RT
backward pass. At 128+64 the corresponding discrepancies fall below 0.7%.

Changes:

- `depth_sampling.randomize_training` supports a deterministic training control;
  evaluation remains deterministic.
- The initial high-resolution Stokes-only control used **128+64, jitter disabled, batch 64**, with
  uniform physical-height coarse placement after the ray audit, and a fresh
  `muram_stokes_only_uniform192_v03` directory. The number of atmosphere
  samples per batch is comparable to the old batch of 256, but the sequential
  RT solve is longer; this is not a speed improvement guarantee.
- `diagnostics.depth_convergence.measure_depth_convergence` reports fixed-pixel
  quadrature bias and stochastic variability while preserving RNG state.
- Thin-path adaptive masses now use `exp(-tau) * -expm1(-delta_tau)`, avoiding
  float32 cancellation that previously erased opacity weights.
- Optical-depth diagnostics preserve signed intervals consistently with the
  formal solver instead of hiding folded paths with absolute distances.

The table above measures the original FALC-shaped coarse placement. Uniform
physical-height placement further reduces the native-column errors at the
same 128+64 budget; see the ray audit. 128+64 remains a starting point, not a
universal accuracy guarantee. Check another doubling on the fitted atmosphere
before trusting sub-percent residuals.

The current faster preset uses **96+48 with coarse jitter**, deterministic fine
quantiles and batch 64. See the [depth budget comparison](stokes-performance-audit.md#reduced-depth-budget)
for measured errors and limitations.

## 3. Forward conventions and SIREN: controls passed

On 128 native columns at identical physical sample positions, generation and
the production inversion forward composition agree to about **2.2e-15 in I
and 1.7e-16 in Q/U/V** in atlas units (float64). A permanent test additionally
passes strong-field columns through actual red-to-blue FITS, the HMI loader,
Carrington/observer velocity conventions, polarization bases, radiance scaling,
and filter integration. It passes at float32 FITS accuracy. This does not test
historical synthetic products or certify absolute atomic physics.

The production 256-wide, eight-layer SIREN with vector-potential magnetic field
was also fitted to four fixed, self-generated strong-field profiles. With
41 deterministic depth nodes, Q/U/V weights 200, clipping 0.1, learning rate
1e-3 and 1000-step warmup, after 300 updates the component errors relative to
target RMS were:

| I | Q | U | V |
|---:|---:|---:|---:|
| 0.40% | 2.39% | 3.17% | 1.35% |

The initial magnetic component RMS values of roughly 60/85/153 G escaped to
strong-field values. Direct magnetic output also fits this control. This is a
self-consistency test, not independent validation or proof of full-image
convergence. It is sufficient to reject the claim that this SIREN/curl path
intrinsically cannot learn Q/U.

Reproduce from the repository root:

```bash
PYTHONPATH=src python -m prom3theus.diagnostics.commands.check_stokes_recovery \
  --representation vector_potential --warmup 1000 --steps 300
```

An independent finite-slab Milne–Eddington analytic test now checks transfer
convergence. Curl parameter gradients also have a finite-difference regression.

A harder control fitted four actual saved native profiles (two strong cores,
two quiet pixels) with 193 deterministic nodes. After 300 updates, loss fell
from **44.07 to 0.02567**. Relative RMS errors were **I 13.5%, Q 32.0%, U 9.8%,
V 3.7%**; predicted polarization amplitudes were about 94–102% of target RMS.
This recovers polarization but remains an imperfect fit, especially I and Q.
It uses the old truncated targets and plane-parallel columns, not a replay of
the full raster or the corrected synthetic experiment. It demonstrates that
near-zero Q/U is avoidable; it does not establish accurate native-atmosphere
recovery. Details and plots are in `data/deep_audit_model/report.md`.

## 4. Continuum normalization: correct operation, consequential weighting

Both target and prediction are divided by the same detached observed proxy.
For linear MSE this is exactly a pixel weight

`1 / max(observed_outer_filter_mean, 0.1)^2`.

It cannot erase the intensity error by normalizing the prediction by itself,
and an exact raw fit remains an exact normalized fit. It can strongly alter
the optimum of a restricted shared model and the optimizer's conditioning.

In the 128 native-column sample, the HMI outer-filter proxy divided by the
raw far-wing intensity has median 0.990, but minimum 0.676. The far wings are
themselves only a practical continuum comparison. Three pixels hit the 0.1
floor; inverse-square weights span **0.332 to 100**, a factor of about 301.
Errors in dark regions therefore have far more influence. With actual noisy
data, uncertainty in the observed denominator also affects the residuals.
After the synthetic boundary correction, two rather than three sampled pixels
hit the floor; the proxy bias and approximately 301-fold weight range persist.

Keep this option explicit and compare enabled/disabled normalization on the
same fixed pixels and initialization. Do not simultaneously change the network,
weights and quadrature. An asinh transform changes the residual geometry again;
it does not fix inaccurate transfer or an inconsistent synthetic boundary.

Current diagnostics log raw atlas-unit per-component residual RMS and
objective-space component MSE. Individual signed profiles are
plotted for strong Q/U, strong V and dark pixels within the bounded
validation sample. Signed spatial medians can cancel real polarization and
are insufficient evidence of fitting quality. The original maps and residuals
still demonstrate the supplied run's underfit.

Optimizer logging reports the shared learning rate. Optional loss balancing
reports its gradient diagnostics when enabled. A component's fraction of scalar
loss alone does not establish its fraction of the parameter update.

## 5. Other fixes and physical limitations

- **Synthetic provenance:** old and new generators shared a serialization schema
  despite numerical changes. Products now store source/resource fingerprints;
  legacy/mismatched products warn even on preparation-cache hits. A hash change
  is conservative evidence to check closure, not proof of a spectral change.
- **Centrifugal term:** normalized momentum counted angular frequency twice.
  Its coefficient is corrected and tested against the dimensional equations
  across time scales. True solar equatorial centrifugal acceleration is only
  about 0.0057 m/s², so this is not a credible explanation for Q/U collapse and
  is inactive in the Stokes-only run.
- **Run script:** `scripts/muram/run_inversion.sh` contained two active inversion
  commands, joint followed by Stokes-only. It now launches only its documented
  joint run; the isolated Stokes command is an explicit commented alternative.
- **Scattering source:** continuum scattering is currently thermalized into
  the LTE Planck source. Its fraction is below roughly 0.1% in sampled ordinary
  photospheric conditions, but hot/dense unphysical line-shell solutions can
  make this approximation inappropriate. It is not established as the present
  failure's cause.
- **Synthetic resolution:** striding selects native columns; it is not spatial
  PSF convolution or detector-footprint averaging. Real HMI comparisons need
  separate treatment of spatial/spectral response, noise and unresolved structure.
- **Runtime:** disabling physics does not remove the LTE transfer calculation
  or the spatial derivatives required by a vector-potential magnetic field.
  AIA also remains enabled for validation in the control preset. Profile
  transfer, curl/backpropagation, data loading and validation should be timed
  separately before interpreting slow iterations as an accidentally active
  physics loss.
- **Geometric registration:** choose one documented native-to-inversion height
  offset. Independently aligning each column's tau=1 surface would erase Wilson
  depression. Agreement at fixed optical depth and agreement at fixed geometric
  height are different recovery tests.

## 6. Relation to established inversions and project direction

HMI provides four Stokes components at six filters. The present model permits
T, pressure, microturbulence, three magnetic components and three velocities
to vary with position, height and time. The data cannot independently determine
all of that structure. At one viewing direction, spectra directly constrain
LOS velocity, not the two transverse velocity components. Good spectral fits
also do not imply unique pressure or geometric height stratifications.

VFISV deliberately reduces free parameters, includes the filter response and
uses LM optimization with restarts/constraints. Its published weights multiply
residuals before squaring; this repository's coefficients multiply squared
residuals directly. Numerical weight values are therefore not interchangeable.
This supports using a compact inversion baseline before expanding the shared
atmosphere. [Centeno et al. 2014](https://arxiv.org/html/1403.3677v1).

SIR's increasing-node cycles provide a useful model for gradually releasing
depth structure. Start with few atmospheric nodes and use response information
to justify additional freedom. [SIR author manual](https://github.com/BasilioRuiz/SIR-code/blob/master/manual/manual.tex).
Recovering a geometric height scale is a further physical inference; sunspot
work has combined optical-depth inversions with force/divergence constraints
to construct it. [Puschmann et al. 2010](https://arxiv.org/abs/1007.2779).

Recommended sequence:

1. **Repair the experiment first.** Regenerate a small target set with verified
   boundaries and provenance; verify fixed-atmosphere closure and depth
   convergence through the actual loader/operator. Include quiet Sun, umbra,
   penumbra, both polarities and several azimuths.
2. **Establish a compact per-pixel baseline.** On synthetic data first hold
   thermodynamics known and fit B/LOS velocity, then release a few temperature
   nodes and broadening parameters. Compare a node-based LM/L-BFGS or ME solution
   against the SIREN on exactly the same profiles. Use multiple starts where
   linear-polarization sensitivity is weak. An independent SIR/STiC forward
   comparison is still needed to validate absolute line physics.
3. **Overfit a fixed representative batch before a full image.** Use deterministic
   converged depth sampling, fixed response, existing SIREN and no physics.
   Report signed component residuals and improvement over zero polarization.
   Set and verify numerical errors below the desired fit precision; assess
   residuals against that floor and real measurement uncertainty, not arbitrary
   total-loss thresholds.
4. **Add spatial sharing and atmospheric freedom gradually.** Initialize from
   the compact inversions. Compare errors at optical depth/response-weighted
   heights before claiming recovery of the full geometric atmosphere.
5. **Add physics only after reliable spectral fitting.** Test the physics
   residuals on the native MURaM truth using the same EOS and units first.
   Radiative MHD is not globally adiabatic or force-free. Reintroduce one
   physically appropriate constraint at a time; monitor the spectral fit and
   actual gradients. Do not put every loss into automatic balancing to conceal
   inconsistent equations or poorly scaled residuals.
6. **Move to observations with a stated error model.** Use noise-normalized
   residuals plus explicit component priorities; assess passband, PSF and
   calibration mismatch separately. Treat an irreducible systematic residual
   as model discrepancy rather than forcing arbitrary atmospheric structure.

Continuity alone does not require velocity to grow with height. The relation
`rho * v * area = constant` follows only under a stationary one-dimensional
flow-tube assumption. The full continuity equation also contains density time
evolution and horizontal transport, and admits zero velocity everywhere.

The immediate priority is **converged and consistent spectral fitting**, while
retaining the SIREN as a viable representation. Full 3D magnetic/dynamic recovery
is a subsequent identifiability and physical-model problem.

## Validation status

Focused objective/normalization, diagnostic sampling/rendering, configuration
control and application checks passed: **80 tests**. The single-run MURaM shell
dispatch regression also passed. The RT audit's 52 focused tests, the
model/training/physics audit's 119 tests and the MURaM audit's 58 tests passed
(these suites overlap; do not sum them as unique coverage). CUDA was not tested.

A broader sweep recorded **765 passed, 7 failed, 7 skipped** in
`data/deep_audit_diagnostics/regression.log`. The remaining failures are:

- A test loading the absent `hmi_local_minimal.yaml` preset.
- Two tests expecting an older set of enabled dynamic boundary penalties.
- A test expecting the old enabled disambiguation settings in the potential preset.
- Two AIA rendering tests whose model fixture lacks `shared_objectives`.
- The two-process Gloo test cannot resolve its local loopback address in this
  environment.

A separate script sweep also finds two failures concerning the existing SDO
run script's command/argument-handling contract. The new MURaM dispatch check
passes. The broad checkout is therefore **not** being reported as fully green.
