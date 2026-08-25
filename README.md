# pinn-me
Differentiable spectropolarimetric inversion with Milne-Eddington and
depth-stratified LTE forward models.

## HMI transmission preprocessing

HMI phase-map data are downloaded and converted to a compact transmission
profile file before an inversion is started.  A registered JSOC email address
is required by the DRMS export service:

```bash
python -m pme.data.hmi_transmission \
  --input '/path/to/hmi_stokes/*.fits' \
  --email name@example.com \
  --output /path/to/hmi_transmission/
```

For every FITS acquisition, the command follows `INVPHMAP` from the matching
`hmi.ME_720s_fd10` record and uses the FITS `HCAMID`. It downloads each unique
calibration only once, stores its full 128x128 field of six profiles, and writes
`manifest.json` mapping acquisitions to calibrations. Set
`data.*.transmission_profile_directory` to this directory. During loading,
every HMI full-disk image or cutout is mapped back onto the detector and assigned
the matching calibration. Dataset workers interpolate that spatial calibration
for the pixels in the current batch, so only batch-local spectral offsets and
weights are transferred to the GPU. The inversion module never opens calibration
files or stores full response maps. The common -0.65 to +0.65 Angstrom line region
is synthesized explicitly for every filter. All remaining modeled passband
throughput is integrated offline to seven blocking-filter standard deviations
with chunked composite quadrature and added as an unpolarized continuum
contribution, following the VFISV optimization.
The phase-map product does not contain the measured fixed-stack/front-window
curve, so these files remain reconstructed responses: they use the published
mean element contrasts, 8.43 Angstrom blocking-filter width, and +16 mAngstrom
untuned-stack center. These assumptions are stored in each profile archive.
Training itself performs no network access. Regenerate the directory with
`--overwrite` whenever the transmission format changes.

## Stokes objective

Spherical inversions use an elementwise mean-squared objective on the
loader-scaled Stokes profiles after the configured linear-I/asinh-QUV transform.
For example:

```yaml
normalization:
  asinh_alphas: {Q: 5.0e-2, U: 5.0e-2, V: 5.0e-2}
```

The normalization setting is stored in checkpoints, and resuming with a
different transform is rejected. Use a fresh `base_path` when changing it.

`train.I/Q/U/V` and `train.stokes_loss` report the selected wavelength-summed
objective. The existing `valid.I/Q/U/V` values remain wavelength-averaged MAE
diagnostics for continuity with older runs; `valid.objective` reports the exact
weighted validation counterpart of `train.stokes_loss`.

## Continuous physics regularization

Physics constraints use a collocation stream that is independent of the Stokes
pixel batches. The sampler draws a fixed global Sobol point set continuously in
the full training time, latitude, and seam-safe longitude bounds, then partitions
that set across distributed ranks. Consequently `num_points` is the total number
of physics points per optimizer step, not a per-GPU count. The same seed and
global step reproduce the same global point set after checkpoint resume.

Constraint schedules live under `physics.constraints`; the top-level `weight`
mapping contains the four Stokes-component weights:

```yaml
physics:
  num_points: 4096
  seed: 17
  domain: full_train
  radius_range_Rs: [1.0, 1.0]
  normalization: raw
  constraints:
    divergence: {type: linear, start: 0.0, end: 1.0e-6, iterations: 100000}

weight:
  I: 1.0
  Q: 1.0
  U: 1.0
  V: 1.0
```

The Stokes and physics terms are evaluated separately and added only at the
final objective. Constraint derivatives are constructed lazily, so a
divergence-only run does not build velocity, induction, current-density, or
second-derivative graphs. Physics configuration and the resolved global domain
are fingerprinted in checkpoints; changing either requires a fresh `base_path`.
The ready-to-run controlled experiment is
`config/hmi/hmi_subframe_20240323_physics.yaml`.

## Spherical profile evaluation

Spherical inversion artifacts retain their forward models, learned calibration
corrections, per-acquisition HMI response mapping, and per-instrument Stokes
normalization. To reproduce an HMI observation with the same geometry and
spectral response used in training:

```python
from pme.evaluation.loader import SPINNMEOutput

output = SPINNMEOutput("/path/to/inversion.pme", instrument_id="HMI")
result = output.load_hmi_observation(
    "/path/to/hmi.s_720s.YYYYMMDD_HHMMSS_TAI.N.I0.fits",
    batch_size=8192,
    denormalize_stokes=True,
)
stokes = result["stokes"]  # [y, x, I/Q/U/V, wavelength]
```

The response file is selected from the acquisition mapping saved with the
inversion. Pass `spectral_response_file=` explicitly if the calibration archive
was moved. `synthesize_observation` provides the lower-level interface for
precomputed coordinates, observer transforms, and sampled response arrays.

## Depth-stratified LTE synthesis

The `pme.lte` package extends the Milne-Eddington implementation with a
differentiable, depth-stratified LTE forward model for the Hinode/SOT-SP
Fe I 6301.5 and 6302.5 Angstrom lines. The forward model is usable without an
inversion network: construct a `StratifiedAtmosphere` on an increasing
`log_tau500` grid, then pass the atmosphere and an absolute air-wavelength grid
in Angstrom to `LTESynthesizer`. The returned tensor uses the final two axes
`[I, Q, U, V]` and wavelength; leading batch axes are preserved.

The Hinode inversion jointly synthesizes both `FeI_6301.5008` and
`FeI_6302.4932` through one atmosphere, pinned STiC thermodynamic lookup,
propagation matrix, and polarized formal solution. Both line IDs are explicit
in the supplied YAML files. The
entry point refuses to train if either record is omitted or its laboratory air
wavelength lies outside the observed FITS wavelength grid.

The LTE interfaces use SI internally unless an argument name documents another
unit. Temperatures are kelvin, velocities are m/s, magnetic-field components
are gauss, gas pressure is pascal, number densities are m^-3, and wavelengths
at the public Hinode/instrument interface are standard-air Angstrom. Before
evaluating photon frequency, Planck radiance, continuum opacity, or Doppler
width, those labels are converted to vacuum Angstrom with the differentiable
VALD/Piskunov inverse of the Morton (2000) standard-air relation. The
`tau500` reference opacity is evaluated at exactly 5000 vacuum Angstrom.
The atmosphere predicts a co-rotating velocity residual and magnetic vector in
one fixed Heliographic-Carrington Cartesian scene frame. Immediately before
synthesis, rigid Carrington rotation is added to the velocity and each
instrument projects the resulting vectors into its per-pixel
`[+Q, +U, toward-observer]` Stokes basis; the radiative-transfer LOS velocity
is the negative toward-observer component and remains positive for a redshift.
The magnetic vector uses a direct linear readout with a scale of `100 G` per
raw unit. Velocity uses a smooth component-wise bound with a `1000 m/s` local
scale and a configured `100 km/s` asymptote. Temperature and gas pressure use
smooth natural-log readouts whose asymptotes are the verified STiC lookup
bounds. A scaled `tanh` retains the configured local natural-log sensitivity
without a hard clamp or an implicit `ln(10)` amplification. The deep
Fourier MLPs use activation-aware,
variance-preserving random hidden weights, Xavier-random readout weights, and
independently random biases. This prevents successive SiLU layers from
suppressing coordinate-dependent Fourier variance into an apparently uniform
DC field. The thermodynamic readout rows start at zero perturbation so the
initial atmosphere is exactly radial FALC; vector readouts remain random. A
single-view Stokes spectrum does not
constrain the transverse velocity without additional dynamical physics.
Stokes azimuth is pi-periodic; Q and U therefore rotate through twice the
physical azimuth. SolarSoft `sp_prep` Level-1 processing already uses `CROTA2`
to rotate Q/U from the FPP frame into the solar frame with +Q along solar
east-west (HPC +X). The Hinode loader therefore uses a zero-degree reference
angle and does not rotate the calibrated profiles a second time.

For each Hinode pixel, the loader derives the ray cosine
`mu = sqrt(1 - (Dsun*sin(rho)/Rsun)^2)` from the Level-1 `XCEN`, `YCEN`,
`CRPIX2`, `CDELT2`, and degree-valued `CROTA2` pointing keywords. In the exact
helioprojective convention, Tx is longitude, Ty is latitude, and
`cos(rho)=cos(Tx)*cos(Ty)`. This uses the exact ray impact parameter
`Dsun*sin(rho)`, rather than either `hypot(Tx,Ty)` or a ratio of angular
small-angle approximations. The apparent solar semidiameter is
`asin(Rsun/Dsun)` using Astropy's geocentric solar ephemeris; the Level-1 files
do not carry an exact Hinode-spacecraft distance. The operational inversion
traces each observer ray through a spherical shell in the solar reference
frame. Intersections are parameterized from the ray's photospheric Carrington
point in solar-radius units, so float32 geometry never subtracts distances of
order 1 AU. The formal solver integrates `alpha500*K` over realized path
length; it does not apply a separate `1/mu` approximation. The atlas calibration
separately uses the same `mu`
with the Neckel continuum center-to-limb law: quiet-Sun intensity selection is
performed after converting the observed continuum to a disk-center-equivalent
level, and the single detector conversion is fitted against the expected local
`I_c(mu)`. The calibrated profiles remain in the common disk-center atlas unit;
they are not divided by `I_c(mu)`. Magnetic and velocity vectors are projected
geometrically into the observer frame, so applying a separate `mu` correction
to their LOS components would be incorrect. MHS is evaluated in the solar frame and contains
no `mu`. The Hinode entry point
exposes no spatial-PSF configuration because a detector kernel would need the
local two-dimensional scan/slit displacement basis in physical Mm coordinates.

The near-side surface intersections are transformed to Heliographic Carrington
Cartesian coordinates. A raster-centred, observer-independent gnomonic chart
conditions the two spatial network inputs without changing the physical scene.
The loader retains each scan-column time, global ray origin and direction,
per-pixel Stokes basis, scene-chart basis, coordinate formulas, and the exact
network affine in raster/checkpoint metadata. A later HMI or second-observer
adapter therefore only needs to provide rays and its calibrated Stokes basis.

The anomalous-dispersion coordinate is red-positive,
`(lambda-lambda_component)/delta_lambda_D` (implemented equivalently in
frequency). This matches the established PINN-ME/SIR wavelength convention.
It is recorded in checkpoint metadata because reversing this odd profile flips
the magneto-optical rho coefficients and can produce an incorrect Stokes-U
shape even though Stokes-I absorption is unchanged.

A minimal standalone calculation follows the public API:

```python
import torch

from pme.lte import LTESynthesizer, StratifiedAtmosphere

log_tau500 = torch.linspace(-5.0, 1.0, 64)
atmosphere = StratifiedAtmosphere(
    log_tau500=log_tau500,
    temperature=(5770.0 + 250.0 * log_tau500).unsqueeze(0),
    velocity_field=torch.zeros(1, log_tau500.numel(), 3),
    magnetic_field=torch.zeros(1, 64, 3),
    microturbulence=torch.full((1, 64), 1.0e3),
    gas_pressure=torch.logspace(-1.0, 5.0, 64).unsqueeze(0),
)
wavelength = torch.linspace(6300.8, 6303.2, 112)
stokes = LTESynthesizer()(atmosphere, wavelength)
```

Hinode spectra must be synthesized on the wavelength array derived from each
FITS header rather than on an idealized fixed grid. The LTE Hinode loader keeps
the complete 112-sample spectral window so that both Fe lines remain present.
For local development, point the small-patch loader or inversion configuration
at a narrow file glob first; a full 1024-step raster is substantially more
expensive than a single pixel or scan-line subset. For example, the opt-in
integration tests use:

```bash
export PINN_LTE_HINODE_GLOB='/path/to/hinode/raster/*.fits'
pytest -m integration tests/lte/test_hinode.py
```

After updating `data.files`, `base_path`, and `work_directory` in the example
configuration, launch the conservative four-pixel inversion with:

```bash
python -m pme.inversion_lte --config config/hinode/lte_small_patch.yaml
```

The same commands are installed as `pinn-me-lte` and `pinn-me-lte-export` by
the packaged project. The Hinode configurations use a heliocentric Carrington
Cartesian atmosphere and a fixed spherical radial domain:

```text
HPC + observer WCS -> Carrington chart (x,y), ray, and Stokes basis
ray + inner/outer radii -> solar-local Carrington points through the shell
F(x,y,r-Rsun) -> log perturbations of radial FALC T, Pgas, xi; plus Cartesian v and B
Cartesian vectors -> local Carrington (r, theta, phi) -> supplied observer basis
alpha500(T,Pgas) integrated over ray distance -> tau500 and polarized transfer
```

`F` uses normalized Carrington-chart `(x,y)` and radial-height coordinates with
variance-preserving random initialization. Thermodynamic readouts are bounded
natural-log perturbations around a pinned radial STiC FALC_82 stratification;
there is no magnetic or velocity seed. The historically named `log_tau500`
array is only an
ordered, dimensionless quadrature parameter from the outer to inner radius; it
does not define geometry or optical depth. Each ray is intersected with the
configured shell from `z=-0.1 Mm` to `z=+1.5 Mm` relative to the solar radius,
then the FALC-spaced quadrature points
are placed between those endpoints. Rays that cannot reach the nominal inner
radius stop just before their tangent point instead of aborting. The formal
solver multiplies its normalized propagation
matrix by `alpha500` and the exact distance between successive intersections.
The diagnostic cumulative `tau500` is the trapezoidal integral of this absolute
opacity along the realized path. No `mu` path-length correction is used for ray
geometry.

Observer-ray synthesis optionally uses deterministic per-ray coarse-to-fine
sampling. A wide coarse trace estimates interval contributions proportional to
`alpha500*exp(-tau500)*delta_s`; detached probability quantiles then add the
configured fine points. Every coarse point is retained, a uniform probability
floor protects initially misplaced formation regions, and the final atmosphere
and polarized transfer are reevaluated differentiably on the merged ordered
grid. This is configured under `training.depth_sampling_config.coarse_to_fine`.

Physics collocation samples shared geometric-height layers in the same Carrington Cartesian
shell, with independent observed rays selected on every layer. The eligible ray
set is recomputed at each radius, so samples follow the actual observed ray
bundle rather than a rectangular longitude-latitude box. MHS balances the full pressure gradient, radial gravity, and magnetic
Lorentz force; `div B` differentiates Carrington magnetic components with
respect to Carrington position. The mean radial `tau500=1` surface is anchored
at the solar radius by integrating continuum opacity from the shell top. Tau
along observer rays is computed independently for synthesis and diagnostics.

There is no explicit smoothness, derivative, node, or sparsity regularizer.
Smoothness comes from the finite-bandwidth Fourier-feature MLPs and smooth
decoders, evaluated at continuously changing coordinates. Public spatial
coordinates are an observer-independent Carrington gnomonic chart in Mm. The entry point stores
one raster-derived affine inside the shared coordinate network; the
isotropic scale is 512 native pixels, mapping the configured full raster to
approximately `x=[-1,1]` and `y=[-0.5,0.5]`; radial height is scaled by `1 Mm`.
These ranges keep inputs order-unity while the full-resolution
configuration limits the shortest representable spatial period to approximately
64 native pixels.

Although the loader retains every scan-column timestamp, the default single-
raster atmosphere is explicitly static and ignores the time coordinate. In a
slit scan, time and scan position are nearly perfectly correlated, so a joint
`F(t,x,y,r)` model would be non-identifiable away
from the observed scan trajectory. The implemented atmosphere is therefore
strictly `F(x,y,r) -> atmosphere`; scan time remains data
provenance rather than a network input.

The chosen frame is co-rotating rather than inertial because the inversion is a
quasi-static snapshot. The learned velocity is a residual expressed in the
co-rotating Carrington Cartesian basis. Polarized synthesis explicitly adds the
rigid sidereal Carrington velocity `Omega_Carrington cross r` before projecting
the total inertial velocity first into the local spherical basis and then into
the supplied per-ray observer Stokes basis. Solar rotation therefore cannot be
absorbed into the learned velocity field. Physics losses continue to use only
the learned co-rotating residual. The transform is instrument-independent; an
observation adapter provides the rays and its right-handed observer basis.

The validation metrics evaluate a deterministic stride-selected subset of the
same inversion pixels without shuffling and on the deterministic depth grid.
`valid.stokes_loss` is the data-fit diagnostic; `valid.loss` additionally uses
the configured fixed magnetohydrostatic-equilibrium, magnetic-divergence, and pressure-boundary weights.
`data.validation_stride` and `validation_batch_size` bound this cost;
`check_val_every_n_epoch` controls the actual Lightning validation cadence as
well as checkpointing. It is a stable data-fit/checkpoint diagnostic, not a
held-out generalization score. All observed pixels remain in the inversion by
design; independent accuracy claims require synthetic truth, withheld pixels,
or an external reference inversion.

The full-resolution YAML does not redefine an epoch as all 524,288 detector
pixels. It draws 16,384 distinct valid pixels uniformly without replacement in
each epoch and redraws that subset on the next epoch. After 50 epochs, a valid
pixel has about 79.6% probability of having participated at least once;
the full raster remains the coordinate domain. Validation uses a separate,
fixed 1/16 area-stride sample: a 4x4 detector lattice producing 128x256 map
samples on the native raster. This avoids spending most wall time repeatedly
synthesizing an essentially redundant dense raster merely to define an epoch.
Both choices and their exact sample counts are stored in checkpoint metadata.

The production polarized solver is the frozen-layer matrix-exponential method.
Unused alternative formal solvers were removed so configurations and
checkpoints have one unambiguous radiative-transfer implementation. Before
training, the pinned STiC/Wittmann solver is evaluated on a regular
`(log10(T), log10(Pgas))` grid. The resulting table stores total continuum
extinction, mass and electron density, physical neutral-H density, and
`n(Fe I)/U(Fe I)`. Runtime uses differentiable tensor-product cubic lookup and
one algebraic Boltzmann factor for each reviewed Fe I lower level. It contains
no charge-neutrality, hydrostatic, Newton, bisection, or other iterative EOS
solve. The separately packaged Barklem/Saha table is a reference and regression
resource; it is not part of the production forward graph.

The configured 25-point grid is a scalable training quadrature, not a claim of
exact depth convergence. In the current physical-radiance convergence test
against a 101-point result, this grid has a Stokes-I RMS error of about
`6e-3 Ic`, while 51 points reduces it to about `1e-3 Ic`.
These figures depend on the atmosphere and table revision, so a quantitative
final inversion must repeat its last refinement on a 51-point-or-denser
randomized grid and verify that the inferred profiles and Stokes residuals are
stable.

The LTE Stokes objective uses the same scaling and component weights as
`config/hmi/hmi_fd_20240323.yaml`: Stokes I remains linear, Q/U/V use
`asinh_alpha=0.01`, the loss is MSE, and I/Q/U/V each have weight 1. The
transform is applied identically to synthesized and observed profiles before
forming residuals. Hinode is calibrated against a checksum-pinned absolute FTS
disk-center atlas and the Neckel continuum center-to-limb variation. The loader
selects quiet-Sun pixels, derives one detector-to-radiance scalar from their
continuum, applies it unchanged to I/Q/U/V, and stores the profiles in the fixed
unit `I_c,atlas(mu=1)`. This keeps values near unity without a per-pixel or
raster-continuum normalization. The synthesizer uses that same immutable atlas
continuum radiance as its output scale; there is no learned radiometric gain.
Validation plots use the exact optimization tensors without additional
normalization. The atlas provenance, quiet-Sun selection definition, and fitted
detector conversion are saved in the checkpoint.

The optimizer uses fixed-weight named objectives. Every enabled loss is
logged at each training step:

- `train.stokes_loss` is the only observation objective.
- `train.magnetohydrostatic_equilibrium` enforces the full vector balance between pressure,
  gravity, and the magnetic Lorentz force in Carrington Cartesian coordinates.
- `train.magnetic_divergence` enforces `div(B)=0` in that same frame.
- `train.mean_radial_optical_depth_anchor` constrains the observed-domain mean
  radial `log10(tau500)` at the solar radius to zero without prescribing a
  pressure at either shell boundary.

The gravity comes from the pinned FALC/STiC resource bundle. The atmosphere predicts magnetic and
velocity vectors in the fixed Carrington Cartesian basis, and only the
radiative-transfer path projects them into each observer's Stokes basis.
The LTE pipeline deliberately does not expose time-dependent momentum, induction,
continuity, or learned tau-surface alternatives.

Stokes and physics weights must be non-negative scalar numbers. Loss schedules
are unsupported. Adam accepts either a scalar `learning_rate` or a per-step
exponential schedule with `start`, `end`, and positive `iterations`; `auto` uses
the trainer's actual estimated optimizer-step count. `train.loss` is the fixed weighted sum;
there are no profile-smoothness or generic parameter penalties.

`training.physics_config.normalization.length_m` sets the derivative length for
`div(B)`. MHS is normalized by the detached layer mean of the pressure-equivalent
sum `P + rho*g*L + |B|^2/mu0`; `div(B)` is normalized by the detached layer mean
of `|B|`. The optical-depth anchor is the square of the observed-domain mean
radial `log10(tau500)` at the solar radius. `valid.pressure_increasing_fraction` reports
the fraction of adjacent validation-depth intervals whose pressure increases
inward. MHS constrains pressure and magnetic field jointly, but does not impose
monotonic temperature or velocity profiles.

During training, the LTE visualization callback writes one update per complete
validation epoch under `work_directory/atmosphere_visualizations`. One Stokes
dashboard contains wavelength-integrated absolute reference maps, prediction
maps on the same color scale, and signed observed/predicted spectra at the
validation ensemble as median signed profiles with 16--84% bands and integrated
predicted-versus-reference scatter; both Fe I laboratory wavelengths are marked.
Integration uses the actual detector wavelength coordinates. Separate
parameter, radial-magnetic-field, and radial-velocity figures place quantities
in columns and selected spherical-shell heights in rows. These atmosphere maps
are evaluated directly on regular physical Carrington longitude-latitude shell
surfaces; they are not reconstructed from detector rays. They include inferred
`T` and `Pgas`, STiC-derived `rho`, microturbulence, `B_r`, and `v_r`.
`visualization.slice_sampling` independently controls their angular resolution,
radial-section resolution, and evaluation batch size. Only the requested shell
surfaces and radial plane are instantiated, never a full
longitude-latitude-radius cube. Stokes maps retain the observer-facing chart in
Mm because those samples are image measurements rather than spherical layers.
Local figures are
saved at the configured DPI and WandB uploads that exact PNG instead of
rerasterizing the live Matplotlib figure at a lower default resolution.
An optional `visualization.meridional_slice.longitude_deg` directly selects a Carrington
longitude and samples a true physical constant-longitude radial plane at the
independent `slice_sampling` resolution.
It adds parameter, `B_r`, and `v_r` sections and never computes optical-depth
slices.
A separate optical-depth diagnostic integrates `alpha500 ds` along those
subsampled validation rays, compares the median and 16--84% range with the FALC radial
shell labels, and maps `log10(tau500)` at the layer nearest `R=R_sun`.
Integrated maps and agreement scatter retain all validation samples; only the
complete 4x112 spectra used for percentile curves are capped by
`ray_sampling.max_profile_samples`. `ray_sampling.max_pixels` separately caps
the observer rays used by the optical-depth diagnostic.

The supplied LTE configurations use a `WandbLogger` with project
`pinn-me-lte`; figures, named losses/metrics, and the complete configuration
are logged to Weights & Biases while the PNG files remain available locally.
Set `logging.type: csv` only for an explicitly offline run. Configure cadence
and layers under the top-level `visualization` mapping, or set
`visualization.enabled: false` to disable the image callback.

The trainer additionally writes best/last checkpoints and an `on_exception.ckpt`
for recoverable failures or interruptions. Resume loading verifies the LTE provenance
and scientific configuration before accepting checkpoint state.
The diagnostic config uses a constant Adam learning rate; the full-resolution
config uses the explicitly configured exponential learning-rate schedule.

The standalone recovery command is a compact forward/objective gradient smoke
test. It fits twelve explicit atmosphere coefficients to a noiseless two-line
Stokes spectrum with fixed pressure; it does **not** exercise the coordinate
network, MHS collocation, Hinode loader, or callbacks and is not presented
as an end-to-end inversion validation:

```bash
pinn-me-lte-recovery --check
```

Trained checkpoints can be evaluated on a denser depth grid than the stochastic
training quadrature. For example, export 101 depth samples with:

```bash
pinn-me-lte-export \
  --checkpoint runs/hinode_lte_active_region_16x16_random/inversion_lte.ckpt \
  --config config/hinode/lte_active_region_16x16.yaml \
  --output runs/hinode_lte_active_region_16x16_random/atmosphere_101.npz \
  --depth-samples 101
```

The NPZ contains temperature, the complete Cartesian velocity vector and its
derived positive-redshift LOS component, microturbulence, Cartesian magnetic
field, the validity mask, coordinates, and provenance metadata on the original
Hinode spatial grid.
Add `--include-stokes` to export the fitted observed-grid Stokes cube and its
residual cube alongside the atmosphere. Checkpoint loading preserves the saved
float32/float64 compute dtype and rejects a different spectral grid. Array
storage is explicitly `float32` by default to bound full-raster memory; pass
`--storage-dtype float64` when that extra precision and size are intended. The
resource directory is taken from the export YAML, so checkpoints are portable
between machines, but every production resource hash must match the checkpoint.
Export compute defaults to `--device auto`: CUDA is preferred, MPS is used only
for float32 atmosphere-only exports, and CPU is the safe fallback. Float64 is
unsupported by MPS, and PyTorch does not provide the `matrix_exp` MPS kernel
needed by the polarized formal solver, so Stokes export uses CUDA or CPU. Pass
an explicit device when scheduler placement must be fixed.
`--include-stokes` on a full raster is intentionally opt-in because predicted,
observed, and residual cubes together require several GiB even in float32.

The instrument model uses a normalized 25 mA Gaussian approximation to the
*measured* pre-launch Hinode/SP spectral response, followed by differentiable
sampling at the exact observed wavelengths. This must not be confused with the
30 mA nominal resolution in the NAOJ specification. The laser measurement has
slightly asymmetric and elevated inner wings, so a Gaussian is a documented
approximation rather than a claim that the measured line-spread function is
Gaussian. The exact source and limitation are stored in
`pme/lte/data/instrument_hinode_sp.json` and copied into checkpoints.

An exact sampled spectral response can be supplied with wavelength offsets,
weights, and mandatory provenance; otherwise the documented Gaussian is used.
The existing `hinode_psf_0.16.fits` product has no web-verifiable provenance in
this repository and a detector kernel has not yet been transformed into the
per-pixel physical Solar-X/Solar-Y basis. Spatial convolution is therefore not
part of the current Hinode entry-point configuration contract.

The reviewed feature inventory is stored in
`pme/lte/data/blend_inventory.json`. Hinode is spaceborne, so terrestrial
telluric lines reported for this wavelength region are not included. The
present forward line model cannot model the documented 6302.09-A molecular
feature that may appear in cool umbrae. The example inversion therefore sets
`training.wavelength_exclude_windows_angstrom: [[6302.04, 6302.14]]`; masked
samples contribute neither to the loss numerator nor its normalization. This
mask and the resolved per-sample objective weights are saved in every
checkpoint.

### Atomic-data provenance and validation status

Atomic line parameters, abundances, partition functions, continuum-opacity
inputs, and Hinode spectral-response metadata are prepared before inversion.
Source URLs, database releases, wavelength medium, units, citations,
transformations, and checksums are recorded in the bundle manifest; generated
tables must not be edited silently. The standalone synthesis API can use the
versioned package resources, while the training entry point requires an
explicit, prepared resource bundle.

The two Fe I records deliberately separate measured database values from
adopted synthesis choices:

- NIST ASD 5.12 supplies the observed air wavelengths, excitation energies,
  level J values, the 6301-A log(gf), and experimental level Landé factors.
  The exact NIST line and level queries, access date, and response hashes are
  in `lines.json` and `sources.json`.
- NIST does not report a transition probability for the 6302-A record returned
  by that query. Its log(gf), and both lines' ABO cross sections and velocity
  exponents, are therefore the published SIR inputs in Yang et al. (2024),
  Table 3. They are labeled as adopted model inputs, not unique laboratory
  measurements.
- Radiative damping uses the classical prescription in the pinned SIR 2015
  `blends2.f` source, including its air-to-vacuum conversion. It is labeled as
  an adopted classical model, not a measured upper-level lifetime. That same
  source explicitly treats Stark broadening as negligible in this wavelength
  region, so the Stark coefficient is deliberately null. Neutral-H ABO
  broadening remains active.

Production continuum coefficients and Fe populations use the abundance file at
the pinned STiC commit: a Grevesse--Anders/Kurucz baseline with the historical
overrides recorded in that file, including `log epsilon(Fe)=7.44`. Scalar
atomic masses used for thermal Doppler widths are reviewed against the CIAAW
2024 tables. The separate Asplund, Amarsi & Grevesse (2021) abundance data,
Barklem--Collet ionization/partition data, and Lightweaver H-minus coefficients
are retained as explicit reference-EOS and opacity diagnostics; production
synthesis does not mix them into the STiC thermodynamic state.

The manifest distinguishes byte-verified bundle-generation inputs from
reviewed citation-only records. Preparation downloads only immutable/pinned
inputs needed to reproduce the tables. Mutable NIST CGI, CIAAW HTML,
publication, SIR, and `sp_prep` URLs document the review provenance but are not
live download dependencies and are not presented as immutable source bytes.
All files actually placed in a prepared bundle are checksum-sealed.

For the instrument, `instrument_hinode_sp.json` distinguishes the official
30 mA nominal resolution and 21.5 mA sampling from the as-built tunable-laser
measurement (approximately 25 mA FWHM and 21.549 mA sampling). The code uses a
25 mA Gaussian approximation and records that the measured response is mildly
asymmetric with non-Gaussian inner wings. It does not assume a fixed stray-light
fraction or a spatial PSF.

The supplied files are CSAC Level-1 calibrated Stokes spectra (data DOI
10.5065/D6T151QF). Their history reports `sp_prep` 1.04, whose calibrated array
already has increasing wavelength but whose header predates the 1.05 WCS fix.
The loader requires a wavelength-equivalent `CUNIT1`, converts `CRVAL1` and
`CDELT1` to Angstrom, and applies the exact later `sp_prep.pro` ROI formula for
positive `CDELT1` and corrected `CRVAL1`. `SPWLSHFT` and `SPWLSFT0` are retained
as provenance for shifts already used by calibration; they are not applied
twice.

The Level-1 FITS keyword `DOP_RCV` is the spacecraft-Sun relative LOS velocity
in m/s, positive for a redshift. SolarSoft `thermd_sbsp.pro`, called by
`sp_prep`, converts it to a spectral-pixel displacement and already removes it
before writing the calibrated Level-1 profile. The loader records every
per-scan `DOP_RCV` value in `observer_velocity_correction`, but intentionally
does not shift the wavelength grid or inferred atmosphere a second time. The
production configuration fixes the residual wavelength offset at zero because
`sp_prep` also registers the slit-averaged Fe I 6301.5 line centre. This chooses
the relative-velocity gauge supplied by Level 1; jointly learning another
raster-wide offset and a constant LOS velocity would be nearly degenerate.

Prepare the complete Hinode/SOT-SP runtime bundle once, before launching any
training job:

```bash
pinn-me-lte-fetch-data \
  --output-dir data/lte_resources/hinode_sp_v1
```

For the supplied Derecho/full-raster paths, the corresponding one-time
preparation and training commands are:

```bash
pinn-me-lte-fetch-data \
  --output-dir /glade/work/rjarolim/data/inversion/hinode_sp/resources

pinn-me-lte --config config/hinode/lte_full_resolution.yaml
```

The fetcher downloads the pinned table-generation inputs concurrently, verifies
their SHA-256 digests, keeps a persistent cache under
`${XDG_CACHE_HOME:-~/.cache}/pinn-me/lte/sources-v1`, and only rewrites changed
outputs. Its default `INFO` log reports every cache hit or download, completed
source counts, conversion stages, approximately ten progress updates through
the long STiC grid calculation, elapsed time, and final bundle validation. Use
`--log-level WARNING` for quiet batch preparation or `--log-level DEBUG` when
diagnosing preparation. It converts the Barklem--Collet fixed-width partition table and the
Lightweaver H-minus arrays into runtime JSON and solves charge neutrality once
on the reference-EOS grid for offline diagnostics. It also loads the Wittmann
implementation, Kurucz partition file, abundances, and FALC atmosphere from one
immutable commit of the official STiC repository. The preparation step
evaluates that iterative reference code over a regular `(log10(T),
log10(Pgas))` grid and stores true continuum absorption, scattering, mass and
electron density, neutral-H density, and `n(Fe I)/U(Fe I)` at 5000 Angstrom and
across the Hinode window. Training imports none of the STiC solver: it only
performs differentiable tensor-product cubic lookup plus wavelength
interpolation in the generated, checksum-verified table.
Repeated preparation reuses the verified cache; `--refresh` explicitly
redownloads all bundle-generation inputs. A current checksum-valid sealed
bundle is recognized before the source cache or network is touched, so moving a
prepared bundle to another machine does not force a redundant download.

For a bundle that already contains a valid schema-2 STiC production table, the
optional Barklem/Saha diagnostic table can be created or repaired offline:

```bash
pinn-me-lte-fetch-data --eos-only \
  --output-dir data/lte_resources/hinode_sp_v1
```

This regenerates `eos_table.json` from the bundle's already verified atomic
inputs, records its SHA-256 digest in `sources.json`, and reseals `bundle.json`.
It is only a reference-EOS-table repair. It requires an already valid schema-2
STiC table. A bundle that lacks that table, or contains the older schema-1
table without neutral-H and Fe-I population fields, must be rebuilt with the
normal fetch command so the pinned STiC sources can be verified and converted.

Every Hinode training YAML then names this shared bundle and its expected
instrument:

```yaml
resources:
  directory: "/absolute/path/to/data/lte_resources/hinode_sp_v1"
  instrument: "Hinode/SOT-SP"
```

At startup, training validates the bundle manifest and every runtime-file
checksum. It performs no network access and no source-table conversion. The
same directory is reusable by all runs with the same instrument model. The
112-point wavelength array is intentionally not cached in this shared bundle:
it is reconstructed from each Level-1 raster's FITS WCS because its reference
wavelength and spectral ROI are observation metadata, while the shared
spectral-response model and calibration conventions belong to the instrument
bundle.

The automated LTE tests cover Saha-Boltzmann/EOS consistency, Voigt and
Faraday-Voigt profiles, Zeeman component patterns, formal-transfer limiting
cases, full-model automatic differentiation, stratified synthetic recovery,
Hinode WCS handling, an opt-in real-Hinode optimizer step, spectral masking,
air-to-vacuum conversion, Planck/opacity SI units, continuum/Stokes
normalization, and instrument-kernel normalization. These numerical checks do not by
themselves establish scientific agreement with an independent inversion code.

At a high level the formulation matches SIR's LTE ingredients—hydrostatic
stratification, LTE level populations, polarized transfer, and simultaneous
fitting of the Fe I 6301/6302 pair—but it is not a maintained SIR golden test.
SIR's bundled line parameters, abundance/EOS/continuum treatment, node-based
atmosphere, and formal solver differ from this implementation. Exact historical
cross-code numbers were therefore removed when production populations were
unified onto the STiC/Wittmann table; retaining them would imply a regression
that the repository cannot reproduce. The local tests instead pin the
implemented velocity, Zeeman, Stokes, air/vacuum, and transfer sign conventions.
A physical Hinode azimuth export must still define and verify the detector +Q
reference direction before Bx/By are interpreted as solar-image axes.

The production graph uses one thermodynamic source. The generated STiC/Wittmann
table supplies total continuum extinction for the tau-height metric and
geometric transfer, mass density for MHS,
electron and neutral-H perturber densities for damping, and `n(Fe I)/U(Fe I)`
for both Fe I lower-level populations. Its FALC conversion supplies the
reference stratification and gravity. The Barklem/Saha table remains a
separate, inspectable regression reference and is never evaluated during
training.

The optical-depth metric uses STiC true absorption plus scattering. The current
LTE source vector nevertheless thermalizes that scattering with `B_lambda`
instead of solving a coherent-scattering source function; this is explicit in
the resource and checkpoint metadata. The forward model contains no molecular
line opacity, so the documented cool-umbra blend window remains masked.
Radiative and neutral-H ABO broadening are active; Stark broadening is
explicitly neglected following the pinned SIR source. The implementation is
LTE-only and does not model Fe I NLTE line weakening, non-thermal scattering,
stray light, or a magnetic filling factor. Spatial convolution is not exposed
by the Hinode entry point, as described above.
These are explicit limitations for quantitative scientific use rather than
hidden fitted terms.
