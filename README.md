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
The atmosphere always predicts Cartesian velocity and magnetic vectors. The
velocity vector is `[v_x, v_y, v_z]` in the observer's Stokes frame with
`+v_z` toward the observer, so the radiative-transfer LOS velocity is `-v_z`
and remains positive for a redshift. Both vectors use direct, unbounded linear
network readouts: the velocity scale is `1000 m/s` per raw unit and the
magnetic scale is `100 G` per raw unit. Temperature and gas pressure use
smooth natural-log readouts whose asymptotes are the verified STiC lookup
bounds. A scaled `tanh` retains the configured local natural-log sensitivity
without a hard clamp or an implicit `ln(10)` amplification. The deep
Fourier MLPs use activation-aware,
variance-preserving random hidden weights, Xavier-random readout weights, and
independently random biases. This prevents successive SiLU layers from
suppressing coordinate-dependent Fourier variance into an apparently uniform
DC field. No output channel or bias is zero-initialized; the explicit physical
decoders set the initial magnitude. A single-view Stokes spectrum does not
constrain `v_x` or `v_y` without additional dynamical physics. The magnetic vector is Cartesian in the
observer's Stokes frame:
`+Bx` defines zero magnetic azimuth (the Stokes `+Q` reference axis), `+By`
defines increasing azimuth, and the positive LOS component is directed toward
the observer. Stokes azimuth is pi-periodic; Q and U therefore rotate through
twice the physical azimuth. The loader does not invent a detector-polarization
rotation, so exported Bx/By must not be labeled solar west/north until that
Hinode reference direction has been independently calibrated.

For each Hinode pixel, the loader derives the ray cosine
`mu = sqrt(1 - (Dsun*sin(rho)/Rsun)^2)` from the Level-1 `XCEN`, `YCEN`,
`CRPIX2`, `CDELT2`, and degree-valued `CROTA2` pointing keywords. In the exact
helioprojective convention, Tx is longitude, Ty is latitude, and
`cos(rho)=cos(Tx)*cos(Ty)`. This uses the exact ray impact parameter
`Dsun*sin(rho)`, rather than either `hypot(Tx,Ty)` or a ratio of angular
small-angle approximations. The apparent solar semidiameter is
`asin(Rsun/Dsun)` using Astropy's geocentric solar ephemeris; the Level-1 files
do not carry an exact Hinode-spacecraft distance. The formal
solver uses `mu` exactly once, as `delta_tau500/mu`, in polarized line
and continuum transfer. The atlas calibration separately uses the same `mu`
with the Neckel continuum center-to-limb law: quiet-Sun intensity selection is
performed after converting the observed continuum to a disk-center-equivalent
level, and the single detector conversion is fitted against the expected local
`I_c(mu)`. The calibrated profiles remain in the common disk-center atlas unit;
they are not divided by `I_c(mu)`. The inferred magnetic and velocity vectors
are already defined in the observer's Stokes frame, so applying an additional
`mu` projection to their LOS components would be incorrect. HSE remains a vertical equation
and therefore contains no `mu`. This is a 1.5-D plane-parallel treatment: an
inclined ray samples a longer path through the same `(x,y)` column and does not
move horizontally through neighbouring network columns. The Hinode entry point
exposes no spatial-PSF configuration because a detector kernel would need the
local two-dimensional scan/slit displacement basis in physical Mm coordinates.

The same pointing solution is converted to physical image-plane coordinates as
`x_mm=D*tan(Tx)` and `y_mm=D*tan(Ty)/cos(Tx)`, with angles in radians and the
same geocentric distance proxy. These are coordinates on one tangent plane
through Sun center for the HPC ray direction
`(cos(Ty)sin(Tx), sin(Ty), cos(Ty)cos(Tx))`. The loader retains each scan-column
time and stores coordinate formulas, ranges, source keywords,
observer-distance values, and the exact network affine in raster/checkpoint
metadata. These are projected helioprojective distances, not a solar-surface
deprojection.

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
the packaged project. The operational inversion composes a tau-to-height field
with one continuous physical atmosphere:

```text
(Solar-X, Solar-Y, log10(tau500)) -> Z -> geometric height z
(Solar-X, Solar-Y, z)             -> F -> T, Pgas, v, B, xi
atmosphere sampled in tau         -> geometric-height LTE polarized transfer
```

`Z` uses a linear 150 km/dex base scale and an MLP that outputs an unconstrained
dimensionless height perturbation scaled by 10% of that base scale (15 km); the
perturbation readout starts at zero. `F`
uses normalized coordinates and variance-preserving random initialization.
There are no fixed atmospheric nodes, reference atmosphere, or magnetic seed.
`Z` is optimized jointly and has one gauge condition,
`mean_FOV[z(log_tau500=0)]=0`, which removes only the global height translation.

The configured `log_tau500` array defines the domain and a deterministic
evaluation grid, not trainable nodes. For every training batch, the transfer
sampler first constructs equally spaced centers from the minimum to maximum
`log_tau500`. The endpoints remain fixed; every interior point is independently
drawn from a uniform stratum bounded exactly by the midpoints to its two
adjacent centers. This produces an ordered nonuniform grid with no secondary
jitter control. The mapping evaluates a physical height at every realized tau
sample. The formal solution multiplies its normalized propagation matrix by
`alpha500` and uses the actual geometric distance between successive mapped
heights as its line element. Refining the sample count changes quadrature
accuracy without changing the number of neural parameters.

Physics collocation is independent of that transfer grid. A volume dataset
draws independent uniform physical `(Solar-X, Solar-Y, log_tau500)` samples
over the complete configured axis-aligned bounds. Each sample passes through
`Z` and then `F`. A separate top-face dataset keeps the gas-pressure boundary at
the configured top optical depth.
Physics validation uses one deterministic linear optical-depth grid.

There is no explicit smoothness, derivative, node, or sparsity regularizer.
Smoothness comes from the finite-bandwidth Fourier-feature MLPs and smooth
decoders, evaluated at continuously changing coordinates. Public spatial
coordinates are helioprojective Solar-X/Solar-Y in Mm. The entry point stores
one raster-derived affine inside both coordinate networks; the
isotropic scale is 512 native pixels, mapping the configured full raster to
approximately `x=[-1,1]` and `y=[-0.5,0.5]`; `log_tau500=[-5,1]` maps to
`[-1,1]` inside `Z`, and `F` receives `z/height_input_scale_m`. These ranges keep inputs order-unity while the full-resolution
configuration limits the shortest representable spatial period to approximately
eight native pixels.

Although the loader retains every scan-column timestamp, the default single-
raster atmosphere is explicitly static and ignores the time coordinate. In a
slit scan, time and scan position are nearly perfectly correlated, so a joint
`F(t,x,y,z)` model would be non-identifiable away
from the observed scan trajectory. The implemented atmosphere is therefore
strictly `Z(x,y,log_tau500)` followed by `F(x,y,z)`; scan time remains data
provenance rather than a network input.

Geometric height and corrugated optical-depth surfaces are inferred. This makes
geometric `div B`, magnetohydrostatic balance, continuity, induction, and
momentum residuals available, although the vector equations remain disabled in
the Hinode example until the transverse Stokes basis has been independently
calibrated to the Solar-X/Y basis.

The validation metrics evaluate a deterministic stride-selected subset of the
same inversion pixels without shuffling and on the deterministic depth grid.
`valid.stokes_loss` is the data-fit diagnostic; `valid.loss` additionally uses
the current scheduled HSE and top-boundary weights.
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

The initial 25-point grid is a scalable training quadrature, not a claim of exact
depth convergence. In the current physical-radiance convergence test against a
101-point result, the 25-point Stokes-I RMS error is about `6e-3 Ic`, while 51
points reduces it to about `1e-3 Ic`.
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

The optimizer uses a configurable set of named objectives. Every enabled loss
and its current weight are logged at each training step:

- `train.stokes_loss` is the only observation objective.
- `train.tau_mapping` enforces
  `alpha500*(-dz/dlog10(tau500)) = ln(10)*tau500` at random volume points.
- `train.hse` differentiates `Pgas` itself and minimizes
  `[dP/dlog10(tau500) - rho*g*(-dz/dlog10(tau500))] / mean_xy(Pgas)` on each
  randomly sampled tau surface. `rho(T,Pgas)` and `alpha500(T,Pgas)` are taken
  from the pinned STiC/Wittmann table.
- `train.pressure_boundary` fixes the pressure integration constant on the
  independent top-face dataset. Its target is not an arbitrary tuning value:
  the resource bundle derives `Pgas=0.0903679136 Pa` at
  `log10(tau500)=-5` from the pinned FALC atmosphere with the same STiC
  opacity convention. It also supplies the paired FALC gravity
  `g=275.4228703 m s^-2`; startup rejects a conflicting override of either
  quantity.

`training.physics_config.equations` enables geometric `divergence_b` in the
Hinode configuration and additionally exposes stationary magnetohydrostatic
`mhs`, `continuity`, ideal `induction`, and
ideal-MHD `momentum` residuals. The MHS residual is the normalized vector force
balance `grad(P) - rho*g - (curl(B) x B)/(4*pi)` in Gaussian cgs units: the
network field stays in gauss, magnetic derivatives are converted from G/m to
G/cm, and all mechanical force densities are converted to dyn/cm^3. It shares
one pressure/magnetic Jacobian with `divergence_b`. The configuration declares
that the magnetic basis matches the Solar-X/Y/z coordinate basis; this assertion
must be backed by an independently calibrated Hinode transverse Stokes-reference
rotation before the result is interpreted physically. Startup refuses geometric
vector equations unless that basis contract is declared. The force and dynamical
equations remain disabled. An energy equation is intentionally absent because the LTE spectra do
not provide a closed heating, conduction, and radiative-cooling model.
The HSE, MHS, and full momentum residuals are mutually exclusive force-balance
models. Applying more than one would silently introduce additional zero-force
assumptions.

Every Stokes and physics `weight` accepts a non-negative number or a schedule.
Supported schedules are `fixed`, `linear`, `exponential`, `step`, and
`smoothstep`; choosing `start < end` or `start > end` gives an ascending or
descending schedule. A scheduled term with current weight zero is not evaluated.
`train.loss` is the configured weighted sum; there are no profile-smoothness or
generic parameter penalties. Training emits no `train.epoch_loss`. Validation
logs the same enabled Stokes and physics components once per completed epoch at
the current schedule weights.

`training.physics_config.normalization` sets the three independent units
`length_m`, `time_s`, and `magnetic_field_gauss`. The derived Gaussian-cgs
units are `v0=L0/t0`, `P0=B0^2/(4*pi)`, `rho0=P0/v0^2`,
`g0=L0/t0^2`, and force-density `f0=P0/L0`. MHS, momentum,
continuity, induction, and `divergence_b` are divided by their corresponding
derived unit. Geometric HSE is normalized only by the detached mean gas
pressure on each sampled optical-depth surface; the pressure boundary is a
dimensionless log10 ratio. `valid.pressure_increasing_fraction` reports
the fraction of adjacent validation-depth intervals whose pressure increases
inward; it should approach one as HSE is satisfied. HSE constrains pressure,
not monotonic temperature, velocity, or magnetic field profiles.

During training, the LTE visualization callback writes one update per complete
validation epoch under `work_directory/atmosphere_visualizations`. One Stokes
dashboard contains wavelength-integrated absolute reference maps, prediction
maps on the same color scale, and signed observed/predicted spectra at the
validation ensemble as median signed profiles with 16--84% bands and integrated
predicted-versus-reference scatter; both Fe I laboratory wavelengths are marked.
Integration uses the actual detector wavelength coordinates. A separate
parameter figure, magnetic-field figure, and velocity figure place quantities
in columns and configured optical-depth layers in rows ordered from high to low
tau. They include inferred `T` and `Pgas` and STiC-derived `rho`, all displayed
in linear physical units with logarithmic color normalization, plus all velocity-vector components,
magnetic quantities, and
microturbulence. Full rasters are uniformly subsampled up to
`max_map_pixels` from the same deterministic validation-derived slit/scan grid
used by the Stokes maps, and evaluation is chunked with
`evaluation_batch_size`. Every map uses the actual 2-D Solar-X/Solar-Y Mm
coordinates rather than detector-index extents. Parameter and Stokes panels
therefore have identical samples and orientation. Local figures are
saved at the configured DPI and WandB uploads that exact PNG instead of
rerasterizing the live Matplotlib figure at a lower default resolution.
An optional `visualization.yz_slice` selects an exact detector `scan_index` and
adds both optical-depth and geometric-height cross sections per visualization
epoch for the parameters, complete magnetic vector and derived angles, and
complete velocity vector. The geometric panels use the learned curvilinear
height mesh and overlay the selected `log_tau500` surfaces as dashed contours.
A separate tau-mapping dashboard shows height and the signed
`-dz/dlog10(tau500)` metric.
Integrated maps and agreement scatter retain all validation samples; only the
complete 4x112 spectra used for percentile curves are capped by
`max_profile_samples`, keeping callback memory bounded on a full raster.

The supplied LTE configurations use a `WandbLogger` with project
`pinn-me-lte`; figures, named losses/metrics, and the complete configuration
are logged to Weights & Biases while the PNG files remain available locally.
Set `logging.type: csv` only for an explicitly offline run. Configure cadence
and layers under the top-level `visualization` mapping, or set
`visualization.enabled: false` to disable the image callback.

The trainer additionally writes best/last checkpoints and an `on_exception.ckpt`
for recoverable failures or interruptions. Resume loading verifies the LTE provenance
and scientific configuration before accepting checkpoint state.
The supplied configs set `lr_params.iterations: auto`, so the exponential
schedule reaches its configured end rate on the trainer's actual final
optimizer step rather than retaining a stale step count from another FOV.

The standalone recovery command is a compact forward/objective gradient smoke
test. It fits twelve explicit atmosphere coefficients to a noiseless two-line
Stokes spectrum with fixed pressure; it does **not** exercise the coordinate
network, HSE collocation, Hinode loader, or callbacks and is not presented
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
geometric transfer, mass density for HSE,
electron and neutral-H perturber densities for damping, and `n(Fe I)/U(Fe I)`
for both Fe I lower-level populations. Its FALC conversion supplies the
matching upper pressure boundary and gravity. The Barklem/Saha table remains a
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
