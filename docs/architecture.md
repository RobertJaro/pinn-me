# PROM3THEUS architecture

**Physics-informed Reconstruction Of Magnetism in 3D THrough EUV and Spectropolarimetry**

[Documentation overview](README.md) · [Project README](../README.md)

`prom3theus` is an LTE inversion framework with an explicit radiative-transfer
backend boundary.  The package name is intentionally not tied to LTE so that an
NLTE backend can be added later without moving observation, instrument, or
optimization code.

## Observation preprocessing

`preprocess.hmi` checks complete filename-matched Stokes sets and crops their
native detector pixels. Filter-response construction is a separate
`prepare hmi-responses` resource operation.

`preprocess.aia` scans FITS headers, caches the standard aiapy correction and
pointing tables, and dispatches independent read/calibrate/crop/write jobs.
Tables are loaded once and shared by the bounded worker pool. The image-store
adapter retains the uncertainty and line-of-sight arrays needed by the inversion;
SunPy coordinate construction is isolated in `instruments.aia_euv.geometry`.
Raw images are never accumulated across the time series. Reuse checks read
store metadata rather than loading all prepared image tensors.

## Dependency direction

```text
CLI -> application runners -> training/inversion
                              |-> radiative transfer
                              |-> observations
                              |-> instruments
                              `-> core geometry and neural models
```

Dependencies must point down this diagram.  In particular:

- `prom3theus.rt` does not import Lightning, Astropy, SunPy, Matplotlib,
  W&B, a data adapter, or the CLI.
- Observation readers produce canonical rasters and batches; they do not
  construct an inversion model.
- Instrument operators consume synthesized spectra; they do not know about
  optimizers or training loops.
- Export reconstructs the same forward model used during training.
- Network access is confined to explicit HMI/AIA acquisition and calibration
  commands. Runtime loading, synthesis, and diagnostics are offline.

## Package responsibilities

| Package | Owns | Does not own |
| --- | --- | --- |
| `core` | Coordinates, constants, neural primitives, solar kinematics | File formats, instruments, training |
| `config` | One strict schema-v3 stream configuration, environment expansion, path resolution | Component construction, defaults chosen by adapters |
| `resources` | The immutable legacy bundle, independently versioned optional scientific sets, and checksum validation | Downloads, runtime table generation |
| `resource_builder` | Flat common-atomic and per-instrument builders, pinned downloads, STiC/Wittmann generation, FTS extraction, and exact reproduction | Runtime imports, training-time generation |
| `rt` | Atmosphere contracts, unified plasma lookup, polarized synthesis, formal transfer, generic optically thin quadrature | Observation I/O, instrument response, optimization |
| `observations` | Independent typed Stokes and scalar-image raster/batch contracts, safe sequence stores, data loading, shared scene metadata | FITS-specific calibration, synthesis |
| `instruments` | Independent Hinode, HMI, and AIA readers/preparation plus their response operators | Optimizers, loss scheduling, artifact policy |
| `download` | Independent observation-download workflows such as AIA time-series/calibration acquisition | Instrument-specific parsing, deployment dates/paths, training |
| `preprocess` | Reusable HMI/AIA preparation workflows, directory scanning, crop-aware reuse checks, and instrument API composition | Deployment paths, fixed crop settings, optimization |
| `application` | Run orchestration, typed configuration translation, stream/scene assembly, setup evaluation, and report scheduling | Numerical equations, optimizer internals |
| `inversion` | Backend-neutral forward compositions, shared objective reduction, runtime component assembly, and constraints | Diagnostic rendering, schema translation, Lightning lifecycle |
| `training` | One stream training lifecycle and resumable reference-stream scheduling | Radiative-transfer equations, instrument parsing |
| `artifacts` | Atomic self-contained persistence and strict reconstruction/export | Observation preparation, training orchestration |
| `diagnostics` | Synthetic recovery, scientific checks, and registered native-grid renderers | Production-run mutation |
| `cli` | Argument parsing and lazy dispatch | Hidden scientific configuration |

HMI and AIA have independent download entry points. The HMI command calls its
instrument acquisition API directly; the AIA time-series workflow in
`prom3theus.download.aia` samples caller-supplied UTC targets and optionally
fetches its own calibration tables. It never imports HMI or scans HMI data.
The shell supplies event times, paths, channels, and tolerance. Independent
instrument preprocessors in `prom3theus.preprocess` receive all input/output
paths and crop settings from the caller. `scripts/sdo/prepare.sh` invokes
`prom3theus prepare hmi` and `prom3theus prepare aia`, keeping deployment
configuration in the shell and Python logic in the installed package. The
lower-level `prepare hmi-subframes`, `prepare hmi-responses`, and
`prepare aia-euv` commands remain available for individual preparation steps.
Runtime loading remains network-free.

The data objective operates directly in the canonical fixed disk-center
atlas-continuum units. Each Stokes residual is divided by an instrument-specific
effective sigma and passed through an elementwise MSE penalty. The sigmas
combine measurement noise with a conservative allowance for unresolved
structure and forward-model discrepancy. No nonlinear transform is applied to
the prediction or observation separately, and no local continuum normalization
is performed. Configured Stokes component weights express relative scientific
preference and are normalized to sum to one before being combined, so their
global scale cannot change the balance between data and physics losses.

The atmosphere is represented by a SIREN coordinate network. Its first sine
frequency is modulated by a configurable radial envelope: normalized horizontal
coordinates, and time in dynamic runs, retain full bandwidth near the solar
surface and are tapered toward the outer shell. The radial coordinate uses the
integral of the same envelope, so its local derivative equals the bandwidth
weight while the coordinate remains strictly monotonic. This supplies a
high-detail near-Sun prior and a progressively smoother outer-field prior while
keeping all coordinate derivatives differentiable for the physics residuals.

## Radiative-transfer boundary

Atmosphere state, ray geometry, spectral grids, and synthesis results are
backend-neutral contracts.  The LTE backend computes populations and source
functions directly from local thermodynamic state.  A future NLTE backend may
own global populations and convergence state behind the same scene-level
synthesis interface.

`StratifiedAtmosphereModel` accepts physical shell bounds and evaluates physical
heights; it owns no observation quadrature grid. Stokes streams own their FALC
reference-coordinate bounds and sample counts. `LTEForwardComposition` converts
those reference coordinates to heights before ray tracing. Refinement and transfer
use physical path lengths. Sampled atmosphere records carry generic ordered labels,
so sample indices cannot be mistaken for derived optical depth. AIA uses its own
geometric-height proposal and the common jitter/refinement tools.

The formal solver never guesses its integration coordinate.  Traced spherical
rays carry monotonically increasing physical distance.  A plane-parallel
solver, when used, is selected explicitly and has its own input contract.

MHS closure and LTE synthesis share one `SolarPlasmaTable` instance. Its
thermodynamic branch matches the bundled STiC state in the photosphere,
continues through a reduced CHIANTI default-coronal-equilibrium electron
mapping, and transitions to the fully ionized ideal STiC mixture. The CHIANTI
resource stores electrons per H; all thermodynamic outputs are reconstructed
from one shared hydrogen density. Native CHIANTI values remain exact through
10 MK, followed by a C1 log-temperature blend over 10--20 MK; the ideal fully
ionized limit is exact at and above 20 MK. Outside either STiC pressure edge,
the thermodynamic composition continuation recovers asymptotically linear
pressure scaling without permitting excess charge or negative heavy-particle
counts.

The radiative branch reproduces STiC exactly through 10 kK. From 10 kK to
31.6 kK, neutral-H, Fe I, and true continuum absorption fade to zero with a C1
quintic window. Scattering transitions over the same interval to Thomson
extinction, `n_e sigma_T`, and remains positive in the corona. Pressure tails
scale populations and scattering with the hybrid-EoS density ratio and true
absorption with its square.

The primary atmosphere decodes temperature, gas pressure, and microturbulence
as unbounded linear residuals in natural-log space around the radial reference.
All consumers use those actual positive values: there is no endpoint clipping,
finite table-support penalty, or separate effective LTE state. Planck emission,
thermal broadening, and damping therefore also retain gradients to the model
outputs throughout the domain.

The radial reference deliberately separates the line-formation grid from its
upper thermodynamic support. Its established 25-point FALC_82 mapping over
`-5 <= log10(tau500) <= 1` remains unchanged for LTE rays. The physical-height
reference additionally restores the 37 native FALC nodes at
`log10(tau500) < -5`, ending at 2.073502459 Mm with `T = 100 kK` and
`p = 0.0319344213 Pa`. From that native top, a quintic smootherstep in
`log(T)` reaches the configured 1 MK value at 2.5 Mm, after which the reference
is isothermal. Pressure is continued hydrostatically through the full domain
with inverse-square gravity and the thermodynamic branch of `SolarPlasmaTable`.

Maintained magnetofluid runs use a Cartesian vector-potential parameterization:
the three magnetic network channels are `A` in G m, and the exposed magnetic
field is `B = curl(A)` in G. This makes `div B` identically zero; an enabled
`magnetic_divergence` entry reports that exact zero without constructing a
redundant magnetic Jacobian. Direct Cartesian `B` decoding remains available
as a compatibility mode for older constructors and checkpoints. Dynamic runs
use the ideal induction equation, momentum, logarithmic continuity
`D ln(rho)/Dt + div(v) = 0`, and the vector-potential magnetic field. The maintained dynamic HMI run additionally applies a
volume current-free (`curl B = 0`) initialization for its first 5,000 optimizer
steps, then removes that term with a hard cutoff so the ideal-MHD solution can
develop currents. It has no resistive term and disables both the soft
adiabatic-pressure residual and the radial magnetic-energy-gradient heuristic.
No explicit energy equation is currently solved.

Physics residuals use characteristic scales evaluated independently at each
sampled height. Transport equations use the combined rate
`1/T + V0/L`. Induction uses the denominator `B_* (1/T + V0/L)`;
logarithmic continuity is a specific rate and divides directly by
`1/T + V0/L`. Direct-`B` `div B` uses `B_*/L`; vector-potential `div B` is exact
zero. Magnetic boundary residuals use a detached
point-local `sqrt(B^2 + B_floor^2) / L` scale. Every
characteristic scale is formed separately at each sampled height and detached
before it divides a residual; magnetic relative scales additionally use the
configured physical floor. The predicted fields and every derivative in the
residual numerator remain attached to autograd. This prevents an equation from
reducing its gradient by changing its own denominator. Vector residuals are
averaged over components rather than summed.

There is no potential- or NLFFF-volume target. At the outer surface, the normal
derivative of the tangential magnetic field is driven to zero while the normal
field remains free. The
`upper_boundary_open_velocity` loss supplies a transmissive velocity boundary:
the normal derivative of every Cartesian velocity component,
`(r_hat dot grad) v`, is driven to zero using the detached `V0/L` scale. It
permits either sign of radial flow and does not create the systematic outward
bias of a one-sided no-inflow penalty.

The dynamic HMI run also draws independent random points on all four angular
side faces with equal face counts. It drives the normal derivative of velocity
and the tangential projection of the magnetic normal derivative to zero. The
top and side magnetic Neumann losses apply at their configured weights from the
first step. Magnetic flux itself remains free to cross every face; side samples
have no shared-height grouping or group-derived normalization. The separate
volume current-free initialization remains a finite-step disambiguation aid.

HMI preparation writes an exact detector cutout and performs no numerical
padding. The physics angular domain is formed from the extrema of valid surface
pixels over all selected acquisitions. A padded FOV therefore means only that
the chosen 1024 by 512 cutout contains contextual margin around the region of
scientific interest; the runtime does not know a separate inner science FOV.

Because the energy equation is disabled, the spectrally invisible upper
temperature would otherwise be underdetermined. A weak upper-volume prior keeps
`log(T)` near the fixed hydrostatic-coronal reference and therefore acts as an
explicit prescribed-temperature stabilization, not as an energy law. A
separate top-surface prior fixes `log(p)` to the reference external pressure.
The model temperature and pressure are attached; both reference targets are
fixed and detached. These two conditions are temporary thermodynamic closure
and boundary data until a complete energy treatment is introduced.

The optional full-volume radial magnetic-energy-gradient loss evaluates
`d(B^2)/dr = 2 B dot dB/dr` from the Cartesian magnetic Jacobian. After
averaging that derivative over each height group and normalizing by the detached
height-group mean `B^2/L`, with the squared magnetic floor applied, it penalizes
only a positive layer-mean gradient. Local energy may increase as a flux system
expands laterally, while layer energy that is constant or decreases outward
receives exactly zero penalty. The term is a soft monotonicity preference rather
than a top-field target, and it is disabled in the maintained dynamic HMI run.

Absolute Doppler shifts have an unavoidable gauge freedom: a common velocity
along the observer LOS and an instrumental wavelength offset produce the same
line displacement. Maintained dynamic runs therefore optimize one explicit
positive-redshift LOS correction after projecting into the Stokes basis. There
is no separate penalty forcing the mean co-rotating atmospheric LOS velocity to
zero; its split from the instrumental offset is therefore determined jointly by
the Stokes fit and active physical equations. The correction does not change the
radial or transverse physical fields used by momentum, induction, and
continuity. Static inversions leave it disabled.

An NLTE implementation would provide the existing forward-backend protocol and
own its population/convergence state in a separate backend package. Supporting
it would require an explicit new configuration and artifact schema version;
NLTE behavior must not be introduced as conditionals inside the LTE solver.

## Observation and instrument boundary

Hinode/SOT-SP and SDO/HMI are independent adapters. Both produce the same
canonical observation model, but neither inherits implementation details from
the other.  Instrument-specific response tensors travel in the batch under an
explicit response mapping.

The maintained Hinode adapter accepts exactly one flat, prepared raster and
constructs a static three-coordinate atmosphere. Nested raster sequences and
time-range selection are rejected. The HMI adapter retains its independent
time-dependent acquisition path.

Schema version 3 adds an independent scalar-image contract rather than
encoding EUV intensities as synthetic Stokes data. Each prepared AIA channel
keeps its native registered grid, actual TAI exposure time, surface anchor,
observer-to-Sun ray, uncertainty, valid mask, and calibration identity. A
`SceneContract` is the only bridge between those physical records and the
shared atmosphere coordinates; AIA is never reprojected onto HMI pixels.

The AIA term evaluates the same atmosphere's unrestricted temperature and gas
pressure along each native ray. The shared thermodynamic EOS supplies electron
density and the independently packaged response supplies `K_c(T)`, giving
`I_c = 10^-10 integral n_e^2 K_c(T) ds_m`. The v1 response is total-only at its
declared reference density. It smoothly tapers to zero at the 10 kK and 1 GK
edges and is exactly zero beyond them; only the observation response is tapered,
not the atmosphere. An identifiable common/relative log-gain model accounts for
residual calibration error without introducing an AIA-only density field.

`JointForwardModel` owns one atmosphere, a module dictionary of observation
terms, and separate shared physics/regularization terms. Its objective
coordinator consumes only modality-neutral `DataTermBatchResult` values. The
current setup runner uses closed typed HMI/AIA construction branches, while
diagnostics use an observation-kind registry. A future third imager should move
construction to the same explicit registration pattern rather than adding
modality checks to objective evaluation.

HMI observation acquisition and response acquisition belong to explicit
download/preparation modules. Runtime readers consume canonical, validated
local inputs and never contact JSOC. Response preparation resolves each input
`T_REC` through the definitive `hmi.B_720s.INVPHMAP` assignment and downloads
the resulting camera-specific `hmi.phasemaps_extended` record; no calibration
FSN is configured by the user.

## Persistence

Evaluation snapshots persist constructor contracts and tensor state without
copying observation arrays. Prepared stores remain independent, immutable and
checksum-verified. Neither format pickles Python model or data-module classes.

The application coordinator in `application/data.py` prepares one joint data
module and publishes `<work_directory>/data_module.pt` with ordinary `torch.save`.
The versioned snapshot contains importable state records, file references, compact
selection metadata, scene/domain summaries, and prepared training-pool specifications.
`observations.persistence.load_data_module` uses `torch.load` and restores these
records without calling provider or raster constructors. Rank zero also follows
this restore path on subsequent runs. Source/scene/resource contract changes require
an explicit rebuild; runtime loader settings bind independently.

`ArrayRef` carries an NPY filename and schema/stat metadata. `ReadSession` owns a
bounded set of process-local read-only mappings and copies into owned batch buffers.
No open mapping, background producer, or provider callback enters the saved module.
Unchanged canonical arrays are referenced directly; derived arrays and packed training
pools have separate immutable files. Random physics samples remain generated from
coordinate bounds on the model device. The data module is a trusted internal Python
snapshot, separate from the public `.p3s` artifact and canonical observation formats.
See [data loading implementation](data-loading-implementation.md) for lifecycle,
distributed execution, validation, and remaining hardware checks.


Scalar images use a parallel image-store schema with the same immutable,
checksum-verified persistence rules. Static AIA/CHIANTI response data live in
`resources/sets/euv/aia_euv_v1`; event-specific pointing/degradation tables are
a separate acquisition bundle, and each prepared store seals the calibration
convention connecting them.

## One application runtime

Every maintained configuration is schema version 3. A single observation is one
entry in `streams`; arbitrary stream IDs distinguish multiple observations of
one type. `application.joint_training.run_joint_inversion` is the sole optimizer
runner. `application.joint_runner` performs setup/evaluation without optimizing.
There is no schema-v2 decoder, legacy runner, old LTE Lightning module, or
compatibility loader for prior P3S/archive formats.

`components.observations` registers observation providers (loading and resource
requirements); `components.forward` registers construction and reconstruction
providers. `config.registry` registers additional strict dataclass schemas and
compatible observation/term pairs before parsing. Registrations are explicit,
with duplicate registrations rejected. The application assembly composes these
providers with the shared scene, atmosphere and physics. New providers own their
scientific requirements; shape compatibility alone is insufficient. Non-raster
providers supply a scene contract and spatial/time sampling support. Stokes
rasters are rebased into the shared scene chart and time origin without changing
their physical ray geometry, polarization bases or observed targets.

`JointForwardModel` owns one atmosphere, stream terms and shared objectives.
Shared physics runs once per optimizer step. Terms own their calibration state
and expose canonical-batch predictions. Metrics use stream-qualified names, so
multiple HMI, Hinode or image streams cannot overwrite one another.

Training logs the total loss plus its active weighted contributions: observation
`data_loss`, enabled physics/regularization components, and enabled calibration
priors. Disabled terms and expired scheduled losses create no series. An active
loss remains visible even when its value is exactly zero. Validation logs only
unweighted `data_loss` per stream plus Stokes residual RMS or AIA asinh RMSE;
render-only streams remain available for validation. Component details stay in
local reports. AIA plots from all configured observations share two stable media galleries:
`AIA observation comparison` and `AIA side view`. Optional contribution
plots remain local;
Stokes and atmosphere plots retain the renderer titles, such as
`Stokes comparison`, `Magnetic field`, and `Optical depth`;
filenames and step numbers never create dashboard keys. There is no `streams/`
image logging namespace;
step-specific directories retain local history. Gradient-finiteness checks remain
active without logging gradient extrema.

`diagnostics.providers` registers rendering callbacks by term type. Setup and training
collect returned artifact records and sends them to logging from `DiagnosticOutput(report, paths)` without scanning report text or guessing
filenames. The built-in renderers retain bounded native-grid evaluation and
per-stream names. Validation preserves model mode and RNG. Diagnostic pixels
are a subset of inversion data, not a held-out training split.

## Sampling

`training.streams.StreamBatchScheduler` uses `training.reference_stream` (or the
scene reference when omitted) to define epochs. All configured streams supply
one batch per step. Other streams continue independent passes across reference
epochs. Loader cursors and CPU sampling RNG states are saved in `last.ckpt`.
Restoration replays delivered batches within the current pass; this costs reads
but avoids storing an entire global permutation in the checkpoint. Built-in
prepared-data reads are deterministic. A new stochastic data provider must
supply equivalent reproducible behavior; arbitrary external mutable loaders
cannot promise exact replay. Worker counts do not change the outer scheduler,
which runs in one process.

`inversion.depth_sampling` owns bounded jitter, contribution-weighted inverse-CDF
sampling and sample merging. LTE supplies attenuation-based interval masses;
AIA supplies channel-specific optically thin emission masses. Coarse and fine
field evaluations both retain gradients; proposal locations are detached. AIA
keeps float64 shell geometry and fixed photospheric/outer-shell endpoints.
Training jitters interior coarse points independently per AIA ray and selects one
random quantile per fine-sampling stratum; validation and export use deterministic
coarse points and midpoint fine quantiles. `DepthRefinement` lives beside the
common sampling functions, so AIA does not import the LTE forward composition. The maintained
AIA configuration uses 96 coarse plus 96 fine samples. Adaptive trapezoidal
quadrature is not guaranteed to outperform a fixed grid for every profile;
dense-reference convergence checks remain necessary.

`rt.optically_thin.trapezoid_weights` computes nodal line elements from the
**final sorted ray distances**, after both perturbation and refinement. For
intervals `ds[i] = s[i+1] - s[i]`, endpoint weights are half their adjacent
interval and interior weights are `(ds[i-1] + ds[i]) / 2`. Integration and
contribution diagnostics use this same helper. Distances stay float64 until
line elements are formed, preventing nearby refined samples from collapsing
when fields are float32. AIA rays are normalized and their shell intersections
use a rationalized root to avoid cancellation near the photosphere. The weights
sum to the physical LOS length, which differs from radial height off disk centre;
no importance-probability correction or second coarse integral is added.
`RayIntegralResult.line_element_m` exposes these weights for auditing.

## Persistence

`output_directory/state.p3s` and `output_directory/last.ckpt` are rolling files,
with no intermediate checkpoint directories or numbered versions. Plots,
validation metrics, caches and logger working files go under `work_directory`.

`artifacts.checkpoint` owns one tensor-only evaluation format. Its context
contains model/term constructor options, the common scene, resource identities,
observation references and resolved configuration. It stores all model/term
parameters and buffers needed for evaluation and no optimizer or observation
pixel arrays. `P3SLoader` reconstructs an `EvaluationModel` independently of
training, verifies resource/tensor contracts, and exposes per-stream prediction
and shared-field queries. Observation stores are accessed lazily for raster
comparison/export. `--stream` selects the stream for raster-based CLI exports.
The full checkpoint additionally holds optimizer/scheduler state and sampling
progress for continuation. Old formats are rejected, not migrated.

The component-extension test registers a third temperature-observation type,
parses its configuration, differentiates its likelihood through the common
atmosphere, saves it, reconstructs it, and predicts through the public loader.
It also runs through the common optimizer using a non-raster observation provider.
This verifies the extension seam independently of HMI/AIA construction.


### Persistent training tensors

`TensorDiskDataset` is the common batch-indexed dataset for final training
values. `TensorDiskDataset.create(tensors, directory, batch_size, seed=0)` accepts
a dictionary (including nested dictionaries) of tensors with the same leading
sample dimension. It applies one shared global permutation and writes one NPY
file per tensor. Reopen it with `TensorDiskDataset(directory, batch_size)`;
`dataset[i]` reads the contiguous i-th batch, including a partial final batch.
The dataset retains paths and small manifest metadata only. Returned batches
own their storage; file mappings are closed after each read.

Training preparation uses the same implementation's `from_batches` path to
stream native observations into the final shuffled files without allocating
a second full training corpus in RAM. `PersistentTensorLoader` persists
channel pools under `work_directory/training-tensors/<stream>`, then releases
its native dataset references. It randomizes chunk IDs with the existing
seeded, resumable schedule. A matching source/configuration/implementation
identity reuses the files on restart. Preparation adapters remain separate
for scene setup and raster diagnostics.


Resume compatibility compares observation identity and persistent model/reference
structure. Fitting controls such as loss weights, objective settings, potential
weight schedules, per-step collocation counts, and loader batch sizes may change.
New loss buffers and learning-rate settings take precedence over saved values;
learned parameters and optimizer history are restored. Changing batch size
restarts data coverage using the new chunk boundaries while retaining the
optimizer step. Data, network architecture, and potential reference-grid changes
remain checked; mismatch errors report the differing contract paths.
