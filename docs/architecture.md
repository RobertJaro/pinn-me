# Architecture

`prom3theus` is an LTE inversion framework with an explicit radiative-transfer
backend boundary.  The package name is intentionally not tied to LTE so that an
NLTE backend can be added later without moving observation, instrument, or
optimization code.

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
- Network access is confined to explicit HMI observation-download and
  response-preparation commands.

## Package responsibilities

| Package | Owns | Does not own |
| --- | --- | --- |
| `core` | Coordinates, constants, neural primitives, solar kinematics | File formats, instruments, training |
| `config` | Exact schema-v2 parsing, environment expansion, path resolution | Component construction, defaults chosen by adapters |
| `resources` | The immutable production bundle, common atomic/thermodynamic namespace, instrument namespaces, and checksum validation | Downloads, runtime table generation |
| `resource_builder` | Flat common-atomic and per-instrument builders, pinned downloads, STiC/Wittmann generation, FTS extraction, and exact reproduction | Runtime imports, training-time generation |
| `rt` | Atmosphere contracts, unified plasma lookup, polarized synthesis, formal transfer | Observation I/O, instrument response, optimization |
| `observations` | Canonical raster/batch contracts, safe sequence stores, data loading | FITS-specific calibration, spectral synthesis |
| `instruments` | Independent Hinode and HMI readers, strict HMI acquisition/cutout/response preparation, response operators | Optimizers, loss scheduling, artifact policy |
| `inversion` | Backend-neutral forward composition, objectives, constraints, run assembly | Diagnostic rendering, schema translation |
| `training` | Lightning lifecycle and optional visualization callbacks | Radiative-transfer equations, instrument parsing |
| `artifacts` | Atomic self-contained persistence and strict reconstruction/export | Observation preparation, training orchestration |
| `diagnostics` | Synthetic recovery and scientific checks | Production-run mutation |
| `cli` | Argument parsing and lazy dispatch | Hidden scientific configuration |

The data objective operates directly in the canonical fixed disk-center
atlas-continuum units. Each Stokes residual is divided by an instrument-specific
effective sigma and passed through an elementwise Huber penalty. The sigmas
combine measurement noise with a conservative allowance for unresolved
structure and forward-model discrepancy. No nonlinear transform is applied to
the prediction or observation separately, and no local continuum normalization
is performed. Configured Stokes component weights express relative scientific
preference and are normalized to sum to one before being combined, so their
global scale cannot change the balance between data and physics losses.

## Radiative-transfer boundary

Atmosphere state, ray geometry, spectral grids, and synthesis results are
backend-neutral contracts.  The LTE backend computes populations and source
functions directly from local thermodynamic state.  A future NLTE backend may
own global populations and convergence state behind the same scene-level
synthesis interface.

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

Dynamic magnetofluid runs decode the three Cartesian magnetic components
directly and use the ideal induction equation, momentum, logarithmic continuity
`D ln(rho)/Dt + div(v) = 0`, and a soft `div B` constraint. Direct decoding
avoids the additional derivatives required by a vector-potential
parameterization. The maintained dynamic HMI run has no resistive term and
disables both the soft adiabatic-pressure residual and the radial
magnetic-energy-gradient heuristic. No explicit energy equation is currently
solved.

Physics residuals use characteristic scales evaluated independently at each
sampled height. Transport equations use the combined rate
`1/T + V0/L`. Induction uses the denominator `B_* (1/T + V0/L)`;
logarithmic continuity is a specific rate and divides directly by
`1/T + V0/L`. `div B` and the current-free top boundary use `B_*/L`. Every
characteristic scale is formed separately at each sampled height and detached
before it divides a residual; magnetic relative scales additionally use the
configured physical floor. The predicted fields and every derivative in the
residual numerator remain attached to autograd. This prevents an equation from
reducing its gradient by changing its own denominator. Vector residuals are
averaged over components rather than summed.

There is no potential- or NLFFF-volume target. At the outer surface, `J = 0` is
enforced by the current-free `curl B` constraint. The
`upper_boundary_open_velocity` loss supplies a transmissive velocity boundary:
the normal derivative of every Cartesian velocity component,
`(r_hat dot grad) v`, is driven to zero using the detached `V0/L` scale. It
permits either sign of radial flow and does not create the systematic outward
bias of a one-sided no-inflow penalty.

The dynamic HMI run also samples all four angular side faces in balanced groups
at shared heights. It drives the normal derivative of velocity to zero and
applies the same `curl B = 0` current-free residual used at the top. The side
current-free loss shares the top loss's training ramp. It replaces, rather than
stacks with, a tangential magnetic Neumann condition so the magnetic boundary is
not redundantly constrained. Magnetic flux itself remains free to cross a side.

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

HMI observation acquisition and response acquisition belong to explicit
download/preparation modules. Runtime readers consume canonical, validated
local inputs and never contact JSOC. Response preparation resolves each input
`T_REC` through the definitive `hmi.B_720s.INVPHMAP` assignment and downloads
the resulting camera-specific `hmi.phasemaps_extended` record; no calibration
FSN is configured by the user.

## Persistence

Observation caches contain a versioned JSON manifest and memory-mappable array
files. A cache stores its sequence as flat, checksum-verified NumPy files next
to the manifest; raster records in the manifest provide names, shapes, dtypes,
hashes, and per-raster metadata. Its source signature includes complete-content
hashes of the selected raw FITS inputs. HMI calibration manifests separately
bind every response archive by SHA-256. Model artifacts contain a versioned JSON
manifest, a checksum-bound tensor-only state dictionary, and an immutable copy
of that observation store in `observations/`. No cache or artifact pickles a
Python data module or model class. Artifacts must satisfy the exact current
schema.

## Configuration

The standard Hinode and HMI runs plus
`configs/hinode_lte_mhs_extrapolation.yaml` are shipped. All use exact schema version 2
and the same hierarchy. Unknown fields, implicit instrument selection, and
command-line text substitution are rejected.

## Public API

The root package exposes only its version.  Users import numerical, observation,
instrument, inversion, or artifact APIs from their owning namespace.  This
keeps importing the LTE solver independent of the training and observational
software stacks.
