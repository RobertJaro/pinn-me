# PROM3THEUS

PROM3THEUS is a clean LTE spectropolarimetric inversion framework built with
PyTorch and PyTorch Lightning. It fits a continuous, depth-stratified solar
atmosphere directly to Stokes observations and can add magnetofluid constraints
at independent collocation points.

This release supports two complete inversion paths:

- Hinode/SOT Spectro-Polarimeter Fe I 6301/6302 A rasters
- SDO/HMI Fe I 6173 A Stokes rasters with acquisition-specific filter profiles

The project uses one strict LTE schema, canonical observation stores, and
checksum-bound tensor artifacts across both inversion paths.

## Design

```text
configuration + CLI
        |
        v
application runner ---- observations ---- instrument response
        |                       |
        v                       v
inversion objective ---- canonical raster/store
        |
        v
LTE backend ---- explicit ray path ---- polarized formal solver
        |
        v
coordinate atmosphere + optional magnetofluid constraints
```

The numerical LTE backend does not depend on Lightning, FITS readers,
visualization, or the command line. Hinode and HMI are independent adapters that
meet the same canonical observation contract. Model artifacts are a versioned
JSON manifest plus tensor-only weights and checksum-protected NumPy arrays.

The package name and forward-model boundary are not LTE-specific. A future NLTE
backend can own its population/convergence state behind the same observation,
instrument, optimization, and artifact layers.

See [the architecture](docs/architecture.md) and the
[complete porting plan](docs/porting-plan.md) for the dependency rules and
removal inventory.

## Installation

Python 3.11 or newer is required.

```bash
python -m pip install -e .
```

Install only the optional stacks needed by your workflow:

```bash
# FITS/SunPy geometry and map readers for Hinode and HMI
python -m pip install -e '.[observations]'

# Diagnostic figures and Weights & Biases logging
python -m pip install -e '.[visualization]'

# HMI Stokes download, Carrington cutouts, and phase-map responses
python -m pip install -e '.[hmi-preparation]'

# Everything needed to run the test suite
python -m pip install -e '.[observations,visualization,hmi-preparation,test]'
```

The checksum-pinned LTE atomic, continuum, reference-spectrum, and instrument
metadata for both supported instruments ship in the package. Validate them at
any time:

```bash
prom3theus resources validate
# Hinode atomic tables and complete instrument-specific resource contract
prom3theus resources validate --instrument hinode_sp
```

Runtime inversions never download scientific inputs.

The resource bundle has an explicit ownership layout:

```text
resources/data/
  bundle.json
  sources.json
  common/
    abundances.json
    lines.json
    stic_continuum_table.json
  hinode_sp/
    blend_inventory.json
    instrument_hinode_sp.json
    solar_reference_630nm.json
  hmi_stokes/
    instrument_hmi.json
    solar_reference_617nm.json
```

The complete checksum-pinned, from-scratch generation workflow is retained
separately under [`resource_builder`](resource_builder/README.md).
It is maintenance tooling rather than runtime package code. Running
`./scripts/resources/rebuild.sh` regenerates both instrument namespaces in
`build/reproduced-lte-resources` and requires byte-for-byte agreement with the
committed production bundle.

Manifests and runtime consumers use these complete relative paths. There is no
fallback lookup into the former flat resource directory.

## Run configurations

The maintained source and distribution contain four public YAML run files:

- [`configs/hinode_lte_mhs.yaml`](configs/hinode_lte_mhs.yaml)
- [`configs/hinode_lte_mhs_extrapolation.yaml`](configs/hinode_lte_mhs_extrapolation.yaml)
- [`configs/hmi_lte_mhs.yaml`](configs/hmi_lte_mhs.yaml)
- [`configs/hmi_lte_dynamic.yaml`](configs/hmi_lte_dynamic.yaml)

They share one strict schema and the same readable section order. The current
inversion configuration schema is version 2. Version 1 encoded finite
thermodynamic decoder bounds and is intentionally rejected rather than silently
loading its weights under the unbounded decoder:

```text
schema_version
solver
resources
observation
atmosphere
synthesis
instrument
training
loss
diagnostics
runtime
logging
```

Unknown or duplicate keys are errors. The solver and instrument are always
explicit. Relative paths are resolved from the configuration file, so commands
behave the same from every working directory.

The `loss` section defines effective `stokes_sigmas` in the stored
disk-center atlas-continuum units and applies a noise-standardized Huber
objective. Relative Stokes weights are normalized to sum to one; changing all
four by the same factor therefore does not alter the data-versus-physics scale.

All maintained YAML files hard-code their GLade input/output paths and Derecho
scratch paths. Ambient environment variables do not redirect these workflows.

Inspect the fully resolved, typed configuration before a run:

```bash
prom3theus validate-config configs/hinode_lte_mhs.yaml
prom3theus validate-config configs/hinode_lte_mhs_extrapolation.yaml
prom3theus validate-config configs/hmi_lte_mhs.yaml
prom3theus validate-config configs/hmi_lte_dynamic.yaml
```

### Hinode MHS extrapolation

The extrapolation configuration uses one atmospheric PINN over the complete
physical shell. LTE synthesis and opacity-guided ray refinement stop at the
configured line-formation top, while independent collocation batches apply MHS
and `div B` throughout the full shell, microturbulence and hydrostatic-coronal
temperature priors only above the line-formation domain, and transmissive-flow
and current-free conditions at the outer boundary. The transmissive velocity
condition drives its normal derivative to zero without selecting inward or
outward flow. The maintained extrapolation configuration reaches 20 Mm while
retaining one PINN for both domains.

### HMI static and dynamic inversions

`hmi_lte_mhs.yaml` fits every selected acquisition with one time-independent
atmosphere constrained by MHS, `div B`, and the top-pressure prior.
`hmi_lte_dynamic.yaml` represents the observed time sequence explicitly over a
single -0.1--50 Mm PINN. HMI line formation remains below 1.5 Mm. The three
Cartesian magnetic components are decoded directly; ideal induction, momentum,
continuity, and a soft `div B` constraint act throughout the full shell. The
maintained run disables the adiabatic-pressure and radial magnetic-energy
heuristics and has no potential- or NLFFF-volume loss. A weak prescribed
temperature stabilization acts only in the invisible upper domain. A
transmissive velocity condition, `J = 0`, and an external-pressure condition
apply at the 50 Mm boundary. Independent grouped samples cover the four angular
side faces at matched height layers. Those faces use zero normal velocity
gradient and `J = 0`; no zero-field or zero-normal-flux condition is imposed.

HMI acquisition selection is explicit in `observation.selection`.
`acquisition_indices: null` uses the complete source sequence; a zero-based
list such as `[10]` loads only those source frames. `validation_raster` uses the
same source-sequence index and must belong to the selected set.

```text
D ln(rho)/Dt + div(v) = 0,
dB/dt - curl(v cross B) = 0,
div(B) = 0.
```

No resistive term is used. Because a complete coronal energy equation is not
yet implemented, the upper-temperature prior is an explicit prescribed
temperature stabilization rather than a conservation equation. The upper
temperature prediction remains trainable, but its log-space target is the fixed
hydrostatic-coronal reference. The top-pressure target is likewise a fixed
external-pressure boundary value; it is not inferred from a magnetic model.
Both targets are detached while the model predictions remain attached.

Physics normalizers are evaluated separately at each sampled height and
detached before division; all predicted fields and residual derivatives remain
attached. `div B` and the current-free top condition use the detached magnetic
scale `B_*/L`, induction uses `B_* (1/T + V0/L)`, and continuity uses the fixed
transport rate. Vector residuals are averaged over components. The top
`upper_boundary_open_velocity` loss penalizes `(r_hat dot grad) v` normalized
by `V0/L`; unlike the previous one-sided no-inflow term, it permits either sign
of radial flow. The analogous side loss uses the outward longitude/latitude-face
normal. Side `curl B` uses the same detached per-height magnetic normalization
as the top current-free condition.

The top gas pressure is not a YAML tuning parameter. The bundled reference keeps
the existing 25-point STiC FALC_82 line-formation mapping on
`-5 <= log10(tau500) <= 1` unchanged, then restores all 37 native FALC samples
with `log10(tau500) < -5` for the physical-height thermodynamic reference. The
true native top is 2.073502459 Mm, where `T = 100 kK` and
`p = 0.0319344213 Pa`. Above that point, a quintic smootherstep in `log(T)`
reaches the configured 1 MK corona at 2.5 Mm and remains isothermal. Pressure is
integrated hydrostatically to the configured outer height using inverse-square
gravity and the same combined plasma closure used by synthesis.

`SolarPlasmaTable` is the single runtime provider for radiative and
thermodynamic plasma quantities. It reproduces the packaged STiC values exactly
from 2512 K through 10 kK. Between 10 kK and 31.6 kK, a quintic C1 window fades
the neutral-H and Fe I reservoirs and true continuum absorption to zero, while
continuum scattering transitions to analytic Thomson extinction, `n_e sigma_T`.
The thermodynamic branch joins the STiC state to a compact CHIANTI 11.0.2
default-coronal-equilibrium electron mapping, remains exact through 10 MK,
blends to the fully ionized ideal STiC mixture over 10--20 MK, and is exactly
ideal at and above 20 MK. Density, electron density, and mean particle mass use
one composition closure throughout.

Temperature, gas pressure, and microturbulence are decoded as unbounded linear
residuals around their references in natural-log space. The provider evaluates
those actual positive model values; it does not clip the model state or add a
table-support loss. Outside the STiC pressure axis, radiative reservoirs are
continued using the hybrid-EoS density ratio: one power for populations and
scattering and two powers for true absorption. This gives the expected linear
and quadratic pressure tails while keeping gradients back to the unrestricted
model outputs. Below the cold STiC edge, composition is frozen but the actual
ideal-gas `P/T` density scaling is retained.

## Preparing HMI responses

The Hinode run consumes calibrated `sp_prep` Level-1 FITS products with an
increasing wavelength WCS. The loader validates their wavelength, calibration,
geometry, and provenance contracts while constructing canonical rasters.

HMI runtime input is also entirely local. Build the phase-map response archive
for the exact HMI FITS acquisitions before inversion (a registered JSOC email is
required for this preparation step):

```bash
export JSOC_EMAIL=you@example.org
prom3theus prepare hmi-responses /path/to/hmi/*.fits \
  --output /path/to/calibration/hmi/2024-03-23
```

No phase-map identifier is supplied by the user. For every input `T_REC`,
preparation queries the matching definitive `hmi.B_720s` record for its
authoritative `INVPHMAP` assignment, then resolves and downloads that exact
camera record from `hmi.phasemaps_extended`. The manifest records the assignment
record, phase-map FSN, calibration record, and archive checksum for every
acquisition.

The HMI YAML points `observation.calibration.transmission_profile_directory` to
that directory. The preparation manifest binds every response archive by
SHA-256; the inversion verifies and selects each acquisition's response without
network access.

## Download, preparation, and run scripts

The repository keeps the workflows as small, instrument-specific shell files.
They use fixed paths inside the checkout and contain no dispatch logic or
overwrite modes. Both inversion launchers retain PBS headers and can be
submitted directly with `qsub`.

For the HMI run, download the exact native-cadence `hmi.S_720s` sequence,
prepare the 1024 by 512 Carrington cutouts used by the configuration, build the
phase-map response archive, and start the inversion:

```bash
export JSOC_EMAIL=you@example.org
./scripts/hmi/download.sh
./scripts/hmi/prepare.sh
./scripts/hmi/run.sh
```

The defaults reproduce the retained 2024-03-23/24 HMI workflow using 20 native
12-minute slots in the half-open interval 22:12--02:12 TAI (last slot 02:00),
Carrington longitude 215 degrees, latitude -12 degrees, and a 1024 by 512 pixel
cutout. Response preparation resolves the calibration assigned to each
acquisition directly from JSOC. Delete or move an existing prepared directory
before intentionally rebuilding it; the scripts never overwrite data.

The 1024 by 512 cutout is exact: preparation does not append, reflect, or fill
extra pixels, and the inversion has no separate science-FOV mask or numerical
padding width. Physics longitude/latitude bounds are the extrema of valid
observed surface pixels across the selected timeline. Consequently, any guard
region is contextual margin chosen when making the large cutout; side boundary
samples lie on that cutout's valid angular envelope.

Hinode acquisition and SolarSoft `sp_prep` calibration remain external because
there is no equivalent trustworthy Python preparation in this package. Place
the calibrated Level-1 columns directly in
the configured directory
`/glade/work/rjarolim/data/inversion/hinode_2011_02/20110214_000004`. Validate
the packaged atomic, continuum, blend, solar-reference, and spectral-response
resources, then run the static MHS inversion:

```bash
./scripts/hinode/load_resources.sh
./scripts/hinode/run.sh
```

The scripts contain no cluster scheduler, machine-specific path, account,
email, or removed namespace assumptions.

## Running an inversion

```bash
prom3theus invert configs/hinode_lte_mhs.yaml
prom3theus invert configs/hinode_lte_mhs_extrapolation.yaml
prom3theus invert configs/hmi_lte_mhs.yaml
prom3theus invert configs/hmi_lte_dynamic.yaml
```

Each run validates configuration, resource checksums, observation geometry,
wavelength support, velocity-frame semantics, and instrument response before
training. Observation cache identities include complete-content hashes of the
selected FITS inputs and external calibration manifest, so changing source data
cannot silently reuse stale arrays. There are no implicit instrument defaults
hidden in the runner.

Validation figures are written to the durable
`solver.output_directory/diagnostics` directory. Each scheduled validation
creates a `*_stokes_validation.png` comparison containing reference and
predicted I, Q, U, and V maps, ensemble line profiles, and integrated-fit
scatter panels.

The durable output is a schema-v1 artifact containing:

```text
artifact/
  manifest.json
  weights.pt              # tensor-only state, SHA-256 recorded in manifest
  observations/
    manifest.json
    r0000_a000.npy
    ... flat, checksum-protected raster arrays ...
```

The artifact manifest records the resolved configuration, resource digests,
observation contract, model constructor, and provenance. Every artifact must
satisfy the exact schema-v1 contract.

## Exporting a result

```bash
prom3theus export /path/to/run/artifact /path/to/atmosphere.npz \
  --depth-samples 101 \
  --batch-size 4096 \
  --include-stokes
```

Export reconstructs the exact forward model from the artifact manifest, checks
the observation store and resource signatures, loads weights with PyTorch's
restricted tensor loader, and evaluates the requested depth grid. For a
multi-raster sequence, the spatial export is evaluated on the validation raster
named by the immutable observation-store manifest.

The default atmosphere arrays follow those observer rays only through the LTE
line-formation domain. To export the upper solution from an extrapolation run,
request the separate full-shell product explicitly:

```bash
prom3theus export /path/to/run/artifact /path/to/atmosphere.npz \
  --include-full-shell \
  --full-shell-samples 161
```

The additional `full_shell_*` arrays sample outer-to-inner geometric height on
radial Carrington columns rooted at every valid validation-raster chart/time
coordinate. Their Cartesian vectors use the heliocentric Carrington frame and
their velocities are co-rotating. They include position, temperature, gas
pressure, mass density, microturbulence, magnetic field, and velocity in
Cartesian and local spherical components. This product does not assign optical
depth or synthesize Stokes profiles above the line-formation domain; its NPZ
metadata records the physical height bounds, sampling convention, and vector
frames. Cost scales with valid pixels times `--full-shell-samples`; lower
`--batch-size` when device memory is limited.

## Numerical API

The public package root deliberately exposes only its version. Import from the
layer that owns an operation:

```python
from prom3theus.config import load_config
from prom3theus.resources import validate_resource_bundle
from prom3theus.rt import LTESynthesizer, RayDistancePath

config = load_config("configs/hinode_lte_mhs.yaml")
resources = validate_resource_bundle()
```

Standalone LTE synthesis always receives an explicit integration path. The
formal solver never guesses whether an array represents optical depth,
geometric height, or ray distance.

## Synthetic recovery and tests

Run the differentiable synthetic recovery check:

```bash
prom3theus recovery --check --steps 500
```

Run the focused suite from the source tree:

```bash
pytest
```

The suite covers the strict maintained configurations, coordinates and neural
primitives, LTE opacity and polarized transfer, ray geometry, both instruments,
safe observation/artifact persistence, inversion constraints, training
orchestration, CLI behavior, package boundaries, and wheel contents.
