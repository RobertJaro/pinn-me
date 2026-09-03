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

## The two run configurations

The maintained source and distribution contain exactly two public YAML run files:

- [`configs/hinode_lte_mhs.yaml`](configs/hinode_lte_mhs.yaml)
- [`configs/hmi_lte_subframe.yaml`](configs/hmi_lte_subframe.yaml)

They share one strict schema and the same readable section order:

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

Both maintained YAML files hard-code their preserved GLade input/output paths
and Derecho scratch paths. Ambient environment variables do not redirect
either workflow.

Inspect the fully resolved, typed configuration before a run:

```bash
prom3theus validate-config configs/hinode_lte_mhs.yaml
prom3theus validate-config configs/hmi_lte_subframe.yaml
```

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
  --output /path/to/calibration/hmi/2024-03-23 \
  --phase-map-fsn 123456789
```

The phase-map FSN is explicit: obtain the authoritative record identifier for
the acquisitions being prepared, then pass it directly. Preparation queries
`hmi.phasemaps_extended`; it does not depend on an inversion-product series.

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
cutout. The preparation script pins the verified
`hmi.phasemaps_extended` calibration FSN `230562565`. Delete or move an existing
prepared directory before intentionally rebuilding it; the scripts never
overwrite data.

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
prom3theus invert configs/hmi_lte_subframe.yaml
```

Each run validates configuration, resource checksums, observation geometry,
wavelength support, velocity-frame semantics, and instrument response before
training. Observation cache identities include complete-content hashes of the
selected FITS inputs and external calibration manifest, so changing source data
cannot silently reuse stale arrays. There are no implicit instrument defaults
hidden in the runner.

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

The suite covers the strict two-file configuration, coordinates and neural
primitives, LTE opacity and polarized transfer, ray geometry, both instruments,
safe observation/artifact persistence, inversion constraints, training
orchestration, CLI behavior, package boundaries, and wheel contents.
