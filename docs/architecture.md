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
| `config` | Exact schema-v1 parsing, environment expansion, path resolution | Component construction, defaults chosen by adapters |
| `resources` | The immutable production bundle, common atomic/thermodynamic namespace, instrument namespaces, and checksum validation | Downloads, runtime table generation |
| `resource_builder` | Flat common-atomic and per-instrument builders, pinned downloads, STiC/Wittmann generation, FTS extraction, and exact reproduction | Runtime imports, training-time generation |
| `rt` | Atmosphere contracts, opacity, polarized LTE synthesis, formal transfer | Observation I/O, instrument response, optimization |
| `observations` | Canonical raster/batch contracts, safe sequence stores, data loading | FITS-specific calibration, spectral synthesis |
| `instruments` | Independent Hinode and HMI readers, strict HMI acquisition/cutout/response preparation, response operators | Optimizers, loss scheduling, artifact policy |
| `inversion` | Backend-neutral forward composition, objectives, constraints, run assembly | Diagnostic rendering, schema translation |
| `training` | Lightning lifecycle and optional visualization callbacks | Radiative-transfer equations, instrument parsing |
| `artifacts` | Atomic self-contained persistence and strict reconstruction/export | Observation preparation, training orchestration |
| `diagnostics` | Synthetic recovery and scientific checks | Production-run mutation |
| `cli` | Argument parsing and lazy dispatch | Hidden scientific configuration |

## Radiative-transfer boundary

Atmosphere state, ray geometry, spectral grids, and synthesis results are
backend-neutral contracts.  The LTE backend computes populations and source
functions directly from local thermodynamic state.  A future NLTE backend may
own global populations and convergence state behind the same scene-level
synthesis interface.

The formal solver never guesses its integration coordinate.  Traced spherical
rays carry monotonically increasing physical distance.  A plane-parallel
solver, when used, is selected explicitly and has its own input contract.

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
local inputs and never contact JSOC.

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

Only `configs/hinode_lte_mhs.yaml` and
`configs/hmi_lte_subframe.yaml` are shipped. Both use exact schema version 1
and the same hierarchy. Unknown fields, implicit instrument selection, and
command-line text substitution are rejected.

## Public API

The root package exposes only its version.  Users import numerical, observation,
instrument, inversion, or artifact APIs from their owning namespace.  This
keeps importing the LTE solver independent of the training and observational
software stacks.
