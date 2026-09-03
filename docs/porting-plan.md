# LTE framework porting plan

## Outcome

The repository becomes a `prom3theus` source-layout package dedicated to LTE
inversions. Its two complete applications are the Hinode/SOT-SP and SDO/HMI
inversion paths.

The package name and radiative-transfer boundary are deliberately neutral so a
future NLTE backend can implement the same scene contract without changing the
observation, instrument, training, or artifact layers.

## Target package layout

```text
src/prom3theus/
  core/             units, coordinates, neural primitives, solar kinematics
  config/           strict schema-v1 model, YAML loading, path resolution
  resources/        checksum-verified LTE atomic and continuum data
  rt/               atmosphere contract, LTE synthesis, formal transfer
  observations/     canonical rasters, datasets, safe array stores
  instruments/
    hinode_sp/       Hinode reader and spectral operator
    hmi/             HMI reader, acquisition/cutout preparation, filter operator
  inversion/        forward composition, objectives, sampling, constraints
  training/         thin Lightning orchestration and optional callbacks
  artifacts/        versioned tensor-only save/load and export
  diagnostics/      synthetic recovery and scientific diagnostics
  cli/               one `prom3theus` command tree
```

Dependencies point inward: the CLI calls application runners; runners compose
the inversion, observation, and instrument layers; those depend on `rt` and
`core`. The numerical `rt` package never imports Lightning, instrument readers,
visualization, or the CLI. Instrument adapters never own optimizers. Persistence
never pickles Python classes.

## Configuration reset

Ship exactly two YAML files:

- `configs/hinode_lte_mhs.yaml`
- `configs/hmi_lte_subframe.yaml`

The Hinode file preserves the full-resolution static MHS run for raster
`20110214_000004`, translated from its earlier operational configuration. The
HMI file converts the retained `lte_subframe_20240324.yaml` workflow. There is
no maintained Hinode time-series configuration or compatibility parser.

Both files use the same top-level order:

1. `schema_version` and `solver`
2. `resources`
3. `observation`
4. `atmosphere`
5. `synthesis` and `instrument`
6. `training` and `loss`
7. `diagnostics`, `runtime`, and `logging`

Schema version 1 is strict: unknown keys, duplicate YAML keys, implicit
instrument choice, and non-LTE solver values are errors. Both preserved runs
retain their literal GLade data/output paths and Derecho scratch paths. Ambient
environment variables never redirect maintained workflows.

## Component review and implementation decisions

### Scientific core and LTE transfer

- Keep the tested stratified atmosphere decoder, polarized line profiles,
  Zeeman patterns, continuum opacity, and formal solver.
- Require an explicit integration path (`RayDistancePath`,
  `GeometricHeightPath`, or `OpticalDepthPath`) instead of inferring geometry
  from tensor shapes or optional arguments.
- Package one checksum-pinned LTE resource bundle covering Fe I 6173 A and the
  Hinode Fe I 6301/6302 A pair.
- Separate payload ownership physically: shared atomic and STiC data live in
  `resources/data/common`, while Hinode/SP and HMI metadata and reference data
  live in `resources/data/hinode_sp` and `resources/data/hmi_stokes`.
- Remove reference-only equation-of-state implementations and runtime fallback
  tables. Production always uses the validated STiC/Wittmann resource contract.

### Observations and instruments

- Hinode and HMI remain independent adapters that produce one canonical
  `ObservationRaster` contract. Hinode accepts one flat prepared raster; HMI
  retains its separate time-dependent acquisition path.
- Runtime readers accept validated local inputs only. Network acquisition of
  HMI observations and response data lives in explicit download/preparation
  commands.
- Preserve scientifically necessary velocity-frame and FITS calibration
  contracts through exact typed interfaces.
- Store prepared rasters as JSON manifests plus checksum-protected `.npy`
  arrays so they are safe, inspectable, and memory-mappable. Bind caches to
  complete-content hashes of their selected FITS and HMI response inputs.

### Inversion and training

- Compose the atmosphere, LTE synthesizer, instrument response, depth sampler,
  Stokes objective, and magnetofluid constraints through explicit interfaces.
- Keep Lightning as orchestration only: training/validation steps, logging, and
  optimizer scheduling. Numerical forward work belongs in `inversion`.
- Keep only constraints used by the two LTE runs. Static
  magnetohydrostatic equilibrium is a distinct pressure--gravity--Lorentz
  equation; it is not approximated with the temporal momentum equation.
  Disabled physics is explicit in YAML; there are no instrument-specific
  hidden defaults.
- Use one application runner accepting the typed `InversionConfig`; remove
  nested dictionary normalization and command-line configuration mutation.

### Artifacts and export

- Save a schema-v1 JSON manifest, a SHA-256-bound tensor-only state dictionary,
  and an immutable copy of the canonical observation store inside each artifact.
- Reconstruct exports exclusively from that manifest. Missing fields or another
  schema version fail with an instruction to rerun the inversion.

### Command line and packaging

- Publish one command, `prom3theus`, with focused subcommands for inversion,
  export, strict config validation, packaged-resource validation, exact HMI
  download/input preparation, and synthetic recovery.
- Use a `src/` package layout. Keep heavy observation, visualization, and HMI
  response-preparation dependencies in optional extras.
- Wheels contain the `prom3theus` package, verified resources, and the two
  public configurations. Source distributions additionally retain the
  architecture documents and static checkout scripts referenced by the README.

## Removal inventory

The port removes these superseded surfaces in full:

- the complete `pme/` namespace, including all Milne--Eddington models,
  spherical-ME training and evaluation, generic loaders, entry points, and LTE
  copies under that namespace;
- all 46 files under `config/` and every configuration variant other than the
  two schema-v1 LTE runs;
- legacy shell launchers and one-off Python scripts replaced by the unified CLI
  plus static, instrument-specific checkout scripts for HMI download,
  preparation, and run and for Hinode resource validation and the MHS run;
- tests and evaluation utilities coupled to the removed namespace and ME
  implementation;
- tracked generated lookup archives and the flat `requirements.txt`, superseded
  by the verified resource package and `pyproject.toml`;
- former console scripts, serialized cache formats, and checkpoint formats.

Raw observations and existing user run directories are data, not source code;
the port does not delete them.

## Verification gates

1. Exactly two source YAML files exist, and both load through the strict schema.
2. Source, entry-point, metadata, and closed-registry inventories match the
   current `prom3theus` package.
3. Core, transfer, observation-store, instrument, inversion, artifact, CLI, and
   packaging tests pass from an installed wheel without repository path hacks.
4. The packaged resource bundle validates every checksum and covers both
   instrument wavelength domains.
5. A smoke inversion can assemble each YAML through its real adapter boundary;
   synthetic LTE recovery remains differentiable and within its tolerance.
6. An artifact can be loaded and exported using only its schema-v1 manifest,
   weights, observation store, and verified resources.

## Execution order

1. Snapshot the dirty worktree and establish the removal inventory.
2. Introduce the `src/prom3theus` package and strict two-file configuration.
3. Port and isolate the scientific core, resources, and LTE transfer solver.
4. Port independent Hinode/HMI adapters and safe observation persistence.
5. Split inversion composition from Lightning orchestration; add artifacts and
   the unified CLI.
6. Remove the superseded source, configuration, script, resource, and test
   surfaces listed in the removal inventory.
7. Rewrite documentation and packaging, then run source, wheel, CLI, scientific,
   and architecture verification.
