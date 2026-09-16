# Initial HMI + AIA integration plan

> Historical design for the initial setup-only milestone. Its schema-v2
> preservation and compatibility requirements were superseded by the explicit
> breaking-change decision on 2026-09-07. The current pipeline and acceptance
> evidence are maintained in [modular-runner-plan.md](modular-runner-plan.md)
> and [architecture.md](architecture.md). Scientific discussion below is retained
> as design context, not a description of the current runner.


## 1. Objective and stopping point

Add AIA EUV images to the current dynamic HMI inversion as a second observation
stream that shares the same unrestricted atmosphere model and thermodynamic EOS.
The first milestone ends with a reproducible setup-only run that:

- downloads and prepares one small, real HMI/AIA sample;
- loads HMI Stokes and AIA images on their own native pixel grids;
- synthesizes HMI Stokes and optically thin AIA 171, 193, and 211 Å images;
- applies explicit AIA uncertainty and calibration models;
- writes initial diagnostic figures and a machine-readable validation report;
- evaluates forward and gradient smoke checks without an optimizer step; and
- never calls `Trainer.fit`, creates a checkpoint, or changes a model parameter.

Actual joint optimization is deliberately outside this milestone.

The initial implementation also does **not** add a coronal energy equation,
off-limb synthesis, EUV absorption, non-equilibrium ionization, PSF-aware patch
training, or AIA 94/131/304/335 Å. Those are follow-up phases.

### 1.1 Implemented v1 status

The setup-only milestone is now implemented, with the following deliberately
narrower scientific scope than the later target design in this document:

- `aia_euv_v1` packages a checksum-verified, total-only 171/193/211 response
  derived from CHIANTI 11.0.2 and the AIA instrument response. It is tabulated
  at fixed `log10(ne/cm^-3)=9` on `log10(T/K)=4...9`, uses piecewise-linear
  interpolation with a one-cell quintic taper to zero at both edges, and
  returns exactly zero outside that interval without modifying the atmosphere.
  The source artifact uses the
  `sun_coronal_2021_chianti` abundance. Component-separated emissivities,
  PCHIP derivatives, a direct in-repository CHIANTI calculation, and analytic
  continuum tails remain a future resource version.
- `download aia-euv` selects the nearest quality-zero records to explicit UTC
  targets. `prepare aia-calibration` separately publishes the pinned V10
  correction table and gap-free event-specific JSOC master-pointing rows.
  Both outputs are immutable and checksummed.
- AIA preparation performs pointing correction, Level-1.5 registration,
  exposure normalization, V10 degradation correction exactly once, and a
  documented inflated diagonal uncertainty approximation. The initial sample
  uses an explicit Carrington longitude/latitude rectangle; a shared projected
  HMI polygon and finalized cross-instrument pairing manifest are not yet
  implemented.
- The generic image contract/store/data module, native-ray optically thin
  composition, AIA term, robust channel-balanced likelihood, identifiable
  three-channel gain model, diagnostic registry, and callback adapter are
  implemented. Fitting batches contain every configured AIA channel, and the
  physics sampling envelope includes the complete inclined AIA ray support out
  to the outer shell. The objective coordinator itself is modality-neutral.
  Runtime construction still uses a closed typed HMI/AIA dispatch, so external
  plugin registration for a third imager remains architectural follow-up.
- `prom3theus invert configs/hmi_aia_dynamic.yaml` trains the shared atmosphere
  with the single-device Lightning runtime, periodic AIA diagnostics, and
  resumable checkpoints. The maintained YAML sets AIA weight to zero: HMI and
  physics train while AIA renders. Positive AIA weight enables image fitting
  and calibration priors, including after a checkpoint resume.
- Pure internal evaluation helpers check stream/shared-loss gradients and LOS
  quadrature without optimization. Their AIA evaluators and renderers are also
  used by training diagnostics; there is no separate setup-only CLI command.

Sections below retain the fuller integration target. Where they call for
component tables, smooth analytic tails, a shared footprint/pairing manifest,
registry-only construction, or multi-device training, those items are
next-stage work rather than claims about `aia_euv_v1`.

## 2. Architectural decisions

1. **Preserve the existing HMI path.** The current HMI full-disk downloader,
   24-segment acquisition validation, Carrington-centered subframing, Stokes
   observation store, LTE module, state-dict names, and schema-v2 execution path
   remain the reference implementation.

2. **Represent each measurement as an observation stream.** A stream has four
   independently owned pieces: a prepared-data adapter, a typed store codec, a
   data module, and an inversion data term. The multimodal runner iterates over
   registered streams and never contains an `if instrument == "aia"` branch.

3. **Represent AIA with its own image contract.** The existing
   `ObservationRaster` and `ObservationStore` are intentionally Stokes-specific.
   AIA must not be encoded as fake Stokes data or hidden in
   `instrument_response`.

4. **Share persistence machinery, not raster schemas.** Keep the public Stokes
   store at version 2 and introduce a generic image-store schema at version 1,
   both implemented on a private checksum/manifest/atomic-array-store engine. This
   avoids duplicating persistence code without weakening either scientific
   contract. The joint run manifest records both store signatures.

5. **Make the scene a first-class contract.** AIA does not read fields from an
   HMI data module. A joint `SceneContract` owns the Carrington basis, solar
   radius, spatial/time affine transforms, physical bounds, and reference-stream
   identity. For this run HMI is configured as the reference stream, but every
   runtime stream binds to it only through the scene interface.

6. **Keep package ownership narrow.** Generic optically thin integration lives
   in `rt`; atmosphere/ray composition lives in `inversion`; AIA response and
   calibration live in `instruments/aia_euv`; data persistence lives in
   `observations`; and Lightning only schedules already assembled data and
   physics terms. The existing polarized `InstrumentOperator` protocol remains
   unchanged because an EUV image operator does not satisfy its spectral and
   polarization contract.

7. **Keep preprocessing and table generation offline from training.** AIA
   calibration, CHIANTI calculations, and downloads happen before a run. The
   training process reads checksum-verified compact resources and prepared
   arrays only.

8. **Do not bound the primary atmosphere for AIA.** Temperature and pressure
   continue to use the unbounded log-residual decoder. The AIA response operator
   is defined for every finite positive model state and never clamps the model's
   temperature or pressure to a table edge.

9. **Correct observations to one response epoch.** Prepared AIA data are
   degradation-corrected to the response-table reference epoch. The forward
   table therefore carries no observation-date degradation. Applying both would
   double-correct the data.

10. **Keep every channel's actual time.** AIA channels are matched independently
   to each HMI `T_OBS`, and the atmosphere is evaluated at the actual AIA
   exposure time rather than assigning all channels the HMI timestamp.

11. **Use on-disk, near-side rays initially.** Each AIA ray starts at its
   photospheric surface intersection and ends at the shared atmosphere's outer
   shell.
   Cool layers contribute negligibly through the temperature response; no
   artificial height gate is placed in the imaging operator.

```text
JointRuntime
├── resource contracts: legacy_lte_v2 + requested optional sets
├── SceneContract <- physical geometry/time from every prepared stream
├── prepared streams
│   ├── HMI adapter -> unchanged Stokes-v2 codec -> Stokes data module
│   └── AIA adapter -> generic image-v1 codec   -> image data module
├── JointForwardModel
│   ├── one shared atmosphere
│   └── observation_terms: ModuleDict
│       ├── Stokes term -> LTE composition + polarized instrument
│       └── AIA term    -> ray composition + AIA response + gain/objective
├── physics evaluator + atmosphere regularization terms
└── joint Lightning runner + pure evaluators + diagnostics callback

stream batch + SceneContract -> matching term -> DataTermBatchResult
```

### 2.1 Modularity acceptance rules

The implementation is not considered modular unless all of the following hold:

- adding another optically thin imager requires a new adapter/response/data-term
  registration, not edits to the multimodal training loop;
- `prom3theus.rt` imports no instrument, observation, configuration, Lightning,
  plotting, SunPy, or Astropy package;
- AIA preparation imports no training or inversion module;
- stream-specific batch validation is owned by each data term;
- the objective coordinator consumes one common `DataTermBatchResult` type and does
  not inspect AIA- or Stokes-specific tensors;
- the schema-v3 scene transform is the only authoritative conversion from
  absolute geometry/time to chart coordinates and relative hours; the
  atmosphere retains neural normalization and decoding, while legacy stored
  Stokes coordinates remain only for schema-v2 and equivalence checks;
- static instrument resources and event-specific observation metadata are never
  stored in the same mutable cache; and
- registered components declare exact resource-set dependencies, whose union is
  validated before any store is opened;
- all schema-v2 HMI/Hinode tests and artifact reconstructions remain unchanged.

### 2.2 Fit to the current framework

| Current seam | Compatibility action | Schema-v3/EUV extension |
| --- | --- | --- |
| `ObservationSpec` / `ObservationRaster` / `ObservationStore` v2 | Keep the exact Stokes contract and bytes | Parallel typed image contract/codec behind `PreparedObservationStream` |
| `ObservationAdapter` registry | Keep the v2 wrapper and existing adapter behavior | Add adapter-owned codec, dependency selection, and data-module restoration |
| polarized `InstrumentOperator` | Leave unchanged | AIA emission operator is owned by its observation data term |
| `LTEForwardComposition` | Leave spectral behavior unchanged | Add a separate generic ray-integral composition |
| `LTEInversionModule` and `run_inversion()` | Leave constructor, state keys, execution, and P3S reconstruction unchanged | Add `JointInversionModule` and `joint_runner.py` |
| exact legacy resource bundle | Keep paths, inventory, and validator unchanged | Load an independent `aia_euv_v1` resource set only for runs that request it |
| legacy callbacks | Keep flat Stokes payload assumptions | Add namespaced evaluator/renderer registration for multimodal runs |

This is an additive migration: shared private helpers may be extracted under
regression tests, but the public schema-v2 route is never routed through the
new multimodal abstractions merely to remove duplication.

## 3. Scope of the CHIANTI and AIA tables

The existing
`src/prom3theus/resources/data/common/chianti_thermodynamic_table.json` only
provides equilibrium electrons per hydrogen nucleus for the hot EOS. It contains
no emissivity and cannot synthesize AIA images.

### 3.1 Required new response resource

Add:

```text
resource_builder/chianti_emission.py
resource_builder/aia_euv.py
src/prom3theus/resources/sets/euv/aia_euv_v1/manifest.json
src/prom3theus/resources/sets/euv/aia_euv_v1/sources.json
src/prom3theus/resources/sets/euv/aia_euv_v1/instrument_aia_euv.json
src/prom3theus/resources/sets/euv/aia_euv_v1/aia_temperature_response.json
```

`resource_builder/chianti_emission.py` owns only atomic-plasma calculations and
returns a versioned spectral-emissivity product. `resource_builder/aia_euv.py`
owns the instrument fold and converts that product into AIA channel responses.
The common CHIANTI builder must not import an AIA module, and runtime AIA code
must not know how the source atomic database was constructed.

Build the first response table with:

- channels 171, 193, and 211 Å;
- CHIANTI 11.0.2 atomic data and `chianti.ioneq`, matching the EOS release;
- bound-bound, free-free, free-bound, and two-photon emissivity;
- the pinned `sun_coronal_1992_feldman_ext.abund` abundance set;
- constant electron pressure, `n_e T = 10^15 K cm^-3`;
- `log10(T/K) = 4.00 ... 9.00` in 0.05-dex steps;
- the pinned AIA V8 full-instrument wavelength response;
- V10 degradation calibration metadata and a fixed reference epoch;
- EVE normalization applied once at that reference epoch;
- crosstalk included; and
- no empirical 94/131 Å `chiantifix`, since those channels are deferred.

The compact table stores the total response, its four physical components, and
general shape-preserving PCHIP derivatives. The current monotonic-only PCHIP
helper cannot be reused unchanged because AIA responses contain extrema and
multiple peaks.

The runtime convention is

\[
I_c = 10^{-10}\int n_e^2 K_c(T)\,ds_{\rm m},
\]

where `n_e` is in m⁻³, `ds_m` is in metres, and `K_c` is stored in the
canonical `DN s^-1 pixel^-1 cm^5` basis. The factor `10^-10` converts the
line-of-sight emission measure from m⁻⁵ to cm⁻⁵. The builder may use
`n_e n_H` internally when evaluating CHIANTI emissivity, but it converts once to
the single authoritative `n_e^2` response before writing the runtime resource.
The conversion assumptions and electron pressure are immutable metadata; a
second selectable normalization is not stored.

This convention uses the electron density already supplied by
`SolarPlasmaTable`. Total hydrogen-nucleus density is introduced later with the
radiative-loss/energy-equation resource rather than as an AIA-only API change.

### 3.2 Behavior outside the atomic grid

The atmosphere remains unrestricted. The response implementation owns its own
scientific continuation:

- below the CHIANTI range, line and continuum contributions join smoothly to
  zero;
- above the range, bound-bound, free-bound, and two-photon contributions fade
  smoothly, while free-free emission joins to a fully ionized analytic
  bremsstrahlung tail; and
- all joins are value- and first-derivative continuous in log temperature.

There is no endpoint saturation of model temperature and no table-domain
exception for a finite positive state. Non-positive or non-finite physical
states remain hard errors.

### 3.3 Deferred radiative-loss resource

Do not generate, package, or validate a radiative-loss table in the initial AIA
milestone. The build-only CHIANTI source cache can later support a separate
`chianti_radiative_loss` builder and
`resources/sets/plasma/chianti_radiative_loss_v1` resource set containing
`Lambda(T)` and its components, reviewed against the actual energy-equation
normalization and continuation. This prevents the imaging integration from
becoming coupled to an energy-equation design that has not yet been selected.

### 3.4 Reproducibility and resource-set changes

All compact, immutable EUV runtime products belong under the
`prom3theus.resources` package. Organize them in a dedicated namespace while
leaving the exact legacy inventory untouched:

```text
src/prom3theus/resources/
├── data/                              # frozen legacy_lte_v2 inventory
└── sets/
    ├── euv/
    │   └── aia_euv_v1/                # initial AIA response set
    └── plasma/                        # future shared EOS/energy resources
```

Each future EUV instrument or reusable runtime component gets an independently
versioned directory below `resources/sets/euv/`, with its own manifest, sources,
files, and declared dependencies. For example, a reusable spectral-emissivity
product promoted from build intermediate to runtime would become
`euv_emissivity_v1`; it would not be inserted into the AIA set. A later
radiative-loss table belongs in a separate
`resources/sets/plasma/chianti_radiative_loss_v1` set because it is plasma/energy
physics rather than an AIA instrument response.

This namespace owns static scientific data, not implementation logic: the AIA
operator remains under `instruments/aia_euv`, generic LOS integration remains
under `rt`, and resource builders remain under `resource_builder`.

The initial independently validated sets are:

```text
legacy_lte_v2  -> existing bundle.json and sources.json, unchanged
aia_euv_v1     -> resources/sets/euv/aia_euv_v1/{manifest.json,sources.json,...}
```

Each registered observation/data-term/instrument factory declares
`required_resource_sets(config)`. The data-only config layer validates only
resource-ID syntax. `build_joint_runtime()` resolves the factories, derives the
exact dependency union, and passes it to `validate_resource_sets(...)` before
opening an observation store. A joint run therefore resolves `legacy_lte_v2`
from `lte_stokes` and `aia_euv_v1` from `aia_optically_thin`, then cross-checks
their intended CHIANTI release. An HMI-only run never loads, hashes, or reports
AIA resources. The existing
`validate_resource_bundle()` remains the schema-v2 compatibility entry point
and continues to inventory only its existing `resources/data` root. The new
set is deliberately outside that root, with its own path resolver and package-
data rule, so adding it cannot appear as an unexpected legacy file.

Update or add:

- `resource_builder/source_cache.py`, `aia_euv.py`, `README.md`, and a separate
  pinned AIA/CHIANTI builder environment;
- an AIA-specific build command rather than adding its large inputs to the
  legacy `_fetch_inputs()` call;
- `src/prom3theus/resources/registry.py` for exact set IDs, relative roots, and
  dependency declarations;
- `src/prom3theus/resources/manifest.py` for set validation;
- `pyproject.toml` package data including the explicit initial pattern
  `sets/euv/aia_euv_v1/*.json`; and
- per-resource-set reproduction and validation tests.

The AIA response `sources.json` pins and hashes the full CHIANTI archive, AIA V8
instrument file, the V10 correction table used to define the response epoch,
abundance file, builder versions, and a small archived SSW reference vector.
`manifest.json` pins the digest of `sources.json` and every runtime file. The AIA
error table is not a response-build input and appears only in the independently
versioned preprocessing-calibration bundle. Each prepared observation records
the actual V10, error, and master-pointing hashes it used. The full CHIANTI archive must
stream to a build-only cache while hashing and extract safely into a temporary
directory; it must not pass through the legacy in-memory payload interface.
Only compact runtime products and their manifests enter the wheel. Raw CHIANTI
archives and large transient emissivity cubes remain in the build cache unless
a later runtime feature explicitly promotes one to its own resource set.

Define a `calibration_convention_id` as a digest of response epoch, degradation
table/version, correction direction and formula, EVE correction, crosstalk,
registered plate scale/pixel-area convention, intensity units, preprocessing
schema version, and pinned `aiapy` response/calibration version. Store it in the
response resource and every prepared AIA observation. Runtime rejects a
mismatch before synthesis.

Acceptance criteria:

- byte-for-byte reproducible resources from pinned sources;
- table components sum to the stored total;
- response peaks and selected nodes agree with the archived SSW/CHIANTI
  reference within a declared tolerance;
- all supported positive temperatures yield finite, non-negative values and
  finite gradients;
- the low- and high-temperature joins are C1;
- an analytic uniform slab reproduces its expected count rate, including the
  exact cgs/SI conversion; and
- training imports neither CHIANTI/Fiasco nor `aiapy`.

## 4. HMI + AIA download and subframe pipeline

### 4.1 HMI

Use `src/prom3theus/download/hmi.py` for direct DRMS downloads and
`src/prom3theus/instruments/hmi/subframe.py` for preprocessing. The downloader
requests camera-3 `hmi.S_720s` with all 24 `I0...V5` segments and delegates
file handling directly to DRMS. It does no staging or FITS validation. Preprocessing
validates complete acquisitions and publishes the subframes.

### 4.2 AIA acquisition

Add:

```text
src/prom3theus/instruments/aia_euv/__init__.py
src/prom3theus/instruments/aia_euv/constants.py
src/prom3theus/instruments/aia_euv/acquisition.py
src/prom3theus/instruments/aia_euv/download.py
src/prom3theus/instruments/aia_euv/subframe.py
src/prom3theus/instruments/aia_euv/calibration.py
src/prom3theus/download/aia.py
src/prom3theus/core/solar_patch.py
```

Download entry points:

```text
prom3theus prepare aia-calibration ...
prom3theus download hmi-stokes --output DIRECTORY --email EMAIL --start TAI --end TAI
prom3theus download aia-euv --output DIRECTORY --email EMAIL --start UTC --end UTC --cadence-seconds 720 --channels 171 193 211
```

The maintained `prom3theus.download.aia.download_aia_observations()` API accepts
start/end dates, cadence, and channels. It issues one standard DRMS FITS export
per channel and downloads directly into the output directory. No nearest-record
search, quality filtering, staging, or manifest creation occurs during download.
`prepare aia` scans the FITS files and performs metadata validation, grouping,
calibration, and subframing. Existing acquisition bundles remain readable.

The implemented download stage uses two independent instrument commands, not a
joint pairing coordinator. HMI selects its TAI record interval; AIA samples an
explicit UTC interval at the requested cadence.
Neither downloader reads the other instrument's files. Exact selected exposure
times remain in instrument-owned provenance. Cross-instrument alignment belongs
to observation assembly/validation, not acquisition.

Never compare HMI TAI and AIA UTC strings directly. Persist the HMI `T_OBS`, AIA
observation/mid-exposure time, time scale, and signed offset for every pairing.

`prepare aia-calibration` creates a separate, versioned preprocessing bundle
containing the event-specific master-pointing records and pinned V10
degradation/error tables. These tables do not belong to the runtime resource
set. Its degradation epoch/convention must match the compact response's
`calibration_convention_id`; the error and pointing digests remain independent
uncertainty/geometry provenance. Preparation must never trigger a hidden
network request.

### 4.3 Instrument-specific subframe preparation

Keep two generic packaged preprocessors: `prom3theus.preprocess.hmi` and
`prom3theus.preprocess.aia`, exposed as `prom3theus prepare hmi` and
`prom3theus prepare aia`. All input/output and response/calibration paths are
caller-supplied; there is no fixed base-directory layout in Python.
Both accept explicit Carrington longitude and
latitude. HMI accepts width/height in detector pixels; AIA accepts width/height
in angular degrees. The deployment values live in `scripts/sdo/prepare.sh`.
There is no joint preparation manifest or cross-instrument file dependency.
Each instrument projects the requested crop at its own observation time and
keeps its native pixel grid. Reuse must validate both the input identity and
the requested crop, including its position. The current shared center does not
imply that the two instrument footprints have exactly the same boundaries.

Prepare each AIA full disk before cropping:

1. load Level 1 counts and retain the original exposure metadata;
2. estimate count uncertainty from the pinned error table;
3. update pointing using the downloaded master-pointing table;
4. register to Level 1.5 and its 0.6-arcsec plate scale;
5. correct degradation to the response-table reference epoch;
6. divide image and propagated uncertainty by exposure time;
7. crop the projected polygon's detector-space bounding rectangle, compute a
   Carrington polygon-membership mask, and combine it with finite, quality, and
   saturation masks; and
8. write the prepared image and provenance manifest atomically.

PSF deconvolution is omitted initially. Deconvolution can amplify noise, and a
spatial PSF cannot be represented honestly in random independent-pixel
training. A forward PSF or documented common rebinning is a pre-training task.

Prepared intensity units are `DN s^-1 pixel^-1`. Preserve finite negative
values and use an explicit valid mask; do not log-transform or silently replace
them. Invalid samples may use finite placeholders in tensors only when their
mask guarantees zero objective weight.

Every prepared record stores raw and preprocessing-bundle hashes, response
`calibration_convention_id`, JSOC record identity, channel, exact times,
exposure, degradation factor, operation order, `aiapy` version, input/output
WCS, footprint-sequence and exposure-group identity, output units, polygon/mask
definition, and uncertainty transformation.

Registration creates correlated errors. Version 1 must either propagate
variance with the actual interpolation weights or explicitly label its
per-pixel uncertainty as a conservative diagonal approximation and include a
separate resampling/model-discrepancy floor. It must never claim that
post-registration pixels have independent exact count noise.

Current `aiapy` requires Python 3.12 while the training package supports Python
3.11. Keep the training floor unchanged and provide a pinned Python 3.12 AIA
preparation extra/environment. Prepared stores and runtime tables remain usable
from the Python 3.11 training environment.

Acceptance criteria:

- rerunning the same request resolves the same records and hashes;
- incomplete channel/time groups fail atomically;
- all pairings satisfy the time tolerance and retain their actual times;
- the degradation factor appears exactly once in the observation-to-response
  convention;
- uncertainty receives the same scalar degradation/exposure transforms as the
  image;
- projecting both products back to the serialized Carrington surface footprint
  using valid polygon-masked pixels agrees within the declared WCS tolerance
  (their time-dependent helioprojective corners are not expected to be
  numerically identical); and
- no preprocessing step reprojects AIA onto HMI pixels.

### 4.4 Durable data stages

Keep three independently versioned, class-free stages:

1. the raw acquisition bundle with exact record identities and checksums;
2. the calibrated Level-1.5 native-grid cutout bundle with preprocessing and
   `SolarPatchSequence` provenance; and
3. the immutable image-observation store optimized for training reads.

Each stage has its own manifest and can be rebuilt from the preceding stage.
The observation adapter performs only stage 2 to stage 3 conversion; network
access and `aiapy` calibration never occur while constructing a training
runtime.

## 5. Small real integration sample

Use the existing event and local HMI reference:

```text
HMI record:       2024-03-23T22:12:00 TAI
HMI I0 T_OBS:     2024-03-23T22:11:58.388 TAI
AIA match target: 2024-03-23T22:11:21.388 UTC
channels:         171, 193, 211 Å
Carrington center: longitude 215°, latitude -12°
HMI crop:         256 x 128 pixels
```

Use a default AIA matching tolerance of 12 seconds and require at most 6 seconds
for this sample when quality-good records are available. AIA crop dimensions
are derived by projecting the exposure group's Carrington surface footprint at
each AIA time, not copied from HMI.

The retained GLade deployment exposes only three shell entry points:
`scripts/sdo/download.sh`, `scripts/sdo/prepare.sh`, and `scripts/sdo/run.sh`.
The first calls `download hmi-stokes` and `download aia-euv` independently.
Dates, AIA cadence/channels/tolerance, email, and output directories are supplied
in the shell file. These commands download observation files only.
The second calls packaged `prepare hmi` and `prepare aia`; AIA preparation
acquires or reuses correction and pointing tables for the downloaded exposures,
with explicit longitude/latitude and instrument-specific subframe dimensions.
The third starts the full inversion, with AIA render-only at weight zero.
There is no shell staging or
embedded Python; instrument APIs publish their own outputs atomically.

Store products below the fixed base:

```text
/glade/work/rjarolim/data/prom3theus/2024_03/hmi/full_disk/
/glade/work/rjarolim/data/prom3theus/2024_03/hmi/subframes/
/glade/work/rjarolim/data/prom3theus/2024_03/hmi/responses/
/glade/work/rjarolim/data/prom3theus/2024_03/aia/level1/
/glade/work/rjarolim/data/prom3theus/2024_03/aia/calibration/
/glade/work/rjarolim/data/prom3theus/2024_03/aia/prepared/
```

AIA targets are supplied independently in UTC; actual HMI and AIA observation
times must be checked when assembling the joint run.
The HMI preprocessor scans and crops every complete acquisition in the input
folder, so the same code handles one slot or a time series. AIA acquisition and
calibration manifests remain instrument-owned provenance records, but no
hand-authored joint request file is used. Normal CI uses synthetic inputs; real
download and preparation require `JSOC_EMAIL`.

## 6. Generic image-observation stream, store, and loader

Add the reusable image layer first, then register AIA against it:

```text
src/prom3theus/observations/streams.py
src/prom3theus/observations/scene.py
src/prom3theus/observations/_array_store.py
src/prom3theus/observations/image_contracts.py
src/prom3theus/observations/image_store.py
src/prom3theus/observations/image_dataset.py
src/prom3theus/observations/image_data.py
src/prom3theus/instruments/aia_euv/geometry.py
src/prom3theus/instruments/aia_euv/timeline.py
src/prom3theus/instruments/aia_euv/observation.py
src/prom3theus/observations/multistream_data.py
```

The existing `ObservationSpec`, `ObservationRaster`, `ObservationStore` version
2, and `StoredObservationDataModule` remain the spectropolarimetric public API.
Do not broaden their fields or make their arrays optional. Add parallel generic
image contracts:

- `ImageObservationSpec` describes an image product, intensity unit, channel
  vocabulary, exposure-group vocabulary, and geometry/calibration conventions;
- `ImageObservationRaster` represents one channel and one exposure, with
  intensity, uncertainty, and valid mask `[H,W]`, physical ray direction and
  photospheric surface anchor `[H,W,3]`, exact absolute observation time,
  channel identifier, exposure-group identifier, and immutable WCS/provenance;
  and
- a small `ObservationDescriptor` protocol exposes only the identifiers and
  `observation_kind` shared by Stokes and image specifications. It is not a
  lowest-common-denominator array schema.

Keep one image raster per channel/exposure so different times and native grids
are not hidden in `[H,W,C]`. Do not persist schema-v3 chart coordinates or
relative times: `SceneContract` derives those from physical geometry and
absolute TAI seconds at runtime. `pixel_index`, `image_index`, and `channel_index` are
deterministic dataset batch fields, not redundant raster arrays.

Extract only checksum, safe-manifest, memory-map, locking, and atomic-publication
mechanics into private `_array_store.py`. Typed codecs define the exact Stokes-v2
and image-v1 layouts. The public Stokes serialization and signatures must remain
byte-for-byte compatible; `ImageObservationStore` version 1 uses the shared
engine without pretending its records are Stokes rasters.

Generalize the registry lifecycle rather than adding a second hard-coded load
function. A typed `ObservationAdapter` declares:

```text
name + observation_kind
codec
build / describe / source_files
preparation_dependencies
store_configuration
build_data_module
```

`load_or_prepare_stream()` delegates serialization and reconstruction to those
adapter-owned hooks and returns a `PreparedObservationStream` containing the
descriptor, data module, store path, and source signature. A cache signature
hashes only `store_configuration(config)`, the adapter's declared preparation
dependencies, and its source files. For AIA, that configuration contains only
stage-2-to-stage-3 conversion inputs and channel selection; batch sizes,
validation selection, worker counts, and future scheduling options are excluded.
A change to a CHIANTI response must not rebuild calibrated AIA images, changing
loader options must reuse the same image store, and an AIA resource change must
not invalidate an HMI store. Keep `load_or_prepare_observation()` and its broad
legacy signature behavior as the unchanged schema-v2 wrapper.

Factor instrument-neutral WCS-to-ray construction out of HMI geometry, while
leaving Stokes bases and observer-velocity handling inside HMI. After all
streams load, a scene assembler constructs one immutable `SceneContract` from
the configured reference stream and every stream's physical geometry. It owns
the Carrington basis, solar radius, absolute time origin, spatial/time affine
normalization, and physics bounds enclosing every stream's complete ray support
through the atmosphere outer shell. This volume can be wider than the union of
the photospheric footprints because oblique EUV rays move laterally with
height. Each `PreparedObservationStream` is then bound to the scene before its
dataset emits chart/time coordinates. The schema-v3 HMI binding derives coordinates
from physical positions/times and verifies them against its legacy stored
coordinates; it does not change the version-2 store. No AIA class imports or
reads an HMI data module.

The scene/model seam is exact:

```text
SceneContract.transform(position_m, absolute_tai_seconds)
    -> chart_time = [chart_x_mm, chart_y_mm, time_hours], geometric_height_m

atmosphere.evaluate_chart_height_points(chart_time, geometric_height_m)
    -> decoded physical atmosphere
```

The atmosphere's public method applies its own configured affine normalization
and the unrestricted physical decoder. Do not call the neural network directly,
and do not pass an already transformed point to `evaluate_position_points()`,
which would repeat the physical-to-chart conversion.

`StoredImageDataModule` flattens valid pixels and derives reconstruction indices
in its dataset. Its validation unit is a complete paired exposure group, so the
171/193/211 images needed by one diagnostic are selected together even when
their shapes and times differ. `MultiStreamDataModule` exposes a mapping from
stream name to data module. The setup-only run requests one deterministic batch
from each stream directly; it does not use `CombinedLoader(max_size_cycle)`,
whose implicit cycling would define training weights. A later training config
must state steps per epoch and each stream's sampling/repetition policy
explicitly.

Acceptance criteria:

- exact typed-store round trips and dependency-scoped signature invalidation;
- changing only loader/validation/scheduling options leaves store signatures
  unchanged;
- UTC/TAI conversion and per-channel absolute-time preservation;
- complete-image reconstruction from dataset-derived batch indices;
- deterministic selection of a complete validation exposure group;
- on-disk ray/surface consistency and cross-instrument scene agreement;
- every configured LOS quadrature point lies inside the declared scene and
  physics-sampling domain;
- adding a second image instrument requires registration, not a store/loader
  branch; and
- HMI cache keys, store bytes, data-module behavior, and runtime outputs remain
  unchanged when AIA is absent or its resource set changes.

## 7. Differentiable image synthesis and the data-term seam

Use generic LOS mechanics with an AIA-specific response implementation:

```text
src/prom3theus/rt/optically_thin.py
src/prom3theus/instruments/aia_euv/response.py
src/prom3theus/instruments/aia_euv/operator.py
src/prom3theus/inversion/ray_integral.py
src/prom3theus/inversion/data_terms/base.py
src/prom3theus/inversion/data_terms/stokes.py
src/prom3theus/inversion/data_terms/aia_euv.py
```

`rt/optically_thin.py` contains only instrument-neutral differentiable path
quadrature and emissivity-shape validation. It does not load resources, know
channel names, sample the neural atmosphere, or import AIA. `AIAResponseTable`
under `instruments/aia_euv` loads the verified compact resource and interpolates
each response component differentiably in log temperature.

`RayIntegralForwardComposition` under `inversion` accepts an emissivity operator
and composes scene geometry, atmosphere evaluation, and the generic LOS
integrator. The AIA operator supplies

\[
\epsilon_c(T,n_e) = 10^{-10}n_e^2K_c(T)
\]

using unrestricted temperature and the electron density returned by the same
`SolarPlasmaTable` as the physics residuals. It has no hydrogen-density input
and no magnetic-field input. The composition:

1. traces each near-side ray from its photospheric intersection to the shared
   atmosphere's outer shell, not the 1.5-Mm LTE line-formation top;
2. makes the stored observer-to-Sun ray convention explicit and reverses it for
   surface-to-observer integration;
3. uses a configurable, nonuniform physical-height grid dense near the lower
   atmosphere;
4. asks `SceneContract` for chart/time coordinates and geometric height from
   physical sample points plus the raster's absolute time;
5. calls `atmosphere.evaluate_chart_height_points()` and then the injected
   emissivity operator; and
6. applies physical trapezoidal quadrature and the one documented SI/cgs
   conversion.

Begin with 192 ray samples and compare against 96 and 384 samples in the dry
run. The ray end is always `atmosphere_outer_shell`; do not duplicate outer
height in the AIA config. The first operator includes only the modeled volume
and reports limitations from foreground/background emission outside the lateral
footprint, emission above the model shell, and absorption by cool material.

### 7.1 Observation data-term protocol

Do not add AIA to the polarized `InstrumentOperator` registry or to
`LTEInversionModule._shared_step()`. Add a separate registry of observation data
terms. Each registered `ObservationDataTerm` is an `nn.Module` that declares a
compatible `observation_kind` and exact resource-set dependencies, validates its
own descriptor and batches, owns its forward composition and objective, and
receives the shared atmosphere and scene when evaluated. A term may own nuisance
parameters, but never owns or duplicates the atmosphere.

Every batch evaluation returns only:

```text
DataTermBatchResult(
    likelihood_loss,
    component_losses,
    metrics,
    sample_count,
    diagnostics=None,
)

ObservationDataTerm.nuisance_losses() -> Mapping[str, Tensor]
```

The joint coordinator computes each configured stream weight times
`likelihood_loss`, calls each term's `nuisance_losses()` exactly once per joint
step, and sums the returned values without inspecting their names. Calibration
priors are therefore not weakened when the AIA data weight is ramped. It logs
namespaced components/metrics but never inspects modality-specific tensors.
`StokesObservationTerm` wraps the current LTE forward/objective behavior for
schema-v3 runs; `AIAEUVObservationTerm` wraps the ray integral, gain model, and
image objective. The existing schema-v2 `LTEInversionModule`, constructor,
registered attributes, and state-dict names remain unchanged. Stateless math
may be extracted for both paths only with exact regression tests; the legacy
module is not rewritten as a special case of the joint module.

Treat atmosphere regularization as a separate registered
`AtmosphereRegularizationTerm` with an explicit scene-collocation sampling
contract, evaluated once per joint step. Do not implicitly reuse the atmosphere
sampled by the Stokes forward pass. The current schema-v2 vector regularizer and
its sampling behavior stay unchanged.

Forward/data-term tests must cover:

- a uniform analytic slab and exact cgs/SI conversion;
- exact shell intersections, ray orientation, and integration length;
- quadrature convergence;
- non-negative finite images;
- finite gradients with respect to temperature, pressure/electron density, and
  atmosphere parameters;
- expected zero direct dependence on magnetic-field values;
- stable response through every interpolation extremum and continuation seam;
- rejection of a descriptor or batch with the wrong observation kind; and
- summation/logging of an arbitrary registered term without coordinator edits;
- changing a stream data weight does not change its nuisance-prior contribution;
  and
- atmosphere regularization is evaluated once without requiring a Stokes batch.

## 8. Calibration nuisance model and AIA objective

Use an unbounded, identifiable log-parameterization with positive physical
gains. For `C` channels, choose a fixed orthonormal contrast matrix
`Q in R^(C x (C-1))` with `Q^T 1 = 0` and write

\[
\log \mathbf{g} = a\mathbf{1} + Q\boldsymbol{\eta},\qquad
I^{\rm compared}_c = g_c I^{\rm synthetic}_c.
\]

Here `a` is the common absolute scale and the `C-1` values in `eta` are the
relative channel corrections. This has no invisible common-offset direction.
Initialize `a` and `eta` to zero, so every gain is exactly one. Do not use
`tanh`, clipping, per-pixel gains, or per-image gains.

Add Gaussian priors in log space. Initial configurable defaults are 25% for the
common absolute scale and 15% for relative channel scales. These are nuisance
priors for residual calibration uncertainty, not permission to absorb a unit,
degradation, abundance, or geometry error. An informative absolute prior is
essential because a common gain is approximately degenerate with density
through `I ~ n_e^2`; the relative-channel prior is tighter because channel
ratios are the useful temperature constraint. These parameters belong only to
`AIAEUVObservationTerm`, after physical synthesis and before the objective.

Use an asinh-MSE image objective rather than raw-intensity MSE or a plain log
loss. It supports zero and negative prepared pixels while controlling active
region dynamic range. Propagate the count uncertainty into transformed space
and add an explicit configurable model-discrepancy floor. Derive and persist a
fixed robust asinh scale per channel from the prepared training data; it is not
learnable. First reduce valid pixels within each channel, then combine channels
with explicit configured weights; native image sizes and valid-pixel counts must
not silently define relative channel weights. Evaluate the calibration prior
once per joint step, not once per channel or repeated loader batch.

Log separately:

- total and per-channel AIA data loss;
- calibration-prior loss;
- raw and gain-adjusted predicted ranges;
- valid fraction and mask/saturation counts;
- robust bias, scatter, and correlation; and
- common, relative, and physical gains.

Acceptance criteria:

- gains are positive and equal one at initialization;
- the contrast matrix has rank `C-1`, orthonormal columns, and exactly zero-sum
  relative log gains;
- priors have zero value and zero gradient at initialization, and finite
  restoring gradients after a nonzero perturbation;
- negative observed values remain usable;
- masked pixels contribute exactly zero;
- scaling one physical gain changes only its intended channel; and
- a synthetic gain-recovery test recovers known factors without changing the
  density field when the atmosphere is frozen.

## 9. Joint runtime and configuration

Introduce configuration schema version 3 for multimodal runs while continuing
to load existing version-2 HMI/Hinode configurations unchanged. Keep the
current `InversionConfig` class intact, add a separate `JointInversionConfig`,
and make `parse_config()` dispatch explicitly on `schema_version`. Never
reinterpret a version-2 field as a stream field.

The setup-only `JointInversionConfig` has these complete top-level owners:

```text
schema_version: Literal[3]
solver: JointSolverConfig                 # kind + output/work directories
scene: SceneConfig                        # reference stream
atmosphere: AtmosphereConfig              # one shared model and shell
physics: PhysicsConfig                    # one shared physics evaluator
atmosphere_regularization: tuple[AtmosphereRegularizationConfig, ...]
streams: tuple[ObservationStreamConfig, ...]
diagnostics: JointDiagnosticsConfig       # enabled renderers and bounded sampling
dry_run: DryRunConfig                     # device, gradients, quadrature study, report
```

Reuse the existing pure-data `AtmosphereConfig`, `PhysicsConfig`, and applicable
diagnostic sampling nodes directly; do not inherit `InversionConfig` or nest
shared physics under the HMI stream. `JointSolverConfig`,
`AtmosphereRegularizationConfig`, `JointDiagnosticsConfig`, and `DryRunConfig`
are new data-only nodes. Training schedule/runtime/logging fields are
intentionally absent until schema-v3 fitting is enabled.

Schema v3 uses discriminated stream records. Each record keeps one observation,
one forward/objective term, and its weight together instead of relying on
parallel HMI/AIA dictionaries with matching keys. The following YAML focuses on
stream structure and is deliberately a partial excerpt; the committed inversion
configuration must populate every top-level node listed above:

```yaml
schema_version: 3

scene:
  reference_stream: photospheric_stokes

streams:
  - id: photospheric_stokes
    observation:
      type: hmi_stokes
      directory: /glade/work/rjarolim/data/prom3theus/2024_03/hmi/subframes
      # Existing HMI observation/calibration/loader fields remain typed here.
    data_term:
      type: lte_stokes
      weight: 1.0
      synthesis:
        # Existing LTE synthesis fields.
      instrument:
        type: hmi_filter_profiles
      objective:
        # Existing Stokes objective fields.

  - id: coronal_euv
    observation:
      type: aia_euv
      directory: /glade/work/rjarolim/data/prom3theus/2024_03/aia/prepared
      channels_angstrom: [171, 193, 211]
      loader:
        batch_size: 256
        validation_group_id: 20240323T221158388-hmi
    data_term:
      type: aia_optically_thin
      weight: 1.0
      synthesis:
        response_resource: aia_euv_v1:aia_temperature_response
        ray_end: atmosphere_outer_shell
        ray_samples: 192
      objective:
        type: asinh_mse
        model_discrepancy_fraction: 0.25
        channel_weights:
          - {channel_angstrom: 171, weight: 1.0}
          - {channel_angstrom: 193, weight: 1.0}
          - {channel_angstrom: 211, weight: 1.0}
        calibration:
          enabled: true
          absolute_prior_fraction: 0.25
          relative_prior_fraction: 0.15
```

Pairing tolerances, degradation switches, and crop choices are preparation
provenance, not runtime knobs. `ray_end` is fixed to the shared atmosphere shell;
the config never carries a second numerical outer height. Pure configuration
normalization requires unique stable stream IDs matching
`[a-z][a-z0-9_]*` (and therefore safe as `ModuleDict` keys), exactly one existing
`scene.reference_stream`, stable validation-group
IDs rather than list ordinals, and a channel-weight record for exactly every
configured AIA channel. Use the existing sequence/dataclass decoder for weight
records; do not rely on unsupported generic dictionary decoding or numeric YAML
mapping-key coercion. Resource references are namespaced exact IDs, never the
ambiguous word `packaged`. It never imports observation, instrument, solver, or
training implementations.

`build_joint_runtime()` then resolves observation and data-term factories,
validates the resource dependency closure and resource-ID ownership, and rejects
incompatible `observation_kind` or reference-descriptor pairs. All runtime
preflight checks finish before any store is opened or data are synthesized.

Add a plain `src/prom3theus/inversion/joint.py::JointForwardModel` that registers
exactly one atmosphere and all observation terms in a `ModuleDict`, without any
Lightning lifecycle. A separate schema-v3
`src/prom3theus/training/joint_lightning.py::JointInversionModule` owns one such
model, invokes terms through the common protocol, and samples/evaluates the
physics and registered atmosphere-regularization terms exactly once per joint
step. Its generic objective is

\[
L = \sum_s w_s L^{\rm data}_s
  + \sum_s\sum_k L^{\rm nuisance}_{s,k}
  + L_{\rm physics} + \sum_j L^{\rm atmosphere}_{j}.
\]

Terms never sample physics or call Lightning logging. The coordinator requests
each term's nuisance-loss mapping exactly once per joint step, adds those losses
without the stream data weight, and logs namespaced metrics from
`DataTermBatchResult`. Named validation loaders remain separate;
deterministic physics validation runs once per validation epoch rather than once
for every loader's first batch.

The setup-only milestone defines the module and pure evaluation entry points but
does not enable schema-v3 fitting. Before training, add an explicit stream
schedule containing steps per epoch, samples per step, repetition policy, and
loss normalization. Do not let a `CombinedLoader` choice implicitly set these
quantities.

The initial milestone does not modify `p3s_context()`, any P3S parser, or any
artifact reconstruction path. `JointRuntime` writes only `dry_run.json`, which
records both observation descriptors/source signatures, AIA response and
calibration digests, exact channel times, LOS quadrature settings, and initial
calibration parameters. A future joint-fitting phase must design and version a
new artifact contract before saving these fields or learned gains. Existing
`LTEInversionModule`, `run_inversion()`, schema-v2 cache keys, and P3S
reconstruction remain unchanged.

## 10. Stream diagnostics and callbacks

Keep all existing schema-v2 callbacks unchanged. Add lifecycle-independent AIA
evaluation/rendering under `diagnostics/aia_euv.py` and a thin
`training/multimodal_callbacks.py::MultimodalDiagnosticsCallback` for schema-v3
runs. A diagnostics registry maps
an observation/data-term kind to its evaluator and renderer. The callback only
schedules collection, calls named `data_module.stream(id)` accessors, and routes
namespaced `DataTermBatchResult.diagnostics`; it contains no Stokes/AIA tensor
branches. Register both a thin schema-v3 wrapper around the existing
Stokes/atmosphere evaluators and the new AIA evaluator; otherwise the callback
would eventually need a hidden Stokes special case. Plotting code is never
imported by a data term.

The AIA evaluator reconstructs one bounded, deterministic validation exposure
group and the renderer shows, for each channel:

- observed intensity;
- raw synthetic intensity;
- gain-adjusted synthetic intensity;
- asinh, fractional, and uncertainty-standardized residuals;
- observed-versus-predicted scatter and residual distribution;
- valid/saturated masks;
- response-weighted contribution versus height for selected rays; and
- current common, relative, and physical calibration gains.

Also render channel ratios and a Carrington/WCS overlay to expose
misregistration. Because native channel grids and times differ, ratio panels
explicitly reproject copies to a declared diagnostics-only grid; reprojected
values never enter a training objective. Annotate every figure with actual AIA
time, paired reference time, time offset, degradation factor, response digest,
and integration bounds.

The callback runs at fit start, scheduled validation boundaries, and fit end.
Internal evaluation helpers invoke
the exact same evaluator and renderer directly without constructing Lightning
callback state. Follow the existing bounded collector pattern: gather only the
requested validation payload in DDP, render on global rank zero, and never
collect all training batches.

Tests cover complete-group reconstruction, channel/raster index separation,
namespaced routing, bounded DDP payloads, scalar logging, diagnostics-only
reprojection, and deterministic figures from fixed inputs. Also test the legacy
callbacks against schema-v2 modules to prevent the new stream accessors from
changing their assumptions.

## 11. Inversion entry point and internal evaluation

`src/prom3theus/application/runner.py::run_inversion()` remains the schema-v2
compatibility path. Joint assembly and evaluation provide:

- `build_joint_runtime()` — resolve all configured factories, derive and
  validate their exact resource-set dependency union, load typed observation
  streams, assemble the scene, create the one shared
  `JointForwardModel` and physics evaluator, and return a plain `JointRuntime`;
  and
- `dry_run_joint_inversion()` — an internal library helper to evaluate that runtime through pure
  methods and return without constructing a Trainer, callback, optimizer, or
  checkpoint writer.

`JointRuntime` contains no Lightning lifecycle state.
`application.joint_training.run_joint_inversion()` composes it with the stream
scheduler, `JointInversionModule`, diagnostics, and resumable checkpoints.
Joint P3S export remains future work.

Run:

```text
prom3theus invert configs/hmi_aia_dynamic.yaml [--rebuild-observations]
```

CLI dispatch is explicit: `validate-config` and `invert` accept schemas 2 and 3.
`invert` selects the matching training runtime and requires a training section
for schema 3. AIA optimization is controlled by its YAML stream weight: zero
keeps forward rendering and diagnostics while disabling likelihood and
calibration-prior gradients. A union-returning `load_config()` is never passed
blindly to the legacy runner.

The internal evaluation helper performs:

1. strict config and resource validation;
2. independent HMI and AIA cache preparation/loading through the adapter
   lifecycle;
3. joint timeline, calibration-convention, geometry, ray-orientation, and full
   LOS-domain validation;
4. one deterministic synthesis batch through every configured observation
   term, including at least one batch for every AIA channel;
5. separate data/component losses and exactly one calibration-prior evaluation;
6. one deterministic physics-validation evaluation;
7. optional `torch.autograd.grad` smoke checks without applying gradients;
8. 96/192/384-node AIA quadrature comparison;
9. direct invocation of the registered AIA evaluator/renderer; and
10. atomic `dry_run.json` output.

The internal evaluation helper never calls a Lightning `_shared_step()`: that method mixes
random physics sampling, reduction, and logging. It uses the same pure term and
diagnostics APIs used by the joint training runtime.

`dry_run.json` includes resource/store digests, shapes, units, time offsets,
valid fractions, observed/predicted ranges, loss components, gain values,
finite-gradient flags and norms, quadrature errors, device, peak memory, and
timings.

Prove non-training behavior by hashing the complete
`runtime.model.state_dict()`
(parameters and persistent buffers) after setup and before evaluation, then
again afterward. Evaluation tests fail if any hash changes, if a Trainer or
optimizer is instantiated, or if checkpoint/P3S files appear.

The initial milestone is complete only when the real sample dry run shows:

- finite HMI Stokes and AIA intensities/losses;
- nonzero AIA sensitivity to temperature and pressure/density;
- gains exactly one and calibration-prior loss exactly zero;
- acceptable LOS quadrature convergence;
- consistent WCS/time metadata;
- no double degradation correction;
- unchanged parameters/buffers and no training artifacts; and
- unchanged HMI-only regression results.

## 12. Test plan and implementation order

Implement in this order, with each stage required to pass before the next:

1. **Freeze compatibility baselines** — record schema-v2 resource metadata,
   HMI cache signatures/store bytes, state-dict keys, numerical outputs, and P3S
   reconstruction tests before extracting shared internals.
2. **Resource-set foundation** — build-only source cache, CHIANTI emissivity and
   AIA response builders, isolated compact resource set, manifest validation,
   interpolation, continuation, and slab tests. Radiative loss is not on this
   milestone's critical path.
3. **Generic observation seams** — typed prepared streams, adapter codecs,
   private array-store mechanics, image raster/store/data module, dependency-
   scoped signatures, scene assembly, and complete LOS-domain tests.
4. **Acquisition** — independent AIA target selection/export, calibration-bundle
   download, SDO pairing coordinator, fake-client tests, and atomic-failure
   tests.
5. **Preparation and sample fetch** — Level 1 to registered/degraded DN/s,
   uncertainty convention, Carrington-footprint projection, three durable data
   stages, provenance, and the one real sample.
6. **Forward model** — generic optically thin quadrature, AIA response operator,
   full-shell ray composition, units, gradients, and continuation tests.
7. **Data terms** — common result contract, Stokes wrapper for v3, AIA objective,
   identifiable unbounded gains, channel-normalized reductions, and synthetic
   recovery tests.
8. **Joint assembly** — separate schema-v3 config/module/runtime, term registry,
   scene injection, physics once per evaluation, named validation streams, and
   HMI-only compatibility regressions. Do not enable fitting yet.
9. **Diagnostics** — registry, pure AIA evaluator/renderer, thin future callback,
   complete-group reconstruction, scalar metrics, and DDP-safe payload tests.
10. **Internal evaluation** — setup-only report, complete-state immutability proof,
    synthetic execution, and then the real-sample execution.

Expected new or expanded tests include:

```text
tests/instruments/aia_euv/test_download.py
tests/instruments/aia_euv/test_preparation.py
tests/instruments/aia_euv/test_response.py
tests/instruments/aia_euv/test_observation.py
tests/observations/test_streams.py
tests/observations/test_scene.py
tests/observations/test_image_contracts.py
tests/observations/test_image_store.py
tests/observations/test_multistream_data.py
tests/rt/test_optically_thin.py
tests/inversion/test_ray_integral.py
tests/inversion/data_terms/test_stokes.py
tests/inversion/data_terms/test_aia_euv.py
tests/inversion/test_joint_runner.py
tests/config/test_joint_schema.py
tests/diagnostics/test_aia_euv.py
tests/training/test_joint_lightning.py
tests/training/test_multimodal_callbacks.py
tests/test_aia_resource_reproduction.py
```

Also update exact CLI, script, configuration, package-data, architecture, and
resource-inventory whitelist tests. Architecture tests must prove that `rt` has
no application imports, the joint coordinator has no AIA/HMI conditional or
import, and adding the isolated AIA resource set cannot change a schema-v2 HMI
cache signature or artifact reconstruction.

## 13. Next steps after the initial dry run

The setup-only milestone establishes a valid AIA likelihood, but it cannot yet
demonstrate an improved magnetic extrapolation. Before full integration:

1. **Resolve spatial response.** Choose patch batches with PSF halos or a
   documented common rebinning, then forward-convolve predictions rather than
   deconvolving noisy data.
2. **Validate the response scientifically.** Compare complete response curves
   with SSW, test abundance and constant-pressure assumptions, and quantify the
   temperature-dependent model discrepancy that a scalar gain cannot absorb.
3. **Run synthetic recovery.** Recover known coronal temperature, density, and
   channel gains from generated HMI+AIA data before using solar observations.
4. **Choose loss weights from gradients.** Measure per-objective parameter
   gradient norms and activate AIA with a low ramped weight; do not select the
   weight from raw loss magnitudes.
5. **Perform short joint fits and ablations.** Compare HMI-only, HMI+AIA, and
   HMI+AIA with calibration fixed across several seeds. Inspect changes to
   thermodynamics and magnetic metrics separately.
6. **Add a coronal energy equation.** Introduce field-aligned conduction,
   CHIANTI radiative loss, a controlled heating model, and a smooth coronal
   activation. This is the main future physical route for AIA-constrained
   thermodynamics to affect the magnetic solution, beyond the existing
   momentum/induction coupling and shared network parameters.
7. **Model time and volume mismatch.** Test AIA temporal averaging against the
   720-second HMI construction and expand the lateral/top volume or add an
   external-emission nuisance model where LOS contribution diagnostics demand
   it.
8. **Add missing image physics.** EUV absorption by cool material, off-limb
   rays, PSF wings, and correlated interpolation errors.
9. **Expand channels carefully.** Add 335 Å after calibration validation,
   94/131 Å with explicit missing-line/cross-calibration treatment, and 304 Å
   only with a separate non-coronal/optical-depth model.
10. **Assess advanced plasma physics.** Density-dependent 2D response tables,
    non-equilibrium ionization, abundance/FIP uncertainty, and multi-viewpoint
    constraints become warranted only after the simpler operator passes the
    ablations.

The criterion for claiming that AIA improves the magnetic extrapolation is an
out-of-sample improvement in magnetic/loop-consistency metrics across repeated
runs, not merely a lower AIA image loss.

## 14. Primary technical references

- [aiapy: preparing AIA Level 1 data](https://aiapy.readthedocs.io/en/latest/preparing_data.html)
- [aiapy: requesting specific JSOC AIA data](https://aiapy.readthedocs.io/en/latest/generated/gallery/download_specific_data.html)
- [aiapy: AIA channel wavelength response](https://aiapy.readthedocs.io/en/latest/api/aiapy.response.Channel.html)
- [CHIANTI guide: AIA temperature responses](https://www.chiantidatabase.org/cug.html)
- [CHIANTI radiative-loss documentation](https://www.chiantidatabase.org/tech_reports/09_rad_loss/chianti_report_09.pdf)
- [AIA V10 response/calibration release notes](https://hesperia.gsfc.nasa.gov/ssw/sdo/aia/response/V10_release_notes.txt)
- [Boerner et al. 2014: photometric and thermal cross-calibration](https://link.springer.com/article/10.1007/s11207-013-0452-z)
- [SunPy response-function unit convention](https://docs.sunpy.org/projects/sunkit-instruments/en/latest/topic_guide/channel_response.html)
