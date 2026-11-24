# Synthesis review and corrections

Three independent reviewers examined physical correctness, computational
efficiency, and code structure. This records the implemented outcome; the
[synthesis guide](muram-synthesis.md) describes normal operation.

## Physical correctness

The review checked native units and vector/axis orientation, geometric-height
transfer, lower-boundary treatment, velocity and polarization conventions,
spatial averaging, AIA emissivity, and persistence of evaluation coordinates.

Corrections:

- Ambiguous or invalid tau boundary means now fail instead of choosing an
  arbitrary upward direction.
- AIA rejects nonfinite/negative tau in its loaded visibility-mask batches.
- Standard AIA FITS reject nonstandard response resources on both write and
  load, preventing incorrect calibration labels. AIA NPZ retains the actual
  response-resource identity.
- Every native snapshot carries a source reference with resolved axes/signs,
  physical units, height origin, time, and stride/binning pixel mapping.
- Source stats are verified before publishing new results and before reusing
  completed ones. External source changes during synthesis prevent publication.

The adopted model remains independent vertical LTE columns with full native
sampling and a Planck lower boundary. Spherical embedding supplies inversion
geometry, not 3D transport. AIA uses equilibrium optically thin emission with a
native tau visibility proxy. These assumptions are explicit rather than being
changed by the refactor. Native height zero is the bottom sample center;
atmospheric evaluation must establish its photospheric registration separately.

## Computational efficiency

The review found unnecessary whole-map allocations outside the source loader.

Corrections:

- A single shared column pass requests the union of fields needed by the chosen
  instruments. SDO renders HMI then AIA from each already loaded batch.
- Pre-instrument Stokes retention is opt-in (`keep_raw_stokes=True`). Normal CLI
  runs keep detector profiles, avoiding the unused full high-resolution cube.
- HMI FITS obtains metadata directly without creating an observation raster and
  its spatial response arrays merely to discard them.
- Uniform instrument responses clone their small base once and remain broadcast
  in memory. Validation checks that base without expanding the entire map.
- Canonical NPY writing handles broadcast/transposed arrays in bounded chunks,
  preserving C-order storage without a full contiguous tensor copy.
- Stokes evaluation reads geometry, velocity corrections, and response inputs
  inside its batch loop. Exported absolute wavelength labels retain float64.

Two CPU workers and one prefetched batch per worker are retained. Native files
are mapped read-only; source identity uses file stats instead of content hashes.
Output maps and geometry still occupy memory. Dense canonical response files
still occupy disk space for compatibility. These changes remove identifiable
allocations; no production GPU throughput improvement is claimed without a
benchmark on the intended machine and snapshot.

## Code structure and persistence

Corrections:

- Removed the obsolete optical-depth resampling implementation.
- Replaced callback coupling with one explicit synthesis loop and small roles:
  shared grid/selection, Stokes renderer, AIA accumulator, result packaging.
- Kept public HMI, Hinode/general Stokes, AIA, and combined SDO entry points.
  Snapshot iterations remain sequential in an ordinary `for` loop.
- Validate instrument options, responses, execution settings, and cheap source
  metadata before replacing existing outputs. The direct Python SDO path also
  checks its device and dtype. A null optional Hinode response is accepted.
- Resume contracts now include generator source/resource provenance. Outputs
  made by different generators cannot be silently mixed.
- Missing native references are recovered from durable product metadata when
  possible; completed outputs without one are rejected rather than appended to
  without evaluation provenance.
- Selected-raster exports retain native synthesis metadata and distinguish
  loading-time spatial binning from synthesis sampling.

The canonical products contain what observable evaluation needs: detector
profiles/intensities, wavelengths, calibration/response, geometry, validity,
and provenance. Atmospheric evaluation reopens the native source through
`open_native_reference`; a derived map alone cannot describe the full atmosphere.
This is a reproducible reference to external native data, not a second copy of
those data or an automatic atmospheric-registration algorithm.

Following the review, optional Python diagnostic accumulation (enabled by default
and always used by the CLI) was added to the same source pass. Native tau=1 maps
and their JPEG overviews have a separate manifest and repair command; they do
not determine whether instrument synthesis is complete. See the synthesis guide
for surface selection, validity flags, pre-binning map resolution, and recovery.

## Verification

Regression coverage includes eager-versus-lazy numerical parity, column ordering
and one shared HMI/AIA pass, worker loading, native-source immutability, full
native depth, unit/orientation conventions, optional raw spectra, bounded
response allocations/writes, FITS/store/NPZ metadata round trips, float64 export
wavelengths, invalid-overwrite preservation, interrupted publication/resume,
source changes during synthesis, and generator-contract changes.

The final suite covers `tests/muram`, `tests/artifacts`, `tests/observations`,
and `tests/cli/test_scripts.py`: **329 passed, 3 skipped** at the original review (2026-11-07).
The three skipped tests require a CUDA GPU. `git diff --check` also passed.

Diagnostic integration subsequently verified 350 distinct test cases across the
full regression run and a final 45-test SDO/diagnostics rerun after correcting
the in-memory snapshot publication path. Three CUDA tests remained skipped.
Coverage includes missing-JPEG repair without synthesis, diagnostic-only map
backfill, source-pass ordering, surface edge cases, unbinned map placement, and
preservation of completed products after a later snapshot fails. The nine-panel
overview was also visually checked using synthetic test maps.
