# MURaM synthesis and evaluation

Synthesis turns native MURaM columns into instrument observables using the
independent LTE reference solver. The inversion uses the production parallel
formal solver. The native atmosphere is retained in its original files; the
synthetic products carry a small reference to those files for later evaluation.

For a separate eager-volume pipeline with inclined Cartesian rays through a spherical
shell, see [ray synthesis](muram-ray-synthesis.md).

## Run synthesis

Hinode/SP uses the packaged two-line configuration:

```bash
PYTHONPATH=src python -m prom3theus.muram.cli synthesize \
  --source-path /path/to/muram/3D \
  --iterations 1000000 2000000 3000000 \
  --out /path/to/hinode_store --output-format store \
  --instrument-config configs/muram/hinode/synthesis.yaml \
  --batch-size 32 --device cuda
```

The configuration specifies Fe I 6301.5008/6302.4932 Å, 112 observed wavelengths
at 21.549 mÅ spacing, and a 25 mÅ Gaussian spectral PSF.
Use `scripts/muram/synthesize_ar11158.pbs` for the AR 11158 inversion store. The [server workflow](../scripts/muram_hinode/README.md) runs the exact LTE inversions on that store. The extended `scripts/muram_hinode/synthesize.sh` adds the AR 11158 cases.

For HMI plus AIA, omit the instrument configuration and store option:

```bash
PYTHONPATH=src python -m prom3theus.muram.cli synthesize \
  --source-path /path/to/muram/3D --iterations 1000000 2000000 \
  --out /path/to/sdo --batch-size 32 --device cuda
```

The default output format is `sdo-fits`. The default batch size is 256;
`--device auto` selects a visible CUDA GPU, otherwise CPU. Float32 is the default
computation dtype; `--dtype float64` selects reference precision. Absolute
wavelength calibration stays float64 in either case.

Stokes synthesis defaults to the lower **3 megameters**, measured upward from the bottom
native cell center. Set `--stokes-height-megameters 4` to change that cutoff. It retains
native cell centers at or below the cutoff, without vertical resampling; a cutoff
above the box includes its full height. Python callers can also pass
`stokes_height_megameters=None` for full height. For Stokes-only runs, source memory-map views are sliced to this height and
horizontal stride before copying into RAM. Diagnostics use the same cropped
volume; crossings above it are reported as missing. Combined HMI+AIA runs retain
the full height for AIA and diagnostics, while Stokes transfer uses the cutoff.

### Logging and progress

The synthesis and diagnostic commands write UTC-timestamped stage logs to stderr,
including snapshot settings, saving times, diagnostic validity counts, and failures.
Use `--log-level DEBUG` for additional details or `--log-level WARNING` to reduce
stage output. Library callers can configure standard Python logging themselves.

Interactive terminals show snapshot, column, and FITS-file progress bars with
elapsed time, throughput, and estimated remaining time. Column progress advances
after each batch is processed. Redirected scheduler logs use plain text instead,
with column updates at most every 30 seconds. `--no-progress` disables bars and
periodic updates, including AIA preparation, while retaining stage logs.

Each column pass reports loading/waiting and processing times. These measure time
observed by the main process; cube loading is timed separately and asynchronous GPU work can overlap,
so they are not separate disk or GPU profiling measurements.

## One snapshot, one eager volume

Iterations run in an ordinary sequential `for` loop. Each iteration follows the
same steps:

1. **Describe the source.** Read its header, resolve axes/components, and determine
   upward orientation from the two tau boundary planes. Record filenames, file
   stats, units, orientation, and pixel mapping. No full-cube content hash is made.
2. **Load the volume.** Slice read-only source mappings to the requested height and XY stride for
   Stokes-only runs, then copy and convert each selected field into memory once. The snapshot owns one shared
   cache of these arrays. Flatten XY logically as `x * ny + y` and iterate over
   column batches in a simple sequential loop. No DataLoader or worker processes
   are used. Combined HMI+AIA runs retain full-height fields for coronal emission.
3. **Render the batch.** Assemble native samples in top-to-bottom order and solve
   polarized LTE transfer on their geometric heights. Apply the requested
   instrument response. For SDO, compute HMI and then AIA from the same loaded
   batch before advancing. The same batch also supplies the native tau=1
   diagnostic accumulator, before instrument or observer-frame transformations.
4. **Assemble the maps.** Place profiles/intensities at their original XY indices,
   expose output maps in YX order, and apply optional spatial averaging.
5. **Publish and close.** Recheck source file stats, write the instrument products
   and provenance, save the diagnostic maps and overview, then release the full cubes and mappings before the
   next iteration. SDO prepares its AIA training store after the sequence.

Selected cubes are loaded before the batch loop. Stokes-only runs load the
configured lower height range for both spectra and diagnostics; combined HMI+AIA
runs retain full-height fields for AIA and diagnostics. The default Stokes grid uses
native heights. The AR 11158 launcher interpolates the column fields at the heights
of the inversion's 32 uniform ray-distance nodes. Stokes uses
native temperature, pressure, magnetic field, and velocity; internal tau is not
needed for its transfer. Native tau is loaded for the diagnostic surface and,
in AIA, for its visibility proxy.
The orientation boundary-plane reads are the only extra atmosphere reads before
the full-field reads. Memory-map access and the OS cache determine physical disk I/O;
loading each field once is not a promise that each disk page is fetched only once.

`--batch-size` controls in-memory column and transfer batches, not cube loading.
The selected converted fields remain resident for the snapshot: approximately
`4 × selected_Nx × selected_Ny × selected_Nz` bytes per float32 field (nine fields with Stokes diagnostics).
Output maps, geometry, transfer intermediates and OS file-cache pages consume
additional memory. Pre-instrument Stokes cubes are not
retained by default. Python callers can request `keep_raw_stokes=True` for a
specific diagnostic; the CLI does not save those cubes.

## Sampling and physical conventions

- With neither `--stride` nor `--binning`, every native horizontal pixel is kept.
  `--stride N` selects columns `[::N, ::N]`; `--binning N` averages N×N synthesized
  pixels. Both lose spatial information. Selected dimensions must be divisible
  by the binning factor. Averaging atmospheres before transfer is not equivalent.
- `streams[].observation.binning` separately averages saved pixels when preparing
  inversion inputs. Set it to 1 for saved resolution. Synthesis and loading
  binning are recorded separately; changing loading resolution requires rebuilding
  prepared observations and a compatible new training run.
- The finest-spaced storage axis is vertical unless `--vertical-axis` overrides
  it. `--component-order` maps primitive vector components to header axes,
  independently of storage transposes. Reflections preserve handedness. Equal,
  invalid, or negative tau boundary means cannot establish orientation and fail.
- Native pressure converts to Pa, velocity/length to SI, and magnetic field to
  gauss using the MURaM conversion factors. The primitive inputs must contain
  velocities rather than momentum densities. No unresolved microturbulence is added.
- Each ray follows one Cartesian column. The spherical scene embedding records
  surface locations and the Stokes basis for inversion; it does not introduce
  lateral interpolation or full 3D transport. Co-rotating flows receive the
  projected Carrington rotation for the stationary synthetic observer.
- Default Stokes transfer evaluates the retained N native cell centers over N−1 geometric intervals,
  with a Planck lower boundary at the bottom center. There is no bottom optical-depth
  check. LTE, the packaged EOS/opacities, and this boundary are model assumptions.
  The requested cutoff, retained sample count, and actual height range are saved
  alongside the full native depth count. The cutoff is relative to the box bottom,
  not the tau=1 surface; its adequacy depends on the snapshot's height registration.
- With `--inversion-depth-grid`, Stokes uses the selected window's fixed sample count
  and shell boundaries. The AR 11158 presets span −1 to +1 Mm about a nominal
  photosphere 7.5 Mm above the box bottom. Transfer starts at the outer shell
  and uses the local Planck source at the lower shell. The observation metadata
  records the registration and both native height endpoints.
- Default HMI uses six idealized 76 mÅ Gaussian filters. Hinode uses the configured
  spectral PSF. No detector noise, extra spatial PSF, or temporal averaging is added.
- AIA integrates `ne(T,P)^2 K(T) dz` over native cell widths, using the equilibrium
  EOS and response. The SI-to-CGS emission-measure factor is `1e-10`. Finite,
  nonnegative native tau is required; `tau <= 1` is an approximate photospheric
  visibility mask, not EUV absorption or non-equilibrium ionization.

Native height zero is the **bottom native cell center**, not the photosphere.
The reference deliberately leaves `photosphere_height_m` unset. Atmospheric
comparisons need an explicit vertical registration to the inversion domain.
The spherical geometry, finite native box, and inversion boundary conditions
are separate assumptions. The HMI depth audit is documented in
[hmi-ray-sampling-audit.md](hmi-ray-sampling-audit.md).

## Saved products

| Mode | Products | Evaluation information retained |
| --- | --- | --- |
| Hinode or HMI `store` | ObservationStore manifest and canonical NPY arrays | Detector Stokes, float64 wavelengths, geometry/basis, masks, instrument/line settings, response arrays where required, native reference and generator provenance |
| Default SDO HMI | 24 Stokes/filter FITS segments plus response NPZ per snapshot | Physical spectral radiance, time/WCS, exact response quadrature, parallel-ray geometry, synthesis contract/native reference in I0 |
| Default SDO AIA | 171/193/211 Å FITS, then `aia/prepared` image store | Reference-response DN/s/pixel, geometry/calibration and native-reference metadata |
| AIA-only `store` | One `muram_<iteration>.npz` per snapshot | Intensity, channels, spacing/time, response identity, native reference and generator provenance in `metadata_json`; not an inversion-ready AIA store |

Native references contain source identity, simulation time, resolved axis/sign mappings, physical
units, geometric origin, and exact stride/binning pixel mapping. They do not
copy the atmosphere. Standard AIA FITS require the standard packaged response;
custom responses remain available in the AIA-only NPZ path.

### Native diagnostic maps and overviews

Every synthesized snapshot also writes two independent products:

- `diagnostics/muram_<iteration>.npz`: float32 maps of `bx/by/bz`, `vx/vy/vz`,
  `temperature`, `pressure`, and `height_m`, plus `valid_mask`, `status`, and
  `metadata_json`. Maps use YX order at the selected column resolution **before
  output binning**. Metadata record units, the native source reference, vector
  frame, surface definition, stride, and synthesis binning.
- `diagnostics/muram_<iteration>.jpg`: a nine-panel overview rendered from that
  NPZ, with display-only downsampling to at most 768 samples per axis. Numeric
  maps are never downsampled for the preview. Magnetic and velocity components
  use shared color scales within each vector.

The accumulator finds the native tau=1 surface once per loaded column, accepting
one strict crossing or one exact native sample. It interpolates all quantities
between the neighboring samples, linearly in log(tau); a bracket containing a
zero endpoint uses linear tau. An exact sample needs no interpolation. Multiple
crossings or multiple exact samples (including a tau=1 plateau) are ambiguous.
No crossing is extrapolated and no transfer depths are changed.

`status` distinguishes valid (0), missing crossing (1), ambiguous crossing (2),
invalid tau (3), and invalid surface values (4). Invalid maps are NaN; invalid
tau takes precedence over crossing flags. These flags describe diagnostics, not
the instrument validity mask. Vectors retain local west/north/upward axes and
co-rotating velocities, before instrumental azimuth or observer corrections.
The height is measured from the bottom native cell center; this surface does
not define a universal spectral-line formation height or register the inversion.

The maps require about 61 MiB plus masks for 1728×1024 selected pixels. Hinode/HMI
loads one additional field (tau) in the same pass; combined SDO already loads it.
AIA-only synthesis additionally loads the six vector fields for these maps.
Python callers can use `diagnostics=False` to omit all map accumulation.

`diagnostics/manifest.json` tracks map/image completion separately from instrument
products. A plotting failure leaves the numeric maps and completed observations
intact. Normal synthesis resume repairs missing diagnostics before skipping a
completed snapshot. A missing JPEG reads only its saved maps; missing maps need
one diagnostic-only native pass, never LTE/AIA synthesis.

To backfill or repair an existing run independently of its generator contract:

```bash
PYTHONPATH=src python -m prom3theus.muram.cli diagnostics \
  --out /path/to/hinode_store --iterations 1000000 2000000 --batch-size 32
```

Omit `--iterations` to process all discoverable snapshots. Saved-map rendering
does not require the original cube files. Map backfill requires the original
native reference and verifies source identity before and after the pass. The
command does not change the instrument products or their synthesis contract.
Overviews require Matplotlib (available with the `visualization` extra).

HMI FITS store physical spectral radiance in W m^-3 sr^-1. The standard HMI
training loader divides it by the packaged atlas continuum. The internal LTE
scale of `1e13` is only a numerical unit. Synthetic AIA is already in reference
count-rate units, so preparation uses unit exposure and unity degradation.
The FITS epoch is 2010-03-24 UTC plus simulation seconds. Embedded parallel-ray
geometry is authoritative for training; the WCS also supports visualization.

Uniform response tensors stay broadcast in memory. Canonical stores still write
dense response arrays for loader compatibility; writes use bounded chunks to
avoid a full contiguous heap copy. FITS writing reads result metadata directly
without constructing a discarded observation raster.

## Resume and source integrity

Existing outputs resume automatically. Completed snapshots are skipped;
an interrupted snapshot restarts from its first batch. `synthesis_run.json`
records the scientific contract and generator code/resource provenance;
`synthesis_status.json` tracks progress and completed products. Source references
are also embedded in saved products so they can be recovered after an interrupted
status update.

Batch size and device may change on resume. Changed synthesis settings, generator
provenance, or native source identity require a new output or explicit
`--overwrite`. That flag replaces the output directory. Configuration and cheap
source checks run before replacement. Source arrays are always opened read-only;
conversion operates on copied columns. No source write, rename, delete, or content
hash is part of synthesis. File identity uses path, size, mtime, and ctime; it
catches ordinary edits/replacements but is not a cryptographic content checksum.
Changing `--stokes-height-megameters` also requires a new output or `--overwrite`.

## Evaluation and export

Two comparisons answer different questions:

1. **Observable recovery:** compare predicted and saved instrument Stokes at the
   same wavelengths, calibration, geometry, mask, and loading resolution. Raw
   high-resolution synthesis spectra are unnecessary for this comparison.
2. **Atmosphere recovery:** reopen the referenced native snapshot, match its
   horizontal samples and physical units/vector basis/velocity frame, and establish
   the vertical registration. Keep all contributing native columns for binned
   observations; a mean native atmosphere does not reproduce their mean spectrum.

Export an inversion save state with its Stokes comparison:

```bash
PYTHONPATH=src python -m prom3theus.muram.cli export \
  /path/to/state.p3s /path/to/evaluation.npz \
  --include-stokes --stokes-batch-size 16 --device cuda
```

The NPZ contains evaluated atmosphere arrays and metadata. `--include-stokes`
adds `observed_stokes`, `predicted_stokes`, `stokes_residual`, and float64
`wavelength_angstrom`. Geometry and instrument responses are read in bounded
Stokes batches. Final exported arrays remain resident while assembling the NPZ.
`--include-full-shell` additionally exports radial columns across the configured
shell; it does not make that shell observationally constrained.

For the selected Stokes raster, `metadata_json.observation.muram_synthesis`
retains the native reference and synthesis sampling, while
`metadata_json.observation.spatial_binning` retains loading reduction. Reopen
that native source lazily when an atmospheric comparison needs it:

```python
from prom3theus.muram.reference import open_native_reference

snapshot = open_native_reference(reference)
try:
    columns = snapshot.synthesis_columns()
    # Select the recorded pixels and evaluate native columns as needed.
finally:
    snapshot.close()
```

An explicitly relocated source can be passed as `source_path=...`; filenames,
sizes, and modification times must be preserved. The native files remain an
external dependency. The descriptor enables reproducible comparisons; it does
not silently choose an optical-depth surface or perform a field-recovery score.

See [the synthesis review](muram-synthesis-review.md) for implemented corrections
and verification coverage.
