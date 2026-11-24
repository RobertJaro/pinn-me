# Cartesian MURaM ray synthesis

This independent Stokes pipeline loads the lower part of a Cartesian atmosphere and traces
straight observing rays through it. The original column pipeline remains available.
This pipeline currently synthesizes HMI or Hinode Stokes; AIA is not part of this entry point.

```bash
PYTHONPATH=src python -m prom3theus.muram.synthesize_rays \
  --source-path /path/to/3D --out /path/to/ray_stokes \
  --instrument-config configs/muram/hinode/synthesis.yaml \
  --iterations 1000000 2000000 --bottom-height-megameters 0 \
  --volume-height-megameters 3 --depth-samples 128 --batch-size 32 --device cuda
```

The Hinode launcher is `scripts/muram/synthesize_rays.sh`. Edit its paths, iterations and height registration before use.

## Physical geometry and sampling

1. Load temperature, pressure, magnetic field and velocity as owned float32 arrays
   once per snapshot, retaining only native cell centers within the lowest
   **3 megameters above the bottom cell center**. `--volume-height-megameters`
   controls this limit. Crop the read-only source views before allocating or copying
   fields, so the upper atmosphere is not materialized in RAM. Python callers can
   pass `volume_height_megameters=None` to `EagerVolume.from_snapshot` for full height. Native source mappings remain read-only. `--stride N`
   subsamples both horizontal axes; `--vertical-stride N` subsamples depth.
   Retained samples keep their original physical coordinates, including both
   endpoints of the cropped box. Strides lose atmospheric structure; interpolation does not recover it.
2. Rigidly place the box at `--longitude-deg` / `--latitude-deg` (default disk
   center). `--bottom-height-megameters` is required: it places the bottom native cell
   center relative to the nominal solar radius. Zero is a chosen registration,
   not an inferred photosphere. Cartesian lengths and vector components retain
   their physical meaning; the volume is never warped onto a spherical grid.
3. Construct an orthographic image covering the projected box. Pixel spacing
   defaults to native horizontal spacing times stride; `--pixel-size-km DX DY`
   overrides it. `--observer-direction X Y Z` points from the Sun toward the
   observer (default `1 0 0`). The complete box must lie on the observer-facing
   solar hemisphere. Rays stay straight in Cartesian space.
4. Intersect rays with a spherical shell and the finite box. By default the shell
   encloses the loaded, height-cropped box; `--shell-height-megameters MIN MAX` selects another radial
   range relative to the same nominal radius. Shell limits select rays within the
   already loaded box; they do not restore atmosphere above the volume cutoff. Interpolation support is the closed hull of
   native cell centers; no half-cell extrapolation is added.
5. Place 128 nominal samples uniformly in physical ray distance between the
   near-side shell intersections. A ray missing the inner sphere terminates at
   its tangent plane. Clip exterior samples to the exact box entry/exit points:
   these repeated endpoints have zero-width intervals. Thus the useful number
   of distinct samples can be smaller than 128. No exterior segment contributes.
6. Trilinearly interpolate in the native Cartesian grid. Rotate vectors to global
   coordinates and then the fixed observer Stokes basis. Add the Carrington
   rotation velocity at each sample and apply the instrument's azimuth convention.
7. Perform polarized LTE transfer using actual ray-distance intervals. Rays
   reaching the Cartesian bottom receive its unpolarized Planck inflow; rays
   entering from a side, top or shell boundary receive zero incoming radiation.
   Missed pixels are zero with a false validity mask. Side-boundary spectra are
   identified separately because vacuum inflow differs from a photospheric boundary.

No optical-depth audit, contribution-based refinement or automatic photosphere
registration is performed. Use `--depth-samples 256` and `512` to assess convergence
for the actual snapshot and viewing geometry. Uniform 128-point sampling is a
starting resolution, especially when a deep shell contains thin contributing layers.

## Execution, memory and progress

Each snapshot runs to completion in a sequential loop. The dataset contains only
flattened image-pixel indices; two spawned DataLoader workers prepare batches
without receiving the atmosphere or CUDA context. The owning process interpolates
and transfers batch-sized atmosphere data to the device. `--batch-size` controls
ray batches and `--workers` controls loader workers (default 4).

The eight resident float32 cubes require about `32 * Nx * Ny * Nz` bytes after
height cropping and striding, plus interpolation, geometry and transfer workspace. Loading logs report
the estimate. Output Stokes arrays are written directly into NPY memory maps;
there is no full-spectrum heap copy or compressed NPZ assembly. OS file-cache
pages can still contribute to resident memory.

UTC stage logs and terminal progress bars cover volume loading, rays and snapshots.
Redirected output uses periodic plain-text ray progress. `--no-progress` disables
bars and periodic updates, while `--log-level` controls stage logging.

## Products and evaluation

Each completed snapshot is published under `muram_<iteration>/`:

- `stokes.npy`: full spectra `[image_y, image_x, IQUV, wavelength]`, physical
  spectral radiance in W m^-3 sr^-1, with `wavelength_angstrom.npy` (air wavelengths).
- `instrument_stokes.npy` and `observed_wavelength_angstrom.npy`: instrument
  response applied to those spectra. Response weights are saved when applicable.
- `valid_mask.npy`, `boundary_type.npy`, `path_length_m.npy`, and
  `distinct_depth_samples.npy`: intersection and transfer-boundary diagnostics.
- `ray_origin_local_m.npy`, `ray_direction_local.npy`, `image_x_m.npy`,
  `image_y_m.npy`: exact output-pixel/ray placement in physical coordinates.
- `detector_basis.npy`: physical observer axes. `synthesis_stokes_basis.npy`:
  axes after the recorded instrument azimuth convention; both saved Stokes arrays
  use this latter basis. Metadata records the rotation and velocity convention.
- `metadata.json`: native source reference, retained native indices and axes,
  box transform, requested volume-height limit, actual loaded height range, shell
  limits, sampling/interpolation choices, instrument settings,
  units and generator provenance. These allow matched-geometry evaluation.

Read the arrays without loading complete spectra:

```python
from prom3theus.muram.synthesize_rays import load_ray_product

arrays, metadata = load_ray_product('/path/to/ray_stokes/muram_1000000')
pixel_stokes = arrays['stokes'][20, 30]
valid = arrays['valid_mask'][20, 30]
```

These are independent numeric products, not an ObservationStore or SDO FITS bundle.
For comparison to another synthesis, match the saved ray geometry, spectral
response, radiometric scale, polarization basis and boundary mask. Original
native files are needed only to evaluate the atmosphere again.

## Resume and source integrity

The output directory has a separate `ray_synthesis_run.json` contract and
`ray_synthesis_status.json`. Completed products with matching native references
are skipped without loading the cubes. Missing array files trigger resynthesis of
that snapshot. Interrupted snapshots restart from their first ray batch.

Source identity is verified with file metadata before and after synthesis, without
hashing complete cubes. Changing sampling, placement, instrument settings, source
identity or generator code requires a new output or `--overwrite`. Batch size,
worker count and device can change on resume. Explicit overwrite validates source
headers and geometry before replacing the output. Output paths must be separate
from the native source directory.
