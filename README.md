# pinn-me
pin the inversion problem down with the Milne Eddington approximation

## HMI transmission preprocessing

HMI phase-map data are downloaded and converted to a compact transmission
profile file before an inversion is started.  A registered JSOC email address
is required by the DRMS export service:

```bash
python -m pme.data.hmi_transmission \
  --input '/path/to/hmi_stokes/*.fits' \
  --email name@example.com \
  --output /path/to/hmi_transmission/
```

For every FITS acquisition, the command follows `INVPHMAP` from the matching
`hmi.ME_720s_fd10` record and uses the FITS `HCAMID`. It downloads each unique
calibration only once, stores its full 128x128 field of six profiles, and writes
`manifest.json` mapping acquisitions to calibrations. Set
`data.*.transmission_profile_directory` to this directory. During loading,
every HMI full-disk image or cutout is mapped back onto the detector and assigned
the matching calibration. Dataset workers interpolate that spatial calibration
for the pixels in the current batch, so only batch-local spectral offsets and
weights are transferred to the GPU. The inversion module never opens calibration
files or stores full response maps. The common -0.65 to +0.65 Angstrom line region
is synthesized explicitly for every filter. All remaining modeled passband
throughput is integrated offline to seven blocking-filter standard deviations
with chunked composite quadrature and added as an unpolarized continuum
contribution, following the VFISV optimization.
The phase-map product does not contain the measured fixed-stack/front-window
curve, so these files remain reconstructed responses: they use the published
mean element contrasts, 8.43 Angstrom blocking-filter width, and +16 mAngstrom
untuned-stack center. These assumptions are stored in each profile archive.
Training itself performs no network access. Regenerate the directory with
`--overwrite` whenever the transmission format changes.

## Stokes objective

Spherical inversions use an elementwise mean-squared objective on the
loader-scaled Stokes profiles after the configured linear-I/asinh-QUV transform.
For example:

```yaml
normalization:
  asinh_alphas: {Q: 5.0e-2, U: 5.0e-2, V: 5.0e-2}
```

The normalization setting is stored in checkpoints, and resuming with a
different transform is rejected. Use a fresh `base_path` when changing it.

`train.I/Q/U/V` and `train.stokes_loss` report the selected wavelength-summed
objective. The existing `valid.I/Q/U/V` values remain wavelength-averaged MAE
diagnostics for continuity with older runs; `valid.objective` reports the exact
weighted validation counterpart of `train.stokes_loss`.

## Continuous physics regularization

Physics constraints use a collocation stream that is independent of the Stokes
pixel batches. The sampler draws a fixed global Sobol point set continuously in
the full training time, latitude, and seam-safe longitude bounds, then partitions
that set across distributed ranks. Consequently `num_points` is the total number
of physics points per optimizer step, not a per-GPU count. The same seed and
global step reproduce the same global point set after checkpoint resume.

Constraint schedules live under `physics.constraints`; the top-level `lambda`
mapping is reserved for the four Stokes components. Legacy mixed mappings are
still read for older experiments, but new runs should use the separated form:

```yaml
physics:
  num_points: 4096
  seed: 17
  domain: full_train
  radius_range_Rs: [1.0, 1.0]
  normalization: raw
  constraints:
    divergence: {type: linear, start: 0.0, end: 1.0e-6, iterations: 100000}

lambda:
  I: 1.0
  Q: 1.0
  U: 1.0
  V: 1.0
```

The Stokes and physics terms are evaluated separately and added only at the
final objective. Constraint derivatives are constructed lazily, so a
divergence-only run does not build velocity, induction, current-density, or
second-derivative graphs. Physics configuration and the resolved global domain
are fingerprinted in checkpoints; changing either requires a fresh `base_path`.
The ready-to-run controlled experiment is
`config/hmi/hmi_subframe_20240323_physics.yaml`.

## Spherical profile evaluation

Spherical inversion artifacts retain their forward models, learned calibration
corrections, per-acquisition HMI response mapping, and per-instrument Stokes
normalization. To reproduce an HMI observation with the same geometry and
spectral response used in training:

```python
from pme.evaluation.loader import SPINNMEOutput

output = SPINNMEOutput("/path/to/inversion.pme", instrument_id="HMI")
result = output.load_hmi_observation(
    "/path/to/hmi.s_720s.YYYYMMDD_HHMMSS_TAI.N.I0.fits",
    batch_size=8192,
    denormalize_stokes=True,
)
stokes = result["stokes"]  # [y, x, I/Q/U/V, wavelength]
```

The response file is selected from the acquisition mapping saved with the
inversion. Pass `spectral_response_file=` explicitly if the calibration archive
was moved. `synthesize_observation` provides the lower-level interface for
precomputed coordinates, observer transforms, and sampled response arrays.
