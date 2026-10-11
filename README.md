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

## Observer geometry

Map-based viewing angles and vector projections use the finite-distance ray
from each surface point to the observer. The same radial/LOS dot product gives
mu for ME synthesis, limb corrections, and visibility masks. The image basis
uses helioprojective tangent directions and image roll; spacecraft Doppler
projection uses the same ray with the existing FITS velocity sign conventions.
Spherical training and validation take mu directly from the vector projection.
Evaluation derives it too and rejects a conflicting explicitly supplied mu.
Geometry caches from earlier versions are rebuilt automatically. Synthetic
spherical data and map-based evaluation use the same projection convention.

## Training data cache

Spherical inversions configure worker counts independently under `data`:

```yaml
data:
  num_workers: 4          # Training and validation DataLoader workers
  num_dataset_workers: 16 # Initial training dataset loading and preparation
```

`num_workers` defaults to 4; set it to `null` to use the CPU count.
When omitted or `null`, `num_dataset_workers` uses the resolved `num_workers`
value. `num_workers: 0` loads batches in the main process; in that case, set
`num_dataset_workers` explicitly to a positive value for initial preparation.

Spherical training stores only float32 Stokes profiles and an integer original
pixel index per retained sample. A frame's WCS, observer position and velocity,
time, image dimensions, and HMI detector mapping are stored once as metadata.
Batch loading reconstructs surface coordinates, mu, vector transforms,
spacecraft Doppler projection, and pixel/detector coordinates. HMI transmission
profiles are still sampled per batch. No per-profile continuum normalization is
stored, so changing the loss does not change the observed profiles in the cache.

Ordered, unfiltered validation stores only Stokes: pixel indices follow from
batch positions, including invalid image pixels. Six-wavelength HMI training
uses 100 bytes per sample before compression (previously 132); validation uses
96. Previous caches rebuild automatically. Overview images are released after
logging, and each training chunk is read only once when assembling a batch.

## Stokes objective

Spherical `model.b_sinh_scale` is specified in gauss and `model.v_sinh_scale`
in km/s. For example, `b_sinh_scale: 10` and `v_sinh_scale: 1` give componentwise
outputs of `10 * sinh(latent_B)` G and `1 * sinh(latent_v)` km/s. The training
module converts these scales to model units using `gauss_per_dB`, `Rs_per_ds`,
and `seconds_per_dt`. Omit a scale or set it to `null` for linear outputs.

Spherical inversions use an elementwise mean-squared objective on the
loader-scaled Stokes profiles after the configured linear-I/asinh-QUV transform.
For example:

```yaml
normalization:
  asinh_alphas: {Q: 5.0e-2, U: 5.0e-2, V: 5.0e-2}
```

The normalization setting is stored in checkpoints, and resuming with a
different transform is rejected. Use a fresh `base_path` when changing it.

For linear scaling instead of asinh, set `normalization.asinh_alphas: null`
and `normalization.stokes_scales: {I: 1.0, Q: 0.01, U: 0.01, V: 0.01}`.
Both predictions and targets are divided by these fixed component scales
after per-profile continuum normalization. With MSE and unit lambda weights,
these scales give effective weights of 1 for I and 10000 for Q/U/V.
Fixed scaling and asinh cannot be enabled together.

With `stokes_loss.continuum_normalization: true`, each predicted and observed
Stokes vector is divided by its own per-pixel 90th-percentile I across wavelengths.
Synthetic normalization happens after instrument integration and configured
corrections, with gradients through the synthetic denominator. Training,
validation, and exported observation synthesis use the same continuum operation;
fixed component scaling/asinh is used only for the loss and validation plots.
Set `stokes_loss.continuum_quantile` to change the percentile (default `0.9`).

Continuum-normalized spherical fits automatically use
`model.source_function: ratio`: one sigmoid parameter a gives b0=1-a and b1=a. Thus b0+b1=1 fixes the
unobservable overall source amplitude at disk center, while the physical
continuum b0+mu*b1 retains its viewing-angle dependence before normalization.
Non-normalized fits retain two independent source coefficients. This does not
force the source ratio or other atmospheric parameters to be spatially uniform.
Use a fresh `base_path` for this objective; older target-only or common-observed-
denominator checkpoints cannot be resumed.

Exported profiles are continuum normalized. Matching an observed absolute scale
requires its per-pixel continuum estimate in the desired units; the instrument
scalar alone is insufficient, so `denormalize_stokes=True` is rejected.

`train.I/Q/U/V` and `train.stokes_loss` report the selected wavelength-summed
objective. The existing `valid.I/Q/U/V` values remain wavelength-averaged MAE
diagnostics for continuity with older runs; `valid.objective` reports the exact
weighted validation counterpart of `train.stokes_loss`.

## Optimizer selection

Spherical training defaults to SOAP. To use native PyTorch Muon (PyTorch >= 2.9)
for hidden weight matrices and AdamW for input/output layers, biases, and
instrument corrections:

```yaml
training:
  lr_params: {start: 1.0e-3, end: 1.0e-4, iterations: 200000}
  optimizer_config: {type: muon, momentum: 0.95, ns_steps: 5, weight_decay: 0.0}
```

Muon uses `adjust_lr_fn=match_rms_adamw`, Nesterov momentum, and AdamW uses
`betas=[0.9, 0.95]`, `eps=1e-8`. Both groups follow the same fractional learning
rate decay. Set `optimizer_config.muon_lr` to tune Muon's starting rate
independently; `lr_params.start` remains AdamW's rate. These are starting values
for an experiment, not a calibrated replacement for SOAP's learning rate.
The two algorithms share one Lightning optimizer for automatic optimization,
gradient handling, scheduling, and checkpoint resume.

Use `optimizer_config: {type: soap}` to select SOAP. Switching optimizers requires
a fresh `base_path`; their checkpoint states are incompatible. The current
90-second HMI configuration selects SOAP. The decay interval does not limit
training steps; set `training.max_steps` for a finite step budget.

## Shared line-profile evaluation

ME synthesis obtains Voigt absorption and Faraday–Voigt dispersion from one
Faddeeva evaluation per Zeeman component. Existing checkpoint buffer names and
the separate profile APIs are retained. Outputs and gradients are tested against
the previous separate-evaluation path. Fixed Zeeman transitions are cached outside
training; physical profiles use nonnegative damping and unit width to skip
reflection and broadcast per-profile arrays. Stokes synthesis shares propagation
coefficients and source terms across I/Q/U/V. Legacy checkpoint keys are preserved.

To benchmark the profile pair's forward and backward computation on the cluster:

```bash
python -m pme.evaluation.benchmark_profiles --device cuda --batch-size 32768
```

The benchmark includes the specialized `physical_unit` path used by synthesis.
Use `--wavelengths` for the number of wavelength samples actually evaluated
before instrument integration. Add `--compile` to test targeted compilation;
this does not enable compilation in training. The local PyTorch 2.12 CPU test
found approximately 1.9x faster profile-pair evaluation, but Inductor compilation
failed on complex arithmetic. Overall GPU training speedup remains to be measured.

## Compiling the real ME solver

The 90-second full-disk configuration enables a limited compilation experiment:

```yaml
training:
  compile_config: {me_solver: true, mode: default}
```

Only the real-valued propagation coefficients and analytic I/Q/U/V solution are
compiled, using Inductor with `fullgraph=True`. Faddeeva uses the previous complex
implementation and remains eager, as do response integration, the MLP, replay,
logging, and the optimizer. The first training call compiles the solver and its
backward computation; subsequent compatible calls reuse the process-local cache.
Errors are surfaced instead of silently falling back. Changed shapes can trigger
additional compilation.

Omit `compile_config` or set `me_solver: false` to disable it. Validation and
exported-model evaluation remain eager. Compiled callables are never serialized;
checkpoint parameter names and optimizer states are unchanged, allowing resume
with compilation enabled or disabled.

Benchmark outputs, gradients, cold-call time, warm forward/backward time and
CUDA peak memory before comparing full training throughput:

```bash
python -m pme.evaluation.benchmark_me_solver --device cuda --batch-size 32768 --wavelengths 54
```

Set `--wavelengths` to the actual total number of samples evaluated before
instrument integration. This benchmark covers the real solver and complete
monochromatic synthesis, not the network, instrument integration, or optimizer.
`--mode reduce-overhead` is an optional CUDA-graph experiment that can use more
memory. For recompilation diagnostics, prefix the command with
`TORCH_LOGS="graph_breaks,recompiles"`. The initial CPU test at 512 x 54 samples
reduced solver forward/backward time from 1.77 to 0.99 ms, and complete synthesis
from 16.1 to 14.9 ms; GPU training throughput remains to be measured.

The compiler regression tests use graph capture/compiled autograd without native
code generation by default. Set `PME_TEST_INDUCTOR=1` to also test generated
kernels with a working compiler toolchain.

## Training metric logging

Training losses and lambda values are averaged on the GPU over 50 batches by
default, then reduced across DDP ranks in one packed collective. Set
`training.metric_log_interval` to change the window (use 1 for per-batch logging).
Learning rate and time-shift scale report their current values at each flush.
Short windows flush at epoch/run end or when a physics schedule changes the
metric set. Accumulators are detached, transient, and reset on checkpoint resume.

## Numerical failure reports

With norm clipping enabled (as in the spherical inversion), non-finite gradient
norms raise before clipping or an optimizer update can contaminate weights and
optimizer state. The existing non-finite loss check also writes a detailed
failure-only report. Per-rank JSON files are saved under
`<checkpoint_directory>/numerical_failures/` (or the trainer root if no checkpoint
directory is configured), with the global step and forward/backward stage.
Reports contain ranges and NaN/Inf counts for model parameters and gradients;
forward failures also include input data, decoded atmosphere parameters,
predicted/observed Stokes and their continuum estimates, and loss components.
Reports perform no distributed collectives and retain no training graphs.
They locate the first detected failure but do not establish its numerical cause
or silently mask invalid samples/gradients. A non-finite global norm can also
result from overflow while individual gradients are still finite.

## Temporary GPU/data-loading diagnostics

Add `--profile` to the spherical inversion command to capture one short window
(10 skipped training batches, 2 warm-up batches, 20 recorded batches). Normal
training continues afterward. Reports appear immediately under
`<base_path>/profiling/`, with separate files per rank:

- `*.txt` and `*.summary.json`: exposed data-loader wait, host-to-device
  transfer, forward/backward and optimizer timing, plus expensive operators.
- `*.trace.json`: CPU/GPU timeline, viewable in Perfetto or Chrome tracing.
- `gpu-activity-*.csv`: GPU utilization and memory sampled every second by
  `nvidia-smi`, when available; one monitor per node, stopped after 40 batches.

Large `train_dataloader_next` waits that coincide with empty GPU lanes indicate
the input pipeline is starving the GPU. Long transfers instead point to the
CPU-to-GPU path. Near-zero data waits with continuous kernels suggest compute is
the bottleneck. Compare ranks for imbalance. Timing scopes are nested (the
optimizer scope includes its forward/backward closure), so do not add them.
The timeline includes NCCL work; high utilization alone does not imply useful
model computation or high hardware occupancy. Worker preprocessing is not
directly profiled; data wait measures the portion not hidden by prefetching.

Unset `CUDA_LAUNCH_BLOCKING` before measuring performance. The profiler adds
overhead during capture, so use it for attribution rather than absolute speed.
Omitting `--profile` disables all diagnostics. All diagnostic code lives in
`pme/train/diagnostics.py`; removal only requires deleting that module, its
tests, and the small argument/Trainer wiring in `pme/inversion_spherical.py`.

## Stokes replay sampling

Replay is currently disabled in the 90-second HMI configuration to measure
training throughput with entirely fresh batches. To enable the optional buffer:

```yaml
training:
  replay_config: {fraction: 0.25, max_replays: 5}
```

After one entirely fresh batch, each instrument batch contains 25% replayed
profiles and 75% fresh profiles, keeping the total batch size unchanged. The
buffer retains profiles with the largest mean squared residual after the usual
continuum normalization and Stokes scaling, allocating slots across acquisition
times. Scores are recomputed each visit; profiles leave as their errors fall
below other candidates or after five replay visits. Small batches or insufficient candidates use
additional fresh profiles. I/Q/U/V lambda weights remain unchanged; replay
deliberately increases the effective weight of difficult profiles without
importance correction.

Only detached inputs are retained, on the training device, up to one replay
fraction of the latest batch per instrument and DDP rank. Predictions and
gradients are recomputed normally. Physics collocation and validation sampling
are unaffected. The buffer is transient and refills after checkpoint resume;
its configuration is recorded in checkpoint metadata. `train.replay_fraction`
reports the actual fraction used. Omit `replay_config` to disable replay.
Reusing the entire retained pool skips sorting and indexing it again; smaller
incoming batches still select a time-balanced subset. Fresh selection and hard
candidate retention remain time balanced.

Fresh data still use the existing full-batch loader; replay replaces a subset
before model evaluation. This first implementation reallocates model updates
but does not reduce disk reads or preprocessing.

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
