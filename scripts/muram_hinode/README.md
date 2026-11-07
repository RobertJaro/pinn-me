# Single-frame MURaM Hinode/SP inversion

Use the already unpacked snapshot `000000` in
`/glade/derecho/scratch/rjarolim/muram_snapshot`, the same frame used by the
single-frame HMI MHS experiment. Run from the repository root on Derecho:

```bash
bash scripts/muram_hinode/synthesize.sh
bash scripts/muram_hinode/check_setup.sh
qsub scripts/muram_hinode/run.sh
```

For an existing interactive GPU allocation, the training command is:

```bash
PYTHONPATH=src python -m prom3theus.muram.cli invert configs/muram/hinode/mhs.yaml
```

The single-frame synthesis selects iteration 0 and saves native-resolution profiles. It
produces both Fe I 6301.5008 and 6302.4932 lines on 112 wavelength samples
(21.549 mÅ spacing), with a 25 mÅ Gaussian spectral PSF. No noise, spatial PSF,
or AIA products are added. Instrument settings and absolute calibration are
stored with the spectra; no real-Hinode FITS preparation is required.
The synthesis settings are in `configs/muram/hinode/synthesis.yaml`.
Native fields are memory-mapped lazily by a dataset of flattened horizontal
pixels. A DataLoader with two CPU workers loads full-depth columns in batches;
`--batch-size` controls both loading and the sequential synthesis loop.
Each batch is read once for synthesis, without overview images, ground-truth maps,
or continuum optical-depth checks. Synthesis uses every native vertical cell
and the physical cell spacing, without log-tau interpolation or top truncation.
All columns and depths are preserved, and outputs are assembled back into 2D maps.
Output spectra, prefetched batches, and OS file-cache pages also use memory.
The inversion presets apply `observation.binning: 4` during loading, before
profile normalization. Set it to `1` for native-resolution training or when
reusing older stores already synthesized with 4×4 binning. Loading never changes
the source store. Changing binning requires fresh work/output directories or an
explicit observation rebuild; do not resume a different-resolution checkpoint.

The output store is
`/glade/derecho/scratch/rjarolim/muram_synthetic_observations/hinode_single_frame`.
Existing outputs are resumed: completed snapshots are skipped and an interrupted
snapshot is restarted. Use `bash scripts/muram_hinode/synthesize.sh --overwrite`
to replace both output directories and regenerate all requested snapshots.
Batch size and device may change on resume; changes to recorded synthesis
settings require a new output directory or `--overwrite`.

The setup command validates the configuration, loads the generated store,
checks the forward contract and gradients, and writes `hinode_setup.json` to
the configured output directory. It performs no optimizer steps. Run it on
a compute node with the inversion dependencies and sufficient memory.

The inversion is static, with one Stokes stream and a −0.5 to 30 megameter domain.
The magnetic field is the curl of a learned vector potential; the explicit
magnetic-divergence, gauge, and potential-boundary penalties are disabled.
`atmosphere.parameters.magnetic_field.scale_vector_potential: 0.01` multiplies
the vector-potential readout before taking its curl, reducing A and B by 100
relative to the same network with the default multiplier of 1.0. This factor
applies throughout training; `scale_gauss` and the model units remain unchanged.
Use a fresh run when changing this decoder scale.
Stokes MSE (weight 1), MHS force balance (weight 0.001), and the upper-boundary
gas-pressure prior (weight 0.0001) are active at full weight from step zero,
with no Stokes-only warmup or physics ramp. The Stokes objective normalizes
by each observed profile's 90th-percentile Stokes I and divides I/Q/U/V by
1/0.1/0.1/0.1 before squaring residuals. Adaptive normalization is detached,
with 1 G magnetic and 1 Pa force-balance floors. These settings are a starting
experiment, not a claim of convergence. Check Stokes recovery before interpreting the
upper atmosphere. There is no UTC selection window: the store contains only
the requested frame and retains its native simulation time.

The PBS script uses the repository checkout at
`/glade/u/home/rjarolim/projects/lte` and the existing `sunerf` environment.
Output and work directories use the distinct name
`muram_hinode_mhs_isotropic_soap_v04`. An existing `last.ckpt` in
the output directory is automatically resumed; choose fresh output and work
directories for a new experiment.

The MHS preset trials 32 coarse depth points without fine refinement and 128
rays per GPU. Check convergence against a denser grid after fitting; this
budget is not a demonstrated fix for mixed magnetic polarities. In the v02
step-5000 snapshot, 16 versus 256 points changed Stokes V by 2.79% of observed
continuum RMS on 16 pixels sampled across the strongest-polarization quartile.
The 128-versus-256 difference was 0.029%; 256 versus 512 was 0.0075%.
At 32 points the Stokes V difference from 256 points was 0.627% of continuum RMS.
These are fixed-snapshot quadrature checks, not recovered-field accuracy.

The ray and line-formation tops are 2.5 Mm; the MHS domain extends to 30 Mm.
The transition-region top is 3.5 Mm, retaining its previous 1 Mm separation;
the configuration requires it to be strictly above the line-formation top. This also
changes the reference-atmosphere transition profile for the new run. A separate
fixed-snapshot test with matched dense depth spacing found that extending the
ray top to 2.5 Mm changed V by 1.56% of continuum RMS. Extending to 3 Mm barely
changed V further, but intensity still changed. Thus the old 1.5 Mm ray top is
not boundary-converged for that fitted atmosphere. The added layers were not
constrained by its Stokes fit, so this sensitivity alone does not validate the
upper atmosphere or establish a remedy for the mixed polarities.

That snapshot used first SIREN frequency 30 and vector-potential multiplier 1,
so it does not test the current frequency 5 / multiplier 0.01 configuration.
Its sampled continuum tau500=1 surfaces lie 223–435 km above geometric height
zero, and Br can reverse with height. Interpret zero-height maps alongside
optical-depth surfaces and the Stokes fit. Analytic bipolar-field controls
verify Cartesian curl, coordinate rotation, Lorentz-force sign, and the MHS
parameter gradient; no equation-sign correction was indicated. The depth
change requires a new fit and does not itself repair the v02 magnetic field.

## Stokes-only control

`configs/muram/hinode/stokes_only.yaml` uses a SIREN atmosphere
network followed by differentiable formal LTE Stokes integration. SOAP's learning
rate decays from 1e-3 to 1e-4 over 10000 steps; MHD constraints are disabled.
The objective uses each profile's own 90th-percentile Stokes I and linear
I/Q/U/V scales of 1/0.1/0.1/0.1. Transfer uses 32 jittered coarse points plus 32 contribution-guided fine points,
64-wavelength chunks, float32 arithmetic, and the local bottom Planck source.
Check depth convergence and bottom optical thickness for the chosen atmosphere.
Outputs and work files use `muram_hinode_isotropic_soap_v01`, separate from
previous SIREN/Adam runs. Existing synthetic observations can be reused.

Update the checkout on the server, then submit:

```bash
cd /glade/u/home/rjarolim/projects/lte
python -m pip install -e .
qsub scripts/muram_hinode/run_stokes_only.sh
```

SOAP is copied from SPINNME (`src/spinnme/train/soap.py`), with betas
(0.95, 0.95), zero weight decay,
and an eigenbasis refresh every 10 updates. The initial step initializes its
preconditioners without changing weights. Scalars and vectors use unpreconditioned
Adam-style updates. No extra forward/backward evaluations are performed.
Preconditioners and their decompositions stay in float32; a failed decomposition
raises instead of retrying in float64. SOAP adds matrix operations and optimizer
state; GPU speed and convergence relative to Adam must be measured on the server.
Do not resume Adam/Muon checkpoints with SOAP; start with a fresh output directory.

The PBS job requests four GPUs and `training.device: cuda`, `training.devices: auto`
launch native Lightning DDP automatically. No `torchrun` command is needed.
Each process binds to its local GPU before constructing the atmosphere, data
loaders, and LTE synthesis windows. Only global rank zero constructs the data module and
prepares missing training pools. It atomically publishes `work_directory/data_module.pt`;
other ranks wait, then reload the same published generation from shared storage.
Restoration bypasses data-provider constructors and reuses the prepared array
files. Startup messages distinguish construction, waiting, and reloading.
With `training.steps_per_epoch: 1000`, each epoch has 1000 optimizer updates,
independent of dataset size and GPU count. Each rank draws fresh observation
pixels uniformly with replacement from the full prepared dataset; repeats within
and between batches are allowed. Random sequences differ by rank and resume from
the committed update. Multi-channel streams retain their channel quotas.
Observation batch size is **per GPU**, with averaged gradients and a common
learning rate. Validation runs at every epoch end; `validation_every_n_steps`
applies only when `steps_per_epoch` is unset. Final validation also runs when
training stops mid-epoch.
Validation, W&B images, and checkpoint files are published by rank zero.

The Stokes-only preset uses 1024 rays per GPU, giving 4096 rays per optimizer
update on the four-GPU job. Each ray has 64 sorted depth points and 63 integration
intervals, with epsilon separation between samples. The coarse coefficients are
reused; only the new fine points require a second atmosphere evaluation. Formal transfer retains a depth and wavelength graph for every ray.
Measure memory and throughput
with `scripts/diagnostics/benchmark_lte_inversion.py` before increasing batches;
wavelength chunks limit individual operations, not the whole backward graph.
The Hinode Stokes-only preset uses a 1000 G magnetic output scale to improve magnetic
updates. Direct-field initialization compensates for this scale, retaining the
same weak initial field. This setting improved all four Stokes residuals in the
bounded local synthetic test; it is an optimization scale, not a field prior.

```bash
qsub scripts/muram_hinode/run_stokes_only.sh
# Or within an interactive allocation with four visible GPUs:
PYTHONPATH=src python -m prom3theus.muram.cli invert configs/muram/hinode/stokes_only.yaml
```

Startup logs identify every rank and its device. Request fewer GPUs by changing
the PBS allocation; automatic selection follows the visible devices. A checkpoint resumed with a
different device count retains its weights/optimizer state and restarts data
coverage under the new rank partition. `max_steps: -1` and `max_epochs: -1` keep
training indefinitely; reaching the final learning rate does not stop training.
