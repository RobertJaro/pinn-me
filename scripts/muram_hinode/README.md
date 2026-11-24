# Single-frame MURaM Hinode/SP inversion

For one snapshot of the X-flare AR 11158 simulation, submit the following
from the repository root on Derecho with the project Python environment active.

```bash
qsub -v PATH scripts/muram/synthesize_ar11158.pbs
```

This job selects iteration 2000000 from the ASD_2023 campaign volume and writes
`/glade/derecho/scratch/rjarolim/muram_synthetic_observations/hinode_ar11158_2000000_uniform32`.
It reads the depth grid from the MHS inversion window. Both inversions and the
synthesis use 32 equally spaced ray distances including the −1 and +1 Mm shell
boundaries. The nominal photosphere is set to 7.5 Mm above the bottom native
cell, so these shell boundaries sample the native columns at 6.5 and 8.5 Mm.
Both inversion presets read this store at native resolution. [Simulation setup](https://arxiv.org/html/2606.25382v1)

After the synthesis finishes, submit the Stokes-only inversion from the same checkout:

```bash
qsub scripts/muram_hinode/run_stokes_only.sh
```

For MHS, use `scripts/muram_hinode/run.sh` in the last command. Both presets use exact LTE coefficients and the polarized formal solver.

For an existing interactive GPU allocation, the training command is:

```bash
PYTHONPATH=src python -m prom3theus.muram.cli invert configs/muram/hinode/mhs.yaml
```

The single-frame synthesis selects iteration 2000000 and saves native-resolution profiles. It
produces both Fe I 6301.5008 and 6302.4932 lines on 112 wavelength samples
(21.549 mÅ spacing), with a 25 mÅ Gaussian spectral PSF. No noise, spatial PSF,
or AIA products are added. Instrument settings and absolute calibration are
stored with the spectra; no real-Hinode FITS preparation is required.
The synthesis settings are in `configs/muram/hinode/synthesis.yaml`.
The renderer linearly interpolates the native column fields at the registered
radial heights. At disk center the ray spacing is 64.5 km. Toward the edges,
the shell geometry sets the ray length just as in the inversion.
For the 64 km vertical mesh, the loader retains the lowest 134 native levels
through 8.512 Mm to bracket the last interpolation point. Native tau=1
diagnostics use these loaded cells. Transfer uses only the 32 sampled points.
The diagnostic maps and overview are saved under `diagnostics/`.
The inversion presets configure `observation.binning` during loading, before
profile normalization. Set it to `1` for native-resolution training or when
reusing older stores already synthesized with 4×4 binning. Loading never changes
the source store. Changing binning requires fresh work/output directories or an
explicit observation rebuild; do not resume a different-resolution checkpoint.
The synthesis contract also stores the native source/frame reference needed for
later atmospheric evaluation. See [the synthesis and evaluation guide](../../docs/muram-synthesis.md)
for the complete pipeline, output contents, and native-height registration.

Existing outputs are resumed: completed snapshots are skipped and an interrupted
snapshot is restarted. The script does not forward command-line arguments.
To replace an output directory, add `--overwrite` to its synthesis command.
Batch size and device may change on resume; changes to recorded synthesis
settings require a new output directory or `--overwrite`.

## In-domain MHS inversion

`configs/muram/hinode/mhs.yaml` uses the Stokes-only atmosphere, instrument,
and training settings. The network predicts the magnetic field directly.
The LTE and physics domains both span −1 to +1 Mm. The MHS ray tracer uses
a 0.2 Mm lateral halo. Each optimizer step samples 2,048 new points uniformly in
physical volume within the observed angular bounds. Both the MHS force
residual and magnetic divergence use those points with weights 0.01 and 0.1, respectively.
Validation uses 384 fixed volume points.

The preset uses 1,024 Stokes rays per GPU and 32 fixed, equally spaced depth
samples per ray. Its output and work directories use
`muram_hinode_ar11158_2000000_mhs_exact_lte_soap_uniform32`.

### Earlier extrapolation experiments

In the v02
step-5000 snapshot, 16 versus 256 points changed Stokes V by 2.79% of observed
continuum RMS on 16 pixels sampled across the strongest-polarization quartile.
The 128-versus-256 difference was 0.029%; 256 versus 512 was 0.0075%.
At 32 points the Stokes V difference from 256 points was 0.627% of continuum RMS.
These are fixed-snapshot quadrature checks, not recovered-field accuracy.

A separate
fixed-snapshot test with matched dense depth spacing found that extending the
ray top to 2.5 megameters changed V by 1.56% of continuum RMS. Extending to 3 megameters barely
changed V further, but intensity still changed. Thus the old 1.5 megameters ray top is
not boundary-converged for that fitted atmosphere. The added layers were not
constrained by its Stokes fit, so this sensitivity alone does not validate the
upper atmosphere or establish a remedy for the mixed polarities.

That snapshot used first SIREN frequency 30 and vector-potential multiplier 1.
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
I/Q/U/V scales of 1/0.1/0.1/0.1. Transfer uses 32 uniformly spaced points,
float32 arithmetic, and the local bottom Planck source across the full wavelength grid.
Check depth convergence and bottom optical thickness for the chosen atmosphere.
Outputs and work files use `muram_hinode_ar11158_2000000_stokes_exact_lte_soap_uniform32`. Both inversions use the new uniform-grid store.

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
update on the four-GPU job. Each ray has 32 sorted depth points and 31 integration
intervals, with epsilon separation between samples. The coarse coefficients are
reused; only the new fine points require a second atmosphere evaluation. Formal transfer retains a depth and wavelength graph for every ray.
Use `python -m prom3theus.diagnostics.commands.benchmark_lte_inversion` to measure memory and
throughput. Reduce the per-GPU ray batch size if the full spectrum exceeds device memory.
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
