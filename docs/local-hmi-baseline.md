# Local HMI inversion and force-free baseline

`scripts/hmi/run_local.py` uses `data/hmi` and defaults to
`configs/hmi_local_constant.yaml`. It selects one 256 × 256 bipolar region of
the March 23, 2024 22:12 TAI acquisition, centered at Carrington longitude
213.2877°, latitude −12.9481°. The existing input is already a cutout; further
cropping accumulates its detector offsets so the HMI response still samples
the correct CCD location.

The first stage is a small, static LTE inversion of all four Stokes components,
with 32-pixel batches on CPU. The default allows 4,000 steps and uses a 128-wide, three-layer
SIREN (34,697 parameters) with first frequency 90 and a 3 Mm height input scale.
This increases horizontal detail while preserving the initial vertical
frequencies of frequency 30 with a 1 Mm height scale. Physical heights and ray
distances remain unchanged. It uses the packaged LTE resources and the actual acquisition-specific
HMI filter response. Logging and online services are disabled during the run.
The standard checkpoint permits resuming with a larger total step count.

The minimal model assumes a magnetic vector constant along each radial column.
`atmosphere.parameters.magnetic_field.reference_height_megameter: 0.15` reads
only the three magnetic network channels at that fixed height, retaining the
same angular coordinates and time. Temperature, pressure, velocity and
microturbulence remain stratified. All Stokes and Cartesian field queries use
this same magnetic readout; the extrapolation therefore uses the field fitted
to the polarized profiles. Oblique rays can still traverse horizontal magnetic
structure. Omitting this option preserves unrestricted height dependence.
The height-independent field is an explicit modeling assumption, not a measured
height profile. It is not constrained to be force-free in the photosphere.

The second stage solves an exterior **spherical potential field**, the α=0
case of a force-free field, from inferred photospheric Br. It uses the existing
spherical Neumann Green solver, with 48 × 48 source cells, 32 × 32 horizontal
output samples, and layers at
0.5, 1, 2, 5, 10 and 20 Mm. Source-cell quadrature evaluates the inverted field
exactly at r=1 using 4 × 4 Gauss quadrature per cell. The surface-limit solution
is saved separately.

This is a two-stage baseline. The LTE network covers −0.1 to 1.5 Mm; the derived
potential model supplies the coronal magnetic field. Tangential magnetic field
need not agree across that interface. There is no nonlinear force-free current,
coronal thermodynamic inference or full-shell neural solution in this baseline.
The source spans the observation's longitude/latitude bounding rectangle; the
inversion supplies predictions in its small unobserved corners. Outside that
rectangle, Br is assumed zero and net flux is retained.
Potential fields are the current-free limit of force-free extrapolation;
see [Zhu, Wiegelmann & Inhester (2020)](https://www.aanda.org/articles/aa/full_html/2020/12/aa39079-20/aa39079-20.html).

## Commands

Use Python ≥3.11 with the project, observation and visualization dependencies.
In this checkout the `nf2` Conda environment has them installed.

```bash
PYTHONPATH=src python scripts/hmi/run_local.py --prepare-only
```

This prepares the region and checks calibration, writing `preparation.json`
without replacing an existing inversion report. It exits with code 2 if the
local response is missing or invalid. It never fetches a replacement or
substitutes an idealized filter. Preparing an official response is an explicit
network operation, performed separately with a registered JSOC email:

```bash
PYTHONPATH=src python -m prom3theus.cli.main prepare hmi-responses \
  data/hmi/*.fits --output data/hmi_responses_local --email YOUR_JSOC_EMAIL
PYTHONPATH=src python scripts/hmi/run_local.py
# Continue the same inversion to a larger total step count if necessary:
PYTHONPATH=src python scripts/hmi/run_local.py --steps 6000
```

The recorded height-independent refit starts from the completed unrestricted
inversion and transfers network weights only:

```bash
PYTHONPATH=src python scripts/hmi/run_local.py \
  --config configs/hmi_local_constant.yaml --steps 1000 \
  --initialize-from-state runs/hmi_local_fine/state.p3s
```

This requires a new output directory without a checkpoint. It verifies matching
network, atmosphere, observations and coordinate normalization, except for the
explicit magnetic height restriction. It starts a fresh optimizer and records
the source P3S checksum and step. A subsequent continuation omits
`--initialize-from-state` and uses the saved checkpoint. Ordinary checkpoint
resumption rejects a change in the magnetic height restriction.

Relative configuration paths resolve from the YAML file. The script locates the
repository from its own path. If a sandbox makes the usual SunPy/Matplotlib
configuration directories unwritable, set `SUNPY_CONFIGDIR` and `MPLCONFIGDIR`
to writable temporary directories before invoking Python.

## Results and acceptance checks

Products go to the chosen configuration's output directory
(`runs/hmi_local_constant/` for the default configuration):

- `report.json`: preparation state and, after optimization, fit and corona checks.
- `setup.json`: initial Stokes loss and finite-gradient checks.
- `initial_reference.npz`: original diagnostic predictions, exact observations,
  pixel indices and initialization provenance, retained across continuations.
- `state.p3s`, `last.ckpt`: trained photospheric model and resumable checkpoint.
- `stokes_comparison.npz`: initial prediction, final prediction, observations and
  pixel indices for the same diagnostic sample.
- `representative_stokes_comparison.npz`: the same quantities for 1,024 pixels
  on an offset 32 × 32 lattice covering the crop, without overlap with the main
  256-pixel lattice.
- `stokes-fit-maps.png`, `stokes-fit-profiles.png`: observed and fitted I/Q/U/V
  with residual maps and representative spectra selected from the observations.
- `corona/coronal-potential.npz`: source cells and Br, surface field, physical
  Cartesian positions and coronal vectors, plus traced field lines.
- `corona/coronal-validation.json` and `corona/coronal-potential.png`: coronal
  numerical checks, amplitudes by height and magnetic connectivity.
- `magnetic_height/magnetic-height-consistency.json` and its sample NPZ: actual
  saved-model queries verifying constant magnetic vectors through the photosphere
  while temperature remains stratified.

The fit must halve its initial weighted loss (the restricted warm-start prediction
for the recorded refit; the seeded unfitted model for an ordinary run) and fit I to 0.05 atlas-continuum
RMS. Each polarized component must fit within the larger of 0.002 continuum
RMS and 75% of its observed RMS; the validation sample must contain measurable
V. Thus perfect intensity with collapsed polarization does not pass. These
explicit baseline limits are not instrument uncertainty estimates. Both the
main and representative samples must pass. All diagnostic pixels can also
appear in training; these are not held-out tests.
The comparison score always uses I weight 10,000 and Q/U/V weights 390,625,
so it remains comparable across optimization stages. The final training
configuration uses I weight 20,000 to prioritize the remaining brightness
error; Q/U/V training weights remain 390,625.
Continuations retain the original predictions and verify that observations and
sample indices match exactly; they do not reset the comparison to an unfitted
network. `setup.json` describes the newly assembled diagnostic runtime, while
the fit report uses the preserved initial reference.

The coronal report checks recovered photospheric Br, nonzero source and top
field, Cartesian finite-difference divergence/curl/Lorentz residuals, and field
lines reaching above 2.5 Mm. It records the source flux imbalance and the local
exterior assumption. A failed fit or coronal check produces exit code 1 and
`failed_validation`; reaching a training step count alone does not establish
success. The checks establish numerical consistency, not independent agreement
with the Sun's coronal magnetic field.
For the height-independent configuration, the saved model must also reproduce
the same magnetic vector at 0, 0.1, 0.15, 0.3 and 1 Mm on 256 angular columns,
within 0.001 G plus a relative tolerance of one part per million.

On September 12, 2026 the official response was downloaded successfully to
`data/hmi_responses_local/`. The definitive acquisition assigns phase-map FSN
230562565, camera 2, from calibration record 2023-10-10 19:26:26 TAI. The response
manifest preserves that assignment and checksums; the earlier calibration date
is expected provenance, not an acquisition mismatch.

The initial 64-wide configuration (`hmi_local_minimal.yaml`) completed 2,000
steps but failed its fit checks: I RMS was 0.0846, Q 0.00709, U 0.00682 and
V 0.01365. Its potential corona passed the numerical checks; the combined run
correctly remains `failed_validation`. `hmi_local_resolved.yaml` increases
width to 128 and first frequency to 30; at 2,000 steps its Q/U/V fit passes,
but I RMS 0.0644 still fails. The higher-frequency configuration addresses the
remaining spatial brightness smoothing. At 4,000 steps it passes both fit
samples and the coronal numerical checks: representative I/Q/U/V RMS is
0.04946 / 0.00432 / 0.00396 / 0.00712 continuum units. A separate height audit,
however, found substantial changes in Br through the line-forming layer. The
height-independent refit removes that unconstrained vertical magnetic freedom
from this minimal baseline.

The recorded higher-frequency run first completed 2,000 steps and was then
resumed with a 4,000-step limit. Its first stage is retained under
`runs/hmi_local_fine/stage_2000/`; I RMS was 0.0521, while Q/U/V passed.
At the saved 3,000-step checkpoint, I RMS was still 0.0503. The final stage
resumes that checkpoint with I training weight increased from 10,000 to
20,000. The original 3,000-step checkpoint and metrics are retained under
`runs/hmi_local_fine/stage_3000/`. These checkpoints preserve the exact starts
of the continuation stages. A fresh run with the final configuration has a
different optimization history from this recorded staged run.

## Accepted local run

The height-independent model completed a 1,000-step refit from the archived
4,000-step unrestricted model on September 12, 2026, with a fresh optimizer
(learning rate 0.0005 to 0.0001). The final result is
`runs/hmi_local_constant/report.json`, status **passed**. This recorded
warm-start trajectory differs from a fresh default 4,000-step run.

| Diagnostic sample | I RMS | Q RMS | U RMS | V RMS | Weighted error, initial → final |
|---|---:|---:|---:|---:|---:|
| 256 primary pixels | 0.04577 | 0.00529 | 0.00502 | 0.00766 | 152.50 → 64.64 |
| 1,024 offset pixels | 0.04705 | 0.00525 | 0.00466 | 0.00753 | 150.34 → 63.54 |

RMS values use atlas-continuum units. These aggregate baseline checks permit
residual errors at individual pixels, especially in fine-scale polarized
profiles; they do not establish a unique vector inversion or coronal truth.

The saved magnetic vector is identical at all five tested photospheric heights,
while temperature remains stratified. The derived corona reaches 20 Mm with
RMS field 75.62 G. At a 0.01 Mm difference step, the
maximum dimensionless divergence/curl/Lorentz residuals are
4.38e-05 / 2.61e-05 / 2.09e-05, using a fixed 1 Mm normalization.
The inferred local source has flux imbalance 17.7%;
its exterior assumption remains explicit in the report.

Final P3S SHA256:
`cdac01a8ae4ec0619edabcbe5d0e57e02eba5033e34e03c816745737bc807095`.

The independent audit in `runs/hmi_local_constant/audit/` reloads the final P3S
and reproduces all saved primary/representative Stokes predictions exactly.
Source Br and sampled exported coronal vectors also reproduce exactly.
Refining source cells from 48 × 48 to 64 × 64 changes coronal vectors by
1.38% at 2 Mm, 0.38% at 5 Mm and 0.069% at 20 Mm; increasing source-cell
quadrature from order 4 to 6 changes them by at most 0.00023%.
