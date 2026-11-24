# Training a pointwise LTE coefficient mapper on MURaM

Train one mapper per spectral window. Start with Hinode, where one evaluation per sampled atmospheric point predicts the combined propagation coefficients across Fe I 6301.5 Å, Fe I 6302.5 Å, their wings and the surrounding continuum. Train a separate HMI mapper for the Fe I 6173.3352 Å window. The two mappers share training code and have independent weights, spectral bases and checkpoints. The inversion still computes the thermal source from temperature, integrates polarized transfer along each ray, and applies the Hinode response after integration. Judge the model by its effect on emergent spectra, inversion gradients, and GPU step time.

## Physical contract

| Quantity | Contract |
| --- | --- |
| Inputs per point | Temperature in K, mass density in kg/m³, microturbulence and line-of-sight velocity in m/s, and three magnetic components in G in the right-handed ray basis. Use the existing sign convention `v_LOS = −dot(v, photon_direction)`. |
| Output per point and wavelength | Seven propagation coefficients in inverse metres, ordered as eta I, Q, U, V followed by rho Q, U, V. Include continuum extinction in eta I. |
| Wavelengths | Vacuum Å in the transfer solver. Construct detector and PSF grids from the Hinode preset in air Å and convert once. Preserve precision using offsets from a fixed origin. Accept shared and per-ray wavelength queries on the supported interval, including observer shifts and PSF wings. |
| Emission and integration | Assemble the existing signed 4 by 4 propagation matrix K. Use the analytic Planck source S and the same fixed radiance scale as the LTE calculation. Local emissivity is K times S. Preserve absolute continuum intensity in training and validation. |

The exact LTE teacher uses temperature and gas pressure from each sampled or perturbed state. Its equation of state supplies the matching mass density. The mapper takes natural logs of physical temperature and density and standardizes the five encoded rest-frame features using the training calibration batch. The checkpoint stores the centers and scales. During inversion the same differentiable equation of state converts the fitted temperature and pressure into mass density before the network call.

Use [LTELocalCoefficients](../src/prom3theus/rt/local_coefficients.py), the [matrix convention](../src/prom3theus/rt/polarization.py), [formal solver](../src/prom3theus/rt/formal_transfer.py), and [Hinode synthesis preset](../configs/muram/hinode/synthesis.yaml) as the reference implementation. For example, recover rho Q, U, V from K[2,3], K[3,1], K[1,2] with zero-based indexing. Test extraction and reconstruction against the full matrix before training.

## Snapshot sampling and perturbations

1. Accept a native snapshot directory and iteration. Reuse MURaMSnapshot and its oriented column reader for units and component order. Read temperature, pressure, all velocity and magnetic components, and continuum optical depth. With one large snapshot, sample columns through the existing memory maps. Keep only the sampled native states and the current batch of training labels in memory.
2. Split contiguous horizontal tiles into training, validation, and final test sets **before** drawing points or perturbations. Keep each entire column, its variants, and each validation ray in a single split. Use nonadjacent held-out regions or a buffer between regions. Save tile coordinates and seeds. If a second snapshot becomes available, reserve it as an external test.
3. Stratify selection by depth using log continuum optical depth and by field strength. Include weak-field granulation, strong-field structures where present, and both tails of the intended line-forming domain. Save the actual counts and input ranges. Use native states as correlated anchors. Sample ray directions and polarization bases over the intended observing geometry and project both vector fields consistently. Record the velocity frame and apply any rotational transformation exactly once.
4. Draw every fresh point from a training snapshot column before evaluating its exact LTE target. The Hinode preset uses Gaussian widths of 0.05 in log temperature, 0.2 in log pressure, and 500 m/s in LOS velocity. For each fresh point, multiply its magnetic field vector by one factor drawn uniformly from [0.9, 1.1]. This preserves field direction and scales the strength by up to 10%. It draws microturbulent speeds from a half-normal distribution with Gaussian width 2500 m/s. Each fresh batch also receives a shared Gaussian wavelength shift with standard deviation 0.288675 grid intervals, matching the RMS of the earlier half-bin shifts. Wavelengths are clipped to the window bounds. Density follows from the EOS. Validation and test points and rays come from their held-out columns.
5. Resample invalid thermodynamic draws using the declared operating domain and track rejection rates. Fit input and target scales on training tiles only. Keep perturbed validation and untouched native columns as distinct evaluation populations. Record where tests exercise extrapolation.

The Hinode trainer seeds a 4,096-point replay buffer with its first full fresh batch. Subsequent 1,024-point updates combine 512 replay points and 512 fresh MURaM points. The YAML settings are `replay_fraction` and `replay_buffer_size`; a zero fraction disables replay. Only fresh points require exact LTE labels. Each cached example retains its density-based mapper inputs, seven coefficient targets, and original wavelength offset. The buffer samples without replacement, weighted by each profile’s last mean normalized training loss across wavelengths and channels. It refreshes the selected scores on reuse and replaces the easiest entries with harder fresh examples. Validation uses its fixed held-out data.

## Labels and PyTorch architecture

For Hinode, pass both line IDs together to a single LTELocalCoefficients instance so each target includes both line contributions and one continuum. Generate fresh labels on the training device under torch.no_grad(). Evaluate the full point batch and wavelength grid at once. Persist a small fixed evaluation set with native indices, perturbed states, wavelength axes, teacher provenance, labels and representative complete rays. Large training-label caches are optional if timings justify reuse.

The Hinode surrogate uses an MLP with four SiLU hidden layers of width 128. The network encodes the local atmosphere once. One learned affine map predicts the knot values of five rest-frame spline profiles from the standardized features. The MLP adds a residual to those same knots before interpolation and physical decoding. Its last layer starts at zero, and both branches train together. Two pairs share the exact magnetic azimuth factors for Q and U. Combining the five amplitudes with the field orientation gives the seven physical coefficients, bounds dichroism by extinction, and preserves field-reversal symmetries. Reconstruct the coefficient profiles in parallel from a differentiable spline basis evaluated at the requested wavelengths. Begin with a basis fit to teacher profiles and measure reconstruction error with optimal weights before training the MLP. Increase knot density near both line cores and across the allowed velocity and magnetic splitting range if the basis alone exceeds the error budget. Save knots, spectral support, feature order and normalizations with the weights. Evaluate gradients with respect to shifted wavelengths as well as atmospheric inputs.

At each depth the spline is sampled at the vacuum wavelength `lambda_rest = lambda_obs * sqrt((1 - v_LOS/c)/(1 + v_LOS/c))`. Positive receding velocity shifts line centers to longer observed wavelengths. Fresh and replay examples use their original moving-atmosphere LTE targets on their observer grids. Velocity enters the mapper through spline sampling, and the Planck source uses the observer wavelength. A shifted query beyond the knot grid takes the nearest endpoint coefficient. Choose a window wide enough to keep the shifted detector samples within the learned range. Existing mapper checkpoints retain their recorded architecture and wavelength convention.

As a first loss, predict log eta I and use an asinh transform for each signed channel, with a fixed per-channel scale estimated from the training split. Average Huber losses equally across the seven transformed channels. Decode to physical units before assembling K. Compute all validation quantities after conversion back to physical units. Check positive extinction, physically admissible dichroism, vanishing polarization at zero field, field-reversal parity, and Q/U transformation under basis rotation. If these fail, enforce the corresponding relations in the output parameterization before using the mapper for inversion.

## Standalone training command

The surrogate YAML specifies the MURaM snapshot and iteration, instrument configuration, output directory, spectral window, and training settings. Relative paths in the YAML resolve from its directory. Lightning selects a single device automatically. [Hinode surrogate configuration](../configs/muram/hinode/surrogate.yaml)

```bash
PYTHONPATH=src python -m prom3theus.surrogate --config configs/muram/hinode/surrogate.yaml
```

On Derecho, submit this training run from the repository root with `qsub scripts/muram/train_surrogate.sh`. To use the trained mapper in an inversion, set the synthesis window’s `coefficient_backend` to `nn_surrogate` and `mapper_checkpoint` to the selected `best.pt`. The 2,049 points span the 7 Å window at 3.42 mÅ spacing. The detector settings use four synthesis intervals per 21.549 mÅ pixel.

For another LTE window, copy the Hinode YAML and update the snapshot, output directory, line IDs, and wavelength bounds. Set `instrument_config` to the detector YAML for that window, or omit it for training without detector validation. The line list and continuum wavelength interval must be supported by the exact LTE resources. Choose bounds that cover the inversion window, including Doppler shifts. The optional detector YAML uses the existing instrument registry. A new window can be trained without writing a detector operator.

## Selecting the inversion backend

Each entry under `synthesis.windows` selects its own coefficient backend. Omitting the field defaults to `coefficient_backend: exact_lte`. To use a pretrained model, add the following fields to the corresponding window in the inversion YAML.

```yaml
coefficient_backend: nn_surrogate
mapper_checkpoint: /absolute/path/to/mapper-run/best.pt
```

Relative checkpoint paths resolve from the inversion YAML directory. The inversion verifies line IDs and wavelength coverage, freezes the mapper and differentiates through its inputs. The two backends share the coarse/fine sampler, thermal source, formal solver, and instrument pipeline. Evaluation snapshots embed the mapper architecture and weights. The inversion resume contract records the source checkpoint digest.

## Monitoring and resuming

The batch size counts atmospheric points for pretraining. Use `--log-every`, `--validate-every` and `--checkpoint-every` to set reporting intervals. Repeat the command with `--resume` and the desired total `--steps` to continue. Start with progress and loss every 50 steps, validation every 5,000 steps, and a resumable checkpoint every 1,000. Use four loader workers for snapshot reads and one training process for GPU label generation and optimization. In a run, first inspect the snapshot and split, establish input scales and spectral support, build fixed evaluation fixtures, then alternate online teacher labeling and student optimization. The Hinode preset uses SOAP with exponential learning-rate decay from 0.001 to 0.0001 over 10,000 steps, then holds the final rate. Report effective throughput including label generation. Generate batches reproducibly from the split, seed and step so DataLoader prefetching cannot change the resumed sequence.

| Run output | Contents |
| --- | --- |
| run.json | Resolved arguments, snapshot layout and identity, split, perturbations, units, line IDs, wavelength convention, basis, teacher resources, software and device information. |
| train.log | Timestamped startup stages, periodic step counts, loss, throughput, ETA, validation results, checkpoint paths, completion or failure. Mirror progress to the terminal. Use plain periodic lines in redirected batch-job logs. |
| metrics.jsonl | Append step, samples seen, learning rate, losses, validation metrics, elapsed time and peak GPU memory at reporting intervals. `loader_wait_ms_per_batch` measures mean host time retrieving the next training batch within the interval. GPU work can overlap that wait. |
| validation data and figures | Fixed held-out points and rays, per-component residuals, absolute continuum comparison, and weak- and strong-field examples. |
| last.pt and best.pt | Both files contain inference weights and the model contract. The resumable last.pt also stores the optimizer, deterministic sample position, replay contents and priorities when enabled, and best model selected so far. Verify the scientific configuration on resume and preserve the best score across restarts. |

## Validation and decision to integrate

1. **Coefficient and basis tests.** Compare all seven channels on unseen points by optical depth and field strength. Evaluate line cores, wings and continuum on 2×, 4× and shifted grids using the same weights. Track relative error for strong signals and absolute error near zero.
2. **Whole-ray tests.** Substitute predicted coefficients into the same formal solver with identical atmospheric samples, intervals, boundary radiation, Planck source, and PSF. Compare each detector Stokes component with LTE on held-out columns and inclined rays. Include weak polarization, strong fields, velocity gradients and absolute continuum contrast.
3. **Gradient tests.** At fixed sampled depths, compare directional derivatives of detector Stokes and of the actual inversion loss with respect to temperature, pressure, velocity and magnetic field. Cover continuum brightness and signed Q/U/V responses. Also test the production coarse-to-fine path where predicted opacity can change fine-point placement.
4. **GPU benchmark.** Run [benchmark_lte_inversion.py](../src/prom3theus/diagnostics/commands/benchmark_lte_inversion.py) on two copies of the same preset with identical pixels and seed and separate output directories. It reports forward and backward time, memory and solver calls. Time local coefficient inference and then complete inversion steps, including backward, with matching hardware, dtype, ray batch, depths and wavelength grid. Warm up both paths. Synchronize at benchmark boundaries and record peak memory. Measure offline teacher cost separately. Report the number of training steps needed to recover pretraining cost from measured per-step savings.
5. **Acceptance.** Agree a channel-specific error budget from the target observation noise or inversion tolerances before selecting a checkpoint. For noiseless synthetic data, explicitly set tolerances in units of a fixed continuum reference. Record high-percentile errors as well as means. Require acceptable spectra, gradient agreement, field recovery on a small matched inversion, and a measured end-to-end speed improvement. A low coefficient training loss alone cannot establish these.

## Implementation order

1. Implement the snapshot point sampler, spatial split, deterministic perturbations and exact teacher extraction. Test units, LOS sign, matrix signs and split isolation on a small fixture.
2. Implement the spectral head and basis reconstruction, then measure the basis error floor on native and perturbed states.
3. Add the standalone PyTorch trainer, file logging, periodic progress, checkpoints and resume. Exercise a short run, restart it, and verify its next batch, metrics and model update against an uninterrupted run.
4. Add held-out ray, gradient and speed evaluations and compare against the exact LTE baseline.
5. Add an opt-in inversion coefficient backend after validation. Freeze mapper weights, retain gradients to the atmosphere, batch every ray at once per coarse or fine evaluation, and integrate once after merging depths. Persist the mapper identity and supported wavelength domain in the run contract.
6. After validating Hinode, use the same trainer to fit an independent HMI mapper on Fe I 6173.3352 Å. Validate its emergent spectra through the HMI filter responses. In a joint inversion, each spectral window loads its own frozen checkpoint and receives local parameters from the shared atmosphere network.
