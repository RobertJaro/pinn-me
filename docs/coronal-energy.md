# Optional coronal energy constraint

`physics.equations.coronal_energy` implements

\[
D_t p+\gamma p\nabla\cdot\mathbf v
=(\gamma-1)(Q-n_e n_H\Lambda(T)-\nabla\cdot\mathbf q),
\qquad
\mathbf q=-\kappa_0 T^{5/2}\hat{\mathbf b}(\hat{\mathbf b}\cdot\nabla T).
\]

It is **disabled** in `configs/hmi_aia_dynamic.yaml`. Disabled means no resource
load, no extra atmosphere evaluation, no second derivatives, and no new logged
loss. HMI and AIA fitting settings are unchanged.

When enabled, it reuses upper-domain collocation points but evaluates only those
above `minimum_height_megameter` (3 Mm). Config validation requires this height
to be above both the reference transition region and the line-formation domain.
This is a fixed spatial selection, not a temperature-dependent escape mask.
There is no photospheric energy constraint and no clipping of model outputs.

## Terms and units

- Local temperature, gas pressure, velocity, and magnetic field come from the
  shared model. Densities come from the shared EOS; `n_H` means hydrogen nuclei,
  obtained as `rho / (mass_u_per_h_nucleus * atomic_mass_unit)`.
- Radiation uses a prepared bolometric CHIANTI table, not AIA channel responses.
  Lambda has units W m^3, and the volumetric loss is W m^-3. Density products
  are evaluated in log space to avoid intermediate float32 overflow.
- Conduction uses SI spatial derivatives and `kappa0 = 1e-11 W m^-1 K^-7/2`.
  The first temperature derivative retains its graph even during validation.
  Near zero magnetic field, the direction uses
  `B / sqrt(B^2 + magnetic_floor_gauss^2)`: this suppresses conduction smoothly,
  not an isotropic unmagnetized transport model.
- Heating is prescribed: `Q = heating_w_m3 * exp(-(h-h_min)/L_Q)`.
  No heating network or fitted parameter is added. The YAML amplitude and scale
  height are trial values, not a calibrated heating model. Set the amplitude to
  zero only for an intentional no-volumetric-heating experiment.
- Time derivatives use seconds and the existing Carrington-frame velocity.
  The residual is normalized by detached local pressure and the existing
  transport timescale, then contributes one `coronal_energy` loss.

## Prepare and enable later

The generated table is included at
`src/prom3theus/resources/sets/plasma/coronal_cooling_v1.json`: 201 temperature
nodes from 1e4 to 1e9 K, with Lambda(1 MK) = 3.57585e-35 W m^3. It was generated
from the existing CHIANTI 11.0.2 database using the `sunerf` environment.
The energy equation remains disabled; no training configuration was activated.

Use the `sunerf` environment with **fiasco 0.8.2** and the existing CHIANTI
**11.0.2** database (the directory containing `ascii/` and
`chianti_11.0.2.h5`). From the repository root:

```sh
conda run --no-capture-output -n sunerf python -m resource_builder.coronal_cooling \
  --database-root /path/to/response_calibration/chianti/11.0.2
```

This writes `src/prom3theus/resources/sets/plasma/coronal_cooling_v1.json`.
The builder includes line and continuum losses, stores log10 SI coefficients,
and records database version, abundance and ionization-equilibrium hashes,
fiasco version, and reference electron density (1e9 cm^-3). No table is
downloaded or calculated during training. The table is loaded once as shared
module buffers, and checkpoints reject a changed table hash.

Set `cooling_table` in the YAML to the generated file (relative to the YAML,
`../src/prom3theus/resources/sets/plasma/coronal_cooling_v1.json`), then set
`enabled: true` and a positive `weight` after reviewing the assumptions below.
The global adiabatic-pressure equation cannot be enabled simultaneously.

## Validity and outstanding scientific choices

This is single-temperature, fixed-gamma, collisional coronal physics with
ionization-equilibrium, optically thin losses. Height alone does not establish
validity: cold condensations, rapid ionization changes, or saturated/nonlocal
heat transport require different closures. Classical Spitzer conduction here
has no flux limiter. It transports energy; it does not replace heating.

The default cooling abundances match the existing AIA response
(`sun_coronal_2021_chianti`), **not** the STiC photospheric EOS mixture. Review
this trace-element enrichment assumption and density dependence before physical
interpretation. The table uses a fixed reference electron density, not a 2D
density-dependent loss function. Log-linear table interpolation has explicit
cold T^2 and hot T^0.5 numerical tails; these do not extend physical validity.

The builder uses FIASCO's integrated free-free and free-bound loss methods
(the latter is the Mewe approximation), all available bound-bound transitions,
and separately integrates the two-photon spectrum. The standard FIASCO
`radiative_loss` method alone omits two-photon emission. Missing atomic datasets
are listed in the resource provenance. Proton excitation is disabled in this
first cooling resource; two-ion population models and level-resolved corrections
are used where available. These choices are recorded, not silently equated to
the spectral AIA response calculation.

No new boundary energy flux condition is imposed. This constrains an open
coronal inverse problem, not a globally energy-closed photosphere-to-corona
simulation. Validate inferred boundary heat/enthalpy fluxes and sensitivity to
heating amplitude/scale before relying on magnetic-field changes.

Tests cover the generated component sum, interpolation, and float32 gradients
through the shared EOS and energy residual. The optimized chunked C IV
calculation agrees with unmodified FIASCO at three temperatures to relative
tolerance 1e-10. The two-photon wavelength quadrature differs from an 8193-point
reference by 0.0043% or less in the He II convergence check.

To rerun the real-database builder tests, set `CHIANTI_DATABASE_ROOT` to the same
database directory and run `python -m pytest
tests/resources/test_coronal_cooling_builder.py` in `sunerf`. The normal runtime
tests need neither FIASCO nor the full database. Synthetic curves remain test
fixtures only and are never supplied as training resources.
