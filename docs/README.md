# PROM3THEUS documentation

<p align="center">
  <img src="assets/prom3theus-emblem.png" alt="PROM3THEUS icon: a gold and red flame forming the letter P" width="200">
</p>

**Physics-informed Reconstruction Of Magnetism in 3D THrough EUV and Spectropolarimetry**

PROM3THEUS reconstructs a shared solar atmosphere from spectropolarimetric and
EUV observations using physics-informed neural networks. It combines LTE Stokes
modeling for Hinode/SOT-SP and SDO/HMI with optically thin SDO/AIA EUV modeling
and optional magnetofluid constraints.

## Guides

- [Project README](../README.md): installation, run configurations, data preparation,
  training, and evaluation.
- [Architecture](architecture.md): component boundaries, observation contracts,
  shared atmosphere, training, and persistence.
- [Local HMI baseline](local-hmi-baseline.md): a small LTE inversion and spherical
  current-free extrapolation to 20 Mm, with explicit acceptance checks.
- [Progressive potential boundaries](progressive-potential-boundaries.md): source Br,
  FFT geometry, update schedule, and checkpoint behavior.
- [Coronal energy constraint](coronal-energy.md): the optional energy equation
  and its physical terms.
- [Scientific resource builder](../resource_builder/README.md): reproducible
  generation of the packaged scientific inputs.

## Design and implementation notes

These documents record implementation decisions and proposals. Consult each
document's status before using it as a description of current behavior.

- [Modular runner implementation](modular-runner-plan.md)
- [Observation loading and sampling plan](data-loading-plan.md)
- [Initial HMI + AIA integration plan](aia-integration-plan.md)
- [LTE framework porting plan](porting-plan.md)

## Logos

The supplied artwork is available as a [horizontal logo](assets/prom3theus-logo.png)
and a [square project icon](assets/prom3theus-emblem.png). Both are PNGs with
transparent backgrounds.
