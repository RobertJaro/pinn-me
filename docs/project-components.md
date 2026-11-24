# PROM3THEUS project component plan

PROM3THEUS fits one solar atmosphere to observations from Hinode/SP, SDO/HMI, or SDO/AIA. The instrument determines what the model predicts. Hinode and HMI measure polarized light, represented by the four Stokes profiles I, Q, U, and V. AIA measures extreme ultraviolet (EUV) intensity in individual channels. A joint run evaluates both kinds of observation against the same temperature, pressure, magnetic field, and velocity model.

## Follow an inversion from input to result

| Stage | What happens | Where to start |
| --- | --- | --- |
| 1. Define the run | Select the instruments, spatial and temporal domain, fitted streams, physical constraints, and output locations. Resolve the YAML before allocating the model. | [Run presets](../configs), [YAML parameters](yaml-parameters.md), `prom3theus validate-config` |
| 2. Prepare observations | Supply a prepared Hinode raster, or acquire and prepare HMI Stokes frames and AIA images. HMI needs filter responses. AIA needs pointing and degradation calibration. | [Acquisition](../src/prom3theus/download), [preprocessing](../src/prom3theus/preprocess), [instrument adapters](../src/prom3theus/instruments) |
| 3. Assemble the scene | Load each stream on its own detector grid. Map its positions, ray directions, and acquisition times into a common scene. Bind each stream to its forward model and objective. | [Scene contract](../src/prom3theus/observations/scene.py), [runtime assembly](../src/prom3theus/application/joint_assembly.py) |
| 4. Predict measurements | Query the atmosphere along each ray. For Stokes, solve polarized LTE transfer and apply the detector response. For AIA, integrate electron density squared times the channel temperature response along the line of sight. | [LTE synthesis window](../src/prom3theus/inversion/lte.py), [AIA observation term](../src/prom3theus/inversion/data_terms/aia_euv.py) |
| 5. Fit the atmosphere | Combine enabled stream losses with configured physics and regularization terms. Backpropagate and update the shared model. The reference stream sets the epoch length. | [Joint forward model](../src/prom3theus/inversion/joint.py), [training module](../src/prom3theus/training/joint.py), `prom3theus invert` |
| 6. Inspect and reuse the result | Resume training from `last.ckpt`. Open `state.p3s` to query fields, predict a stream, export arrays, compare with an HMI vector record, or render time evolution. Raster comparisons also need the referenced observations. | [Save-state loader](../src/prom3theus/artifacts/loader.py), [export](../src/prom3theus/artifacts/export.py), [diagnostics](../src/prom3theus/diagnostics) |

## Choose a data path

| Input | Intended use | Starting point |
| --- | --- | --- |
| Observed Hinode/SP | Fit a static atmosphere from one raster. The presets cover magnetohydrostatic (MHS) inversion and extrapolation. | [Hinode configurations](../configs/hinode) |
| Observed SDO time series | The dynamic preset fits HMI and renders AIA with zero AIA training weight. The energy preset fits both streams and enables the coronal energy constraint. | [Dynamic run](../configs/sdo/hmi_aia_dynamic.yaml), [energy run](../configs/sdo/hmi_aia_energy.yaml) |
| MURaM columns | Generate observations from a known simulation atmosphere with the independent reference Stokes solver. Write a single-instrument store or synthetic SDO products for inversion, then compare the reconstruction with the native snapshot. | [Column synthesis workflow](muram-synthesis.md), [Hinode presets](../configs/muram/hinode), [mapper training](muram-lte-coefficient-mapper-plan.md) |
| MURaM volume viewed along inclined rays | Trace straight rays through the Cartesian volume to synthesize HMI or Hinode Stokes. Each snapshot produces spectral arrays, validity and boundary masks, and ray geometry. | [Ray synthesis workflow](muram-ray-synthesis.md), [ray entry point](../src/prom3theus/muram/synthesize_rays.py) |

For column synthesis, run `python -m prom3theus.muram.cli synthesize`. For inversion of a `muram_stored` stream, the same module's `invert` command registers the synthetic observation type before reading the YAML. Synthetic SDO FITS and prepared AIA stores use the ordinary HMI and AIA adapters. The Hinode [synthesis settings](../configs/muram/hinode/synthesis.yaml) configure the synthetic instrument.

## Find the component that owns a change

| Runtime package | Responsibility |
| --- | --- |
| [cli](../src/prom3theus/cli) | Parse commands and dispatch configuration validation, resource checks, acquisition, preparation, inversion, recovery, and export or comparison. |
| [config](../src/prom3theus/config) | Define and validate schema-v5 run settings, resolve paths and environment variables, and register configuration types. |
| [application](../src/prom3theus/application) | Coordinate the run lifecycle, including data setup, scene and model assembly, training startup, warm starts, evaluation, and report scheduling. |
| [components](../src/prom3theus/components) | Connect configured observation types to loaders and forward terms. Forward providers also reconstruct those terms from a saved model. |
| [download](../src/prom3theus/download) and [preprocess](../src/prom3theus/preprocess) | Acquire SDO inputs and coordinate time-series preparation, cropping, and reuse of prepared products. |
| [instruments](../src/prom3theus/instruments) | Own detector-specific geometry, calibration, readers, and response operators in the Hinode/SP, HMI, and AIA adapters. |
| [observations](../src/prom3theus/observations) | Define Stokes and scalar-image records, observation identity, scene coordinates, persistent stores, datasets, and batch loading. |
| [core](../src/prom3theus/core) | Supply shared coordinate transforms, solar kinematics, neural-network primitives, numerical checks, and parallel execution utilities. |
| [resources](../src/prom3theus/resources) | Load and verify the packaged atomic, plasma, and instrument inputs through versioned manifests and checksums. |
| [rt](../src/prom3theus/rt) | Implement the atmosphere model, equation of state, line opacity and polarization, formal Stokes transfer, optically thin integration, and potential-field and energy physics. |
| [inversion](../src/prom3theus/inversion) | Assemble predictions and losses. Ray and depth samplers select evaluation points, data terms score measurements, and constraints evaluate magnetofluid equations and boundaries. |
| [training](../src/prom3theus/training) | Schedule stream batches, run Lightning optimization, reduce metrics, and track sampling progress for resumption. |
| [artifacts](../src/prom3theus/artifacts) | Define the evaluation save format, validate saved state, reconstruct the model, and export atmosphere and optional Stokes arrays. |
| [diagnostics](../src/prom3theus/diagnostics) | Render observed and predicted signals, field maps, side views, and time series. Compare recovered fields and report physics residuals. |
| [validation](../src/prom3theus/validation) | Supply independent numerical reference transfer, depth-convergence checks, and synthetic recovery checks. |
| [muram](../src/prom3theus/muram) | Read native simulation snapshots, synthesize column and ray products, and retain source references, diagnostics, and provenance. |

## Repository support

- **Scientific inputs.** The offline builders reproduce atomic tables, instrument calibrations, and coronal cooling data. The runtime consumes their packaged outputs. [Resource builder](../resource_builder/README.md)
- **Run recipes.** YAML files specify experiments. Shell and Python scripts connect them to particular datasets and provide preparation, synthesis, profiling, and recovery workflows. [Configurations](../configs) · [Scripts](../scripts)
- **Verification.** Tests are grouped by subsystem. Cross-cutting tests cover architecture boundaries, distribution contents, and reproduction of resources. [Test suite](../tests)
- **Distribution.** Python dependencies, optional extras, the installed command, and packaged presets are declared together. The manifest controls source-distribution contents. [Project metadata](../pyproject.toml) · [Manifest](../MANIFEST.in) · [License](../LICENSE)
- **Local workspaces.** `data/` holds local scientific inputs and preparation files. `runs/` and `results/` hold local products. Each inversion configuration selects its actual work and output directories. [Repository ignore rules](../.gitignore)
- **Further reading.** The project README covers installation and commands. The architecture guide traces a numerical update, and the documentation index links scientific guides and historical design notes. [Project README](../README.md) · [Architecture](architecture.md) · [Documentation index](README.md)
