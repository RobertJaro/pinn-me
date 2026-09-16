# Modular runner implementation

Implemented 2026-09-07. This replaces the earlier migration proposal; the change
intentionally removes backwards compatibility.

There is one optimizer runner: `application.joint_training.run_joint_inversion`.
A Stokes-only run is a schema-v3 configuration containing one stream. Joint
Stokes/EUV runs and additional registered observation types use the same
assembly, training lifecycle, scheduler, diagnostics and snapshot format.
`application.joint_runner` is its setup/forward diagnostic operation, not another
optimizer implementation.

## Extension contracts

1. Register strict observation and term dataclass schemas plus their compatible
   pair in `config.registry` before parsing configuration.
2. Register an `ObservationProvider` for preparation and resource requirements.
   Return canonical batch loaders and immutable observation references. Raster
   sources use the built-in scene construction; other sources supply an explicit
   `SceneContract` and a `sampling_support(scene, chunk_size)` iterator describing
   their spatial/time support for physics sampling.
3. Register a `ForwardProvider` for construction and snapshot reconstruction.
   Its term evaluates the likelihood against the shared atmosphere and exposes
   deterministic canonical-batch prediction. Constructor contracts contain only
   serializable values; model parameters and buffers live in the tensor state.
4. Optionally register a diagnostic renderer that returns per-stream reports and
   `DiagnosticOutput(report, paths)`. The common lifecycle handles scheduling and
   logging using only the explicitly declared paths.

The common runner does not require a new type switch for an additional source.
Providers remain responsible for instrument calibration, units, geometry,
resource requirements and likelihood semantics.

## Shared behavior

- One atmosphere and scene; Stokes sources are rebased into that common chart
  and time origin while retaining physical rays, polarization bases and targets.
- Shared physics evaluates once per optimizer step. Stream-qualified metrics
  prevent collisions when multiple observations have the same kind.
- A reference stream defines epochs. Other streams retain independent sampling
  positions and RNG state across epochs; continuation restores their progress.
- LTE and AIA use common bounded jitter, contribution-weighted inverse-CDF
  refinement and sample merging. Each forward model supplies its own physical
  contribution weights. Validation and export are deterministic.
- Only rolling `output_directory/state.p3s` and `output_directory/last.ckpt` are
  written. Plots and validation working files live under `work_directory`.
- Evaluation snapshots reconstruct the atmosphere and all registered terms
  without a Trainer or observation files. Stored observations are opened lazily
  for raster comparisons and exports.

## Removed components

The schema-v2 decoder and configurations, standalone LTE runner and Lightning
module, duplicate callbacks, old artifact model/export entry points, checkpoint
migration paths and their obsolete lifecycle tests have been removed. Maintained
configuration examples use schema v3. Existing runs must start with the new
configuration and artifact contracts.

## Validation and limits

Tests cover single/joint optimization and continuation, shuffled sampling replay,
stream-qualified metrics, scene rebasing, AIA jitter/refinement/gradients,
snapshot reconstruction and a third non-raster observation that runs through the
same optimizer and public loader. Numerical RT and instrument tests remain.
Packaging tests check isolated wheel imports and source distributions.

Adaptive trapezoidal sampling does not guarantee lower error for every profile;
production comparisons still need dense-reference convergence checks. Exact
loader replay assumes deterministic prepared-data reads, and replaying a partial
pass incurs reads. Extension registrations must be installed before loading a
snapshot that uses them. This refactor is not a production GPU performance or
scientific equivalence benchmark.

See [architecture.md](architecture.md) for the maintained system description.


## Pipeline acceptance audit (2026-09-07)

| Criterion | Executable evidence |
| --- | --- |
| One optimizer lifecycle; retired modules absent and not imported | `tests/test_architecture.py`; single/joint training and resume in `tests/application/test_joint_runner.py` |
| New observation type requires registration, not runner edits | Third non-raster source runs, saves and predicts in `tests/artifacts/test_stream_state.py` |
| Registration is strict and cannot partially mutate dispatch or shadow built-ins | `tests/config/test_registry.py` |
| RT and AIA preparation respect dependency boundaries; coordinator has no instrument-specific imports | `tests/test_architecture.py` |
| Shared atmosphere, shared physics, stream-qualified metrics | `tests/inversion/test_joint.py`, `tests/training/test_metrics.py` |
| Independent reference-stream epochs and reproducible shuffled continuation | `tests/training/test_stream_scheduler.py` |
| Shared coarse/fine sampling, bounded jitter, deterministic evaluation and differentiable AIA synthesis | `tests/inversion/test_ray_integral.py` and existing LTE forward tests |
| Common physical scene/time, including radius checks on the unchanged-chart path | `tests/observations/test_scene_rebase.py` and ray-support tests in `tests/application/test_joint_runner.py` |
| Rolling root-level P3S/checkpoint files; diagnostics under work directory | Joint runner and third-provider integration tests |
| P3S reconstructs Stokes, AIA and a third term without observation I/O or a Trainer | `tests/artifacts/test_stream_state.py` |
| Logger consumes exact declared artifacts with stream-qualified names | `tests/diagnostics/test_providers.py` |
| Built wheel supports isolated imports and maintained configurations | `tests/test_packaging.py` |

The second cleanup removed the unused LTE batch validator, an unused sampling
wrapper, dashboard metric pass-through code, incidental dry-run re-exports,
and an artificial evaluation-report wrapper. Checkpoint lookup now checks the
single rolling file directly. Scene setup and rebasing use one time-origin
decoder. Scientific resource identifiers and active LTE synthesis, instrument,
observation-store and export code remain because the joint pipeline uses them.

Passing these automated checks does not substitute for production GPU benchmarks,
large-event memory measurements or scientific convergence studies. Those are
separate experiments; no claim of measured production parity is made here.
