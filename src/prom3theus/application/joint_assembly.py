"""Assemble joint models from validated configuration and observation streams."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import Any

from torch import nn

from prom3theus.components.forward import _default_term_builder
from prom3theus.components.observations import (
    _default_stream_loader,
    _required_resource_sets,
)
from prom3theus.config.joint_schema import (
    JointInversionConfig,
    STICTableSupportRegularizationConfig,
    VectorMagnitudeRegularizationConfig,
    VectorPotentialRegularizationConfig,
)
from prom3theus.inversion.assembly import build_physics_assembly
from prom3theus.inversion.data_terms import (
    AtmosphereRegularizationTerm,
    PhysicsConstraintTerm,
    SharedObjectiveTerm,
)
from prom3theus.inversion.data_terms.potential_boundary import (
    ProgressivePotentialBoundary,
)
from prom3theus.inversion.joint import JointForwardModel
from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.observations import SceneContract
from prom3theus.resources import LEGACY_LTE_RESOURCE_SET_ID, validate_resource_sets
from prom3theus.rt import StratifiedAtmosphereModel

from .batches import deterministic_validation_batch
from .configuration import atmosphere_options
from .joint_contracts import (
    JointRunnerFactories,
    JointRuntime,
    LoadedJointStream,
    SharedTermAssembly,
)
from .joint_streams import (
    _default_scene_builder,
    _joint_observation_bounds,
    _observation_times_hours,
)
from .runtime import _device, _move_value, state_dict_sha256


from .data import _startup_stage


def _atmosphere_options(config: JointInversionConfig, scene: SceneContract) -> dict:
    coordinate = scene.atmosphere_coordinate_metadata
    return {
        **atmosphere_options(config.atmosphere),
        **{
            name: coordinate[name]
            for name in (
                "spatial_coordinate_center_mm",
                "spatial_coordinate_scale_mm",
                "time_coordinate_center_hours",
                "time_coordinate_scale_hours",
                "scene_geometry_config",
            )
        },
    }


def _default_atmosphere_builder(
    config: JointInversionConfig,
    scene: SceneContract,
    resources: Mapping[str, Mapping[str, Any]],
) -> nn.Module:
    del resources
    options = _atmosphere_options(config, scene)
    model = StratifiedAtmosphereModel(**options).float()
    model.construction = options
    return model


def _reference_gravity(
    resources: Mapping[str, Mapping[str, Any]], *, required: bool
) -> float | None:
    try:
        return (
            float(
                resources[LEGACY_LTE_RESOURCE_SET_ID]["falc_top_boundary"][
                    "gravity_cm_s2"
                ]
            )
            / 100.0
        )
    except (KeyError, TypeError, ValueError):
        if required:
            raise ValueError(
                "Active force balance requires verified FALC gravity metadata."
            ) from None
        return None


def _physics_options(
    config: JointInversionConfig,
    domain: SphericalShellDomain,
    gravity_m_per_s2: float | None,
) -> dict[str, Any]:
    physics = config.physics
    equations = physics.equations.to_dict()
    upper_active = any(
        equations[name]["enabled"] and equations[name]["weight"] > 0.0
        for name in (
            "upper_domain_microturbulence_prior",
            "upper_domain_temperature_prior",
            "coronal_energy",
        )
    )
    upper_domain = None
    if upper_active:
        geometry = config.atmosphere.geometry
        upper_domain = domain.with_height_bounds(
            (
                geometry.line_formation_outer_height_megameter,
                geometry.outer_height_megameter,
            )
        ).configuration()
    return {
        **physics.collocation.to_dict(),
        "vector_basis_matches_spatial_coordinates": (
            physics.vector_basis_matches_spatial_coordinates
        ),
        "magnetic_current_free_steps": physics.magnetic_current_free_steps,
        "magnetic_current_free_final_factor": physics.magnetic_current_free_final_factor,
        "loss_start_step": physics.loss_start_step,
        "loss_ramp_steps": physics.loss_ramp_steps,
        "robust_loss_delta": physics.robust_loss_delta,
        "adiabatic_index": physics.adiabatic_index,
        "normalization": physics.normalization.to_dict(),
        "equations": equations,
        "gravity_m_per_s2": gravity_m_per_s2,
        "sampling_domain": domain.configuration(),
        "upper_sampling_domain": upper_domain,
    }


def _default_shared_terms_builder(
    config: JointInversionConfig,
    streams: Mapping[str, LoadedJointStream],
    scene: SceneContract,
    resources: Mapping[str, Mapping[str, Any]],
    *, summaries=None,
) -> SharedTermAssembly:
    geometry = config.atmosphere.geometry
    height_power = config.physics.collocation.height_sampling_power
    observation_bounds = (_joint_observation_bounds(streams, scene) if summaries is None
                          else summaries["observation_bounds"])
    full_domain = SphericalShellDomain.from_observation_bounds(
        observation_bounds,
        (geometry.outer_height_megameter, geometry.inner_height_megameter),
    )
    line_domain = full_domain.with_height_bounds(
        (
            geometry.inner_height_megameter,
            geometry.line_formation_outer_height_megameter,
        )
    )
    equations = config.physics.equations.to_dict()
    physics_active = any(
        item["enabled"] and item["weight"] > 0.0 for item in equations.values()
    )
    gravity_required = any(
        equations[name]["enabled"] and equations[name]["weight"] > 0.0
        for name in (
            "hydrostatic_equilibrium",
            "magnetohydrostatic_equilibrium",
            "momentum",
        )
    )
    gravity = _reference_gravity(resources, required=gravity_required)
    terms: dict[str, SharedObjectiveTerm] = {}
    physics_metadata = None
    if physics_active:
        physics = build_physics_assembly(
            _physics_options(config, full_domain, gravity),
            reference_gravity_m_per_s2=gravity,
        )
        # Keep the assembled objective training-ready.  The setup-only
        # evaluation temporarily disables higher-order derivative graphs below.
        terms["physics"] = PhysicsConstraintTerm(physics)
        physics_metadata = physics.model_configuration()
    potential_metadata = None
    potential = config.physics.potential_boundary
    if potential.enabled:
        stokes_streams = {
            name: loaded for name, loaded in streams.items()
            if loaded.prepared.descriptor.observation_kind == "stokes"
        }
        if not stokes_streams:
            raise ValueError("Potential boundaries require a Stokes observation footprint")
        if not geometry.inner_height_megameter <= potential.source_height_megameter <= geometry.line_formation_outer_height_megameter:
            raise ValueError("Potential source height must lie in the line-formation domain")
        source_domain = SphericalShellDomain.from_observation_bounds(
            (_joint_observation_bounds(stokes_streams, scene) if summaries is None
             else summaries["stokes_bounds"]),
            (geometry.outer_height_megameter, geometry.inner_height_megameter),
        )
        terms["potential_boundary"] = ProgressivePotentialBoundary(
            potential.to_dict(), full_domain, source_domain, scene.scene_basis,
            time_dependent=geometry.time_dependent,
            observation_times_hours=(
                (_observation_times_hours(stokes_streams, scene) if summaries is None
                 else summaries["observation_times_hours"])
                if geometry.time_dependent else None
            ),
        )
        potential_metadata = {
            "configuration": potential.to_dict(),
            "geometry": terms["potential_boundary"].geometry_metadata,
        }
    for regularization in config.atmosphere_regularization:
        domain = (
            line_domain if regularization.domain == "line_formation" else full_domain
        )
        sample = domain.deterministic_grouped(
            regularization.height_layers,
            regularization.sample_count // regularization.height_layers,
        )
        smoothness_step_m = None
        if isinstance(regularization, STICTableSupportRegularizationConfig):
            weights = {
                "temperature": regularization.temperature_weight,
                "gas_pressure": regularization.gas_pressure_weight,
            }
        elif isinstance(regularization, VectorMagnitudeRegularizationConfig):
            weights = {
                "magnetic": regularization.magnetic_weight,
                "velocity": regularization.velocity_weight,
            }
            smoothness_step_m = None
        elif isinstance(regularization, VectorPotentialRegularizationConfig):
            weights = {
                "gauge": regularization.gauge_weight,
                "smoothness": regularization.smoothness_weight,
            }
            smoothness_step_m = regularization.smoothness_step_megameter * 1.0e6
        else:  # pragma: no cover - closed typed schema
            raise TypeError(
                f"Unsupported atmosphere regularization {type(regularization).__name__}."
            )
        terms[regularization.id] = AtmosphereRegularizationTerm(
            kind=regularization.type,
            position_m=sample["position_m"],
            time_hours=sample["time_hours"],
            component_weights=weights,
            smoothness_step_m=smoothness_step_m,
        )
    return SharedTermAssembly(
        terms=terms,
        metadata={
            "observation_bounds": observation_bounds,
            "full_domain": full_domain.metadata(height_power=height_power),
            "line_formation_domain": line_domain.metadata(height_power=height_power),
            "physics": physics_metadata,
            "potential_boundary": potential_metadata,
        },
    )


def build_joint_runtime(
    config: JointInversionConfig,
    *,
    rebuild_observations: bool = False,
    factories: JointRunnerFactories | None = None,
    for_training: bool = False,
) -> JointRuntime:
    """Load every stream and assemble one shared, unfitted atmosphere."""

    if not isinstance(config, JointInversionConfig):
        raise TypeError(
            "build_joint_runtime requires a validated JointInversionConfig."
        )
    factories = factories or JointRunnerFactories()
    if for_training and config.training is None:
        raise ValueError(
            "Joint optimization requires an explicit training configuration."
        )
    validate = factories.resource_validator or validate_resource_sets
    load_stream = factories.stream_loader or partial(
        _default_stream_loader,
        time_window=config.scene.time_window,
    )
    build_scene = factories.scene_builder or _default_scene_builder
    build_atmosphere = factories.atmosphere_builder or _default_atmosphere_builder
    build_term = factories.term_builder or _default_term_builder
    build_shared = factories.shared_terms_builder or _default_shared_terms_builder

    config.solver.work_directory.mkdir(parents=True, exist_ok=True)
    with _startup_stage("validate resources"):
        resources = dict(validate(_required_resource_sets(config)))
    from .data import load_joint_data
    with _startup_stage("restore or prepare data module"):
        data_module = load_joint_data(
            config, resources, load_stream, build_scene,
            rebuild_observations=rebuild_observations, for_training=for_training,
            compute_summaries=factories.shared_terms_builder is None,
        )
    loaded, scene = data_module.streams, data_module.scene
    if for_training:
        from prom3theus.observations.bulk import configure_readers
        configure_readers(config.training.reader_workers)
    controls = config.training if for_training else config.dry_run
    device = _device(controls.device)
    import os
    if device.type == "cuda" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    with _startup_stage("build atmosphere and load physics resources"):
        atmosphere = build_atmosphere(config, scene, resources)
    if not isinstance(atmosphere, nn.Module):
        raise TypeError("atmosphere_builder must return torch.nn.Module.")
    terms = {
        stream.id: build_term(stream, loaded[stream.id], atmosphere, scene)
        for stream in config.streams
    }
    with _startup_stage("scan domain geometry and build shared objectives"):
        shared = build_shared(config, MappingProxyType(loaded), scene, resources,
                              **({"summaries": data_module.summaries} if factories.shared_terms_builder is None else {}))
    if not isinstance(shared, SharedTermAssembly):
        raise TypeError("shared_terms_builder must return SharedTermAssembly.")
    weights = {stream.id: stream.data_term.weight for stream in config.streams}
    weight_schedules = {
        stream.id: stream.data_term.weight_schedule.to_dict()
        for stream in config.streams
        if stream.data_term.weight_schedule is not None
    }
    model = JointForwardModel(
        atmosphere,
        terms,
        weights,
        shared_terms=shared.terms,
        weight_schedules=weight_schedules,
    ).to(device)
    with _startup_stage("collect validation batches"):
        validation_batches = {
            stream.id: _move_value(
                deterministic_validation_batch(
                    loaded[stream.id].data_module,
                    max_samples=(
                        controls.validation_samples_per_stream
                        if for_training
                        else controls.max_samples_per_stream
                    ),
                ),
                device,
            )
            for stream in config.streams
        }
    return JointRuntime(
        data_module=data_module,
        config=config,
        resources=MappingProxyType(resources),
        streams=MappingProxyType(loaded),
        scene=scene,
        model=model,
        validation_batches=MappingProxyType(validation_batches),
        device=device,
        shared_metadata=shared.metadata,
        state_sha256_at_build=None if for_training else state_dict_sha256(model),
    )
