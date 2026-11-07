from copy import deepcopy

import pytest

from prom3theus.training.joint import _restore_cuda_rng_state, _scientific_contract


def contract():
    return {
        "configuration": {
            "atmosphere": {"network": {"hidden_dimension": 64}},
            "physics": {
                "potential_boundary": {
                    "enabled": True,
                    "grid_size": 64,
                    "weight": 1.0,
                    "start_step": 500,
                },
                "equations": {"divergence": {"enabled": True, "weight": 1.0}},
                "collocation": {"volume_points_per_step": 128},
            },
            "streams": [
                {
                    "observation": {"loader": {"batch_size": 256}},
                    "data_term": {
                        "objective": {"qu_warmup_steps": 8000},
                        "weight": 1.0,
                        "weight_schedule": {
                            "initial_weight": 10.0,
                            "initial_steps": 3000,
                            "ramp_steps": 3000,
                        },
                    },
                }
            ],
        },
        "streams": {"hmi": "data-signature"},
    }


def test_resume_cuda_rng_state_handles_fewer_visible_devices(monkeypatch):
    import torch

    saved = [torch.tensor([index], dtype=torch.uint8) for index in range(3)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)

    _restore_cuda_rng_state(saved)

    assert len(restored) == 1
    assert len(restored[0]) == 1
    torch.testing.assert_close(restored[0][0], saved[0])


def test_training_changes_are_allowed_without_mutating_recorded_config():
    previous = contract()
    original = deepcopy(previous)
    current = deepcopy(previous)
    cfg = current["configuration"]
    cfg["physics"]["potential_boundary"].update(weight=0.2, start_step=1000, jitter_fraction=0.5)
    cfg["physics"]["equations"]["divergence"].update(enabled=False, weight=0.0)
    cfg["physics"]["collocation"]["volume_points_per_step"] = 512
    cfg["streams"][0]["observation"]["loader"]["batch_size"] = 512
    cfg["streams"][0]["data_term"]["objective"]["qu_warmup_steps"] = 0
    cfg["streams"][0]["data_term"]["weight_schedule"].update(
        initial_weight=20.0,
        initial_steps=1000,
    )
    assert _scientific_contract(previous) == _scientific_contract(current)
    assert previous == original


def test_model_data_and_reference_geometry_changes_remain_incompatible():
    previous = contract()
    for path, value in [
        (("configuration", "atmosphere", "network", "hidden_dimension"), 128),
        (("configuration", "physics", "potential_boundary", "grid_size"), 32),
        (("streams", "hmi"), "changed-data"),
    ]:
        changed = deepcopy(previous)
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        assert _scientific_contract(previous) != _scientific_contract(changed)


def test_vector_potential_scale_default_is_compatible_but_changes_are_not():
    previous = contract()
    previous["configuration"]["atmosphere"]["parameters"] = {
        "magnetic_field": {"representation": "vector_potential", "scale_gauss": 1000.0},
    }
    previous["atmosphere"] = {"magnetic_representation": "vector_potential"}
    current = deepcopy(previous)
    magnetic = current["configuration"]["atmosphere"]["parameters"]["magnetic_field"]
    magnetic["scale_vector_potential"] = 1.0
    current["atmosphere"]["scale_vector_potential"] = 1.0
    assert _scientific_contract(previous) == _scientific_contract(current)
    magnetic["scale_vector_potential"] = 0.01
    assert _scientific_contract(previous) != _scientific_contract(current)
    magnetic["scale_vector_potential"] = 1.0
    current["atmosphere"]["scale_vector_potential"] = 0.01
    assert _scientific_contract(previous) != _scientific_contract(current)


def test_resume_allows_disabling_los_offset_fit_but_preserves_calibration_contract():
    previous = contract()
    instrument = {"line_of_sight_velocity_correction_m_per_s": 0.,
                  "optimize_line_of_sight_velocity_correction": True}
    previous["configuration"]["streams"][0]["data_term"]["instrument"] = instrument
    previous["terms"] = {"hmi": {"type": "lte_stokes", "options": {
        "instrument_line_of_sight_velocity_correction_m_per_s": 0.,
        "optimize_instrument_line_of_sight_velocity_correction": True,
    }}}
    original = deepcopy(previous)
    current = deepcopy(previous)
    current["configuration"]["streams"][0]["data_term"]["instrument"]["optimize_line_of_sight_velocity_correction"] = False
    current["terms"]["hmi"]["options"]["optimize_instrument_line_of_sight_velocity_correction"] = False
    assert _scientific_contract(previous) == _scientific_contract(current)
    assert previous == original
    current["configuration"]["streams"][0]["data_term"]["instrument"]["line_of_sight_velocity_correction_m_per_s"] = 100.
    assert _scientific_contract(previous) != _scientific_contract(current)


def test_new_disabled_plain_equation_matches_absent_checkpoint_default():
    previous = contract()
    current = deepcopy(previous)
    current["configuration"]["physics"]["equations"]["magnetic_force_free"] = {
        "enabled": False, "weight": 0.0,
    }
    original = deepcopy(current)
    assert _scientific_contract(previous) == _scientific_contract(current)
    assert "magnetic_force_free" not in _scientific_contract(current)["configuration"]["physics"]["equations"]
    assert current == original


def test_missing_magnetic_reference_height_matches_none_without_mutation():
    previous = contract()
    previous["configuration"]["atmosphere"]["parameters"] = {
        "magnetic_field": {"scale_gauss": 1000.0},
    }
    current = deepcopy(previous)
    current["configuration"]["atmosphere"]["parameters"]["magnetic_field"]["reference_height_megameter"] = None
    original = deepcopy(current)
    assert _scientific_contract(previous) == _scientific_contract(current)
    assert current == original


@pytest.mark.parametrize("before,after", [(None, 0.0), (None, 0.15), (0.0, 0.15), (0.15, 0.3)])
def test_numeric_magnetic_reference_height_changes_require_new_run(before, after):
    previous = contract()
    previous["configuration"]["atmosphere"]["parameters"] = {
        "magnetic_field": {"scale_gauss": 1000.0, "reference_height_megameter": before},
    }
    current = deepcopy(previous)
    current["configuration"]["atmosphere"]["parameters"]["magnetic_field"]["reference_height_megameter"] = after
    assert _scientific_contract(previous) != _scientific_contract(current)


def test_structured_equation_options_remain_part_of_resume_contract():
    previous = contract()
    current = deepcopy(previous)
    structured = {
        "enabled": False, "weight": 0.0,
        "cooling_resource": "plasma_v1:coronal_cooling", "minimum_height_megameter": 3.0,
    }
    current["configuration"]["physics"]["equations"]["coronal_energy"] = structured
    assert _scientific_contract(previous) != _scientific_contract(current)
    for name, value in (
        ("cooling_resource", "plasma_v2:coronal_cooling"),
        ("minimum_height_megameter", 5.0),
    ):
        changed = deepcopy(current)
        changed["configuration"]["physics"]["equations"]["coronal_energy"][name] = value
        assert _scientific_contract(current) != _scientific_contract(changed)


def test_resume_keeps_new_loss_buffers_and_applies_requested_learning_rate():
    from types import SimpleNamespace

    import torch

    from prom3theus.training.joint import JointInversionModule

    model = torch.nn.Module()
    model.field = torch.nn.Parameter(torch.tensor(3.0))
    model.objective = torch.nn.Module()
    model.objective.register_buffer("test_scale", torch.tensor([0.2]))
    model.shared_objectives = torch.nn.ModuleDict()
    settings = SimpleNamespace(
        learning_rate=0.01,
        final_learning_rate=0.001,
        learning_rate_decay_steps=None,
        max_steps=100,
        max_epochs=-1,
        migrate_data_order=False,
    )
    context = {
        "contract": contract(),
        "configuration": {"training": {"learning_rate": 0.1}},
    }
    module = JointInversionModule(model, settings, context)
    state = {k: v.clone() for k, v in module.state_dict().items()}
    state["model.objective.test_scale"][:] = 9.0
    state["model.field"] = torch.tensor(5.0)
    checkpoint = {"formal_transfer_format_version": 1, "state_dict": state, "prom3theus_joint": context}
    module.on_load_checkpoint(checkpoint)
    module.load_state_dict(checkpoint["state_dict"])
    assert model.field.item() == 5.0
    torch.testing.assert_close(model.objective.test_scale, torch.tensor([0.2]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    module._trainer = SimpleNamespace(
        global_step=50,
        estimated_stepping_batches=100,
        optimizers=[optimizer],
        lr_scheduler_configs=[SimpleNamespace(scheduler=schedule)],
    )
    module.on_train_start()
    assert abs(optimizer.param_groups[0]["lr"] - 0.01 * 0.1**0.5) < 1e-12
    assert schedule.base_lrs == [0.01]


def test_volume_to_surface_sampling_is_a_training_change():
    old, new = contract(), contract()
    old["configuration"]["physics"]["potential_boundary"].update(
        interior={"enabled": True, "points": 1024},
        top_points=256,
        side_points=512,
        normalization_height_count=32,
        normalization_points_per_height=64,
    )
    new["configuration"]["physics"]["potential_boundary"].update(
        photosphere={"enabled": True, "end_step": 8000},
        top_grid_size=16,
        side_horizontal_points=32,
        side_height_points=16,
        seed=0,
    )
    assert _scientific_contract(old) == _scientific_contract(new)


def test_sampling_migration_rebuilds_only_derived_buffers_and_keeps_same_geometry():
    from types import SimpleNamespace

    import torch

    from prom3theus.config.schema import PotentialBoundaryConfig
    from prom3theus.inversion.data_terms.potential_boundary import (
        ProgressivePotentialBoundary,
    )
    from prom3theus.inversion.sampling import SphericalShellDomain
    from prom3theus.training.joint import JointInversionModule

    domain = SphericalShellDomain(
        0.0, (-0.04, 0.04), (-0.04, 0.04), (0.0, 2.0), (0.0, 20.0), 696e6
    )
    options = PotentialBoundaryConfig(
        grid_size=4, top_grid_size=2, side_horizontal_points=2, side_height_points=2
    ).to_dict()
    term = ProgressivePotentialBoundary(
        options, domain, domain, torch.eye(3), observation_times_hours=[0.0, 2.0]
    )
    model = torch.nn.Module()
    model.field = torch.nn.Parameter(torch.tensor(3.0))
    model.shared_objectives = torch.nn.ModuleDict({"potential_boundary": term})
    module = JointInversionModule(model, SimpleNamespace(), {})
    prefix = "model.shared_objectives.potential_boundary."
    original = deepcopy(module.state_dict())
    checkpoint = {
        "state_dict": deepcopy(original),
        "optimizer_states": [{"sentinel": 7}],
        "global_step": 1234,
    }
    checkpoint["state_dict"]["model.field"] = torch.tensor(55.0)
    # Old normalized objectives must not silently inherit raw-G² semantics.
    checkpoint["state_dict"][prefix + "_extra_state"].update(
        version=7, last_update=1000, frozen=True
    )
    with pytest.raises(ValueError, match="objective changed"):
        module._restore_potential_sampling(checkpoint)
    checkpoint["state_dict"][prefix + "_extra_state"]["version"] = 8
    with pytest.raises(ValueError, match="objective changed"):
        module._restore_potential_sampling(checkpoint)
    checkpoint["state_dict"][prefix + "_extra_state"]["version"] = 9
    with pytest.raises(ValueError, match="objective changed"):
        module._restore_potential_sampling(checkpoint)
    # Geometry changes under the current objective may still rebuild targets.
    checkpoint["state_dict"][prefix + "_extra_state"]["version"] = 10
    checkpoint["state_dict"][prefix + "interior_positions"] = torch.zeros(10, 3)
    checkpoint["state_dict"][prefix + "targets"].fill_(99.0)
    module._restore_potential_sampling(checkpoint)
    module.load_state_dict(checkpoint["state_dict"])
    assert model.field.item() == 55.0
    assert term.last_update == -1 and not term.frozen
    assert term.targets.count_nonzero() == 0
    assert (
        checkpoint["optimizer_states"] == [{"sentinel": 7}]
        and checkpoint["global_step"] == 1234
    )
    assert prefix + "interior_positions" not in checkpoint["state_dict"]
    # Identical surface geometry retains references; even a shape-preserving
    # coordinate change must reset them instead of silently mislabeling targets.
    checkpoint["state_dict"][prefix + "targets"].fill_(19.0)
    checkpoint["state_dict"][prefix + "_extra_state"]["last_update"] = 1000
    module._restore_potential_sampling(checkpoint)
    assert checkpoint["state_dict"][prefix + "targets"].eq(19).all()
    checkpoint["state_dict"][prefix + "positions"] = (
        checkpoint["state_dict"][prefix + "positions"].clone() + 1
    )
    module._restore_potential_sampling(checkpoint)
    assert checkpoint["state_dict"][prefix + "_extra_state"]["last_update"] == -1


def _formal_window_contract(chunk_size=64):
    document = contract()
    window = {
        'id': 'fe', 'line_ids': ['FeI_6302.4932'],
        'depth_count': 257, 'wavelength_chunk_size': chunk_size,
        'transfer_precision': 'float64',
        'wavelength_bounds_vacuum_angstrom': [6300., 6307.],
        'shell_height_bounds_megameter': [-1., 1.5],
    }
    document['configuration']['synthesis'] = {'windows': [deepcopy(window)]}
    # P3S construction contexts also record window options separately.
    document['synthesis_windows'] = {'fe': deepcopy(window)}
    return document


def test_resume_allows_spectral_batch_changes_without_mutating_archived_contract():
    previous = _formal_window_contract(64)
    current = _formal_window_contract(32)
    recorded_previous, recorded_current = deepcopy(previous), deepcopy(current)
    assert _scientific_contract(previous) == _scientific_contract(current)
    assert previous == recorded_previous
    assert current == recorded_current
    # Missing operational defaults also mean the same scientific model.
    current['configuration']['synthesis']['windows'][0].pop('wavelength_chunk_size')
    current['synthesis_windows']['fe'].pop('wavelength_chunk_size')
    assert _scientific_contract(previous) == _scientific_contract(current)


@pytest.mark.parametrize('location', ['configuration', 'construction'])
@pytest.mark.parametrize('key,value', [('jitter_fraction', 1.), ('transfer_precision', 'float32')])
def test_resume_allows_sampling_and_precision_controls_without_mutating_archived_contract(location, key, value):
    previous = _formal_window_contract()
    current = deepcopy(previous)
    window = (current['configuration']['synthesis']['windows'][0]
              if location == 'configuration' else current['synthesis_windows']['fe'])
    window[key] = value
    recorded = deepcopy(current)
    assert _scientific_contract(previous) == _scientific_contract(current)
    assert current == recorded


@pytest.mark.parametrize('key,value', [
    ('depth_count', 129),
    ('shell_height_bounds_megameter', [-1., 2.]),
    ('wavelength_bounds_vacuum_angstrom', [6300., 6308.]),
])
@pytest.mark.parametrize('location', ['configuration', 'construction'])
def test_resume_rejects_changed_formal_physics_and_numerics(key, value, location):
    previous = _formal_window_contract()
    current = deepcopy(previous)
    window = (current['configuration']['synthesis']['windows'][0]
              if location == 'configuration' else current['synthesis_windows']['fe'])
    window[key] = value
    assert _scientific_contract(previous) != _scientific_contract(current)


def test_missing_refinement_preserves_old_windows_but_enabling_changes_contract():
    previous = _formal_window_contract()
    current = deepcopy(previous)
    refinement = {"enabled": False, "fine_sample_count": 1, "uniform_weight_floor": 0.}
    current['configuration']['synthesis']['windows'][0]['coarse_to_fine'] = dict(refinement)
    current['synthesis_windows']['fe']['coarse_to_fine'] = dict(refinement)
    assert _scientific_contract(previous) == _scientific_contract(current)
    current['configuration']['synthesis']['windows'][0]['coarse_to_fine']['enabled'] = True
    assert _scientific_contract(previous) != _scientific_contract(current)
