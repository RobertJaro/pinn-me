from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from prom3theus.application import joint_assembly
from prom3theus.config import load_config
from prom3theus.config.schema import PotentialBoundaryConfig
from prom3theus.inversion.sampling import SphericalShellDomain


@pytest.mark.parametrize('equation', ['magnetic_divergence', 'induction'])
def test_builder_uses_only_stokes_source_footprint(monkeypatch, equation):
    config = load_config('configs/hmi_aia_dynamic.yaml')
    equations = config.physics.equations
    equations = replace(equations, **{field.name: replace(getattr(equations, field.name), enabled=False, weight=0.) for field in fields(equations)})
    equations = replace(equations, **{equation: replace(getattr(equations, equation), enabled=True, weight=1.)})
    config = replace(config, atmosphere_regularization=(), physics=replace(config.physics,
        equations=equations, magnetic_current_free_steps=0,
        potential_boundary=PotentialBoundaryConfig(enabled=True, grid_size=8, top_grid_size=2, side_horizontal_points=2, side_height_points=2)))
    streams = {name: SimpleNamespace(prepared=SimpleNamespace(descriptor=SimpleNamespace(observation_kind=kind)))
               for name, kind in [('photosphere', 'stokes'), ('corona', 'image')]}
    calls = []

    def bounds(selected, scene):
        calls.append(set(selected))
        width = .04 if len(selected) == 1 else .06
        result = SphericalShellDomain(0., (-width, width), (-width, width), (0., 2.), (0., 50.), 696e6).configuration()
        result.pop('height_Mm')
        return result

    monkeypatch.setattr(joint_assembly, '_joint_observation_bounds', bounds)
    def observed_times(selected, scene):
        assert set(selected) == {'photosphere'}
        return [.2, .7, 1.8]
    monkeypatch.setattr(joint_assembly, '_observation_times_hours', observed_times)
    result = joint_assembly._default_shared_terms_builder(config, streams,
        SimpleNamespace(scene_basis=torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])), {})
    assert calls == [{'photosphere', 'corona'}, {'photosphere'}]
    assert 'potential_boundary' in result.terms
    physics = result.terms['physics']
    potential = result.terms['potential_boundary']
    torch.testing.assert_close(potential.times, torch.tensor([0., .2, .7, 1.8, 2.]))
    assert not hasattr(physics, 'magnetic_normalization')

    # Refreshing or changing the reference must not rescale volume residuals.
    from prom3theus.rt import StratifiedAtmosphereModel
    model = StratifiedAtmosphereModel(
        shell_height_bounds_Mm=(50., -.1), time_dependent=True,
        scene_geometry_config={"solar_radius_m": 696e6, "scene_basis": torch.eye(3)},
        model_config={"type": "mlp", "dim": 10, "n_layers": 2,
                      "activation": "silu", "encoding_config": {"type": "identity"}},
    )
    sample = physics.sampling_domain.deterministic_grouped(2, 4)
    monkeypatch.setattr(physics, '_samples', lambda parameter: {
        'volume_position_m': sample['position_m'],
        'volume_time_hours': sample['time_hours'],
    })
    before = physics.evaluate(model).loss.detach()
    assert torch.isfinite(before) and before > 0
    potential.set_step(potential.options['start_step'])
    potential.prepare(model)
    assert potential.update_count == 1
    torch.testing.assert_close(physics.evaluate(model).loss.detach(), before, rtol=0, atol=0)
    potential.normalization.fill_(1e-6)
    torch.testing.assert_close(physics.evaluate(model).loss.detach(), before, rtol=0, atol=0)


@pytest.mark.parametrize('options', [
    {'time_samples': 3}, {'update_every_n_steps': 0}, {'freeze_step': 1001}, {'blend': 1.1},
    {'grid_size': 3}, {'side_height_points': 0}, {'source_height_megameter': -.1}, {'source_height_megameter': .1}, {'geometry': 'buried_plane_fft'}, {'start_step': 1.5},
])
def test_rejects_invalid_potential_schedule_or_grid(options):
    with pytest.raises((TypeError, ValueError)):
        PotentialBoundaryConfig(**options)
