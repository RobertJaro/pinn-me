from dataclasses import replace

import pytest
import torch

from prom3theus.application.configuration import atmosphere_options
from prom3theus.config import load_config
from prom3theus.rt import StratifiedAtmosphereModel


@pytest.mark.parametrize('configuration', ['configs/hmi_aia_dynamic.yaml', 'configs/hmi_lte_dynamic.yaml'])
def test_dynamic_configs_have_uniform_spatial_metric_and_independent_time(configuration):
    atmosphere = load_config(configuration).atmosphere
    atmosphere = replace(atmosphere, network=replace(atmosphere.network, hidden_dimension=8, hidden_layers=2))
    options = atmosphere_options(atmosphere)
    model = StratifiedAtmosphereModel(
        **options,
        spatial_coordinate_center_mm=(2., -3.),
        spatial_coordinate_scale_mm=(90., 120.),
        time_coordinate_center_hours=5., time_coordinate_scale_hours=2.,
        scene_geometry_config={'solar_radius_m': 696e6, 'scene_basis': torch.eye(3)},
    )
    assert isinstance(model.network.coordinate_weighting, torch.nn.Identity)
    assert options['uniform_spatial_scaling'] is True
    assert options['model_config']['radial_weighting_config'] is None

    def encoded(point):
        # Common physical units (Mm) for x/y/height; time is in hours.
        coords = point[[0, 1, 3]][None]
        height = point[2].reshape(1, 1) * 1e6
        value = model._network_inputs_at_height(coords, height)[0, 0]
        return model.network.coordinate_weighting(value)

    for height in (0., 1., 10., 50.):
        point = torch.tensor([22., 7., height, 7.], dtype=torch.float64)
        jacobian = torch.autograd.functional.jacobian(encoded, point)
        torch.testing.assert_close(jacobian, torch.diag(torch.tensor([.1, .1, .1, .5], dtype=point.dtype)))
        torch.testing.assert_close(encoded(point), torch.tensor([2., 1., height / 10, 1.], dtype=point.dtype))
    # Model scaling must not rewrite the scene geometry used for ray construction.
    torch.testing.assert_close(model.spatial_coordinate_scale_mm, torch.tensor([90., 120.]))
    assert model.reference_metadata()['spatial_coordinates']['network_scale_mm'] == [10., 10.]
    assert model.reference_metadata()['time_coordinate_scale_hours'] == 2.

    # Model serialization preserves the new scaling flag and disabled warp.
    import io
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    coords = torch.tensor([[22., 7., 7.]])
    height = torch.tensor([[10e6]])
    torch.testing.assert_close(restored._network_inputs_at_height(coords, height), model._network_inputs_at_height(coords, height))
    assert isinstance(restored.network.coordinate_weighting, torch.nn.Identity)
