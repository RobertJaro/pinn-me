import math

import pytest
import torch

from pme.lte.atmosphere import (
    GeometricHeightModel,
    StratifiedAtmosphere,
    StratifiedAtmosphereModel,
)


def _atmosphere(depth=17, batch_shape=(2,), dtype=torch.float64):
    log_tau500 = torch.linspace(-5.0, 1.0, depth, dtype=dtype)
    shape = (*batch_shape, depth)
    temperature = torch.full(shape, 5700.0, dtype=dtype)
    velocity = torch.zeros(*shape, 3, dtype=dtype)
    microturbulence = torch.full(shape, 1000.0, dtype=dtype)
    magnetic_field = torch.zeros(*shape, 3, dtype=dtype)
    magnetic_field[..., 2] = 800.0
    return StratifiedAtmosphere(
        log_tau500=log_tau500,
        temperature=temperature,
        velocity_field=velocity,
        microturbulence=microturbulence,
        magnetic_field=magnetic_field,
    )


def test_stratified_atmosphere_preserves_batch_depth_and_vector_shapes():
    atmosphere = _atmosphere(depth=17, batch_shape=(2, 3))
    assert atmosphere.log_tau500.shape == (17,)
    assert atmosphere.temperature.shape == (2, 3, 17)
    assert atmosphere.velocity_field.shape == (2, 3, 17, 3)
    assert atmosphere.v_los.shape == (2, 3, 17)
    assert atmosphere.microturbulence.shape == (2, 3, 17)
    assert atmosphere.magnetic_field.shape == (2, 3, 17, 3)
    assert atmosphere.gas_pressure is None


def test_stratified_atmosphere_rejects_reversed_depth_grid():
    atmosphere = _atmosphere(depth=9)
    with pytest.raises(ValueError, match="increase|top.*bottom|depth"):
        StratifiedAtmosphere(
            log_tau500=atmosphere.log_tau500.flip(0),
            temperature=atmosphere.temperature,
            velocity_field=atmosphere.velocity_field,
            microturbulence=atmosphere.microturbulence,
            magnetic_field=atmosphere.magnetic_field,
        )


def test_direct_log_tau_model_is_continuous_and_has_no_height_mapping():
    grid = torch.linspace(-5.0, 1.0, 9)
    model = StratifiedAtmosphereModel(
        grid,
        coordinate_mode="log_tau",
        model_config={
            "dim": 12,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    coords = torch.tensor(((0.0, -0.2, 0.1), (0.0, 0.3, -0.1)))
    arbitrary_q = torch.tensor((-4.73, -2.11, 0.37))
    atmosphere = model(coords, arbitrary_q)
    paired = model.evaluate_points(coords, arbitrary_q.expand(2, -1))

    assert model.height_mapping is None
    assert atmosphere.geometric_height_m is None
    assert model.network_input_names == ("x", "y", "log_tau500")
    for name in (
        "temperature", "velocity_field", "magnetic_field",
        "microturbulence", "gas_pressure",
    ):
        torch.testing.assert_close(getattr(atmosphere, name), paired[name])
    atmosphere.temperature.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_geometric_height_mapping_starts_from_linear_base_and_learns_perturbation():
    depth_grid = torch.linspace(-5.0, 1.0, 11)
    coords = torch.tensor(((0.0, -0.3, 0.2), (0.0, 0.4, -0.1)))
    mapping = GeometricHeightModel(
        depth_grid,
        base_scale_m_per_log_tau=1.5e5,
        gauge_reference_coords=coords,
        model_config={
            "dim": 10,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    common_height = mapping(coords, depth_grid)

    assert common_height.shape == (2, 11)
    assert torch.isfinite(common_height).all()
    expected_height = -1.5e5 * depth_grid
    torch.testing.assert_close(common_height, expected_height.expand_as(common_height))
    gauge_height = mapping(coords, torch.tensor([0.0]))[:, 0]
    torch.testing.assert_close(gauge_height, torch.zeros_like(gauge_height))
    paired_depth = torch.tensor(
        ((-5.0, -3.7, -1.2, 1.0), (-5.0, -4.1, -0.4, 1.0)),
        requires_grad=True,
    )
    paired_height = mapping(coords, paired_depth)
    differentiated_height, differentiated_metric = mapping.height_and_metric(
        coords, paired_depth
    )
    torch.testing.assert_close(differentiated_height, paired_height)
    torch.testing.assert_close(
        mapping.metric_m_per_log_tau(coords, paired_depth), differentiated_metric
    )
    torch.testing.assert_close(
        differentiated_metric,
        differentiated_metric.new_full(differentiated_metric.shape, 1.5e5),
    )
    dz_dq = torch.autograd.grad(
        paired_height,
        paired_depth,
        grad_outputs=torch.ones_like(paired_height),
        create_graph=True,
    )[0]
    assert torch.isfinite(dz_dq).all()
    torch.testing.assert_close(differentiated_metric, -dz_dq, rtol=2e-6, atol=2e-2)

    (paired_height.square().mean() / 1.0e12).backward()
    gradients = [parameter.grad for parameter in mapping.network.parameters()]
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for gradient in gradients
    )
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_height_mapping_is_one_mlp_with_normalized_xyz_inputs():
    grid = torch.linspace(-5.0, 1.0, 17)
    mapping = GeometricHeightModel(
        grid,
        model_config={
            "dim": 8,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    coords = torch.tensor(((0.0, -0.3, 0.2), (0.0, 0.4, -0.1)))
    inputs = mapping._network_inputs_from_paired_q(
        coords, mapping._expanded_q(coords, grid)
    )

    assert mapping.network.out_layer.out_features == 1
    assert inputs.shape == (2, 17, 3)
    torch.testing.assert_close(inputs[..., 2].amin(), inputs.new_tensor(-1.0))
    torch.testing.assert_close(inputs[..., 2].amax(), inputs.new_tensor(1.0))
    metadata = mapping.metadata()
    assert metadata["base_scale_m_per_log_tau"] == 1.5e5
    assert metadata["perturbation_scale_m"] == 1.5e4
    assert metadata["perturbation_scale_fraction_of_base"] == 0.1
    assert metadata["type"].startswith("linear base mapping")


def test_atmosphere_is_composed_as_xy_to_height_to_physical_fields():
    depth_grid = torch.linspace(-5.0, 1.0, 9)
    model = StratifiedAtmosphereModel(
        depth_grid,
        height_input_scale_m=8.0e5,
        height_mapping_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            },
        },
        model_config={
            "dim": 10,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    coords = torch.tensor(((0.0, -0.2, 0.3), (0.0, 0.4, -0.1)))
    atmosphere = model(coords)
    mapped_height = model.height_mapping(coords, depth_grid)
    direct_fields = model.evaluate_at_height(coords, mapped_height)

    torch.testing.assert_close(atmosphere.geometric_height_m, mapped_height)
    assert model.network_input_names == ("x", "y", "z")
    for name in (
        "temperature", "velocity_field", "magnetic_field", "microturbulence", "gas_pressure"
    ):
        torch.testing.assert_close(getattr(atmosphere, name), direct_fields[name])

    paired_depth = torch.tensor(
        ((-4.8, -2.4, 0.3), (-4.2, -1.1, 0.7))
    )
    paired_fields = model.evaluate_points(coords, paired_depth)
    paired_direct = model.evaluate_at_height(
        coords, paired_fields["geometric_height_m"]
    )
    for name, value in paired_direct.items():
        torch.testing.assert_close(paired_fields[name], value)

    objective = (
        atmosphere.temperature.mean() / 6000.0
        + torch.log(atmosphere.gas_pressure).mean()
        + atmosphere.v_los.square().mean() / 3000.0**2
    )
    objective.backward()
    for network in (model.height_mapping.network, model.network):
        gradients = [parameter.grad for parameter in network.parameters()]
        assert any(
            gradient is not None and torch.count_nonzero(gradient) > 0
            for gradient in gradients
        )
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in gradients
        )


def test_atmosphere_model_initialization_shapes_physical_values_and_gradients():
    torch.manual_seed(0)
    depth_grid = torch.linspace(-5.0, 1.0, 19)
    model = StratifiedAtmosphereModel(depth_grid)
    coords = torch.tensor(
        [[0.0, -0.1, 0.2], [0.4, 0.3, -0.2], [-0.5, 0.1, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    atmosphere = model(coords)

    assert torch.count_nonzero(model.network.out_layer.weight) > 0
    assert torch.count_nonzero(model.network.out_layer.bias) > 0
    assert model.reference_metadata()["name"] == "random coordinate-network initialization"
    assert atmosphere.temperature.shape == (3, 19)
    assert atmosphere.velocity_field.shape == (3, 19, 3)
    assert atmosphere.v_los.shape == (3, 19)
    assert atmosphere.microturbulence.shape == (3, 19)
    assert atmosphere.gas_pressure.shape == (3, 19)
    assert atmosphere.magnetic_field.shape == (3, 19, 3)
    assert torch.isfinite(atmosphere.temperature).all()
    assert torch.all(atmosphere.temperature > 0)
    assert torch.all(
        (atmosphere.microturbulence > 10**1.0)
        & (atmosphere.microturbulence < 10**4.0)
    )
    assert torch.all(atmosphere.gas_pressure > 0)
    assert torch.count_nonzero(atmosphere.magnetic_field) > 0
    assert torch.count_nonzero(atmosphere.v_los) > 0
    assert torch.count_nonzero(atmosphere.velocity_field) > 0
    # Ordinary framework initialization plus explicit physical output scales
    # keep the initial vectors reasonable without zeroing any output bias.
    assert atmosphere.velocity_field.abs().max() < 1_000.0
    assert atmosphere.magnetic_field.abs().max() < 500.0
    assert torch.var(atmosphere.temperature) > 0

    objective = (
        atmosphere.temperature.mean() / 6000.0
        + atmosphere.v_los.square().mean() / 3000.0**2
        + atmosphere.microturbulence.mean() / 1000.0
        + torch.log(atmosphere.gas_pressure).mean()
        + atmosphere.magnetic_field.square().mean() / 2000.0**2
    )
    objective.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)


def test_atmosphere_model_is_continuous_coordinate_field_not_fixed_depth_nodes():
    base_grid = torch.linspace(-5.0, 1.0, 25)
    model = StratifiedAtmosphereModel(
        base_grid,
        model_config={
            "dim": 12,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    coords = torch.tensor([[0.0, -0.2, 0.3], [0.1, 0.4, -0.1]])
    arbitrary_grid = torch.tensor(
        [-5.0, -4.61, -3.83, -2.92, -1.77, -0.64, 0.18, 0.73, 1.0]
    )
    atmosphere = model(coords, log_tau500=arbitrary_grid)

    assert atmosphere.temperature.shape == (2, arbitrary_grid.numel())
    torch.testing.assert_close(atmosphere.log_tau500, arbitrary_grid)
    assert torch.var(atmosphere.temperature) > 0
    # Changing the number and position of samples only changes evaluation;
    # there is no trainable parameter tensor indexed by the depth samples.
    denser = model(coords, log_tau500=torch.linspace(-5.0, 1.0, 41))
    assert denser.depth == 41
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


def test_atmosphere_is_differentiable_in_x_y_and_tau_without_regularization():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 25),
        model_config={
            "dim": 16,
            "n_layers": 3,
            "activation": "silu",
            "encoding_config": {
                "type": "fourier",
                "num_frequencies": [3, 3, 3],
                "max_frequencies": [4, 4, 3],
            },
        },
    )
    coords = torch.tensor(
        [[0.0, -0.17, 0.23], [0.1, 0.31, -0.29]], requires_grad=True
    )
    depth = torch.tensor(
        [-5.0, -4.37, -3.12, -1.84, -0.41, 0.36, 1.0],
        requires_grad=True,
    )
    atmosphere = model(coords, log_tau500=depth)
    objective = (
        atmosphere.temperature.mean() / 6000.0
        + atmosphere.v_los.mean() / 3000.0
        + atmosphere.magnetic_field.mean() / 2000.0
        + atmosphere.microturbulence.mean() / 1000.0
    )
    objective.backward()

    assert coords.grad is not None and torch.isfinite(coords.grad).all()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
    assert torch.count_nonzero(coords.grad[..., 1:]) > 0
    torch.testing.assert_close(coords.grad[..., 0], torch.zeros_like(coords.grad[..., 0]))
    assert torch.count_nonzero(depth.grad) > 0


def test_paired_depth_evaluation_returns_pointwise_coordinate_derivatives():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 13),
        model_config={
            "dim": 12,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    coords = torch.tensor(((0.0, -0.2, 0.3), (0.0, 0.4, -0.1)))
    paired_depth = torch.tensor(
        ((-4.7, -2.1, 0.4), (-4.2, -1.3, 0.8)), requires_grad=True
    )
    fields = model.evaluate_points(coords, paired_depth)
    log_pressure = torch.log(fields["gas_pressure"])
    derivative = torch.autograd.grad(
        log_pressure,
        paired_depth,
        grad_outputs=torch.ones_like(log_pressure),
        create_graph=True,
    )[0]
    assert derivative.shape == paired_depth.shape
    assert torch.isfinite(derivative).all()
    assert torch.count_nonzero(derivative) == derivative.numel()


def test_static_atmosphere_is_invariant_to_degenerate_scan_time_coordinate():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 9),
        model_config={
            "dim": 8,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    first = model(torch.tensor(((0.0, 0.2, -0.1),)))
    later = model(torch.tensor(((3.0, 0.2, -0.1),)))
    for name in (
        "temperature", "velocity_field", "magnetic_field", "microturbulence", "gas_pressure",
        "geometric_height_m",
    ):
        torch.testing.assert_close(getattr(first, name), getattr(later, name))
    assert model.reference_metadata()["network_inputs"] == ["x", "y", "z"]


def test_atmosphere_rejects_nonsmooth_relu_activation():
    with pytest.raises(ValueError, match="smooth activation"):
        StratifiedAtmosphereModel(
            torch.linspace(-5.0, 1.0, 9),
            model_config={"activation": "relu"},
        )


def test_physical_decoders_are_smoothly_bounded_by_stic_domain():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 9),
        temperature_log10_bounds=(3.4, 4.0),
        temperature_log_scale=0.2,
        velocity_scale_m_per_s=1_000.0,
        magnetic_scale_gauss=100.0,
        microturbulence_log10_bounds=(1.0, 4.0),
        gas_pressure_log10_bounds=(-1.5, 6.0),
        gas_pressure_log_scale=1.0,
        model_config={
            "dim": 8,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    raw = torch.tensor((4.0, 3.0, 2.0, -2.0, 3.0, 4.0, 2.0, 4.0, 2.0))
    fields = model._decode_raw(raw)

    temperature_log_center = 0.5 * math.log(10.0) * (3.4 + 4.0)
    temperature_log_half_range = 0.5 * math.log(10.0) * (4.0 - 3.4)
    expected_temperature = math.exp(
        temperature_log_center
        + temperature_log_half_range
        * math.tanh(4.0 * 0.2 / temperature_log_half_range)
    )
    torch.testing.assert_close(
        fields["temperature"], torch.tensor(expected_temperature)
    )
    assert 1.0 < torch.log10(fields["microturbulence"]) < 4.0
    pressure_log_center = 0.5 * math.log(10.0) * (-1.5 + 6.0)
    pressure_log_half_range = 0.5 * math.log(10.0) * (6.0 - (-1.5))
    expected_pressure = math.exp(
        pressure_log_center
        + pressure_log_half_range
        * math.tanh(2.0 / pressure_log_half_range)
    )
    torch.testing.assert_close(
        fields["gas_pressure"], torch.tensor(expected_pressure)
    )
    torch.testing.assert_close(
        fields["velocity_field"], torch.tensor((3000.0, 2000.0, -2000.0))
    )
    torch.testing.assert_close(
        fields["magnetic_field"], torch.tensor((300.0, 400.0, 200.0))
    )
    decoder = model.reference_metadata()["vector_decoder"]
    assert "unbounded component-wise linear" in decoder
    assert "randomly initialized" in decoder

    extreme = torch.zeros((2, 9), dtype=torch.float64)
    extreme[:, 0] = torch.tensor((-1.0e6, 1.0e6))
    extreme[:, 8] = torch.tensor((-1.0e6, 1.0e6))
    bounded = model._decode_raw(extreme)
    torch.testing.assert_close(
        torch.log10(bounded["temperature"]),
        torch.tensor((3.4, 4.0), dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-15,
    )
    torch.testing.assert_close(
        torch.log10(bounded["gas_pressure"]),
        torch.tensor((-1.5, 6.0), dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_deep_fourier_initialization_retains_spatial_vector_contrast():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 9),
        spatial_coordinate_center_mm=(0.0, 0.0),
        spatial_coordinate_scale_mm=(1.0, 1.0),
        model_config={
            "dim": 64,
            "n_layers": 8,
            "activation": "silu",
            "encoding_config": {
                "type": "fourier",
                "num_frequencies": (12, 12, 6),
                "max_frequencies": (32.0, 32.0, 8.0),
                "include_input": True,
            },
        },
    )
    axis = torch.linspace(-1.0, 1.0, 20)
    y_coord, x_coord = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack(
        (torch.zeros_like(x_coord), x_coord, y_coord), dim=-1
    ).reshape(-1, 3)
    atmosphere = model(coords)

    velocity_spatial_std = atmosphere.velocity_field.std(dim=0).median()
    magnetic_spatial_std = atmosphere.magnetic_field.std(dim=0).median()
    assert velocity_spatial_std > 1.0
    assert magnetic_spatial_std > 0.1
    assert "variance-preserving" in model.reference_metadata()[
        "network_initialization"
    ]


def test_velocity_decoder_is_linear_and_unbounded():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 9),
        velocity_scale_m_per_s=1_000.0,
        model_config={
            "dim": 8,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    raw = torch.zeros(9, requires_grad=True)
    velocity = model._decode_raw(raw)["velocity_field"]
    slope = torch.autograd.grad(velocity.sum(), raw)[0][1:4]
    torch.testing.assert_close(slope, torch.full((3,), 1_000.0))

    extreme = raw.detach().clone()
    extreme[1:4] = torch.tensor((1.0e6, -1.0e6, 1.0e6))
    decoded = model._decode_raw(extreme)["velocity_field"]
    torch.testing.assert_close(
        decoded, torch.tensor((1.0e9, -1.0e9, 1.0e9))
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("temperature", float("nan"), "temperature"),
        ("temperature", 0.0, "temperature"),
        ("microturbulence", -1.0, "microturbulence"),
        ("gas_pressure", 0.0, "gas_pressure"),
        ("velocity_field", float("inf"), "velocity_field"),
        ("magnetic_field", float("nan"), "magnetic_field"),
    ),
)
def test_stratified_atmosphere_rejects_nonphysical_values(field, value, message):
    depth = 3
    values = {
        "log_tau500": torch.linspace(-2.0, 0.0, depth),
        "temperature": torch.full((1, depth), 6000.0),
        "velocity_field": torch.zeros(1, depth, 3),
        "microturbulence": torch.full((1, depth), 1000.0),
        "magnetic_field": torch.zeros(1, depth, 3),
        "gas_pressure": torch.full((1, depth), 1000.0),
    }
    values[field].reshape(-1)[0] = value
    with pytest.raises(ValueError, match=message):
        StratifiedAtmosphere(**values)
