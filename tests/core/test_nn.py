import pytest
import torch

from prom3theus.core import FourierEncoding, MLPModel, SIRENModel


def test_fourier_encoding_is_deterministic_and_differentiable():
    encoding = FourierEncoding(
        2,
        num_frequencies=(1, 2),
        max_frequencies=(2.0, 4.0),
        include_input=True,
    )
    coordinates = torch.tensor([[0.25, -0.5]], requires_grad=True)
    features = encoding(coordinates)
    assert features.shape == (1, 8)
    features.square().sum().backward()
    assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"num_frequencies": 2.5}, TypeError, "must be integers"),
        ({"max_frequencies": float("inf")}, ValueError, "must be finite"),
        ({"min_frequency": float("nan")}, ValueError, "must be finite"),
        ({"include_input": "false"}, TypeError, "must be a boolean"),
    ],
)
def test_fourier_encoding_rejects_coercive_or_nonfinite_options(
    options, error, message
):
    with pytest.raises(error, match=message):
        FourierEncoding(2, **options)


def test_mlp_accepts_only_canonical_encoding_and_activation_names():
    model = MLPModel(
        3,
        4,
        dim=16,
        n_layers=3,
        encoding_config={"type": "identity"},
        activation="silu",
    )
    assert model(torch.zeros(5, 3)).shape == (5, 4)
    with pytest.raises(ValueError, match="expected 'fourier' or 'identity'"):
        MLPModel(2, 1, encoding_config={"type": "none"})
    with pytest.raises(ValueError, match="expected one of"):
        MLPModel(2, 1, activation="swish")
    with pytest.raises(TypeError, match="must be integers"):
        MLPModel(2, 1, dim=8.5)
    with pytest.raises(TypeError, match="must be a mapping"):
        MLPModel(2, 1, encoding_config=[])


def test_siren_uses_canonical_initialization_and_radial_bandwidth_weighting():
    model = SIRENModel(
        3,
        2,
        dim=16,
        n_layers=3,
        first_omega_0=30.0,
        hidden_omega_0=1.0,
        radial_weighting_config={
            "radial_dimension": 2,
            "radial_bounds": (-0.1, 5.0),
            "near_sun_weight": 1.0,
            "outer_weight": 0.1,
            "exponent": 1.0,
        },
    )
    assert model(torch.zeros(5, 3)).shape == (5, 2)
    assert model.in_layer.weight.abs().max().item() <= 1.0 / 3.0

    coordinates = torch.tensor([[2.0, -3.0, -0.1], [2.0, -3.0, 5.0]])
    weighted = model.coordinate_weighting(coordinates)
    torch.testing.assert_close(weighted[0], coordinates[0])
    torch.testing.assert_close(weighted[1, :2], 0.1 * coordinates[1, :2])
    expected_outer_radius = -0.1 + 5.1 * (0.1 + 0.9 / 2.0)
    torch.testing.assert_close(
        weighted[:, 2], torch.tensor([-0.1, expected_outer_radius])
    )


def test_siren_radial_weighting_is_differentiable():
    model = SIRENModel(
        3,
        1,
        dim=8,
        n_layers=2,
        radial_weighting_config={
            "radial_dimension": 2,
            "radial_bounds": (0.0, 1.0),
        },
    )
    coordinates = torch.tensor([[0.2, -0.1, 0.5]], requires_grad=True)
    model(coordinates).sum().backward()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()


def test_integral_radial_warp_is_monotonic_with_envelope_derivative():
    model = SIRENModel(
        3,
        1,
        dim=8,
        radial_weighting_config={
            "radial_dimension": 2,
            "radial_bounds": (-0.1, 5.0),
            "near_sun_weight": 1.0,
            "outer_weight": 0.1,
            "exponent": 2.0,
        },
    )
    weighting = model.coordinate_weighting
    radius = torch.linspace(-0.1, 5.0, 51, requires_grad=True)
    coordinates = torch.stack(
        (torch.zeros_like(radius), torch.zeros_like(radius), radius), dim=-1
    )
    warped = weighting.radial_coordinate(coordinates)
    derivative = torch.autograd.grad(warped.sum(), radius)[0]

    assert torch.all(warped[1:] > warped[:-1])
    torch.testing.assert_close(derivative, weighting.envelope(coordinates))
