import pytest
import torch

from prom3theus.core import FourierEncoding, MLPModel


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
