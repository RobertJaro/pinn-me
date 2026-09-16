"""Small neural-network building blocks shared by inversion models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import math
from numbers import Integral, Real

import torch
from torch import nn


def _per_dimension(value, in_dim: int, name: str, cast):
    values = (
        list(value)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes))
        else [value] * in_dim
    )
    if len(values) != in_dim:
        raise ValueError(
            f"{name} must contain one value per input dimension ({in_dim}); "
            f"got {values}."
        )
    return [cast(item) for item in values]


def _frequency_count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("num_frequencies entries must be integers.")
    return int(value)


def _finite_frequency(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Fourier frequency bounds must be real numbers.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Fourier frequency bounds must be finite.")
    return result


class FourierEncoding(nn.Module):
    """Deterministic, coordinate-wise multiresolution Fourier features."""

    def __init__(
        self,
        in_dim: int,
        num_frequencies=None,
        max_frequencies=None,
        min_frequency=1.0,
        include_input: bool = True,
    ):
        super().__init__()
        if isinstance(in_dim, bool) or not isinstance(in_dim, Integral):
            raise TypeError("in_dim must be an integer.")
        in_dim = int(in_dim)
        if in_dim < 1:
            raise ValueError("in_dim must be positive.")
        if type(include_input) is not bool:
            raise TypeError("include_input must be a boolean.")
        if num_frequencies is None:
            num_frequencies = [8, *([64] * (in_dim - 1))] if in_dim == 4 else 64
        if max_frequencies is None:
            max_frequencies = [2, *([2048] * (in_dim - 1))] if in_dim == 4 else 2048

        counts = _per_dimension(
            num_frequencies, in_dim, "num_frequencies", _frequency_count
        )
        maxima = _per_dimension(
            max_frequencies, in_dim, "max_frequencies", _finite_frequency
        )
        minima = _per_dimension(
            min_frequency, in_dim, "min_frequency", _finite_frequency
        )
        if any(count < 0 for count in counts):
            raise ValueError("num_frequencies cannot be negative.")

        frequencies = []
        for count, minimum, maximum in zip(counts, minima, maxima):
            if count == 0:
                values = torch.empty(0, dtype=torch.float32)
            elif minimum <= 0 or maximum < minimum:
                raise ValueError(
                    "Fourier frequencies require 0 < min_frequency <= max_frequencies."
                )
            elif count == 1:
                values = torch.tensor([maximum], dtype=torch.float32)
            else:
                values = 2 ** torch.linspace(
                    torch.log2(torch.tensor(minimum)),
                    torch.log2(torch.tensor(maximum)),
                    count,
                    dtype=torch.float32,
                )
            frequencies.append(values)

        for index, values in enumerate(frequencies):
            self.register_buffer(f"frequencies_{index}", values, persistent=True)
        self.in_dim = int(in_dim)
        self.include_input = bool(include_input)
        self.out_dim = (in_dim if self.include_input else 0) + 2 * sum(counts)

    @property
    def frequencies(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            getattr(self, f"frequencies_{index}") for index in range(self.in_dim)
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected coordinates ending in {self.in_dim} values, "
                f"received shape {tuple(coordinates.shape)}."
            )
        features = [coordinates] if self.include_input else []
        for coordinate, frequencies in zip(
            coordinates.unbind(dim=-1), self.frequencies
        ):
            if frequencies.numel() == 0:
                continue
            phase = torch.pi * coordinate[..., None] * frequencies.to(coordinates)
            features.extend((torch.sin(phase), torch.cos(phase)))
        if not features:
            return coordinates.new_empty((*coordinates.shape[:-1], 0))
        return torch.cat(features, dim=-1)


class MLPModel(nn.Module):
    """Coordinate MLP with an explicit ``fourier`` or ``identity`` encoding."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dim: int = 256,
        n_layers: int = 6,
        encoding_config: Mapping | None = None,
        activation: str = "silu",
    ):
        super().__init__()
        dimensions = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "dim": dim,
            "n_layers": n_layers,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in dimensions.values()
        ):
            raise TypeError("MLP dimensions and layer count must be integers.")
        in_dim, out_dim, dim, n_layers = (
            int(dimensions[name]) for name in ("in_dim", "out_dim", "dim", "n_layers")
        )
        if in_dim < 1 or out_dim < 1 or dim < 1:
            raise ValueError("in_dim, out_dim, and dim must be positive.")
        if n_layers < 1:
            raise ValueError("n_layers must be at least one.")

        if encoding_config is not None and not isinstance(encoding_config, Mapping):
            raise TypeError("encoding_config must be a mapping or null.")
        config = deepcopy(dict(encoding_config or {}))
        encoding_type = str(config.pop("type", "fourier")).lower()
        if encoding_type == "fourier":
            self.encoding = FourierEncoding(in_dim, **config)
            encoded_dim = self.encoding.out_dim
        elif encoding_type == "identity":
            if config:
                raise TypeError(
                    f"Identity encoding does not accept options: {sorted(config)}"
                )
            self.encoding = nn.Identity()
            encoded_dim = in_dim
        else:
            raise ValueError(
                f"Unknown MLP encoding {encoding_type!r}; expected 'fourier' or 'identity'."
            )

        activations = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        if not isinstance(activation, str):
            raise TypeError("activation must be a string.")
        try:
            activation_class = activations[activation.lower()]
        except KeyError as error:
            raise ValueError(
                f"Unknown MLP activation {activation!r}; expected one of "
                f"{sorted(activations)}."
            ) from error

        self.in_layer = nn.Linear(encoded_dim, dim)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dim, dim) for _ in range(n_layers - 1)
        )
        self.activations = nn.ModuleList(activation_class() for _ in range(n_layers))
        self.out_layer = nn.Linear(dim, out_dim)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        value = self.activations[0](self.in_layer(self.encoding(coordinates)))
        for layer, activation in zip(self.hidden_layers, self.activations[1:]):
            value = activation(layer(value))
        return self.out_layer(value)


class Sine(nn.Module):
    """SIREN sine activation with an explicit angular frequency."""

    def __init__(self, omega_0: float = 1.0):
        super().__init__()
        omega_0 = _finite_frequency(omega_0)
        if omega_0 <= 0.0:
            raise ValueError("omega_0 must be positive.")
        self.omega_0 = omega_0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * value)


class RadialCoordinateWeighting(nn.Module):
    """Reduce coordinate bandwidth toward the outer shell.

    Non-radial coordinates are multiplied by a smooth envelope.  The radial
    coordinate uses the integral of that envelope, preserving a strictly
    monotonic coordinate whose local derivative, and therefore local SIREN
    bandwidth, is exactly the same envelope.
    """

    def __init__(
        self,
        in_dim: int,
        radial_dimension: int,
        radial_bounds: tuple[float, float],
        near_sun_weight: float = 1.0,
        outer_weight: float = 0.1,
        exponent: float = 1.0,
    ):
        super().__init__()
        if isinstance(in_dim, bool) or not isinstance(in_dim, Integral):
            raise TypeError("in_dim must be an integer.")
        if isinstance(radial_dimension, bool) or not isinstance(
            radial_dimension, Integral
        ):
            raise TypeError("radial_dimension must be an integer.")
        in_dim, radial_dimension = int(in_dim), int(radial_dimension)
        if in_dim < 1 or not 0 <= radial_dimension < in_dim:
            raise ValueError("radial_dimension must index an input coordinate.")
        if len(radial_bounds) != 2:
            raise ValueError("radial_bounds must contain (inner, outer).")
        inner, outer = (_finite_frequency(value) for value in radial_bounds)
        near = _finite_frequency(near_sun_weight)
        far = _finite_frequency(outer_weight)
        power = _finite_frequency(exponent)
        if not inner < outer:
            raise ValueError("radial_bounds must increase from inner to outer.")
        if near <= 0.0 or far <= 0.0 or far > near:
            raise ValueError(
                "Radial weights require 0 < outer_weight <= near_sun_weight."
            )
        if power <= 0.0:
            raise ValueError("Radial weighting exponent must be positive.")

        self.in_dim = in_dim
        self.radial_dimension = radial_dimension
        self.near_sun_weight = near
        self.outer_weight = far
        self.exponent = power
        self.register_buffer("radial_bounds", torch.tensor((inner, outer)))

    def envelope(self, coordinates: torch.Tensor) -> torch.Tensor:
        bounds = self.radial_bounds.to(coordinates)
        fraction = (
            (coordinates[..., self.radial_dimension] - bounds[0])
            / (bounds[1] - bounds[0])
        ).clamp(0.0, 1.0)
        return self.outer_weight + (self.near_sun_weight - self.outer_weight) * (
            1.0 - fraction
        ).pow(self.exponent)

    def radial_coordinate(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return the integral warp whose derivative is :meth:`envelope`."""

        bounds = self.radial_bounds.to(coordinates)
        inner, outer = bounds.unbind()
        radius = coordinates[..., self.radial_dimension]
        extent = outer - inner
        fraction = ((radius - inner) / extent).clamp(0.0, 1.0)
        difference = self.near_sun_weight - self.outer_weight
        integral = self.outer_weight * (radius - inner) + (
            difference
            * extent
            / (self.exponent + 1.0)
            * (1.0 - (1.0 - fraction).pow(self.exponent + 1.0))
        )
        warped = inner + integral

        # Continue linearly beyond the represented shell.  These branches meet
        # the in-domain primitive with matching value and first derivative.
        outer_warped = inner + extent * (
            self.outer_weight + difference / (self.exponent + 1.0)
        )
        warped = torch.where(
            radius < inner,
            inner + self.near_sun_weight * (radius - inner),
            warped,
        )
        return torch.where(
            radius > outer,
            outer_warped + self.outer_weight * (radius - outer),
            warped,
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected coordinates ending in {self.in_dim} values, "
                f"received shape {tuple(coordinates.shape)}."
            )
        scale = self.envelope(coordinates)[..., None]
        mask = torch.ones(
            self.in_dim, dtype=coordinates.dtype, device=coordinates.device
        )
        mask[self.radial_dimension] = 0.0
        weighted = coordinates * (1.0 + mask * (scale - 1.0))
        radial_delta = (
            self.radial_coordinate(coordinates)
            - coordinates[..., self.radial_dimension]
        )
        return weighted + radial_delta[..., None] * (1.0 - mask)


class SIRENModel(nn.Module):
    """Sinusoidal representation network with radius-dependent bandwidth."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dim: int = 256,
        n_layers: int = 6,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 1.0,
        radial_weighting_config: Mapping | None = None,
    ):
        super().__init__()
        dimensions = (in_dim, out_dim, dim, n_layers)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in dimensions
        ):
            raise TypeError("SIREN dimensions and layer count must be integers.")
        in_dim, out_dim, dim, n_layers = map(int, dimensions)
        if min(in_dim, out_dim, dim) < 1 or n_layers < 1:
            raise ValueError("SIREN dimensions and layer count must be positive.")
        first_omega_0 = _finite_frequency(first_omega_0)
        hidden_omega_0 = _finite_frequency(hidden_omega_0)
        if first_omega_0 <= 0.0 or hidden_omega_0 <= 0.0:
            raise ValueError("SIREN frequencies must be positive.")
        if radial_weighting_config is None:
            self.coordinate_weighting = nn.Identity()
        elif isinstance(radial_weighting_config, Mapping):
            self.coordinate_weighting = RadialCoordinateWeighting(
                in_dim=in_dim, **deepcopy(dict(radial_weighting_config))
            )
        else:
            raise TypeError("radial_weighting_config must be a mapping or null.")

        self.in_layer = nn.Linear(in_dim, dim)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dim, dim) for _ in range(n_layers - 1)
        )
        self.activations = nn.ModuleList(
            [Sine(first_omega_0), *[Sine(hidden_omega_0) for _ in range(n_layers - 1)]]
        )
        self.out_layer = nn.Linear(dim, out_dim)
        self._initialize(hidden_omega_0)

    def _initialize(self, hidden_omega_0: float) -> None:
        """Apply the initialization derived for SIREN networks."""

        with torch.no_grad():
            first_bound = 1.0 / self.in_layer.in_features
            self.in_layer.weight.uniform_(-first_bound, first_bound)
            self.in_layer.bias.uniform_(-first_bound, first_bound)
            for layer in (*self.hidden_layers, self.out_layer):
                bound = math.sqrt(6.0 / layer.in_features) / hidden_omega_0
                layer.weight.uniform_(-bound, bound)
                layer.bias.uniform_(-bound, bound)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        value = self.coordinate_weighting(coordinates)
        value = self.activations[0](self.in_layer(value))
        for layer, activation in zip(self.hidden_layers, self.activations[1:]):
            value = activation(layer(value))
        return self.out_layer(value)


__all__ = [
    "FourierEncoding",
    "MLPModel",
    "RadialCoordinateWeighting",
    "SIRENModel",
    "Sine",
]
