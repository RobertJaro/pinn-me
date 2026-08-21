"""Smooth neural representation of a depth-stratified LTE atmosphere.

The operational representation is one continuous coordinate network
``F(x, y, log_tau500)``.  The retained Hinode scan time is deliberately ignored
for a single-raster inversion because it is degenerate with scan position.
An experimental geometric-height composition remains available explicitly,
but is not part of the default Hinode inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn

from pme.model import MLPModel


def _initialize_smooth_coordinate_mlp(model: MLPModel, activation: str) -> None:
    """Initialize a deep smooth MLP without suppressing coordinate variance.

    PyTorch's default linear-layer initialization is too contractive when it is
    repeated through a deep SiLU/GELU coordinate network.  The network remains
    random, but its output is then dominated by the final bias and initially
    appears spatially uniform.  Use activation-aware, variance-preserving
    hidden weights and an ordinary Xavier readout.  Biases intentionally remain
    independent random variables; no physical output channel is zeroed.
    """

    activation = str(activation).lower()
    hidden_layers = (model.in_layer, *model.hidden_layers)
    with torch.no_grad():
        for layer in hidden_layers:
            if activation == "tanh":
                nn.init.xavier_uniform_(
                    layer.weight,
                    gain=nn.init.calculate_gain("tanh"),
                )
            else:
                # PyTorch has no dedicated SiLU/GELU gain.  The ReLU gain is
                # the standard variance-preserving approximation for these
                # smooth, non-saturating activations.
                nn.init.kaiming_uniform_(
                    layer.weight,
                    a=0.0,
                    nonlinearity="relu",
                )
            bound = 1.0 / math.sqrt(layer.in_features)
            layer.bias.uniform_(-bound, bound)

        nn.init.xavier_uniform_(model.out_layer.weight)
        output_bound = 1.0 / math.sqrt(model.out_layer.in_features)
        model.out_layer.bias.uniform_(-output_bound, output_bound)


def _as_float_tensor(value, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _coordinate_pair(value, *, name: str, positive: bool) -> torch.Tensor:
    tensor = _as_float_tensor(value, name=name)
    if tensor.shape != (2,):
        raise ValueError(f"{name} must contain [Solar-X, Solar-Y].")
    if positive and torch.any(tensor <= 0):
        raise ValueError(f"{name} values must be positive.")
    return tensor


@dataclass(frozen=True)
class StratifiedAtmosphere:
    """Physical atmosphere sampled on a common optical-depth grid.

    Parameters
    ----------
    log_tau500:
        One-dimensional ``log10(tau_500)`` grid, strictly increasing from the
        top of the atmosphere to the bottom, shape ``[depth]``.
    temperature:
        Temperature in kelvin, shape ``[..., depth]``.
    velocity_field:
        Velocity vector ``[v_x, v_y, v_z]`` in the observer's Stokes frame,
        in m/s and with shape ``[..., depth, 3]``. ``+v_z`` points toward the
        observer, so the positive-redshift velocity consumed by radiative
        transfer is ``v_los = -v_z``. The transverse components are not
        constrained by a single-view Stokes spectrum in the absence of
        additional dynamical physics.
    microturbulence:
        Microturbulent speed in m/s, shape ``[..., depth]``.
    magnetic_field:
        Magnetic vector ``[B_x, B_y, B_los]`` in the observer's Stokes frame,
        in gauss and with shape ``[..., depth, 3]``. ``+B_x`` defines zero
        magnetic azimuth (the Stokes ``+Q`` reference axis), and ``+B_y``
        defines increasing azimuth. Converting these transverse components to
        solar-image west/north requires an independently verified Hinode
        polarization-reference rotation that is intentionally not guessed
        here.
    gas_pressure:
        Gas pressure in pascal, shape ``[..., depth]``.  In an inversion this
        is predicted by the coordinate network and constrained by hydrostatic
        equilibrium; it is not solved iteratively inside radiative transfer.
    """

    log_tau500: torch.Tensor
    temperature: torch.Tensor
    velocity_field: torch.Tensor
    microturbulence: torch.Tensor
    magnetic_field: torch.Tensor
    gas_pressure: torch.Tensor | None = None
    geometric_height_m: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.log_tau500.ndim != 1:
            raise ValueError("log_tau500 must be one-dimensional.")
        if self.log_tau500.numel() < 2:
            raise ValueError("log_tau500 must contain at least two depth points.")
        if not torch.isfinite(self.log_tau500).all():
            raise ValueError("log_tau500 must contain only finite values.")
        if not torch.all(self.log_tau500[1:] > self.log_tau500[:-1]):
            raise ValueError("log_tau500 must increase strictly from top to bottom.")

        depth = self.log_tau500.numel()
        scalar_fields = {
            "temperature": self.temperature,
            "microturbulence": self.microturbulence,
        }
        if self.gas_pressure is not None:
            scalar_fields["gas_pressure"] = self.gas_pressure
        if self.geometric_height_m is not None:
            scalar_fields["geometric_height_m"] = self.geometric_height_m
        for name, value in scalar_fields.items():
            if value.shape[-1:] != (depth,):
                raise ValueError(
                    f"{name} must end in depth dimension {depth}; got {tuple(value.shape)}."
                )
            if value.shape[:-1] != self.temperature.shape[:-1]:
                raise ValueError(f"{name} and temperature must have matching batch dimensions.")
            if value.device != self.temperature.device:
                raise ValueError(f"{name} and temperature must be on the same device.")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values.")

        if torch.any(self.temperature <= 0):
            raise ValueError("temperature must be strictly positive.")
        if torch.any(self.microturbulence < 0):
            raise ValueError("microturbulence cannot be negative.")
        if self.gas_pressure is not None and torch.any(self.gas_pressure <= 0):
            raise ValueError("gas_pressure must be strictly positive.")

        expected_vector_shape = (*self.temperature.shape, 3)
        if self.velocity_field.shape != expected_vector_shape:
            raise ValueError(
                "velocity_field must have shape [..., depth, 3]; "
                f"expected {expected_vector_shape}, got {tuple(self.velocity_field.shape)}."
            )
        if self.velocity_field.device != self.temperature.device:
            raise ValueError("velocity_field and temperature must be on the same device.")
        if not torch.isfinite(self.velocity_field).all():
            raise ValueError("velocity_field must contain only finite values.")
        expected_magnetic_shape = expected_vector_shape
        if self.magnetic_field.shape != expected_magnetic_shape:
            raise ValueError(
                "magnetic_field must have shape [..., depth, 3]; "
                f"expected {expected_magnetic_shape}, got {tuple(self.magnetic_field.shape)}."
            )
        if self.magnetic_field.device != self.temperature.device:
            raise ValueError("magnetic_field and temperature must be on the same device.")
        if not torch.isfinite(self.magnetic_field).all():
            raise ValueError("magnetic_field must contain only finite values.")
        if self.log_tau500.device != self.temperature.device:
            raise ValueError("log_tau500 and atmospheric fields must be on the same device.")

    @property
    def depth(self) -> int:
        return int(self.log_tau500.numel())

    @property
    def batch_shape(self) -> torch.Size:
        return self.temperature.shape[:-1]

    @property
    def v_los(self) -> torch.Tensor:
        """Positive-redshift LOS speed used by the radiative-transfer solver."""

        return -self.velocity_field[..., 2]

class GeometricHeightModel(nn.Module):
    """Direct smooth mapping ``Z(x, y, log10(tau500)) -> z``.

    One ordinary coordinate MLP represents the complete mapping.  Its spatial
    inputs and optical-depth input are normalized to order-unity intervals and
    its scalar output is converted to metres by ``height_scale_m``.  The only
    gauge condition removes the mean height at one optical depth over a fixed
    FOV quadrature.  It fixes the unobservable global translation without
    flattening that optical-depth surface or restricting its corrugation.

    No hand-designed depth basis or monotonic transform is used.  The physical
    optical-depth equation constrains both the magnitude and sign of ``-dz/dq``.
    """

    def __init__(
        self,
        log_tau500,
        *,
        height_scale_m: float = 1_000_000.0,
        gauge_log_tau500: float = 0.0,
        gauge_reference_coords=None,
        spatial_coordinate_center_mm=(0.0, 0.0),
        spatial_coordinate_scale_mm=(1.0, 1.0),
        model_config: Mapping | None = None,
    ):
        super().__init__()
        grid = _as_float_tensor(log_tau500, name="log_tau500")
        if grid.ndim != 1 or grid.numel() < 2 or not torch.all(grid[1:] > grid[:-1]):
            raise ValueError("Geometric-height mapping requires an increasing depth grid.")
        if not float(height_scale_m) > 0:
            raise ValueError("height_scale_m must be positive.")
        gauge_log_tau500 = float(gauge_log_tau500)
        if not float(grid[0]) <= gauge_log_tau500 <= float(grid[-1]):
            raise ValueError("gauge_log_tau500 must lie inside the represented domain.")
        self.register_buffer("log_tau500", grid)
        self.register_buffer(
            "spatial_coordinate_center_mm",
            _coordinate_pair(
                spatial_coordinate_center_mm,
                name="spatial_coordinate_center_mm",
                positive=False,
            ),
        )
        self.register_buffer(
            "spatial_coordinate_scale_mm",
            _coordinate_pair(
                spatial_coordinate_scale_mm,
                name="spatial_coordinate_scale_mm",
                positive=True,
            ),
        )
        if gauge_reference_coords is None:
            center = self.spatial_coordinate_center_mm.detach().clone()
            gauge_coords = torch.cat((center.new_zeros(1), center)).reshape(1, 3)
            gauge_reference_type = "single spatial-center fallback"
        else:
            gauge_coords = _as_float_tensor(
                gauge_reference_coords,
                name="gauge_reference_coords",
            )
            if gauge_coords.ndim != 2 or gauge_coords.shape[1] != 3:
                raise ValueError("gauge_reference_coords must have shape [points, 3].")
            if gauge_coords.shape[0] < 1:
                raise ValueError("gauge_reference_coords must contain at least one point.")
            gauge_reference_type = "fixed valid-FOV equal-pixel quadrature"
        self.register_buffer("gauge_reference_coords", gauge_coords)
        self.height_scale_m = float(height_scale_m)
        self.gauge_log_tau500 = gauge_log_tau500
        self.gauge_reference_type = gauge_reference_type
        config = dict(model_config or {
            "dim": 64,
            "n_layers": 4,
            "activation": "swish",
            "encoding_config": {
                "type": "fourier",
                "num_frequencies": [16, 16, 16],
                "max_frequencies": [64, 64, 16],
                "include_input": True,
            },
        })
        model_type = config.pop("type", "mlp")
        if model_type != "mlp":
            raise ValueError(
                "GeometricHeightModel supports model type 'mlp' only."
            )
        self.network = MLPModel(
            in_dim=3,
            out_dim=1,
            **config,
        )
        _initialize_smooth_coordinate_mlp(
            self.network,
            activation=str(config.get("activation", "silu")),
        )

    @property
    def top_log_tau500(self) -> torch.Tensor:
        return self.log_tau500[0]

    @property
    def bottom_log_tau500(self) -> torch.Tensor:
        return self.log_tau500[-1]

    def _normalized_xy(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim < 2 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [..., time, x, y].")
        normalized_xy = (
            coords[..., 1:] - self.spatial_coordinate_center_mm.to(coords)
        ) / self.spatial_coordinate_scale_mm.to(coords)
        return normalized_xy

    def _expanded_q(self, coords: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if q.ndim == 1:
            view_shape = (*([1] * len(coords.shape[:-1])), q.numel())
            return q.reshape(view_shape).expand(*coords.shape[:-1], q.numel())
        if q.shape[:-1] == coords.shape[:-1]:
            return q
        raise ValueError(
            "log_tau500 must be a common [depth] grid or paired "
            "coords.shape[:-1]+[points]."
        )

    def _network_inputs_from_paired_q(
        self,
        coords: torch.Tensor,
        paired_q: torch.Tensor,
    ) -> torch.Tensor:
        normalized_xy = self._normalized_xy(coords)
        points = paired_q.shape[-1]
        xy = normalized_xy.unsqueeze(-2).expand(
            *coords.shape[:-1], points, normalized_xy.shape[-1]
        )
        q_center = 0.5 * (self.top_log_tau500 + self.bottom_log_tau500)
        q_half_range = 0.5 * (self.bottom_log_tau500 - self.top_log_tau500)
        normalized_q = (paired_q - q_center.to(paired_q)) / q_half_range.to(paired_q)
        return torch.cat((xy, normalized_q[..., None]), dim=-1)

    def _uncentered_height_m(
        self,
        coords: torch.Tensor,
        paired_q: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self._network_inputs_from_paired_q(coords, paired_q)
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *paired_q.shape, 1
        )
        return self.height_scale_m * raw[..., 0]

    def _gauge_mean_m(self, reference: torch.Tensor) -> torch.Tensor:
        coords = self.gauge_reference_coords.to(reference)
        q = reference.new_full((*coords.shape[:-1], 1), self.gauge_log_tau500)
        return self._uncentered_height_m(coords, q).mean()

    def _validate_inputs(
        self,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        q = log_tau500.to(device=coords.device, dtype=coords.dtype)
        if not torch.isfinite(q).all() or torch.any(q < self.log_tau500[0]) or torch.any(
            q > self.log_tau500[-1]
        ):
            raise ValueError("log_tau500 lies outside the geometric-height domain.")
        return coords, self._expanded_q(coords, q)

    def height_and_metric(
        self,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``z`` and ``-dz/dlog10(tau500)`` from the direct MLP."""

        coords, paired_q = self._validate_inputs(coords, log_tau500)
        if not paired_q.requires_grad:
            paired_q = paired_q.detach().requires_grad_(True)
        height = self._uncentered_height_m(coords, paired_q)
        height = height - self._gauge_mean_m(height)
        derivative = torch.autograd.grad(
            height,
            paired_q,
            grad_outputs=torch.ones_like(height),
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        metric = -derivative
        return height, metric

    def metric_m_per_log_tau(
        self,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Return the learned metric ``-dz/dlog10(tau500)``."""

        return self.height_and_metric(
            coords, log_tau500, create_graph=create_graph
        )[1]

    def forward(self, coords: torch.Tensor, log_tau500: torch.Tensor) -> torch.Tensor:
        coords, paired_q = self._validate_inputs(coords, log_tau500)
        height = self._uncentered_height_m(coords, paired_q)
        return height - self._gauge_mean_m(height)

    def metadata(self) -> dict:
        return {
            "type": "direct coordinate MLP Z(x,y,log10(tau500))",
            "unit": "m",
            "orientation": "z increases upward; log_tau500 increases inward",
            "gauge": (
                f"mean_FOV[z(log_tau500={self.gauge_log_tau500})] = 0"
            ),
            "gauge_limitation": (
                "the mean-height gauge fixes only a global translation; spatial "
                "corrugation remains learned but is not a measured Wilson depression "
                "without sufficient horizontal physics or external information"
            ),
            "gauge_reference_type": self.gauge_reference_type,
            "gauge_reference_point_count": int(self.gauge_reference_coords.shape[0]),
            "gauge_log_tau500": self.gauge_log_tau500,
            "height_scale_m": self.height_scale_m,
            "optical_depth_transform": "linear configured interval -> [-1, 1]",
            "monotonicity": "constrained by the configured tau-mapping physics loss",
            "metric_api": "automatic differentiation of the direct coordinate MLP",
            "runtime_iterations": 0,
            "spatial_coordinates": {
                "public_input": "helioprojective Solar-X/Solar-Y in Mm",
                "network_center_mm": self.spatial_coordinate_center_mm.detach().cpu().tolist(),
                "network_scale_mm": self.spatial_coordinate_scale_mm.detach().cpu().tolist(),
                "network_transform": "(xy_mm - center_mm) / scale_mm",
            },
        }


class StratifiedAtmosphereModel(nn.Module):
    """Coordinate MLP producing a smooth stratified atmosphere.

    The MLP uses activation-aware variance-preserving random initialization:
    neither output weights nor output biases are zeroed, and no fixed
    temperature stratification or magnetic vector is added.
    Positive thermodynamic variables use smooth natural-log decoders bounded
    by the calibrated STiC lookup domain.
    Cartesian magnetic and velocity vectors use direct unit-scaled linear
    decoders. No output bias is zeroed and no hard clamp is used.
    """

    output_names = (
        "temperature", "v_x", "v_y", "v_z", "b_x", "b_y", "b_los",
        "microturbulence", "gas_pressure",
    )

    def __init__(
        self,
        log_tau500,
        temperature_log10_bounds=(3.4, 4.0),
        temperature_log_scale: float = 0.2,
        velocity_scale_m_per_s: float = 1_000.0,
        magnetic_scale_gauss: float = 100.0,
        microturbulence_log10_bounds=(1.0, 4.0),
        gas_pressure_log10_bounds=(-1.5, 6.0),
        gas_pressure_log_scale: float = 1.0,
        spatial_coordinate_center_mm=(0.0, 0.0),
        spatial_coordinate_scale_mm=(1.0, 1.0),
        coordinate_mode: str = "geometric_height",
        height_input_scale_m: float = 1_000_000.0,
        height_mapping_config: Mapping | None = None,
        model_config: Mapping | None = None,
    ):
        super().__init__()
        log_tau500 = _as_float_tensor(log_tau500, name="log_tau500")
        if log_tau500.ndim != 1 or log_tau500.numel() < 2:
            raise ValueError("log_tau500 must be a one-dimensional grid with at least two points.")
        if not torch.all(log_tau500[1:] > log_tau500[:-1]):
            raise ValueError("log_tau500 must increase strictly from top to bottom.")

        temperature_log10_bounds = tuple(map(float, temperature_log10_bounds))
        microturbulence_log10_bounds = tuple(map(float, microturbulence_log10_bounds))
        gas_pressure_log10_bounds = tuple(map(float, gas_pressure_log10_bounds))
        if len(temperature_log10_bounds) != 2 or not (
            temperature_log10_bounds[0] < temperature_log10_bounds[1]
        ):
            raise ValueError("temperature_log10_bounds must be an increasing pair.")
        if len(microturbulence_log10_bounds) != 2 or not (
            0 <= microturbulence_log10_bounds[0] < microturbulence_log10_bounds[1]
        ):
            raise ValueError(
                "microturbulence_log10_bounds must be a non-negative increasing pair."
            )
        if len(gas_pressure_log10_bounds) != 2 or not (
            gas_pressure_log10_bounds[0] < gas_pressure_log10_bounds[1]
        ):
            raise ValueError("gas_pressure_log10_bounds must be an increasing pair.")
        scales = {
            "temperature_log_scale": temperature_log_scale,
            "velocity_scale_m_per_s": velocity_scale_m_per_s,
            "magnetic_scale_gauss": magnetic_scale_gauss,
            "gas_pressure_log_scale": gas_pressure_log_scale,
        }
        if any(float(value) <= 0 for value in scales.values()):
            raise ValueError(f"Atmosphere physical scales must be positive; got {scales}.")

        self.register_buffer("log_tau500", log_tau500)
        self.temperature_log10_bounds = temperature_log10_bounds
        self.temperature_log_scale = float(temperature_log_scale)
        self.velocity_scale_m_per_s = float(velocity_scale_m_per_s)
        self.magnetic_scale_gauss = float(magnetic_scale_gauss)
        self.microturbulence_log10_bounds = microturbulence_log10_bounds
        self.gas_pressure_log10_bounds = gas_pressure_log10_bounds
        self.gas_pressure_log_scale = float(gas_pressure_log_scale)
        self.register_buffer(
            "spatial_coordinate_center_mm",
            _coordinate_pair(
                spatial_coordinate_center_mm,
                name="spatial_coordinate_center_mm",
                positive=False,
            ),
        )
        self.register_buffer(
            "spatial_coordinate_scale_mm",
            _coordinate_pair(
                spatial_coordinate_scale_mm,
                name="spatial_coordinate_scale_mm",
                positive=True,
            ),
        )
        coordinate_mode = str(coordinate_mode).lower()
        if coordinate_mode not in ("log_tau", "geometric_height"):
            raise ValueError(
                "coordinate_mode must be 'log_tau' or 'geometric_height'."
            )
        self.coordinate_mode = coordinate_mode
        if not float(height_input_scale_m) > 0:
            raise ValueError("height_input_scale_m must be positive.")
        self.height_input_scale_m = float(height_input_scale_m)
        height_config = dict(height_mapping_config or {})
        duplicated_coordinate_options = {
            "spatial_coordinate_center_mm",
            "spatial_coordinate_scale_mm",
        } & set(height_config)
        if duplicated_coordinate_options:
            raise ValueError(
                "Configure the shared physical-coordinate transform on "
                "StratifiedAtmosphereModel, not inside height_mapping_config: "
                f"{sorted(duplicated_coordinate_options)}."
            )
        if self.coordinate_mode == "log_tau":
            if height_config:
                raise ValueError(
                    "height_mapping_config is not used when coordinate_mode='log_tau'."
                )
            self.height_mapping = None
            self.network_input_names = ("x", "y", "log_tau500")
        else:
            self.height_mapping = GeometricHeightModel(
                log_tau500,
                spatial_coordinate_center_mm=(
                    self.spatial_coordinate_center_mm.detach().cpu().tolist()
                ),
                spatial_coordinate_scale_mm=(
                    self.spatial_coordinate_scale_mm.detach().cpu().tolist()
                ),
                **height_config,
            )
            self.network_input_names = ("x", "y", "z")

        config = dict(model_config or {})
        model_type = config.pop("type", "mlp")
        if model_type != "mlp":
            raise ValueError("StratifiedAtmosphereModel currently supports model type 'mlp' only.")
        activation = str(config.get("activation", "silu")).lower()
        smooth_activations = {"silu", "swish", "gelu", "tanh"}
        if activation not in smooth_activations:
            raise ValueError(
                "The LTE atmosphere must use a smooth activation; choose one of "
                f"{sorted(smooth_activations)}, got {activation!r}."
            )
        self.network = MLPModel(
            in_dim=len(self.network_input_names),
            out_dim=len(self.output_names),
            **config,
        )
        _initialize_smooth_coordinate_mlp(self.network, activation=activation)

    def _evaluation_grid(self, log_tau500=None) -> torch.Tensor:
        """Return a validated evaluation grid in the model's dtype/device.

        ``self.log_tau500`` defines the represented optical-depth interval and
        the reference quadrature grid.  It does not define trainable nodes: the
        coordinate MLP may be evaluated at any strictly increasing points
        inside the same interval.
        """

        if log_tau500 is None:
            return self.log_tau500
        grid = torch.as_tensor(
            log_tau500,
            dtype=self.log_tau500.dtype,
            device=self.log_tau500.device,
        )
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError("log_tau500 must be a one-dimensional grid with at least two points.")
        if not torch.isfinite(grid).all():
            raise ValueError("log_tau500 must contain only finite values.")
        if not torch.all(grid[1:] > grid[:-1]):
            raise ValueError("log_tau500 must increase strictly from top to bottom.")
        if grid[0] < self.log_tau500[0] or grid[-1] > self.log_tau500[-1]:
            raise ValueError(
                "Evaluation log_tau500 must remain inside the configured atmosphere interval "
                f"[{float(self.log_tau500[0])}, {float(self.log_tau500[-1])}]."
            )
        return grid

    def _network_inputs(
        self,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coords.ndim < 1 or coords.shape[-1] != 3:
            raise ValueError(f"coords must end in [time, x, y]; got {tuple(coords.shape)}.")
        if not torch.is_floating_point(coords):
            coords = coords.float()
        if self.coordinate_mode == "geometric_height":
            geometric_height = self.height_mapping(coords, log_tau500)
            return self._network_inputs_at_height(coords, geometric_height), geometric_height
        normalized_xy = (
            coords[..., 1:] - self.spatial_coordinate_center_mm.to(coords)
        ) / self.spatial_coordinate_scale_mm.to(coords)
        if log_tau500.ndim == 1:
            paired_q = log_tau500.reshape(
                *([1] * len(coords.shape[:-1])), log_tau500.numel()
            ).expand(*coords.shape[:-1], log_tau500.numel())
        elif log_tau500.shape[:-1] == coords.shape[:-1]:
            paired_q = log_tau500
        else:
            raise ValueError(
                "log_tau500 must be a common depth grid or paired with coords."
            )
        q_center = 0.5 * (self.log_tau500[0] + self.log_tau500[-1])
        q_half_range = 0.5 * (self.log_tau500[-1] - self.log_tau500[0])
        normalized_q = (paired_q - q_center.to(paired_q)) / q_half_range.to(paired_q)
        spatial = normalized_xy.unsqueeze(-2).expand(
            *coords.shape[:-1], paired_q.shape[-1], 2
        )
        return torch.cat((spatial, normalized_q[..., None]), dim=-1), None

    def _network_inputs_at_height(
        self,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> torch.Tensor:
        if self.coordinate_mode != "geometric_height":
            raise RuntimeError(
                "evaluate_at_height is unavailable for a log_tau atmosphere."
            )
        normalized_xy = (
            coords[..., 1:] - self.spatial_coordinate_center_mm.to(coords)
        ) / self.spatial_coordinate_scale_mm.to(coords)
        represented_coords = normalized_xy
        if geometric_height_m.shape[:-1] != coords.shape[:-1]:
            raise ValueError("geometric_height_m must have shape coords.shape[:-1]+[points].")
        depth = geometric_height_m.shape[-1]
        spatial = represented_coords.unsqueeze(-2).expand(
            *coords.shape[:-1], depth, represented_coords.shape[-1]
        )
        normalized_height = geometric_height_m / self.height_input_scale_m
        return torch.cat((spatial, normalized_height[..., None]), dim=-1)

    def forward(self, coords: torch.Tensor, log_tau500=None) -> StratifiedAtmosphere:
        """Evaluate the continuous atmosphere at an arbitrary depth grid."""

        grid = self._evaluation_grid(log_tau500)
        inputs, geometric_height = self._network_inputs(coords, grid)
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )
        fields = self._decode_raw(raw)
        return StratifiedAtmosphere(
            log_tau500=grid,
            geometric_height_m=geometric_height,
            **fields,
        )

    def _decode_raw(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode network channels into physical quantities."""

        def bounded_exponential(
            value: torch.Tensor,
            log10_bounds: tuple[float, float],
            local_log_scale: float,
        ) -> torch.Tensor:
            """Map a raw value smoothly into fixed physical log bounds.

            The multiplier inside tanh is chosen so d(log(output))/d(raw) at
            raw=0 equals ``local_log_scale``. This retains well-conditioned
            natural-log gradients while guaranteeing lookup-valid outputs.
            """

            log_lower = math.log(10.0) * log10_bounds[0]
            log_upper = math.log(10.0) * log10_bounds[1]
            log_center = 0.5 * (log_lower + log_upper)
            log_half_range = 0.5 * (log_upper - log_lower)
            log_value = log_center + log_half_range * torch.tanh(
                value * (local_log_scale / log_half_range)
            )
            return torch.exp(log_value)

        temperature = bounded_exponential(
            raw[..., 0], self.temperature_log10_bounds, self.temperature_log_scale
        )
        velocity_field = self.velocity_scale_m_per_s * raw[..., 1:4]
        magnetic_field = self.magnetic_scale_gauss * raw[..., 4:7]
        micro_fraction = torch.sigmoid(raw[..., 7])
        log_microturbulence = self.microturbulence_log10_bounds[0] + micro_fraction * (
            self.microturbulence_log10_bounds[1] - self.microturbulence_log10_bounds[0]
        )
        microturbulence = torch.pow(raw.new_tensor(10.0), log_microturbulence)
        gas_pressure = bounded_exponential(
            raw[..., 8], self.gas_pressure_log10_bounds, self.gas_pressure_log_scale
        )
        return {
            "temperature": temperature,
            "velocity_field": velocity_field,
            "microturbulence": microturbulence,
            "magnetic_field": magnetic_field,
            "gas_pressure": gas_pressure,
        }

    def evaluate_points(
        self,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate paired per-sample depth coordinates for PINN residuals.

        Unlike :meth:`forward`, ``log_tau500`` has shape
        ``coords.shape[:-1] + [points]``. Each output therefore depends on its
        own independent depth coordinate, allowing one reverse-mode derivative
        to return the pointwise ``d/dlog_tau500`` values without summing over
        pixels that share a common quadrature grid.
        """

        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        log_tau500 = log_tau500.to(
            device=self.log_tau500.device, dtype=self.log_tau500.dtype
        )
        if coords.ndim < 2 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [..., 3].")
        if log_tau500.shape[:-1] != coords.shape[:-1]:
            raise ValueError(
                "Paired log_tau500 must have shape coords.shape[:-1] + [points]."
            )
        if not torch.isfinite(log_tau500).all():
            raise ValueError("Paired log_tau500 must be finite.")
        if torch.any(log_tau500 < self.log_tau500[0]) or torch.any(
            log_tau500 > self.log_tau500[-1]
        ):
            raise ValueError("Paired log_tau500 lies outside the represented interval.")
        inputs, geometric_height = self._network_inputs(coords, log_tau500)
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )
        result = self._decode_raw(raw)
        if geometric_height is not None:
            result["geometric_height_m"] = geometric_height
        return result

    def evaluate_at_height(
        self,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate ``F(x,y,z)`` at paired physical heights in metres."""

        if self.coordinate_mode != "geometric_height":
            raise RuntimeError(
                "evaluate_at_height is unavailable for a log_tau atmosphere."
            )
        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        height = geometric_height_m.to(device=coords.device, dtype=coords.dtype)
        inputs = self._network_inputs_at_height(coords, height)
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )
        return self._decode_raw(raw)

    def reference_metadata(self) -> dict:
        return {
            "name": "random coordinate-network initialization",
            "reference": None,
            "provenance_note": "no fixed atmosphere or magnetic seed is added",
            "network_initialization": (
                "activation-aware variance-preserving random hidden weights, "
                "Xavier random readout weights, and independently random biases"
            ),
            "log_tau500": self.log_tau500.detach().cpu().tolist(),
            "temperature_log10_bounds": list(self.temperature_log10_bounds),
            "temperature_log_scale": self.temperature_log_scale,
            "temperature_parameterization": (
                "exp(log_center + log_half_range*tanh("
                "temperature_log_scale*raw/log_half_range)); smooth and bounded"
            ),
            "microturbulence_log10_bounds": list(self.microturbulence_log10_bounds),
            "gas_pressure_log10_bounds": list(self.gas_pressure_log10_bounds),
            "gas_pressure_log_scale": self.gas_pressure_log_scale,
            "gas_pressure_parameterization": (
                "exp(log_center + log_half_range*tanh("
                "gas_pressure_log_scale*raw/log_half_range)); smooth, bounded, "
                "and constrained by physics"
            ),
            "velocity_scale_m_per_s": self.velocity_scale_m_per_s,
            "magnetic_scale_gauss": self.magnetic_scale_gauss,
            "vector_decoder": (
                "velocity and magnetic field are unbounded component-wise linear "
                "outputs; output weights and biases remain randomly initialized; "
                "both vectors always have three components"
            ),
            "velocity_observability": (
                "single-view LTE Stokes synthesis constrains only -v_z (positive "
                "redshift); v_x and v_y require additional dynamical physics"
            ),
            "coordinate_mode": self.coordinate_mode,
            "spatial_coordinates": {
                "public_input": "helioprojective Solar-X/Solar-Y in Mm",
                "network_center_mm": self.spatial_coordinate_center_mm.detach().cpu().tolist(),
                "network_scale_mm": self.spatial_coordinate_scale_mm.detach().cpu().tolist(),
                "network_transform": "(xy_mm - center_mm) / scale_mm",
            },
            "network_inputs": list(self.network_input_names),
            "depth_network_transform": (
                "log_tau500 mapped linearly from the configured interval to [-1, 1]"
                if self.coordinate_mode == "log_tau"
                else "z_normalized = z_m / height_input_scale_m"
            ),
            "height_input_scale_m": (
                self.height_input_scale_m
                if self.coordinate_mode == "geometric_height"
                else None
            ),
            "height_mapping": (
                self.height_mapping.metadata()
                if self.height_mapping is not None
                else None
            ),
        }
