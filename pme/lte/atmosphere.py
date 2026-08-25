"""Smooth neural representation of a spherical LTE atmosphere.

The operational representation is a heliocentric Carrington atmosphere
``F(x, y, r - R_sun)``. Thermodynamics are learned as bounded logarithmic
perturbations of a radial reference stratification. Observer rays are sampled
inside a fixed spherical shell and optical depth is obtained by integrating
absolute opacity along those physical paths. The retained Hinode scan time is
deliberately ignored because it is degenerate with scan position in one raster.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn

from pme.lte.geometry import (
    RayTraceResult,
    chart_height_to_position_m,
    chart_to_direction,
    direction_to_chart_mm,
    intersect_sphere_near_side,
    intersect_sphere_near_side_from_local_point,
    position_to_chart_height,
    validate_scene_basis,
)
from pme.model import MLPModel


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0

# Pinned STiC FALC_82 reference reduced to 25 points over log(tau500)=-5..1.
# Heights follow dm=-rho dz and are shifted so tau500=1 is z=0; pressure is
# g times column mass. Source provenance/checksums live in the LTE bundle.
FALC_REFERENCE_LOG_TAU500 = tuple(-5.0 + 0.25 * index for index in range(25))
FALC_REFERENCE_HEIGHT_M = (
    1694040.720, 1482004.879, 1228647.068, 906105.577, 659652.144,
    566986.638, 515665.796, 473754.974, 434954.627, 397259.443,
    360316.540, 323389.396, 286344.876, 249055.839, 211220.854,
    173545.194, 135342.068, 96734.772, 59431.618, 26488.648,
    0.0, -20849.247, -37954.699, -53546.012, -68815.437,
)
FALC_REFERENCE_TEMPERATURE_K = (
    7456.570, 7023.570, 6524.910, 5735.543, 4707.653,
    4524.964, 4503.208, 4523.450, 4562.799, 4615.340,
    4673.397, 4736.631, 4804.665, 4878.092, 4960.151,
    5057.491, 5211.207, 5423.944, 5708.676, 6105.000,
    6558.360, 7089.503, 7636.435, 8148.910, 8659.479,
)
FALC_REFERENCE_GAS_PRESSURE_PA = (
    0.0903679136, 0.207198031, 0.769446443, 6.12251408, 42.8947702,
    98.7395337, 159.142071, 234.861013, 334.998783, 470.775912,
    654.134411, 905.246499, 1249.59409, 1721.82404, 2368.86264,
    3256.22303, 4466.93582, 6087.39481, 8096.31031, 10283.9793,
    12305.3506, 14025.0095, 15489.1232, 16847.1586, 18194.6094,
)
FALC_REFERENCE_MICROTURBULENCE_M_PER_S = (
    5999.580, 5173.763, 3945.847, 2221.887, 1193.338,
    902.770, 778.128, 691.655, 621.713, 553.325,
    528.791, 548.213, 586.271, 626.370, 756.351,
    889.794, 1045.084, 1203.432, 1355.503, 1484.954,
    1587.349, 1661.127, 1699.278, 1729.933, 1751.808,
)


def _linear_interpolate(
    nodes: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
    *,
    extrapolate: bool = False,
) -> torch.Tensor:
    """Piecewise-linear interpolation, differentiable inside each interval."""

    if not extrapolate:
        query = query.clamp(nodes[0], nodes[-1])
    upper = torch.searchsorted(nodes, query.detach().contiguous(), right=True).clamp(
        1, nodes.numel() - 1
    )
    lower = upper - 1
    x0, x1 = nodes[lower], nodes[upper]
    fraction = (query - x0) / (x1 - x0)
    return values[lower] + fraction * (values[upper] - values[lower])


class RadialReferenceAtmosphere(nn.Module):
    """One-dimensional atmosphere supplying a physical radial baseline."""

    def __init__(self, config: str | Mapping = "falc_82"):
        super().__init__()
        if isinstance(config, str):
            if config.lower() != "falc_82":
                raise ValueError("reference_atmosphere_config must be 'falc_82' or a mapping.")
            data = {
                "log_tau500": FALC_REFERENCE_LOG_TAU500,
                "height_m": FALC_REFERENCE_HEIGHT_M,
                "temperature_k": FALC_REFERENCE_TEMPERATURE_K,
                "gas_pressure_pa": FALC_REFERENCE_GAS_PRESSURE_PA,
                "microturbulence_m_per_s": FALC_REFERENCE_MICROTURBULENCE_M_PER_S,
            }
            self.name = "STiC FALC_82"
        else:
            data = dict(config)
            self.name = str(data.pop("name", "configured radial reference"))
        required = {
            "log_tau500", "height_m", "temperature_k",
            "gas_pressure_pa", "microturbulence_m_per_s",
        }
        unknown, missing = set(data) - required, required - set(data)
        if unknown or missing:
            raise TypeError(
                f"Invalid radial-reference fields; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}."
            )
        q = _as_float_tensor(data["log_tau500"], name="reference log_tau500")
        height = _as_float_tensor(data["height_m"], name="reference height_m")
        temperature = _as_float_tensor(data["temperature_k"], name="reference temperature_k")
        pressure = _as_float_tensor(data["gas_pressure_pa"], name="reference gas_pressure_pa")
        micro = _as_float_tensor(
            data["microturbulence_m_per_s"], name="reference microturbulence_m_per_s"
        )
        if not all(value.shape == q.shape for value in (height, temperature, pressure, micro)):
            raise ValueError("All radial-reference arrays must have the same shape.")
        if q.ndim != 1 or q.numel() < 2 or not torch.all(q[1:] > q[:-1]):
            raise ValueError("Reference log_tau500 must be a strictly increasing vector.")
        if not torch.all(height[1:] < height[:-1]):
            raise ValueError("Reference height must decrease as optical depth increases.")
        if torch.any(temperature <= 0) or torch.any(pressure <= 0) or torch.any(micro <= 0):
            raise ValueError("Reference thermodynamic quantities must be positive.")
        self.register_buffer("log_tau500", q)
        self.register_buffer("height_descending_m", height)
        self.register_buffer("height_ascending_m", height.flip(0))
        self.register_buffer("log_temperature_ascending", torch.log(temperature.flip(0)))
        self.register_buffer("log_pressure_ascending", torch.log(pressure.flip(0)))
        self.register_buffer("log_microturbulence_ascending", torch.log(micro.flip(0)))

    def height_from_log_tau(self, q: torch.Tensor) -> torch.Tensor:
        return _linear_interpolate(
            self.log_tau500.to(q), self.height_descending_m.to(q), q
        )

    def metric_from_log_tau(self, q: torch.Tensor) -> torch.Tensor:
        nodes = self.log_tau500.to(q)
        heights = self.height_descending_m.to(q)
        upper = torch.searchsorted(nodes, q.detach().contiguous(), right=True).clamp(
            1, nodes.numel() - 1
        )
        lower = upper - 1
        return -(heights[upper] - heights[lower]) / (nodes[upper] - nodes[lower])

    def logs_at_height(
        self, height_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = self.height_ascending_m.to(height_m)
        return tuple(
            _linear_interpolate(
                nodes, values.to(height_m), height_m, extrapolate=True
            )
            for values in (
                self.log_temperature_ascending,
                self.log_pressure_ascending,
                self.log_microturbulence_ascending,
            )
        )

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "coordinate": "radius minus solar radius in metres",
            "tau_one_height_m": 0.0,
            "height_bounds_m": [
                float(self.height_descending_m[0]),
                float(self.height_descending_m[-1]),
            ],
            "parameterization": "bounded log perturbations around the radial reference",
        }


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
        raise ValueError(f"{name} must contain [spatial-X, spatial-Y].")
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
        Velocity vector in m/s and with shape ``[..., depth, 3]`` in the
        Heliographic Carrington Cartesian frame when spherical geometry is enabled.
        The synthesis adapter projects this vector into the observer frame;
        positive-redshift velocity is then the negative observer-directed
        component. The transverse components are not constrained by a
        single-view Stokes spectrum without additional dynamical physics.
    microturbulence:
        Microturbulent speed in m/s, shape ``[..., depth]``.
    magnetic_field:
        Magnetic vector in gauss and with shape ``[..., depth, 3]``. It is in
        the Heliographic Carrington Cartesian frame when spherical geometry is
        enabled, and is projected to ``[B_Q, B_U, B_LOS]`` immediately before
        polarized synthesis.
    gas_pressure:
        Gas pressure in pascal, shape ``[..., depth]``.  In an inversion this
        is predicted by the coordinate network and constrained by magnetohydrostatic
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

class StratifiedAtmosphereModel(nn.Module):
    """Coordinate MLP producing a smooth stratified atmosphere.

    The MLP uses activation-aware variance-preserving random initialization:
    neither output weights nor output biases are zeroed, and no fixed
    temperature stratification or magnetic vector is added.
    Positive thermodynamic variables use smooth natural-log decoders bounded
    by the calibrated STiC lookup domain.
    Cartesian magnetic vectors use a direct unit-scaled linear decoder.
    Velocity components use a smooth arctangent bound with the configured
    local linear scale, preventing invalid relativistic Doppler factors while
    retaining nonzero gradients outside the ordinary photospheric range.
    """

    output_names = (
        "temperature", "v_x", "v_y", "v_z", "b_x", "b_y", "b_z",
        "microturbulence", "gas_pressure",
    )

    def __init__(
        self,
        log_tau500,
        temperature_log10_bounds=(3.4, 4.0),
        temperature_log_scale: float = 0.2,
        velocity_scale_m_per_s: float = 1_000.0,
        velocity_max_m_per_s: float = 100_000.0,
        magnetic_scale_gauss: float = 100.0,
        microturbulence_log10_bounds=(1.0, 4.0),
        gas_pressure_log10_bounds=(-1.5, 6.0),
        gas_pressure_log_scale: float = 1.0,
        spatial_coordinate_center_mm=(0.0, 0.0),
        spatial_coordinate_scale_mm=(1.0, 1.0),
        height_input_scale_m: float = 1_000_000.0,
        shell_height_bounds_Mm: tuple[float, float] | None = None,
        tangent_margin_m: float = 1_000.0,
        reference_atmosphere_config: str | Mapping | None = "falc_82",
        microturbulence_log_scale: float = 0.2,
        scene_geometry_config: Mapping | None = None,
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
            "velocity_max_m_per_s": velocity_max_m_per_s,
            "magnetic_scale_gauss": magnetic_scale_gauss,
            "gas_pressure_log_scale": gas_pressure_log_scale,
            "microturbulence_log_scale": microturbulence_log_scale,
        }
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in scales.values()):
            raise ValueError(f"Atmosphere physical scales must be positive; got {scales}.")
        if math.sqrt(3.0) * float(velocity_max_m_per_s) >= SPEED_OF_LIGHT_M_PER_S:
            raise ValueError(
                "sqrt(3)*velocity_max_m_per_s must remain below the speed of light "
                "so every projected velocity is strictly subluminal."
            )

        self.register_buffer("log_tau500", log_tau500)
        self.temperature_log10_bounds = temperature_log10_bounds
        self.temperature_log_scale = float(temperature_log_scale)
        self.velocity_scale_m_per_s = float(velocity_scale_m_per_s)
        self.velocity_max_m_per_s = float(velocity_max_m_per_s)
        self.magnetic_scale_gauss = float(magnetic_scale_gauss)
        self.microturbulence_log10_bounds = microturbulence_log10_bounds
        self.gas_pressure_log10_bounds = gas_pressure_log10_bounds
        self.gas_pressure_log_scale = float(gas_pressure_log_scale)
        self.microturbulence_log_scale = float(microturbulence_log_scale)
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
        if not float(height_input_scale_m) > 0:
            raise ValueError("height_input_scale_m must be positive.")
        self.height_input_scale_m = float(height_input_scale_m)
        if reference_atmosphere_config is None:
            raise ValueError("The spherical LTE atmosphere requires a radial reference.")
        self.reference_atmosphere = RadialReferenceAtmosphere(
            reference_atmosphere_config
        )
        tangent_margin_m = float(tangent_margin_m)
        if not math.isfinite(tangent_margin_m) or tangent_margin_m <= 0:
            raise ValueError("tangent_margin_m must be finite and positive.")
        self.tangent_margin_m = tangent_margin_m
        if scene_geometry_config is None:
            raise ValueError("The spherical LTE atmosphere requires scene_geometry_config.")
        scene_config = dict(scene_geometry_config)
        unknown_scene = set(scene_config) - {"solar_radius_m", "scene_basis"}
        if unknown_scene:
            raise TypeError(f"Unknown scene-geometry options: {sorted(unknown_scene)}")
        solar_radius_m = float(scene_config["solar_radius_m"])
        if not math.isfinite(solar_radius_m) or solar_radius_m <= 0:
            raise ValueError("scene_geometry_config.solar_radius_m must be positive.")
        self.register_buffer(
            "solar_radius_m", log_tau500.new_tensor(solar_radius_m)
        )
        self.solar_radius_value_m = solar_radius_m
        reference_endpoint_heights = self.reference_atmosphere.height_from_log_tau(
            self.log_tau500[[0, -1]]
        )
        reference_height_bounds_Mm = tuple(
            float(height) / 1.0e6 for height in reference_endpoint_heights
        )
        if shell_height_bounds_Mm is None:
            shell_height_bounds_Mm = reference_height_bounds_Mm
        shell_height_bounds_Mm = tuple(map(float, shell_height_bounds_Mm))
        if len(shell_height_bounds_Mm) != 2 or not (
            math.isfinite(shell_height_bounds_Mm[0])
            and math.isfinite(shell_height_bounds_Mm[1])
            and shell_height_bounds_Mm[0] > shell_height_bounds_Mm[1]
        ):
            raise ValueError(
                "shell_height_bounds_Mm must be [outer_height, inner_height] "
                "as offsets from R_sun in Mm, with outer_height > inner_height."
            )
        if self.log_tau500[0] < 0 < self.log_tau500[-1] and not (
            shell_height_bounds_Mm[0] > 0 > shell_height_bounds_Mm[1]
        ):
            raise ValueError(
                "A shell whose depth grid crosses log_tau500=0 must have a positive "
                "outer height and negative inner height."
            )
        self.shell_height_bounds_Mm = shell_height_bounds_Mm
        reference_outer_m, reference_inner_m = (
            float(value) * 1.0e6 for value in reference_height_bounds_Mm
        )
        self.outer_height_scale = (
            shell_height_bounds_Mm[0] * 1.0e6 / reference_outer_m
        )
        self.inner_height_scale = (
            shell_height_bounds_Mm[1] * 1.0e6 / reference_inner_m
        )
        self.scene_basis_values = tuple(
            tuple(float(component) for component in row)
            for row in scene_config["scene_basis"]
        )
        self.register_buffer(
            "scene_basis", validate_scene_basis(scene_config["scene_basis"])
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
        # Exact initial radial thermodynamics; velocity and magnetic rows retain
        # random readouts so polarized gradients do not start at a symmetry point.
        with torch.no_grad():
            for index in (0, 7, 8):
                self.network.out_layer.weight[index].zero_()
                self.network.out_layer.bias[index].zero_()

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
        geometric_height = self.depth_to_height(log_tau500, coords.shape[:-1])
        return self._network_inputs_at_height(coords, geometric_height), geometric_height

    def _network_inputs_at_height(
        self,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> torch.Tensor:
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

    def depth_to_height(
        self,
        depth_coordinate: torch.Tensor,
        leading_shape=(),
    ) -> torch.Tensor:
        """Map the shell parameter monotonically from outer to inner height.

        In ``physical_shell`` mode the historically named ``log_tau500`` grid
        is only a dimensionless, ordered quadrature parameter.  It has no
        optical-depth meaning; tau500 is derived from the predicted opacity.
        """

        q = torch.as_tensor(depth_coordinate, device=self.log_tau500.device,
                            dtype=self.log_tau500.dtype)
        reference_height = self.reference_atmosphere.height_from_log_tau(q)
        height = torch.where(
            q <= 0,
            reference_height * self.outer_height_scale,
            reference_height * self.inner_height_scale,
        )
        if q.ndim == 1 and leading_shape:
            height = height.reshape(*([1] * len(leading_shape)), q.numel()).expand(
                *leading_shape, q.numel()
            )
        return height

    def depth_metric_from_log_tau(self, q: torch.Tensor) -> torch.Tensor:
        """Return ``-dz/dlog10(tau500)`` for the configured shell mapping."""

        q = torch.as_tensor(
            q, device=self.log_tau500.device, dtype=self.log_tau500.dtype
        )
        reference_metric = self.reference_atmosphere.metric_from_log_tau(q)
        return torch.where(
            q <= 0,
            reference_metric * self.outer_height_scale,
            reference_metric * self.inner_height_scale,
        )

    def forward(self, coords: torch.Tensor, log_tau500=None) -> StratifiedAtmosphere:
        """Evaluate the continuous atmosphere at an arbitrary depth grid."""

        grid = self._evaluation_grid(log_tau500)
        inputs, geometric_height = self._network_inputs(coords, grid)
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )
        fields = self._decode_raw(raw, geometric_height_m=geometric_height)
        return StratifiedAtmosphere(
            log_tau500=grid,
            geometric_height_m=geometric_height,
            **fields,
        )

    def _decode_raw(
        self,
        raw: torch.Tensor,
        *,
        geometric_height_m: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode network channels into physical quantities."""

        def bounded_reference_perturbation(
            value: torch.Tensor,
            reference_log: torch.Tensor,
            log10_bounds: tuple[float, float],
            local_log_scale: float,
        ) -> torch.Tensor:
            """Apply a bounded perturbation with raw=0 exactly at reference."""

            lower = value.new_tensor(math.log(10.0) * log10_bounds[0])
            upper = value.new_tensor(math.log(10.0) * log10_bounds[1])
            eps = 32.0 * torch.finfo(value.dtype).eps
            fraction = ((reference_log - lower) / (upper - lower)).clamp(eps, 1.0 - eps)
            logit = torch.log(fraction) - torch.log1p(-fraction)
            raw_scale = local_log_scale / (
                (upper - lower) * fraction * (1.0 - fraction)
            )
            return torch.exp(
                lower + (upper - lower) * torch.sigmoid(logit + raw_scale * value)
            )

        if geometric_height_m is None:
            raise RuntimeError("Radial-reference decoding requires geometric height.")
        reference_logs = self.reference_atmosphere.logs_at_height(
            geometric_height_m.to(raw)
        )
        temperature = bounded_reference_perturbation(
            raw[..., 0], reference_logs[0],
            self.temperature_log10_bounds, self.temperature_log_scale,
        )
        velocity_argument = raw[..., 1:4] * (
            0.5
            * math.pi
            * self.velocity_scale_m_per_s
            / self.velocity_max_m_per_s
        )
        velocity_field = (
            self.velocity_max_m_per_s
            * (2.0 / math.pi)
            * torch.atan(velocity_argument)
        )
        magnetic_field = self.magnetic_scale_gauss * raw[..., 4:7]
        microturbulence = bounded_reference_perturbation(
            raw[..., 7], reference_logs[2],
            self.microturbulence_log10_bounds, self.microturbulence_log_scale,
        )
        gas_pressure = bounded_reference_perturbation(
            raw[..., 8], reference_logs[1],
            self.gas_pressure_log10_bounds, self.gas_pressure_log_scale,
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
        result = self._decode_raw(raw, geometric_height_m=geometric_height)
        result["geometric_height_m"] = geometric_height
        return result

    def evaluate_at_height(
        self,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate ``F(x,y,z)`` at paired physical heights in metres."""

        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        height = geometric_height_m.to(device=coords.device, dtype=coords.dtype)
        inputs = self._network_inputs_at_height(coords, height)
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )
        return self._decode_raw(raw, geometric_height_m=height)

    def position_from_coords_tau(
        self,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map paired chart/optical-depth samples to solar Cartesian positions."""

        del create_graph
        height = self.depth_to_height(log_tau500, coords.shape[:-1])
        if height.ndim == log_tau500.ndim:
            height = height[..., None]
        metric = self.depth_metric_from_log_tau(log_tau500)
        if metric.ndim == log_tau500.ndim:
            metric = metric[..., None]
        geometry_height = height[..., 0]
        position = chart_height_to_position_m(
            coords[..., 1:],
            geometry_height,
            torch.tensor(self.scene_basis_values, dtype=coords.dtype, device=coords.device),
            self.solar_radius_m,
        )
        return position, height[..., 0], metric[..., 0]

    def position_from_coords_height(
        self, coords: torch.Tensor, geometric_height_m: torch.Tensor
    ) -> torch.Tensor:
        """Map chart coordinates and radial height directly to Cartesian points."""

        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        height = geometric_height_m.to(device=coords.device, dtype=coords.dtype)
        return chart_height_to_position_m(
            coords[..., 1:],
            height,
            torch.tensor(
                self.scene_basis_values, dtype=coords.dtype, device=coords.device
            ),
            self.solar_radius_m,
        )

    def evaluate_position_rsun(
        self, position_rsun: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Evaluate the field at Carrington Cartesian positions in solar radii."""

        position = position_rsun.to(
            device=self.log_tau500.device, dtype=self.log_tau500.dtype
        )
        geometry_basis = torch.tensor(
            self.scene_basis_values, dtype=position.dtype, device=position.device
        )
        chart = direction_to_chart_mm(
            position, geometry_basis, self.solar_radius_m
        )
        height = (
            torch.linalg.vector_norm(position, dim=-1) - 1.0
        ) * self.solar_radius_m
        coords = torch.cat((torch.zeros_like(chart[..., :1]), chart), dim=-1)
        inputs = self._network_inputs_at_height(coords, height[..., None])
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )[..., 0, :]
        return self._decode_raw(raw, geometric_height_m=height)

    def evaluate_position_points(self, position_m: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate the unique solar-Cartesian field at physical 3-D points."""

        position = position_m.to(
            device=self.log_tau500.device, dtype=self.log_tau500.dtype
        )
        return self.evaluate_position_rsun(position / self.solar_radius_m)

    def ray_shell_positions(
        self,
        ray_origin_m: torch.Tensor,
        ray_direction: torch.Tensor,
        geometric_height_m: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Intersect observed rays with shell heights in solar-local coordinates."""

        dtype = self.log_tau500.dtype
        device = self.log_tau500.device
        origin = ray_origin_m.to(device=device, dtype=dtype)
        direction = ray_direction.to(device=device, dtype=dtype)
        height = geometric_height_m.to(device=device, dtype=dtype)
        coords = coords.to(device=device, dtype=dtype)
        if origin.shape != direction.shape or origin.shape[-1] != 3:
            raise ValueError("Ray origins and directions must have matching [..., 3] shapes.")
        if height.shape != origin.shape[:-1]:
            raise ValueError("geometric_height_m must match the ray leading dimensions.")
        if coords.shape != (*origin.shape[:-1], 3):
            raise ValueError("coords must match the ray leading dimensions and end in 3.")
        ray = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        basis = torch.tensor(
            self.scene_basis_values, dtype=dtype, device=device
        )
        reference_rsun = chart_to_direction(
            coords[..., 1:], basis, self.solar_radius_m
        )
        impact = torch.linalg.vector_norm(
            torch.linalg.cross(reference_rsun, ray, dim=-1), dim=-1
        )
        requested_radius = 1.0 + height / self.solar_radius_m
        reachable_radius = torch.maximum(
            requested_radius,
            impact + self.tangent_margin_m / self.solar_radius_m,
        )
        offset = intersect_sphere_near_side_from_local_point(
            reference_rsun, ray, reachable_radius
        )
        if not torch.isfinite(offset).all():
            raise RuntimeError("At least one observer ray misses the requested shell radius.")
        return (reference_rsun + offset[..., None] * ray) * self.solar_radius_m

    def trace_rays(
        self,
        coords: torch.Tensor,
        ray_origin_m: torch.Tensor,
        ray_direction: torch.Tensor,
        log_tau500: torch.Tensor,
    ) -> tuple[StratifiedAtmosphere, RayTraceResult]:
        """Sample physical points along observer rays through the atmosphere."""

        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        # The observer coordinate is used only to validate the supplied batch.
        # Intersections below are parameterized from a point at one solar
        # radius, never from a spacecraft-scale coordinate.
        origin = ray_origin_m.to(device=coords.device, dtype=coords.dtype)
        direction = ray_direction.to(device=coords.device, dtype=coords.dtype)
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        grid = self._evaluation_grid(log_tau500)
        if origin.shape != (*coords.shape[:-1], 3) or direction.shape != origin.shape:
            raise ValueError("Ray origins/directions must match coords leading dimensions.")

        geometry_basis = torch.tensor(
            self.scene_basis_values, dtype=coords.dtype, device=coords.device
        )
        # coords[..., 1:] identifies the ray at the photosphere. In this local
        # Carrington parameterization, positions are O(1) R_sun and offsets are
        # O(shell thickness / R_sun), preserving both in float32.
        surface_reference_rsun = chart_to_direction(
            coords[..., 1:], geometry_basis, self.solar_radius_m
        )

        requested_height = self.depth_to_height(grid, coords.shape[:-1])
        expanded_reference = surface_reference_rsun.unsqueeze(-2).expand(
            *requested_height.shape, 3
        )
        expanded_direction = direction.unsqueeze(-2).expand_as(expanded_reference)
        impact = torch.linalg.vector_norm(
            torch.linalg.cross(expanded_reference, expanded_direction, dim=-1), dim=-1
        )
        requested_radius = 1.0 + requested_height / self.solar_radius_m
        outer_radius = requested_radius[..., :1]
        inner_radius = requested_radius[..., -1:]
        effective_inner = torch.maximum(
            inner_radius,
            impact[..., :1] + self.tangent_margin_m / self.solar_radius_m,
        )
        if torch.any(effective_inner >= outer_radius):
            raise RuntimeError(
                "At least one observer ray does not enter the configured outer atmosphere."
            )
        outer_offset = intersect_sphere_near_side_from_local_point(
            surface_reference_rsun, direction, outer_radius[..., 0]
        )
        inner_offset = intersect_sphere_near_side_from_local_point(
            surface_reference_rsun, direction, effective_inner[..., 0]
        )
        fraction = (outer_radius - requested_radius) / (outer_radius - inner_radius)
        offset_rsun = outer_offset[..., None] + fraction * (
            inner_offset - outer_offset
        )[..., None]
        local_distance_m = (offset_rsun - outer_offset[..., None]) * self.solar_radius_m
        if not torch.isfinite(local_distance_m).all():
            raise RuntimeError("The configured physical ray sampling is non-finite.")
        position_rsun = expanded_reference + offset_rsun[..., None] * expanded_direction
        position = position_rsun * self.solar_radius_m
        chart, actual_height = position_to_chart_height(
            position, geometry_basis, self.solar_radius_m
        )
        fields = self.evaluate_position_rsun(position_rsun)
        atmosphere = StratifiedAtmosphere(
            log_tau500=grid,
            geometric_height_m=actual_height.to(coords),
            **fields,
        )
        return atmosphere, RayTraceResult(
            position_m=position,
            distance_m=local_distance_m,
            chart_xy_mm=chart,
            geometric_height_m=actual_height,
            maximum_surface_residual_m=(
                effective_inner - inner_radius
            ).amax() * self.solar_radius_m,
        )

    def reference_metadata(self) -> dict:
        radial_reference = self.reference_atmosphere.metadata()
        return {
            "name": "radial reference atmosphere plus learned perturbations",
            "reference": radial_reference,
            "provenance_note": (
                "pinned STiC FALC_82 thermodynamics; no magnetic or velocity seed"
            ),
            "network_initialization": (
                "activation-aware variance-preserving random hidden weights, "
                "zero thermodynamic perturbation readouts and random vector readouts"
            ),
            "log_tau500": self.log_tau500.detach().cpu().tolist(),
            "temperature_log10_bounds": list(self.temperature_log10_bounds),
            "temperature_log_scale": self.temperature_log_scale,
            "temperature_parameterization": "smooth bounded log perturbation around radial reference",
            "microturbulence_log10_bounds": list(self.microturbulence_log10_bounds),
            "gas_pressure_log10_bounds": list(self.gas_pressure_log10_bounds),
            "gas_pressure_log_scale": self.gas_pressure_log_scale,
            "gas_pressure_parameterization": "smooth bounded log perturbation around radial reference",
            "velocity_scale_m_per_s": self.velocity_scale_m_per_s,
            "velocity_max_m_per_s": self.velocity_max_m_per_s,
            "magnetic_scale_gauss": self.magnetic_scale_gauss,
            "vector_decoder": (
                "velocity uses a smoothly bounded component-wise arctangent output "
                "with the configured local linear scale and nonzero tail gradients; "
                "magnetic field remains an unbounded component-wise linear output; "
                "output weights and biases remain randomly initialized; both vectors "
                "always have three components"
            ),
            "velocity_observability": (
                "single-view LTE Stokes synthesis constrains only the negative "
                "projection onto the toward-observer axis (positive redshift); "
                "the two observer-transverse combinations require additional "
                "dynamical physics"
            ),
            "velocity_frame_convention": (
                "physical velocity projected to the observer, with components "
                "expressed in the instantaneous Carrington Cartesian basis; it is "
                "not a Carrington coordinate time derivative"
            ),
            "depth_coordinate_role": "ordered computational shell parameter; tau500 is derived from opacity",
            "shell_height_bounds_Mm": list(self.shell_height_bounds_Mm),
            "tangent_margin_m": self.tangent_margin_m,
            "spherical_geometry": {
                "enabled": True,
                "solar_radius_m": self.solar_radius_value_m,
                "reference_frame": "heliocentric Heliographic Carrington Cartesian (co-rotating)",
                "ray_geometry_units": (
                    "dimensionless solar radii with signed offsets from the "
                    "photospheric ray point"
                ),
                "scene_basis_rows_solar_cartesian": self.scene_basis.detach().cpu().tolist(),
                "field_vector_basis": "Heliographic Carrington Cartesian [Xc,Yc,Zc]",
            },
            "spatial_coordinates": {
                "public_input": "observer-independent Carrington gnomonic chart in Mm",
                "network_center_mm": self.spatial_coordinate_center_mm.detach().cpu().tolist(),
                "network_scale_mm": self.spatial_coordinate_scale_mm.detach().cpu().tolist(),
                "network_transform": "(xy_mm - center_mm) / scale_mm",
            },
            "network_inputs": list(self.network_input_names),
            "network_outputs": list(self.output_names),
            "depth_network_transform": "z_normalized = z_m / height_input_scale_m",
            "height_input_scale_m": self.height_input_scale_m,
        }
