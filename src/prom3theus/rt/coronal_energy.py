"""Optically thin cooling and field-aligned coronal conduction.

The legacy physical residual is retained for external callers.  The training
path uses normalized atmosphere fields and fixed dimensionless source
coefficients; SI is crossed only at the cooling/EOS table adapters.  This is a
single-temperature, fixed-gamma coronal closure, not a photospheric energy
equation. No AIA response, calibration, or intensity scale enters here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from prom3theus.core import ATOMIC_MASS_UNIT, MLPModel
from prom3theus.resources import CORONAL_COOLING_RESOURCE, resolve_resource_reference

# Fixed reference scales used only to keep the heating network's inputs to
# O(1) across the quiet-Sun-to-sunspot and corona-to-chromosphere dynamic
# ranges; neither is a physical scale that enters the returned heating rate.
HEATING_NETWORK_REFERENCE_GAUSS = 100.0
HEATING_NETWORK_REFERENCE_DENSITY_KG_M3 = 1.0e-12

# Shared ceiling for every exp() in this module that could see an
# out-of-domain input (a point below coronal_energy's own minimum height,
# evaluated anyway so its loss can be masked by height rather than
# pre-filtered out of the batch). Kept well below float32's ~88 exp overflow
# threshold because some of these exp() calls chain (HeatingNetwork's
# amplitude feeds into a second exp() for the height-decay envelope) or
# multiply a large prefactor (a megameter-scale heating_scale_height_m in metres) --
# exp(20) ~ 4.9e8 is still many orders of magnitude past any physically
# meaningful heating/radiative-loss variation, with ample headroom left for
# those downstream multiplications to stay finite too.
_MAX_EXP_ARGUMENT = 20.0


@dataclass(frozen=True)
class CoronalEnergyOptions:
    """Runtime options independent of the application configuration layer."""

    # A plain ``str`` is resolved as a "resource_set:logical_name" reference
    # through the checksum-verified resource registry (the default is the
    # bundled, production CHIANTI-derived table); a ``Path`` is read literally,
    # which unit tests use to exercise a small synthetic cooling curve.
    cooling_resource: str | Path = CORONAL_COOLING_RESOURCE
    minimum_height_megameter: float = 3.0
    conductivity_w_m_k72: float = 1.0e-11
    magnetic_floor_gauss: float = 0.1
    # Reference heating-law anchors: the field-dependent heating network in
    # `HeatingNetwork` perturbs these multiplicatively, and reduces to them
    # exactly at construction (see its zero-initialized output layer).
    heating_w_m3: float = 1.0e-5
    heating_scale_height_megameter: float = 30.0
    heating_network_hidden_dim: int = 64
    heating_network_hidden_layers: int = 4
    heating_network_activation: str = "silu"

    def __post_init__(self):
        for name in (
            "minimum_height_megameter",
            "conductivity_w_m_k72",
            "magnetic_floor_gauss",
            "heating_scale_height_megameter",
            "heating_w_m3",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and name != "heating_w_m3")
            ):
                raise ValueError(f"Invalid coronal energy option {name}")
        for name in ("heating_network_hidden_dim", "heating_network_hidden_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid coronal energy option {name}")
        if self.heating_network_activation not in {"silu", "gelu", "tanh"}:
            raise ValueError(
                "Invalid coronal energy option heating_network_activation"
            )


class HeatingNetwork(nn.Module):
    """Volumetric heating amplitude and decay length as functions of ``(|B|, rho)``.

    The height dependence stays the fixed exponential profile: only its
    amplitude and scale height are perturbed by a small network of the local
    field strength and mass density, so the source term cannot degenerate into
    an unconstrained function of position and time -- see ``CoronalEnergy``
    for how the two outputs are combined with height. Density is a physically
    motivated second input (most heating candidates -- Ohmic and especially
    Alfven-wave/turbulent dissipation -- scale with density through the local
    Alfven speed, not field strength alone), unlike temperature or pressure,
    which the energy equation itself solves for and are therefore excluded to
    avoid the source term trivially tracking whatever state the data terms
    already produce. The output layer is zero-initialized, so a freshly
    constructed instance reproduces the fixed reference heating law exactly
    until training moves its parameters.
    """

    def __init__(self, options: CoronalEnergyOptions):
        super().__init__()
        self.heating_w_m3 = float(options.heating_w_m3)
        self.heating_scale_height_m = float(
            options.heating_scale_height_megameter * 1.0e6
        )
        self.magnetic_floor_gauss = float(options.magnetic_floor_gauss)
        self.network = MLPModel(
            in_dim=2,
            out_dim=2,
            dim=options.heating_network_hidden_dim,
            n_layers=options.heating_network_hidden_layers,
            activation=options.heating_network_activation,
            encoding_config={"type": "identity"},
        )
        with torch.no_grad():
            self.network.out_layer.weight.zero_()
            self.network.out_layer.bias.zero_()

    def forward(
        self,
        magnetic_field_gauss: torch.Tensor,
        mass_density_kg_m3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(amplitude_w_m3, scale_height_m)`` for a batch of points."""

        magnitude = torch.sqrt(
            magnetic_field_gauss.square().sum(-1) + self.magnetic_floor_gauss**2
        )
        log_field = torch.log10(magnitude / HEATING_NETWORK_REFERENCE_GAUSS)
        log_density = torch.log10(
            mass_density_kg_m3 / HEATING_NETWORK_REFERENCE_DENSITY_KG_M3
        )
        raw = self.network(torch.stack((log_field, log_density), dim=-1))
        # Clamped, not just exponentiated: points far outside the corona
        # (e.g. below coronal_energy's minimum_height_megameter, evaluated
        # anyway so the loss reduction can mask them by height rather than
        # pre-filtering the batch) can feed this network wildly
        # out-of-distribution inputs. exp(_MAX_EXP_ARGUMENT) is already many
        # orders of magnitude past any physical heating rate, so the clamp
        # only ever engages there -- it keeps the output finite instead of
        # overflowing to inf, which a masked-out point could otherwise still
        # poison shared parameters' gradients with (0 * inf = nan).
        amplitude = self.heating_w_m3 * torch.exp(raw[..., 0].clamp(max=_MAX_EXP_ARGUMENT))
        scale_height = self.heating_scale_height_m * torch.exp(
            raw[..., 1].clamp(max=_MAX_EXP_ARGUMENT)
        )
        return amplitude, scale_height


class CoronalCoolingTable(nn.Module):
    """Prepared CHIANTI Lambda(T), defined per electron and H nucleus.

    Log-linear interpolation preserves positivity. Outside the table, explicit
    power-law continuations (cold T^2, hot T^0.5) avoid clipping model outputs.
    These numerical tails do not extend the physical validity of the closure.
    """

    def __init__(self, resource):
        super().__init__()
        resource = (
            resolve_resource_reference(resource)
            if isinstance(resource, str)
            else Path(resource)
        )
        payload = resource.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        document = json.loads(payload)
        if (
            document.get("schema_version") != 1
            or document.get("unit") != "W m^3"
            or document.get("density_convention") != "n_e * n_H"
            or not document.get("provenance")
        ):
            raise ValueError(
                "Invalid coronal cooling table units, convention, or provenance"
            )
        axis = torch.tensor(document["log10_temperature_k"], dtype=torch.float32)
        values = torch.tensor(document["log10_lambda_w_m3"], dtype=torch.float32)
        if (
            axis.ndim != 1
            or axis.numel() < 2
            or values.shape != axis.shape
            or not torch.isfinite(axis).all()
            or not torch.isfinite(values).all()
            or not torch.all(axis[1:] > axis[:-1])
        ):
            raise ValueError(
                "Cooling table requires finite, increasing temperature samples"
            )
        self.register_buffer("log_temperature", axis)
        self.register_buffer("log_lambda", values)
        self.register_buffer(
            "source_digest",
            torch.tensor(list(bytes.fromhex(self.sha256)), dtype=torch.uint8),
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        digest = state_dict.get(prefix + "source_digest")
        if digest is not None and not torch.equal(
            digest.cpu(), self.source_digest.cpu()
        ):
            error_msgs.append(
                "Checkpoint cooling table does not match the prepared resource"
            )
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, temperature):
        x = temperature.log10()
        axis, values = self.log_temperature.to(x), self.log_lambda.to(x)
        index = torch.searchsorted(axis, x.contiguous()).clamp(1, axis.numel() - 1)
        fraction = (x - axis[index - 1]) / (axis[index] - axis[index - 1])
        result = torch.lerp(values[index - 1], values[index], fraction)
        result = torch.where(x < axis[0], values[0] + 2.0 * (x - axis[0]), result)
        result = torch.where(x > axis[-1], values[-1] + 0.5 * (x - axis[-1]), result)
        return result


def _mass_u_per_h_nucleus(thermodynamic_eos) -> float:
    """Return the mean atomic mass per H nucleus for either EOS shape.

    ``HybridSolarEOS`` (and simple test doubles) expose this directly.
    ``SolarPlasmaTable``, the production unified-lookup wrapper, only exposes
    it on the ``HybridSolarEOS`` instance it wraps as ``.eos``.
    """

    value = getattr(thermodynamic_eos, "mass_u_per_h_nucleus", None)
    return value if value is not None else thermodynamic_eos.eos.mass_u_per_h_nucleus


def coordinate_gradient(value, coordinates, *, create_graph=True):
    """Pointwise derivative, including constant/unused-coordinate fields."""
    if not value.requires_grad:
        return coordinates * 0.0
    derivative = torch.autograd.grad(
        value.sum(),
        coordinates,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return coordinates * 0.0 if derivative is None else derivative


def spitzer_heat_flux(temperature, gradient_k_m, magnetic_gauss, coefficient, floor):
    # A smooth weak-field suppression avoids an undefined direction at B=0.
    # This is not an isotropic weak-field transport model.
    direction = magnetic_gauss / torch.sqrt(
        magnetic_gauss.square().sum(-1, keepdim=True) + floor**2
    )
    conductivity = coefficient * 1.0e15 * (temperature / 1.0e6).pow(2.5)
    return (
        -conductivity[:, None]
        * direction
        * (direction * gradient_k_m).sum(-1, keepdim=True)
    )


class CoronalEnergy(nn.Module):
    """One normalized dynamic pressure-equation residual.

    Heating keeps the fixed exponential height profile but its amplitude and
    scale height are learned functions of the local field strength and mass
    density through ``HeatingNetwork``, rather than the two fixed constants
    used previously.
    """

    def __init__(self, options):
        super().__init__()
        self.options = options
        if options.cooling_resource is None:
            raise ValueError("coronal_energy requires a prepared cooling_resource")
        self.cooling = CoronalCoolingTable(options.cooling_resource)
        self.heating = HeatingNetwork(options)

    def residual(
        self,
        atmosphere,
        position_m,
        time_hours,
        gamma,
        time_scale_s,
        *,
        create_graph,
        model_time_scale_s: float | None = None,
    ):
        """Return the energy residual in normalized model units when available."""
        if all(
            hasattr(atmosphere, name)
            for name in (
                "evaluate_position_rsun_normalized",
                "mass_density_normalized",
                "electron_density_normalized",
                "temperature_scale_k",
                "gas_pressure_scale_pa",
                "density_scale_kg_m3",
                "electron_density_scale_m3",
                "height_input_scale_m",
                "velocity_scale_m_per_s",
                "magnetic_scale_gauss",
                "solar_radius_model",
            )
        ):
            return self._residual_normalized(
                atmosphere,
                position_m,
                time_hours,
                gamma,
                time_scale_s,
                create_graph=create_graph,
                model_time_scale_s=model_time_scale_s,
            )
        return self._residual_physical(
            atmosphere,
            position_m,
            time_hours,
            gamma,
            time_scale_s,
            create_graph=create_graph,
            model_time_scale_s=model_time_scale_s,
        )

    def _residual_normalized(
        self,
        atmosphere,
        position_m,
        time_hours,
        gamma,
        time_scale_s,
        *,
        create_graph,
        model_time_scale_s: float | None,
        return_components: bool = False,
    ):
        """Evaluate the coronal energy equation in model units.

        Atomic cooling and the conductivity coefficient remain in their native
        physical table/operator units inside this adapter.  The atmosphere
        fields, derivatives, source normalization, and returned residual are
        dimensionless. With ``return_components=True``, also returns the
        transport/heating/radiation/conduction breakdown before it is summed
        and divided by the local pressure scale, for diagnostic rendering.
        """
        parameter = next(atmosphere.parameters())
        coordinates = (
            torch.cat(
                (
                    (position_m / atmosphere.solar_radius_m).to(parameter),
                    time_hours.to(parameter).reshape(-1, 1),
                ),
                -1,
            )
            .detach()
            .requires_grad_(True)
        )
        fields = atmosphere.evaluate_position_rsun_normalized(
            coordinates[:, :3], time_hours=coordinates[:, 3:4]
        )
        temperature = fields["temperature"]
        pressure = fields["gas_pressure"]
        velocity = fields["velocity_field"]
        magnetic = fields["magnetic_field"]
        length_scale_m = float(atmosphere.height_input_scale_m)
        time_scale_model_s = float(
            model_time_scale_s
            if model_time_scale_s is not None
            else getattr(atmosphere, "time_scale_s", 3_600.0)
        )
        spatial_factor = length_scale_m / float(atmosphere.solar_radius_m)
        temporal_factor = time_scale_model_s / 3_600.0

        temperature_gradient_model = coordinate_gradient(
            temperature, coordinates, create_graph=True
        )
        temperature_gradient_model = (
            temperature_gradient_model[:, :3] * spatial_factor
        )
        # Keep conduction dimensionless.  The fixed coefficient below converts
        # the Spitzer operator's reference conductivity into P0/T0 units; no
        # physical heat-flux tensor is formed in the training graph.
        temperature_reference_model = 1.0e6 / atmosphere.temperature_scale_k
        conductivity_shape = temperature.div(
            temperature_reference_model
        ).pow(2.5)
        magnetic_floor_model = (
            self.options.magnetic_floor_gauss / atmosphere.magnetic_scale_gauss
        )
        direction = magnetic / torch.sqrt(
            magnetic.square().sum(-1, keepdim=True) + magnetic_floor_model**2
        )
        conductive_flux_shape = (
            -conductivity_shape[:, None]
            * direction
            * (direction * temperature_gradient_model).sum(-1, keepdim=True)
        )
        divergence_flux_shape = sum(
            coordinate_gradient(
                conductive_flux_shape[:, i], coordinates, create_graph=create_graph
            )[:, i]
            * spatial_factor
            for i in range(3)
        )
        log_pressure = pressure.log()
        pressure_gradient = coordinate_gradient(
            log_pressure, coordinates, create_graph=create_graph
        )
        pressure_gradient_model = pressure_gradient[:, :3] * spatial_factor
        pressure_time_model = pressure_gradient[:, 3] * temporal_factor
        velocity_gradient = torch.stack(
            [
                coordinate_gradient(velocity[:, i], coordinates, create_graph=True)[
                    :, :3
                ]
                * spatial_factor
                for i in range(3)
            ],
            dim=1,
        )
        divergence_velocity_model = torch.diagonal(
            velocity_gradient, dim1=-2, dim2=-1
        ).sum(dim=-1)
        density = atmosphere.mass_density_normalized(temperature, pressure)
        electrons = atmosphere.electron_density_normalized(temperature, pressure)
        temperature_si = temperature * atmosphere.temperature_scale_k
        # The SI density product must enter the exponent, not multiply it: the
        # normalized densities and a coronal Lambda leave exp() below float32's
        # smallest normal, while the coefficient alone is above its largest.
        # Summing the logarithms keeps the physical W m^-3 loss representable.
        log_radiation_coefficient = (
            math.log(atmosphere.electron_density_scale_m3)
            + math.log(atmosphere.density_scale_kg_m3)
            - math.log(
                _mass_u_per_h_nucleus(atmosphere.thermodynamic_eos) * ATOMIC_MASS_UNIT
            )
        )
        # Clamped for the same reason as HeatingNetwork's exp(): a point far
        # below minimum_height_megameter (evaluated anyway; masked out only
        # at the loss reduction) can push this argument arbitrarily high.
        radiation = torch.exp(
            (
                electrons.log()
                + density.log()
                + self.cooling(temperature_si) * 2.302585092994046
                + log_radiation_coefficient
            ).clamp(max=_MAX_EXP_ARGUMENT)
        )
        height_model = (
            torch.linalg.vector_norm(coordinates[:, :3], dim=-1)
            - 1.0
        ) * atmosphere.solar_radius_model
        magnetic_gauss = magnetic * atmosphere.magnetic_scale_gauss
        density_kg_m3 = density * atmosphere.density_scale_kg_m3
        heating_amplitude, heating_scale_height_m = self.heating(
            magnetic_gauss, density_kg_m3
        )
        # Also clamped: well below minimum_height_megameter this exponent is
        # large and positive (the profile is only designed to decay upward),
        # and a learned scale_height can shrink toward zero.
        heating = heating_amplitude * torch.exp(
            (
                -(
                    height_model
                    - self.options.minimum_height_megameter * 1.0e6 / length_scale_m
                )
                / (heating_scale_height_m / length_scale_m)
            ).clamp(max=_MAX_EXP_ARGUMENT)
        )
        rate_time = float(time_scale_s) / time_scale_model_s
        rate_advective = (
            float(time_scale_s)
            * atmosphere.velocity_scale_m_per_s
            / length_scale_m
        )
        dynamic = pressure * (
            rate_time * pressure_time_model
            + rate_advective
            * (
                (velocity * pressure_gradient_model).sum(dim=-1)
                + gamma * divergence_velocity_model
            )
        )
        conductivity_coefficient = (
            self.options.conductivity_w_m_k72
            * 1.0e15
            * atmosphere.temperature_scale_k
            / length_scale_m**2
        )
        # Same decomposition style as the other normalized force-balance
        # equations (e.g. momentum's "-J×B"): each term keeps the sign it
        # contributes to the residual, so the labelled terms sum exactly to
        # `dynamic - (gamma - 1) * source` -- verify algebraically before
        # changing any sign here.
        gamma_factor = (gamma - 1.0) * float(time_scale_s) / atmosphere.gas_pressure_scale_pa
        heating_term = -gamma_factor * heating
        radiation_term = gamma_factor * radiation
        conduction_term = gamma_factor * conductivity_coefficient * divergence_flux_shape
        pressure_scale = pressure.detach().clamp_min(torch.finfo(pressure.dtype).tiny)
        residual = (dynamic + heating_term + radiation_term + conduction_term) / pressure_scale
        # No finiteness check here: the caller may evaluate this at points
        # outside coronal_energy's valid domain (below minimum_height_megameter)
        # on purpose, sharing the volume equations' collocation batch instead
        # of pre-filtering it -- those points are expected to be masked out of
        # the loss afterward, not necessarily finite. The caller is
        # responsible for checking finiteness of whatever subset it actually
        # uses (see MagnetofluidConstraints.upper_domain).
        if not return_components:
            return residual
        components = {
            "D_t p + γp∇·v": dynamic,
            "-(γ-1)Q_heat": heating_term,
            "+(γ-1)n_e n_H Λ": radiation_term,
            "+(γ-1)∇·F_c": conduction_term,
        }
        return residual, components, pressure_scale

    def _residual_physical(
        self,
        atmosphere,
        position_m,
        time_hours,
        gamma,
        time_scale_s,
        *,
        create_graph,
        model_time_scale_s: float | None,
    ):
        parameter = next(atmosphere.parameters())
        coordinates = (
            torch.cat(
                (
                    (position_m / atmosphere.solar_radius_m).to(parameter),
                    time_hours.to(parameter).reshape(-1, 1),
                ),
                -1,
            )
            .detach()
            .requires_grad_(True)
        )
        fields = atmosphere.evaluate_position_rsun(
            coordinates[:, :3], time_hours=coordinates[:, 3:4]
        )
        temperature, pressure = fields["temperature"], fields["gas_pressure"]
        velocity, magnetic = fields["velocity_field"], fields["magnetic_field"]
        radius = atmosphere.solar_radius_m
        # The first temperature derivative must retain a graph even in validation:
        # divergence of heat flux requires a second spatial derivative.
        grad_t = coordinate_gradient(temperature, coordinates)[:, :3] / radius
        flux = spitzer_heat_flux(
            temperature,
            grad_t,
            magnetic,
            self.options.conductivity_w_m_k72,
            self.options.magnetic_floor_gauss,
        )
        divergence_flux = sum(
            coordinate_gradient(flux[:, i], coordinates, create_graph=create_graph)[
                :, i
            ]
            / radius
            for i in range(3)
        )
        dp = coordinate_gradient(pressure.log(), coordinates, create_graph=create_graph)
        divergence_velocity = sum(
            coordinate_gradient(velocity[:, i], coordinates, create_graph=create_graph)[
                :, i
            ]
            / radius
            for i in range(3)
        )
        eos = atmosphere.thermodynamic_eos
        density = eos.mass_density(temperature, pressure)
        electrons = eos.electron_density(temperature, pressure)
        hydrogen = density / (_mass_u_per_h_nucleus(eos) * ATOMIC_MASS_UNIT)
        # Multiply in log space: neither n_e*n_H nor a tiny Lambda need exist
        # as an intermediate float32 value.
        radiation = torch.exp(
            electrons.log()
            + hydrogen.log()
            + self.cooling(temperature) * 2.302585092994046
        )
        height_m = (coordinates[:, :3].norm(dim=-1) - 1.0) * radius
        heating_amplitude, heating_scale_height_m = self.heating(magnetic, density)
        heating = heating_amplitude * torch.exp(
            -(height_m - self.options.minimum_height_megameter * 1.0e6)
            / heating_scale_height_m
        )
        transport = dp[:, 3] / 3600.0 + (velocity * dp[:, :3]).sum(-1) / radius
        # Divide the entire physical pressure residual by a detached pressure
        # scale, preserving the zeros and the physical p-gradient contribution.
        residual = (
            pressure * (transport + gamma * divergence_velocity)
            - (gamma - 1.0) * (heating - radiation - divergence_flux)
        ) * (time_scale_s / pressure.detach())
        if not torch.isfinite(residual).all():
            raise FloatingPointError("Non-finite coronal energy residual")
        return residual
