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

from prom3theus.core import ATOMIC_MASS_UNIT


@dataclass(frozen=True)
class CoronalEnergyOptions:
    """Runtime options independent of the application configuration layer."""

    cooling_table: str | Path | None = None
    minimum_height_megameter: float = 3.0
    conductivity_w_m_k72: float = 1.0e-11
    magnetic_floor_gauss: float = 0.1
    heating_w_m3: float = 1.0e-5
    heating_scale_height_megameter: float = 30.0

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


class CoronalCoolingTable(nn.Module):
    """Prepared CHIANTI Lambda(T), defined per electron and H nucleus.

    Log-linear interpolation preserves positivity. Outside the table, explicit
    power-law continuations (cold T^2, hot T^0.5) avoid clipping model outputs.
    These numerical tails do not extend the physical validity of the closure.
    """

    def __init__(self, path):
        super().__init__()
        payload = Path(path).read_bytes()
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
    """One normalized dynamic pressure-equation residual; no learned heating."""

    def __init__(self, options):
        super().__init__()
        self.options = options
        if options.cooling_table is None:
            raise ValueError("coronal_energy requires a prepared cooling_table")
        self.cooling = CoronalCoolingTable(options.cooling_table)

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
    ):
        """Evaluate the coronal energy equation in model units.

        Atomic cooling and the conductivity coefficient remain in their native
        physical table/operator units inside this adapter.  The atmosphere
        fields, derivatives, source normalization, and returned residual are
        dimensionless.
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
        radiation = torch.exp(
            electrons.log()
            + density.log()
            + self.cooling(temperature_si) * 2.302585092994046
        )
        height_model = (
            torch.linalg.vector_norm(coordinates[:, :3], dim=-1)
            - 1.0
        ) * atmosphere.solar_radius_model
        heating = self.options.heating_w_m3 * torch.exp(
            -(
                height_model
                - self.options.minimum_height_megameter * 1.0e6 / length_scale_m
            )
            / (self.options.heating_scale_height_megameter * 1.0e6 / length_scale_m)
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
        radiation_coefficient = (
            atmosphere.electron_density_scale_m3
            * atmosphere.density_scale_kg_m3
            / (
                atmosphere.thermodynamic_eos.mass_u_per_h_nucleus
                * ATOMIC_MASS_UNIT
            )
        )
        conductivity_coefficient = (
            self.options.conductivity_w_m_k72
            * 1.0e15
            * atmosphere.temperature_scale_k
            / length_scale_m**2
        )
        source = float(time_scale_s) / atmosphere.gas_pressure_scale_pa * (
            heating
            - radiation_coefficient * radiation
            - conductivity_coefficient * divergence_flux_shape
        )
        residual = (
            dynamic - (gamma - 1.0) * source
        ) / pressure.detach().clamp_min(torch.finfo(pressure.dtype).tiny)
        if not torch.isfinite(residual).all():
            raise FloatingPointError("Non-finite coronal energy residual")
        return residual

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
        hydrogen = density / (eos.mass_u_per_h_nucleus * ATOMIC_MASS_UNIT)
        # Multiply in log space: neither n_e*n_H nor a tiny Lambda need exist
        # as an intermediate float32 value.
        radiation = torch.exp(
            electrons.log()
            + hydrogen.log()
            + self.cooling(temperature) * 2.302585092994046
        )
        height_m = (coordinates[:, :3].norm(dim=-1) - 1.0) * radius
        heating = self.options.heating_w_m3 * torch.exp(
            -(height_m - self.options.minimum_height_megameter * 1.0e6)
            / (self.options.heating_scale_height_megameter * 1.0e6)
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
