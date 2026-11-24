"""Differentiable, window-local emulator of dimensional polarized LTE coefficients."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from prom3theus.core import SPEED_OF_LIGHT
from prom3theus.core.nn import SIRENModel

from .local_coefficients import LTELocalCoefficients, LocalRadiationCoefficients
from .polarized_exponential import independent_coefficients, propagation_matrix
from .opacity import planck_lambda
from .wavelength import CenteredWavelength

FORMAT = "prom3theus.lte_mapper.v2"
_PRESSURE_FORMAT = "prom3theus.lte_mapper.v1"


class CoefficientMapper(nn.Module):
    """One atmosphere encoding per point and a linear spline across a whole window."""

    def __init__(
        self, *, line_ids, bounds, knots, width, thermodynamic_input="mass_density",
        architecture="silu", hidden_layers=3, first_omega_0=30.0, hidden_omega_0=1.0,
        spectral_frame="rest", linear_residual=True,
    ):
        super().__init__()
        lo, hi = bounds
        if (
            not line_ids
            or not 0 < lo < hi
            or not math.isfinite(hi)
            or knots < 2
            or width < 1
            or type(hidden_layers) is not int or hidden_layers < 1
        ):
            raise ValueError(
                "Invalid mapper line list, wavelength bounds or architecture"
            )
        if thermodynamic_input not in {"mass_density", "gas_pressure"}:
            raise ValueError(
                f"Unknown mapper thermodynamic input: {thermodynamic_input}"
            )
        self.thermodynamic_input = thermodynamic_input
        if spectral_frame not in ("rest", "observer"):
            raise ValueError(f"Unknown mapper spectral frame: {spectral_frame}")
        self.spectral_frame = spectral_frame
        input_dim = 5 if spectral_frame == "rest" else 6
        self.spec = dict(
            thermodynamic_input=thermodynamic_input,
            line_ids=list(line_ids),
            bounds=list(bounds),
            knots=knots,
            width=width,
        )
        if spectral_frame == "rest":
            self.spec["spectral_frame"] = spectral_frame
        # Three hidden layers are implicit in existing checkpoint and resume specs.
        if hidden_layers != 3:
            self.spec["hidden_layers"] = hidden_layers
        self.origin = math.floor(lo)
        self.low_offset = lo - self.origin
        self.spacing = (hi - lo) / (knots - 1)
        if architecture == "siren":
            self.network = SIRENModel(
                input_dim, 5 * knots, dim=width, n_layers=hidden_layers,
                first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0,
            )
            self.spec.update(architecture=architecture, first_omega_0=first_omega_0,
                             hidden_omega_0=hidden_omega_0)
        elif architecture in ("silu", "tanh"):
            activation = nn.SiLU
            if architecture == "tanh":
                activation = nn.Tanh
                self.spec["architecture"] = architecture
            layers = [nn.Linear(input_dim, width), activation()]
            for _ in range(hidden_layers - 1):
                layers.extend([nn.Linear(width, width), activation()])
            self.network = nn.Sequential(*layers, nn.Linear(width, 5 * knots))
        else:
            raise ValueError(f"Unknown surrogate architecture: {architecture}")
        self.linear = None
        if linear_residual:
            self.spec["linear_residual"] = True
            self.linear = nn.Linear(input_dim, 5 * knots)
            self.register_buffer("linear_scale", torch.ones(5))
            # Start with the affine prediction and learn corrections to its knot values.
            readout = self.network.out_layer if architecture == "siren" else self.network[-1]
            nn.init.zeros_(readout.weight)
            nn.init.zeros_(readout.bias)
        self.register_buffer("input_center", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))
        self.register_buffer("log_extinction_center", torch.tensor(-12.0))
        self.register_buffer("log_extinction_scale", torch.tensor(1.0))
        self.register_buffer("relative_scale", torch.ones(6))

    def features(self, points):
        """Transform physical inputs (K, kg/m³, m/s, G) before standardization.

        Legacy pressure checkpoints use Pa for the second component.
        """
        temperature, thermodynamic_value, xi, v, bx, by, bz = points.unbind(-1)
        # Smooth rotation and polarity invariants. Angular factors are applied analytically.
        transverse = (bx / 1000).square() + (by / 1000).square()
        longitudinal = (bz / 1000).square()
        features = [temperature.log(), thermodynamic_value.log(), (xi / 1000).square()]
        if self.spectral_frame == "observer":
            features.append(v / 1000)
        return torch.stack([*features, torch.log1p(transverse), torch.log1p(longitudinal)], -1)

    def prepare_atmosphere(self, atmosphere, *, direction, stokes_basis):
        temperature, thermodynamic_value, xi, v, b = LTELocalCoefficients._point_fields(
            atmosphere,
            direction,
            stokes_basis,
            thermodynamic_field=self.thermodynamic_input,
        )
        return torch.stack((temperature, thermodynamic_value, xi, v, *b.unbind(-1)), -1)

    def prepare_spectrum(self, wavelength):
        if isinstance(wavelength, CenteredWavelength):
            offset = wavelength.offset_angstrom + (
                wavelength.origin_angstrom - self.origin
            )
        else:
            offset = wavelength - self.origin
        coordinate = (offset - self.low_offset) / self.spacing
        # The window contract covers all configured spectral queries. Mark any
        # unsupported query explicitly, even when host-side checks are disabled.
        upper = self.spec["knots"] - 1
        valid = (coordinate >= -1e-4) & (coordinate <= upper + 1e-4)
        coordinate = coordinate.clamp(0, upper)
        left = coordinate.long().clamp_max(upper - 1)
        fraction = torch.where(valid, coordinate - left, torch.nan)
        return left, fraction, offset + self.origin

    def profile_indices(self, velocity_los, spectrum):
        """Locate each observer query on the point's spline, extending edge knots as constants."""
        left, fraction, vacuum = spectrum
        if self.spectral_frame == "observer":
            return left, fraction
        beta = velocity_los / SPEED_OF_LIGHT
        doppler = torch.sqrt((1 - beta) / (1 + beta))
        # lambda_rest = lambda_obs * D. Add the rationalized D - 1 to the
        # centered grid coordinate to preserve small shifts, including in float32.
        delta = -2 * beta / ((1 + beta) * (1 + doppler))
        upper = self.spec["knots"] - 1
        coordinate = (left + fraction + delta[..., None] * vacuum / self.spacing).clamp(0, upper)
        # Invalid observer queries retain their NaN fraction with a safe gather index.
        left = coordinate.long().clamp(0, upper - 1)
        return left, coordinate - left

    def coefficients(self, points, spectrum):
        features = (self.features(points) - self.input_center) / self.input_scale
        profile_shape = (self.spec["knots"], 5)
        weights = self.network(features).unflatten(-1, profile_shape)
        if self.linear is not None:
            baseline = self.linear(features).unflatten(-1, profile_shape)
            weights = baseline * self.linear_scale + weights
        left, fraction = self.profile_indices(points[..., 3], spectrum)
        shape = (*points.shape[:-1], left.shape[-1], 5)
        index = left.unsqueeze(-1).expand(shape)
        values = torch.lerp(
            weights.gather(-2, index),
            weights.gather(-2, index + 1),
            fraction.to(weights).unsqueeze(-1),
        )
        log_i, linear, circular, dispersion_linear, dispersion_circular = values.unbind(
            -1
        )
        extinction = (
            (self.log_extinction_center + self.log_extinction_scale * log_i)
            .clamp(-80, 50)
            .exp()
        )
        bx, by, bz = (points[..., 4:] / 1000).unbind(-1)
        # Fixed field scale makes all polarized channels vanish smoothly at B=0.
        denom = 1 + bx.square() + by.square() + bz.square()
        q = ((bx.square() - by.square()) / denom).unsqueeze(-1)
        u = (2 * bx * by / denom).unsqueeze(-1)
        v = (bz / denom.sqrt()).unsqueeze(-1)
        dichroism = torch.stack((linear * q, linear * u, circular * v), -1)
        dichroism = dichroism / (1 + dichroism.square().sum(-1, keepdim=True)).sqrt()
        dispersion = torch.stack(
            (dispersion_linear * q, dispersion_linear * u, dispersion_circular * v), -1
        )
        return torch.cat(
            (
                extinction.unsqueeze(-1),
                extinction.unsqueeze(-1) * dichroism,
                extinction.unsqueeze(-1) * dispersion,
            ),
            -1,
        )

    def evaluate_spectrum(self, prepared, spectrum, *, radiance_scale, compact=False):
        coefficients = self.coefficients(prepared, spectrum)
        temperature, vacuum = torch.broadcast_tensors(
            prepared[..., 0, None], spectrum[2]
        )
        thermal = planck_lambda(
            temperature, vacuum, radiance_scale=radiance_scale, paired_wavelengths=True
        )
        zero = torch.zeros_like(thermal)
        return LocalRadiationCoefficients(
            coefficients if compact else propagation_matrix(coefficients),
            torch.stack((thermal, zero, zero, zero), -1),
        )

    def _coefficient_loss(self, prediction, target):
        def transformed(values):
            extinction = values[..., :1].clamp_min(1e-35)
            return torch.cat(
                (
                    (extinction.log() - self.log_extinction_center)
                    / self.log_extinction_scale,
                    torch.asinh(values[..., 1:] / extinction / self.relative_scale),
                ),
                -1,
            )

        return nn.functional.smooth_l1_loss(
            transformed(prediction), transformed(target), reduction="none"
        )

    def profile_loss_channels(self, prediction, target):
        """Normalized loss per atmospheric point and channel, averaged over wavelength."""
        return self._coefficient_loss(prediction, target).mean(dim=-2)

    def loss_channels(self, prediction, target):
        return self._coefficient_loss(prediction, target).reshape(-1, 7).mean(0)

    @classmethod
    def from_saved_spec(cls, spec):
        # Older artifacts learned observer-frame profiles, originally from pressure.
        return cls(**{
            "thermodynamic_input": "gas_pressure", "spectral_frame": "observer",
            "linear_residual": False, **spec,
        })

    @classmethod
    def load(cls, path):
        payload = torch.load(Path(path), weights_only=True, map_location="cpu")
        if payload.get("format") not in (FORMAT, _PRESSURE_FORMAT):
            raise ValueError("Expected a trained LTE coefficient mapper checkpoint")
        spec = payload["spec"]
        if payload["format"] == FORMAT and "thermodynamic_input" not in spec:
            raise ValueError("Missing mapper thermodynamic input in checkpoint")
        # Loading a frozen model must preserve the inversion sampling RNG.
        with torch.random.fork_rng(devices=[]):
            model = cls.from_saved_spec(spec)
        model.load_state_dict(payload["mapper"])
        return model
