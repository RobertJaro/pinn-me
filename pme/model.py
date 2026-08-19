from collections.abc import Iterable
from copy import deepcopy
import math

import torch
import torch.nn.functional as F
from torch import nn


def _per_dimension(value, in_dim, name, cast):
    values = list(value) if isinstance(value, Iterable) and not isinstance(value, (str, bytes)) \
        else [value] * in_dim
    if len(values) != in_dim:
        raise ValueError(f'{name} must contain one value per input dimension ({in_dim}); got {values}.')
    return [cast(item) for item in values]


class FourierEncoding(nn.Module):
    """Deterministic, coordinate-wise multiresolution Fourier features."""

    def __init__(self, in_dim, num_frequencies=None, max_frequencies=None,
                 min_frequency=1.0, include_input=True):
        super().__init__()
        if num_frequencies is None:
            num_frequencies = [8, *([64] * (in_dim - 1))] if in_dim == 4 else 64
        if max_frequencies is None:
            max_frequencies = [2, *([2048] * (in_dim - 1))] if in_dim == 4 else 2048

        counts = _per_dimension(num_frequencies, in_dim, 'num_frequencies', int)
        maxima = _per_dimension(max_frequencies, in_dim, 'max_frequencies', float)
        minima = _per_dimension(min_frequency, in_dim, 'min_frequency', float)
        if any(count < 0 for count in counts):
            raise ValueError('num_frequencies cannot be negative.')

        frequencies = []
        for count, minimum, maximum in zip(counts, minima, maxima):
            if count == 0:
                values = torch.empty(0, dtype=torch.float32)
            elif minimum <= 0 or maximum < minimum:
                raise ValueError('Fourier frequencies require 0 < min_frequency <= max_frequencies.')
            elif count == 1:
                values = torch.tensor([maximum], dtype=torch.float32)
            else:
                values = 2 ** torch.linspace(
                    torch.log2(torch.tensor(minimum)),
                    torch.log2(torch.tensor(maximum)),
                    count,
                    dtype=torch.float32,
                )
            frequencies.append(nn.Parameter(values, requires_grad=False))

        self.frequencies = nn.ParameterList(frequencies)
        self.include_input = bool(include_input)
        self.out_dim = (in_dim if self.include_input else 0) + 2 * sum(counts)

    def forward(self, coords):
        features = [coords] if self.include_input else []
        for coordinate, frequencies in zip(coords.unbind(dim=-1), self.frequencies):
            if frequencies.numel() == 0:
                continue
            phase = torch.pi * coordinate[..., None] * frequencies
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(features, dim=-1)


class Swish(nn.Module):
    """Element-wise Swish activation with fixed beta=1."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class MLPModel(nn.Module):
    """Coordinate MLP with explicit multiresolution Fourier features."""

    def __init__(self, in_dim, out_dim, dim=256, n_layers=6,
                 encoding_config=None, activation='silu'):
        super().__init__()
        if n_layers < 1:
            raise ValueError('n_layers must be at least one.')

        encoding_config = deepcopy(encoding_config) if encoding_config is not None else {}
        encoding_type = encoding_config.pop('type', 'fourier').lower()
        if encoding_type == 'fourier':
            self.encoding = FourierEncoding(in_dim, **encoding_config)
            encoded_dim = self.encoding.out_dim
        elif encoding_type in ('identity', 'none'):
            if encoding_config:
                raise TypeError(f'Identity encoding does not accept options: {sorted(encoding_config)}')
            self.encoding = nn.Identity()
            encoded_dim = in_dim
        else:
            raise ValueError(f'Unknown MLP encoding: {encoding_type!r}.')

        activations = {
            'swish': Swish,
            'silu': nn.SiLU,
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
        }
        try:
            activation_class = activations[activation.lower()]
        except KeyError as error:
            raise ValueError(f'Unknown MLP activation: {activation!r}.') from error

        self.in_layer = nn.Linear(encoded_dim, dim)
        self.hidden_layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers - 1))
        self.activations = nn.ModuleList(activation_class() for _ in range(n_layers))
        self.out_layer = nn.Linear(dim, out_dim)

    def forward(self, coords):
        x = self.activations[0](self.in_layer(self.encoding(coords)))
        for layer, activation in zip(self.hidden_layers, self.activations[1:]):
            x = activation(layer(x))
        return self.out_layer(x)


class SineLayer(nn.Module):
    """SIREN layer with the initialization from Sitzmann et al. (2020)."""

    def __init__(self, in_dim, out_dim, omega_0=1.0, is_first=False):
        super().__init__()
        if omega_0 <= 0:
            raise ValueError('SIREN omega_0 must be positive.')
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_dim, out_dim)
        bound = 1 / in_dim if is_first else math.sqrt(6 / in_dim) / self.omega_0
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenModel(nn.Module):
    """SIREN with independent angular input frequencies per coordinate."""

    def __init__(self, in_dim, out_dim, dim=256, n_layers=8,
                 input_omegas=None, hidden_omega_0=1.0,
                 output_init_scale=1.0):
        super().__init__()
        if n_layers < 1:
            raise ValueError('n_layers must be at least one.')
        if hidden_omega_0 <= 0:
            raise ValueError('SIREN hidden_omega_0 must be positive.')
        if output_init_scale <= 0:
            raise ValueError('SIREN output_init_scale must be positive.')
        if input_omegas is None:
            input_omegas = [30.0] * in_dim
        input_omegas = torch.tensor(
            _per_dimension(input_omegas, in_dim, 'input_omegas', float), dtype=torch.float32
        )
        if torch.any(input_omegas <= 0):
            raise ValueError('SIREN input_omegas must be positive.')
        self.register_buffer('input_omegas', input_omegas)

        self.in_layer = SineLayer(in_dim, dim, omega_0=1.0, is_first=True)
        self.hidden_layers = nn.ModuleList(
            SineLayer(dim, dim, omega_0=hidden_omega_0)
            for _ in range(n_layers - 1)
        )
        self.out_layer = nn.Linear(dim, out_dim)
        # The standard SIREN bound is appropriate for another sine layer, but
        # is unnecessarily broad for physical outputs. Keep its dependence on
        # the hidden frequency while allowing a smaller linear readout.
        output_bound = (
            math.sqrt(6 / dim) / float(hidden_omega_0) * float(output_init_scale)
        )
        with torch.no_grad():
            self.out_layer.weight.uniform_(-output_bound, output_bound)
            self.out_layer.bias.zero_()

    def forward(self, coords):
        x = self.in_layer(coords * self.input_omegas)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out_layer(x)


class MEModel(MLPModel):

    def __init__(self, in_coords, **kwargs):
        super().__init__(in_dim=in_coords, out_dim=9, **kwargs)

    def forward(self, x):
        params = super().forward(x)
        theta = params[..., 1:2] * torch.pi
        chi = params[..., 2:3] * torch.pi
        sin_inc2 = torch.sin(theta).square()
        return {
            'b_field': params[..., 0:1] * 1e3,
            'theta': theta,
            'chi': chi,
            'sin_inc2': sin_inc2,
            'cos_inc': torch.cos(theta),
            'sin_inc2_sin2azi': sin_inc2 * torch.sin(2 * chi),
            'sin_inc2_cos2azi': sin_inc2 * torch.cos(2 * chi),
            'vmac': torch.sigmoid(params[..., 3:4]) * 20e3,
            'damping': torch.sigmoid(params[..., 4:5]),
            'b0': torch.sigmoid(params[..., 5:6]),
            'b1': torch.sigmoid(params[..., 6:7]),
            'vdop': params[..., 7:8] * 1e4,
            'kl': torch.sigmoid(params[..., 8:9]) * 100,
        }


class _BaseMESphericalModel:

    def __init__(self, vector_potential=False, b_sinh_scale=None, v_sinh_scale=None):
        if b_sinh_scale is not None and b_sinh_scale <= 0:
            raise ValueError('b_sinh_scale must be positive or None.')
        if v_sinh_scale is not None and v_sinh_scale <= 0:
            raise ValueError('v_sinh_scale must be positive or None.')
        if vector_potential and b_sinh_scale is not None:
            raise ValueError(
                'b_sinh_scale cannot be used with vector_potential because a nonlinear '
                'post-curl mapping would no longer guarantee a divergence-free field.'
            )
        self.vector_potential = vector_potential
        self.b_sinh_scale = b_sinh_scale
        self.v_sinh_scale = v_sinh_scale

    def _scale_magnetic_vector(self, b):
        if self.b_sinh_scale is None:
            return b
        return torch.sinh(b) * float(self.b_sinh_scale)

    def _scale_velocity_vector(self, v):
        if self.v_sinh_scale is None:
            return v
        return torch.sinh(v) * float(self.v_sinh_scale)

    def _decode_output(self, params, x):
        if params.shape[-1] != 11:
            raise RuntimeError(
                'Expected 11 spherical atmosphere outputs, '
                f'received {params.shape[-1]}. '
                'Legacy scale-channel checkpoints are incompatible with the linear parameterization.'
            )
        if self.vector_potential:
            a = params[..., 0:3]
            a_jac_matrix = jacobian(a, x)
            dAy_dx = a_jac_matrix[:, 1, 1]
            dAz_dx = a_jac_matrix[:, 2, 1]
            dAx_dy = a_jac_matrix[:, 0, 2]
            dAz_dy = a_jac_matrix[:, 2, 2]
            dAx_dz = a_jac_matrix[:, 0, 3]
            dAy_dz = a_jac_matrix[:, 1, 3]
            b_x = (dAz_dy - dAy_dz)[..., None]
            b_y = (dAx_dz - dAz_dx)[..., None]
            b_z = (dAy_dx - dAx_dy)[..., None]
        else:
            b = self._scale_magnetic_vector(params[..., 0:3])
            b_x, b_y, b_z = b.split(1, dim=-1)
            a_jac_matrix = None

        v = self._scale_velocity_vector(params[..., 7:10])
        output = {
            'b_x': b_x,
            'b_y': b_y,
            'b_z': b_z,
            'vmac': F.softplus(params[..., 3:4]) * 1000.0,
            'damping': torch.sigmoid(params[..., 4:5]),
            'b0': torch.sigmoid(params[..., 5:6]),
            'b1': torch.sigmoid(params[..., 6:7]),
            'v_x': v[..., 0:1],
            'v_y': v[..., 1:2],
            'v_z': v[..., 2:3],
            'kl': 1e-3 + F.softplus(params[..., 10:11]),
            'a_jac_matrix': a_jac_matrix,
        }
        return output


class MESphericalModel(MLPModel, _BaseMESphericalModel):

    def __init__(self, vector_potential=False, b_sinh_scale=None,
                 v_sinh_scale=None, **kwargs):
        _BaseMESphericalModel.__init__(
            self, vector_potential=vector_potential,
            b_sinh_scale=b_sinh_scale, v_sinh_scale=v_sinh_scale,
        )
        MLPModel.__init__(self, in_dim=4, out_dim=11, **kwargs)

    def forward(self, x):
        return self._decode_output(MLPModel.forward(self, x), x)


class MESphericalSirenModel(SirenModel, _BaseMESphericalModel):

    def __init__(self, vector_potential=False, b_sinh_scale=None,
                 v_sinh_scale=None, **kwargs):
        _BaseMESphericalModel.__init__(
            self, vector_potential=vector_potential,
            b_sinh_scale=b_sinh_scale, v_sinh_scale=v_sinh_scale,
        )
        SirenModel.__init__(self, in_dim=4, out_dim=11, **kwargs)

    def forward(self, x):
        return self._decode_output(SirenModel.forward(self, x), x)


def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    return torch.stack(jac_matrix, dim=1)


class NormalizationModule(nn.Module):

    def __init__(self, asinh_alphas=None):
        super().__init__()

        if isinstance(asinh_alphas, dict):
            asinh_alphas = [asinh_alphas[k] for k in ('Q', 'U', 'V')]
        elif isinstance(asinh_alphas, (float, int)):
            asinh_alphas = [asinh_alphas] * 3

        if asinh_alphas is not None:
            if len(asinh_alphas) != 3:
                raise ValueError('asinh_alphas must contain the Q, U, and V transition scales')
            if any(alpha <= 0 for alpha in asinh_alphas):
                raise ValueError('asinh_alphas must be strictly positive')
            asinh_alphas = torch.tensor(asinh_alphas, dtype=torch.float32).reshape(1, 3, 1)

        self.register_buffer('asinh_alphas', asinh_alphas)

    def forward(self, stokes):
        stokes_i = stokes[..., 0:1, :]
        stokes_q = stokes[..., 1:2, :]
        stokes_u = stokes[..., 2:3, :]
        stokes_v = stokes[..., 3:4, :]

        if self.asinh_alphas is not None:
            polarization = torch.cat([stokes_q, stokes_u, stokes_v], dim=-2)
            polarization = torch.asinh(polarization / self.asinh_alphas) / torch.asinh(1 / self.asinh_alphas)
            stokes_q, stokes_u, stokes_v = polarization.split(1, dim=-2)

        return torch.cat([stokes_i, stokes_q, stokes_u, stokes_v], dim=-2)


class VelocityCorrectionModel(MLPModel):

    def __init__(self, **kwargs):
        super().__init__(in_dim=1, out_dim=1, dim=16, n_layers=3,
                         encoding_config={'type': 'identity'}, **kwargs)

    def forward(self, x):
        return super().forward(x) * 1e3


class LimbCorrectionModel(MLPModel):

    def __init__(self, **kwargs):
        super().__init__(in_dim=1, out_dim=4, dim=16, n_layers=3,
                         encoding_config={'type': 'identity'}, **kwargs)
        with torch.no_grad():
            self.out_layer.weight[3].zero_()
            self.out_layer.bias[3] = torch.logit(torch.tensor(0.99))

    def forward(self, mu):
        x = super().forward(mu)
        return {
            'c_b0': torch.sigmoid(x[..., 0:1]),
            'c_b1': torch.sigmoid(x[..., 1:2]),
            'c_vdop': x[..., 2:3] * 100,
            'filling_factor': torch.sigmoid(x[..., 3:4]),
        }
