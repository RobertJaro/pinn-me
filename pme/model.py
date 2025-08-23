import numpy as np
import torch
from torch import nn

from pme.encoding import PeriodicBoundary, GaussianPositionalEncoding, ProgressiveFourierEncoding, PositionalEncoding
from pme.train.siren import SirenModel


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class MEModel(nn.Module):

    def __init__(self, in_coords, dim=256, encoding='gaussian_positional', activation='sine', num_layers=8):
        super().__init__()
        # encoding layer
        if encoding == "periodic":
            posenc = PeriodicBoundary()
            d_in = nn.Linear(in_coords + 2, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        if encoding == "positional":
            posenc = PositionalEncoding(num_freqs=20, d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "gaussian":
            posenc = GaussianPositionalEncoding(d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "linear":
            self.d_in = nn.Linear(in_coords, dim)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # hidden layers
        lin = [nn.Linear(dim, dim) for _ in range(num_layers)]
        self.linear_layers = nn.ModuleList(lin)

        # output layer
        self.d_out = nn.Linear(dim, 9)

        # activation functions
        if activation == "swish":
            self.in_activation = Swish()
            self.activations = nn.ModuleList([Swish() for _ in range(num_layers)])
        elif activation == "sine":
            self.in_activation = Sine()
            self.activations = nn.ModuleList([Sine() for _ in range(num_layers)])
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # output activations
        self.softplus = nn.Softplus()
        self.register_buffer("c", torch.tensor(3e8))

    def forward(self, x):
        x = self.in_activation(self.d_in(x))
        for l, a in zip(self.linear_layers, self.activations):
            x = a(l(x))
        params = self.d_out(x)
        #
        b_field = params[..., 0:1] * 1e3
        theta = params[..., 1:2] * torch.pi
        chi = params[..., 2:3] * torch.pi
        vmac = torch.sigmoid(params[..., 3:4]) * 20e3
        damping = torch.sigmoid(params[..., 4:5]) * 1
        b0 = torch.sigmoid(params[..., 5:6])
        b1 = torch.sigmoid(params[..., 6:7])
        vdop = params[..., 7:8] * 1e4
        kl = torch.sigmoid(params[..., 8:9]) * 100
        #
        output = {
            "b_field": b_field,
            "theta": theta,
            "chi": chi,
            "vmac": vmac,
            "damping": damping,
            "b0": b0,
            "b1": b1,
            "vdop": vdop,
            "kl": kl,
        }

        return output


class GenericModel(nn.Module):

    def __init__(self, in_dim, out_dim, dim=512, encoding='gaussian', activation='sine', n_layers=8):
        super().__init__()
        # encoding layer
        self.posenc = None
        if encoding == "periodic":
            posenc = PeriodicBoundary()
            d_in = nn.Linear(in_dim + 2, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        if encoding == "positional":
            posenc = PositionalEncoding(num_freqs=20, d_input=in_dim)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "gaussian":
            posenc = GaussianPositionalEncoding(d_input=in_dim, scale=64)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "progressive_fourier":
            posenc = ProgressiveFourierEncoding(d_input=in_dim)
            d_in = nn.Linear(posenc.d_output, dim)
            self.posenc = posenc
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "linear":
            self.d_in = nn.Linear(in_dim, dim)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # hidden layers
        lin = [nn.Linear(dim, dim) for _ in range(n_layers)]
        self.linear_layers = nn.ModuleList(lin)

        # output layer
        self.d_out = nn.Linear(dim, out_dim)

        # activation functions
        if activation == "swish":
            self.in_activation = Swish()
            self.activations = nn.ModuleList([Swish() for _ in range(n_layers)])
        elif activation == "sine":
            self.in_activation = Sine()
            self.activations = nn.ModuleList([Sine() for _ in range(n_layers)])
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def step(self, global_step):
        if self.posenc is not None:
            self.posenc.step(global_step)

    def forward(self, x):
        x = self.in_activation(self.d_in(x))
        for l, a in zip(self.linear_layers, self.activations):
            x = a(l(x))
        out = self.d_out(x)
        return out


class MESphericalModel(SirenModel):

    def __init__(self, vector_potential=False, **kwargs):
        super().__init__(in_dim=4, out_dim=13, **kwargs)
        self.vector_potential = vector_potential

    def forward(self, x):
        params = super().forward(x)
        #
        if self.vector_potential:
            a = params[..., 0:3]
            jac_matrix = jacobian(a, x)
            dAy_dx = jac_matrix[:, 1, 1]
            dAz_dx = jac_matrix[:, 2, 1]
            dAx_dy = jac_matrix[:, 0, 2]
            dAz_dy = jac_matrix[:, 2, 2]
            dAx_dz = jac_matrix[:, 0, 3]
            dAy_dz = jac_matrix[:, 1, 3]
            # B = curl(A)
            b_x = (dAz_dy - dAy_dz)[..., None]
            b_y = (dAx_dz - dAz_dx)[..., None]
            b_z = (dAy_dx - dAx_dy)[..., None]
        else:
            b_x = params[..., 0:1]
            b_y = params[..., 1:2]
            b_z = params[..., 2:3]

        vmac = torch.sigmoid(params[..., 4:5]) * 20e3
        damping = torch.sigmoid(params[..., 5:6]) * 1
        b0 = torch.sigmoid(params[..., 6:7])
        b1 = torch.sigmoid(params[..., 7:8])

        v_x = params[..., 9:10]
        v_y = params[..., 10:11]
        v_z = params[..., 11:12]
        kl = torch.sigmoid(params[..., 12:13]) * 100
        #
        output = {
            "b_x": b_x,
            "b_y": b_y,
            "b_z": b_z,
            "vmac": vmac,
            "damping": damping,
            "b0": b0,
            "b1": b1,
            "v_x": v_x,
            "v_y": v_y,
            "v_z": v_z,
            "kl": kl,
        }

        return output


def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    jac_matrix = torch.stack(jac_matrix, dim=1)
    return jac_matrix


class NormalizationModule(nn.Module):

    def __init__(self, value_range):
        super().__init__()
        self.register_buffer("value_range", torch.tensor(value_range, dtype=torch.float32)[None, :, None, :])
        self.register_buffer("stretch", torch.tensor(np.arcsinh(1e3), dtype=torch.float32))

    def forward(self, stokes):
        stokes = stokes / self.value_range[..., 1]  # normalize by max value (I = [0, 1]; QUV = [-1, 1])
        stokes = torch.asinh(stokes * 1e3) / self.stretch
        return stokes


class INormalizationModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("stretch", torch.tensor(np.arcsinh(1e2), dtype=torch.float32))

    def forward(self, stokes):
        # total_intensity = stokes[..., 0:1, :].sum(-1, keepdim=True) + 1e-6  # avoid division by zero
        # normalized_stokes = stokes / total_intensity  # normalize by total intensity
        normalized_stokes = torch.asinh(stokes * 1e2) / self.stretch
        return normalized_stokes


class ProjectionModel(SirenModel):
    def __init__(self, Mm_per_ds, max_shift_Mm=1, **kwargs):
        super().__init__(4, 1, dim=32, n_layers=4, **kwargs)
        self.max_shift = max_shift_Mm / Mm_per_ds  # convert to model units

    def forward(self, x):
        x = super().forward(x)
        x = torch.tanh(x) * self.max_shift
        return x


class VelocityCorrectionModel(SirenModel):
    def __init__(self, **kwargs):
        super().__init__(1, 1, dim=32, n_layers=4, w0_initial=1, **kwargs)

    def forward(self, x):
        x = super().forward(x) * 1e3  # scale to m/s
        return x


class LimbCorrectionModel(SirenModel):
    def __init__(self, **kwargs):
        super().__init__(1, 6, dim=32, n_layers=4, w0_initial=1, **kwargs)

    def forward(self, mu):
        x = super().forward(mu)
        c_b0 = 10 ** x[..., 0:1]  # limb correction for B0
        c_b1 = 10 ** x[..., 1:2]  # limb correction for B1
        c_vmac = 10 ** x[..., 2:3]  # limb correction for v_mac
        c_damping = 10 ** x[..., 3:4]  # limb correction for damping
        c_kl = 10 ** x[..., 4:5]  # limb correction for kl
        c_vdop = 200 * x[..., 5:6]  # limb correction for convective blue shift
        # c_vdop = self.limb_shift_velocity(mu)
        return {'c_b0': c_b0, 'c_b1': c_b1, 'c_vmac': c_vmac, 'c_damping': c_damping, 'c_kl': c_kl, 'c_vdop': c_vdop}
