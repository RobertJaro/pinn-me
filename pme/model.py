import torch
from torch import nn
from torch.nn import Identity, Sequential

from pme.encoding import PeriodicBoundary, GaussianPositionalEncoding, ProgressiveFourierEncoding, PositionalEncoding, \
    ProgressiveSpatiotemporalEncoding, SpatiotemporalEncoding, ProgressiveGaussianEncoding, \
    ProgressivePositionalEncoding, SphericalEncoding, BankGaussianEncoding
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
            d_in = nn.Linear(posenc.out_dim, dim)
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

    def __init__(self, in_dim, out_dim, dim=512, encoding_config=None, activation='sine', n_layers=8):
        super().__init__()
        # encoding layer
        encoding_config = {'type': 'gaussian'} if encoding_config is None else encoding_config
        encoding_type = encoding_config.pop('type')

        if encoding_type == "positional":
            self.posenc = PositionalEncoding(in_dim=in_dim, **encoding_config)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "spatiotemporal":
            self.posenc = SpatiotemporalEncoding(d_input=in_dim)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "gaussian":
            self.posenc = GaussianPositionalEncoding(d_input=in_dim, **encoding_config)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "spherical":
            spherical_encoding = SphericalEncoding()
            gaussian_encoding = GaussianPositionalEncoding(d_input=spherical_encoding.out_dim, **encoding_config)
            self.posenc = Sequential(spherical_encoding, gaussian_encoding)
            posenc_dim = gaussian_encoding.out_dim
        elif encoding_type == "progressive_fourier":
            self.posenc = ProgressiveFourierEncoding(d_input=in_dim)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "progressive_spatiotemporal":
            self.posenc = ProgressiveSpatiotemporalEncoding(d_input=in_dim)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "progressive_gaussian":
            self.posenc = ProgressiveGaussianEncoding(in_dim=in_dim, **encoding_config)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "bank_gaussian":
            self.posenc = BankGaussianEncoding(in_dim=in_dim, **encoding_config)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "progressive_positional":
            self.posenc = ProgressivePositionalEncoding(in_dim=in_dim, **encoding_config)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "none" or encoding_type == "identity":
            self.posenc = Identity()
            posenc_dim = in_dim
        else:
            raise ValueError(f"Unknown encoding: {encoding_type}")

        self.d_in = nn.Linear(posenc_dim, dim)

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
        if self.posenc is not None and hasattr(self.posenc, 'step'):
            self.posenc.step(global_step)

    def forward(self, x):
        x = self.posenc(x)
        x = self.in_activation(self.d_in(x))
        for l, a in zip(self.linear_layers, self.activations):
            x = a(l(x))
        out = self.d_out(x)
        return out


class MESphericalModel(SirenModel):

    def __init__(self, vector_potential=False, scale=False, **kwargs):
        super().__init__(in_dim=4, out_dim=14, **kwargs)
        self.vector_potential = vector_potential
        self.scale = scale

    def forward(self, x):
        # forward pass through generic model
        params = super().forward(x)
        #
        if self.vector_potential:
            a_scale = torch.exp(params[..., 3:4]) if self.scale else 1.0
            a = params[..., 0:3] * a_scale
            a_jac_matrix = jacobian(a, x)
            dAy_dx = a_jac_matrix[:, 1, 1]
            dAz_dx = a_jac_matrix[:, 2, 1]
            dAx_dy = a_jac_matrix[:, 0, 2]
            dAz_dy = a_jac_matrix[:, 2, 2]
            dAx_dz = a_jac_matrix[:, 0, 3]
            dAy_dz = a_jac_matrix[:, 1, 3]
            # B = curl(A)
            b_x = (dAz_dy - dAy_dz)[..., None]
            b_y = (dAx_dz - dAz_dx)[..., None]
            b_z = (dAy_dx - dAx_dy)[..., None]
        else:
            b_scale = torch.exp(params[..., 3:4]) if self.scale else 1.0
            b_x = params[..., 0:1] * b_scale
            b_y = params[..., 1:2] * b_scale
            b_z = params[..., 2:3] * b_scale
            a_jac_matrix = None

        vmac = torch.exp(params[..., 4:5] + 9)
        damping = torch.sigmoid(params[..., 5:6]) * 1
        b0 = torch.sigmoid(params[..., 6:7])
        b1 = torch.sigmoid(params[..., 7:8])

        v_scale = torch.exp(params[..., 8:9]) if self.scale else 1.0
        v_x = params[..., 9:10] * v_scale
        v_y = params[..., 10:11] * v_scale
        v_z = params[..., 11:12] * v_scale
        kl = torch.sigmoid(params[..., 12:13]) * 100
        #
        eta = torch.exp(params[..., 13:14])
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
            "a_jac_matrix": a_jac_matrix,
            "eta": eta
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

    def __init__(self, alphas=None, **kwargs):
        super().__init__()
        alphas = [1e-1, 1e-2, 1e-2, 1e-2] if alphas is None else alphas
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))

    def forward(self, stokes, Ic):
        I = stokes[..., 0:1, :]
        Q = stokes[..., 1:2, :]
        U = stokes[..., 2:3, :]
        V = stokes[..., 3:4, :]

        # normalize to continuum
        # Ic = Ic.clamp_min(1e-3)  # prevent division by zero or very small numbers
        I = I / Ic
        Q = Q / Ic
        U = U / Ic
        V = V / Ic

        # asinh scaling for polarization
        # Q = torch.asinh(Q / self.alphas[1]) / torch.asinh(1 / self.alphas[1])
        # U = torch.asinh(U / self.alphas[2]) / torch.asinh(1 / self.alphas[2])
        # V = torch.asinh(V / self.alphas[3]) / torch.asinh(1 / self.alphas[3])

        return torch.cat([I, Q, U, V], dim=-2)


class VelocityCorrectionModel(GenericModel):
    def __init__(self, **kwargs):
        encoding_config = {'type': 'none'}
        super().__init__(1, 1, dim=16, n_layers=3, encoding_config=encoding_config, **kwargs)

    def forward(self, x):
        x = super().forward(x) * 1e3  # scale to m/s
        return x


class LimbCorrectionModel(GenericModel):
    def __init__(self, **kwargs):
        encoding_config = {'type': 'identity'}
        super().__init__(1, 6, dim=16, n_layers=3, encoding_config=encoding_config, **kwargs)

    def forward(self, mu):
        x = super().forward(mu)
        c_b0 = torch.sigmoid(x[..., 0:1])  # limb correction for B0
        c_b1 = torch.sigmoid(x[..., 1:2])  # limb correction for B1
        c_vdop = x[..., 5:6] * 100  # limb correction for convective blue shift
        # c_vdop = self.limb_shift_velocity(mu)
        return {'c_b0': c_b0, 'c_b1': c_b1, 'c_vdop': c_vdop}


class DisambiguationModel(SirenModel):
    def __init__(self, **kwargs):
        encoding_config = {'type': 'spatiotemporal'}
        super().__init__(4, 1, dim=64, encoding_config=encoding_config, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        x = torch.tanh(x)
        return x
