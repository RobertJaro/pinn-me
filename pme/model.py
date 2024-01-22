import torch
from torch import nn


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

    def __init__(self, in_coords, dim, pos_encoding=False):
        super().__init__()
        if pos_encoding:
            posenc = PositionalEncoding(8, 20)
            d_in = nn.Linear(in_coords * 40, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            self.d_in = nn.Linear(in_coords, dim)
        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, 10)
        self.activation = Swish()  # torch.tanh
        self.softplus = nn.Softplus()
        self.register_buffer("c", torch.tensor(3e8))

    def forward(self, x):
        x = self.activation(self.d_in(x))
        for l in self.linear_layers:
            x = self.activation(l(x))
        params = self.d_out(x)
        #
        b_field = self.softplus(params[..., 0:1]) * 100
        theta = torch.sigmoid(params[..., 1:2]) * 180
        chi = torch.sigmoid(params[..., 2:3]) * 180
        vmac = self.softplus(params[..., 3:4])
        damping = self.softplus(params[..., 4:5])
        b0 = torch.sigmoid(params[..., 5:6])
        b1 = torch.sigmoid(params[..., 6:7])
        mu = torch.sigmoid(params[..., 7:8])
        vdop = torch.tanh(params[..., 8:9]) * 0
        kl = self.softplus(params[..., 9:10])
        #
        output = {
            "b_field": b_field,
            "theta": theta,
            "chi": chi,
            "vmac": vmac,
            "damping": damping,
            "b0": b0,
            "b1": b1,
            "mu": mu,
            "vdop": vdop,
            "kl": kl,
        }

        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.

    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """

    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2 ** torch.linspace(-max_freq, max_freq, num_freqs)
        self.register_buffer("freqs", freqs)  # (num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (batch, num_samples, in_features)
        Outputs:
            out: (batch, num_samples, 2*num_freqs*in_features)
        """
        x_proj = x.unsqueeze(dim=-2) * self.freqs.unsqueeze(dim=-1)  # (num_rays, num_samples, num_freqs, in_features)
        x_proj = x_proj.reshape(*x.shape[:-1], -1)  # (num_rays, num_samples, num_freqs*in_features)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)  # (num_rays, num_samples, 2*num_freqs*in_features)
        return out

def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    jac_matrix = torch.stack(jac_matrix, dim=1)
    return jac_matrix