import torch
from torch import nn

from pme.model import GenericModel


class ArtifactCorrectionModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        encoding_config = {'type': 'gaussian', 'scales': 4.0, 'num_frequencies': 16}
        self.model = GenericModel(in_dim=2, out_dim=6, dim=64, n_layers=4,
                                  encoding_config=encoding_config, *args, **kwargs)

    def forward(self, pix, stokes):
        """
        pix: (..., 2) normalized pixel coordinates
        stokes: (..., 4, wl) Stokes I, Q, U, V
        """
        # normalize hpc to [-1, 1]
        x = self.model(pix)  # (..., 6)

        # scale outputs to small ranges
        x = x * 0.01

        gain_I = torch.ones_like(x[..., 0:1]) #torch.exp(x[..., 0:1])
        bias_I = torch.zeros_like(x[..., 0:1]) #x[..., 1:2]
        gain_Q = torch.ones_like(x[..., 0:1]) #torch.exp(x[..., 0:1])
        gain_U = torch.ones_like(x[..., 1:2]) #torch.exp(x[..., 1:2])
        gain_V = torch.ones_like(x[..., 2:3]) #torch.exp(x[..., 2:3])
        alpha_Q = x[..., 3:4]
        alpha_U = x[..., 4:5]
        alpha_V = x[..., 5:6]

        I_corr = stokes[..., 0:1, :] * gain_I[..., None] + bias_I[..., None]
        Q_corr = stokes[..., 1:2, :] * gain_Q[..., None] + alpha_Q[..., None] * stokes[..., 0:1, :]
        U_corr = stokes[..., 2:3, :] * gain_U[..., None] + alpha_U[..., None] * stokes[..., 0:1, :]
        V_corr = stokes[..., 3:4, :] * gain_V[..., None] + alpha_V[..., None] * stokes[..., 0:1, :]
        stokes_corr = torch.cat([I_corr, Q_corr, U_corr, V_corr], dim=-2)

        correction_params = torch.cat([gain_I, bias_I,
                                       gain_Q, alpha_Q,
                                       gain_U, alpha_U,
                                       gain_V, alpha_V], dim=-1)
        return {'stokes_corr': stokes_corr, 'correction_params': correction_params}
