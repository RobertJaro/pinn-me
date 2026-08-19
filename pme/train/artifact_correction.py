import torch
from torch import nn

from pme.model import MLPModel


class ArtifactCorrectionModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = MLPModel(in_dim=2, out_dim=7, dim=64, n_layers=4,
                              encoding_config={
                                  'type': 'fourier',
                                  'num_frequencies': 32,
                                  'max_frequencies': 128,
                              }, *args, **kwargs)
        # Start from one spatially uniform, non-saturated filling factor. The
        # correction channels retain their standard initialization.
        with torch.no_grad():
            self.model.out_layer.weight[-1].zero_()
            self.model.out_layer.bias[-1].zero_()

    def forward(self, pix, stokes):
        """
        pix: (..., 2) normalized pixel coordinates
        stokes: (..., 4, wl) Stokes I, Q, U, V
        """
        # normalize hpc to [-1, 1]
        output = self.model(pix)  # (..., 7)

        # scale outputs to small ranges
        x = output[..., :6] * 0.01
        filling_factor = torch.sigmoid(output[..., 6:7])

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
        full_correction = torch.cat([I_corr, Q_corr, U_corr, V_corr], dim=-2)
        stokes_corr = stokes + filling_factor[..., None] * (full_correction - stokes)

        correction_params = torch.cat([gain_I, bias_I,
                                       gain_Q, alpha_Q,
                                       gain_U, alpha_U,
                                       gain_V, alpha_V,
                                       filling_factor], dim=-1)
        return {
            'stokes_corr': stokes_corr,
            'correction_params': correction_params,
            'filling_factor': filling_factor,
        }
