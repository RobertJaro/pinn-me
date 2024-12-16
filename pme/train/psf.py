import numpy as np
import torch
from astropy.io import fits
from torch import nn


class PSF(nn.Module):

    def __init__(self, shape, file):
        super().__init__()
        assert len(shape) == 2 and shape[0] % 2 == 1 and shape[1] % 2 == 1, "Invalid PSF shape"
        psf = torch.ones(*shape, dtype=torch.float32) * -1
        psf[shape[0] // 2, shape[1] // 2] = 1
        #
        psf = np.load(file)['PSF']
        psf = torch.tensor(psf, dtype=torch.float32)
        self.psf = nn.Parameter(psf, requires_grad=False)
        self.activation = nn.Softplus()

    def forward(self):
        return self.psf


class LoadPSF(nn.Module):

    def __init__(self, file, crop=None):
        super().__init__()
        #
        if file.endswith('.npz'):
            psf = np.load(file)['PSF']
        elif file.endswith('.npy'):
            psf = np.load(file)
        elif file.endswith('.fits'):
            psf = fits.getdata(file).astype(np.float32)
        else:
            raise ValueError(f"Invalid PSF file: {file}")
        if crop is not None:
            psf = psf[crop[0]:crop[1], crop[2]:crop[3]]
        psf = psf / np.sum(psf) # assure that the PSF is normalized
        psf = torch.tensor(psf, dtype=torch.float32)
        self.psf = nn.Parameter(psf, requires_grad=False)
        self.psf_shape = psf.shape

    def forward(self):
        return self.psf


class NoPSF(nn.Module):

    def __init__(self):
        super().__init__()

        psf = torch.ones(1, 1, dtype=torch.float32)
        self.psf = nn.Parameter(psf, requires_grad=False)
        self.psf_shape = psf.shape

    def forward(self):
        return self.psf
