import glob
import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

files = sorted(glob.glob("/glade/work/rjarolim/data/phi_fdt/**/*.fits.gz", recursive=True))
out_path = "/glade/work/rjarolim/spinn_me/phi_fdt_data"
os.makedirs(out_path, exist_ok=True)

norms = [LogNorm(vmin=1e-4) for _ in range(4)]

for f in tqdm(files):
    stokes_data = fits.getdata(f)
    # --> (x, y, stokes, wl)
    stokes = np.transpose(stokes_data, (2, 3, 1, 0))
    stokes[stokes[:, :, 0, :].sum(-1) <= 1e-3] = np.nan  # mask out stokes I < 1e-3

    stokes_I = np.abs(stokes[:, :, 0, :]).sum(-1)
    stokes_Q = np.abs(stokes[:, :, 1, :]).sum(-1)
    stokes_U = np.abs(stokes[:, :, 2, :]).sum(-1)
    stokes_V = np.abs(stokes[:, :, 3, :]).sum(-1)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    ax = axs[0]
    im = ax.imshow(stokes_I, cmap='viridis', norm=norms[0], origin='lower')
    ax.set_title('Stokes I')
    fig.colorbar(im, ax=ax)

    ax = axs[1]
    im = ax.imshow(stokes_Q, cmap='viridis', norm=norms[1], origin='lower')
    ax.set_title('Stokes Q')
    fig.colorbar(im, ax=ax)

    ax = axs[2]
    im = ax.imshow(stokes_U, cmap='viridis', norm=norms[2], origin='lower')
    ax.set_title('Stokes U')
    fig.colorbar(im, ax=ax)

    ax = axs[3]
    im = ax.imshow(stokes_V, cmap='viridis', norm=norms[2], origin='lower')
    ax.set_title('Stokes V')
    fig.colorbar(im, ax=ax)

    plt.suptitle(f)
    plt.tight_layout()
    plt.savefig(f"{out_path}/stokes_{f.split('/')[-1].replace('.fits.gz', '.jpg')}", dpi=100)
    plt.close(fig)
