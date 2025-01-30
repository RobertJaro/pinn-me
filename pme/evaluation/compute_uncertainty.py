import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pme.evaluation.loader import PINNMEOutput

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--ref_stokes', type=str, help='the path to the reference Stokes profile')
parser.add_argument('--output', type=str, help='the path to the output file')
parser.add_argument('--reference', type=str, help='the path to the reference file')

args = parser.parse_args()

if args.ref_stokes.endswith('.fits'):
    stokes_vector_ref = fits.getdata(args.ref_stokes).astype(np.float32)
elif args.ref_stokes.endswith('.npz'):
    stokes_vector_ref = np.load(args.ref_stokes)['stokes_profiles'].astype(np.float32)
else:
    raise ValueError('Invalid reference Stokes profile file format')

pinnme = PINNMEOutput(args.input)
# parameters = pinnme.load_cube(compute_jacobian=True, batch_size=1024)
parameters = pinnme.load_time(pinnme.times[9], compute_jacobian=True, batch_size=4096)

stokes_vector_pred = np.stack([parameters['I'], parameters['Q'], parameters['U'], parameters['V']], axis=-2)
stokes_vector_pred = stokes_vector_pred[0] # remove time dimension


stokes_diff = ((stokes_vector_pred - stokes_vector_ref) ** 2).sum(axis=(-1, -2))

response_functions = np.stack([v for k, v in parameters.items() if 'jacobian' in k], axis=-1)
parameter_keys = [k.replace('jacobian_', '') for k in parameters.keys() if 'jacobian' in k]
response_functions = response_functions[0] # remove time dimension
m = response_functions.shape[-1]

response = m * (response_functions ** 2).sum(axis=(-2, -3))
response = np.moveaxis(response, (0, 1, 2), (1, 2, 0))

uncertainty = np.sqrt(stokes_diff[None] / (response + 1e-4))

fig, axs = plt.subplots(uncertainty.shape[0] + 1, 1, figsize=(10, 10))

im = axs[0].imshow(np.sqrt(stokes_diff), cmap='viridis')
divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
axs[0].set_title('Stokes difference')

for i, ax in enumerate(axs[1:]):
    parameter_key = parameter_keys[i]
    im = ax.imshow(uncertainty[i], cmap='hot', norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(f'Uncertainty {parameter_key}')

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'uncertainty.png'), dpi=300)
plt.close()