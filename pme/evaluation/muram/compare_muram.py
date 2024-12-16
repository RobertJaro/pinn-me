import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pme.evaluation.loader import to_cartesian

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file')

args = parser.parse_args()

input_path = args.input
output_path = args.output

os.makedirs(output_path, exist_ok=True)

paths = {
    'MURaM': os.path.join(input_path, 'data/MURaM_SIRPlage_parameters_logtau_-2.npz'),
    'PINN-ME': os.path.join(input_path, 'muram_sunspot_v01.npz'),
    'PINN-ME PSF': os.path.join(input_path, 'muram_sunspot_psf_v01.npz'),
    'PyMilne 1D': os.path.join(input_path, 'pymilne_MURaM_AR_1D_clear_v01.npz'),
    'PyMilne 2D': os.path.join(input_path, 'pymilne_MURaM_AR_2D_PSF_v01.npz'),
}

results = {}
for k, path in paths.items():
    data = np.load(path)
    b = data['b_field'].squeeze()
    theta = data['theta'].squeeze()
    chi = data['chi'].squeeze()
    if k == 'MURaM':
        theta = np.deg2rad(theta)
        chi = np.deg2rad(chi)
        # b = block_reduce(b, (4, 4), np.mean)
        # theta = block_reduce(theta, (4, 4), np.mean)
        # chi = block_reduce(chi, (4, 4), np.mean)
    if k == 'PyMilne 2D':
        b = b.T
        theta = theta.T
        chi = chi.T
    b_xyz = to_cartesian(b, theta, chi % np.pi)
    b_los = b * np.cos(theta)
    b_trv = b * np.sin(theta)
    azi = chi % np.pi
    results[k] = {'b_xyz': b_xyz, 'b_los': b_los, 'b_trv': b_trv, 'azi': azi}

########################################################################################################################
# plot example images

subframe = {'x': 60, 'y': 4, 'w': 80, 'h': 40}
b_max = 2.5e3  # np.max(np.abs(results['PINN-ME']['b_los']))
Mm_per_pix = 0.192 * 4
b_ref = results['PINN-ME']['b_los']
extent = [0, b_ref.shape[1] * Mm_per_pix, 0, b_ref.shape[0] * Mm_per_pix]

plot_kwargs = {'extent': extent, 'origin': 'lower'}

fig, axs = plt.subplots(5, 3, figsize=(12, 5))

im_los = axs[0, 0].imshow(results['MURaM']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[0, 1].imshow(results['MURaM']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[0, 2].imshow(np.rad2deg(results['MURaM']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

im_los = axs[1, 0].imshow(results['PINN-ME']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[1, 1].imshow(results['PINN-ME']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[1, 2].imshow(np.rad2deg(results['PINN-ME']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

im_los = axs[2, 0].imshow(results['PINN-ME PSF']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[2, 1].imshow(results['PINN-ME PSF']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[2, 2].imshow(np.rad2deg(results['PINN-ME PSF']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

im_los = axs[3, 0].imshow(results['PyMilne 1D']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[3, 1].imshow(results['PyMilne 1D']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[3, 2].imshow(np.rad2deg(results['PyMilne 1D']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

im_los = axs[4, 0].imshow(results['PyMilne 2D']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[4, 1].imshow(results['PyMilne 2D']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[4, 2].imshow(np.rad2deg(results['PyMilne 2D']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

divider = make_axes_locatable(axs[0, 0])
cax = divider.append_axes('top', size='15%', pad=0.05)
fig.colorbar(im_los, cax=cax, orientation='horizontal', label='$B_{LOS}$ [G]')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

divider = make_axes_locatable(axs[0, 1])
cax = divider.append_axes('top', size='15%', pad=0.05)
fig.colorbar(im_trv, cax=cax, orientation='horizontal', label='$B_{TRV}$ [G]')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

divider = make_axes_locatable(axs[0, 2])
cax = divider.append_axes('top', size='15%', pad=0.05)
fig.colorbar(im_azi, cax=cax, orientation='horizontal', label='Azimuth [deg]')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

[ax.set_xlabel('X [Mm]', fontsize=8) for ax in axs[-1, :]]
[ax.set_ylabel('Y [Mm]', fontsize=8) for ax in axs[:, 0]]

[ax.set_xticklabels([]) for ax in axs[:-1, :].ravel()]
[ax.set_yticklabels([]) for ax in axs[:, 1:].ravel()]

# add rectangles to indicate the subframe
for ax in axs.ravel():
    ax.add_patch(plt.Rectangle((subframe['x'], subframe['y']), subframe['w'], subframe['h'],
                               edgecolor='black', facecolor='none'))

fig.tight_layout()
fig.savefig(f'{output_path}/comparison.png', transparent=True, dpi=300)
plt.close(fig)

########################################################################################################################
# plot subframe

plot_kwargs = {'extent': extent, 'origin': 'lower'}

fig, axs = plt.subplots(3, 5, figsize=(10, 3.5))

im_los = axs[0, 0].imshow(results['MURaM']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[1, 0].imshow(results['MURaM']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[2, 0].imshow(np.rad2deg(results['MURaM']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)
im_los = axs[0, 1].imshow(results['PINN-ME']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[1, 1].imshow(results['PINN-ME']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[2, 1].imshow(np.rad2deg(results['PINN-ME']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)
im_los = axs[0, 2].imshow(results['PINN-ME PSF']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[1, 2].imshow(results['PINN-ME PSF']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[2, 2].imshow(np.rad2deg(results['PINN-ME PSF']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)
im_los = axs[0, 3].imshow(results['PyMilne 1D']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[1, 3].imshow(results['PyMilne 1D']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[2, 3].imshow(np.rad2deg(results['PyMilne 1D']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)
im_los = axs[0, 4].imshow(results['PyMilne 2D']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[1, 4].imshow(results['PyMilne 2D']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[2, 4].imshow(np.rad2deg(results['PyMilne 2D']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

[ax.set_xlabel('X [Mm]', fontsize=8) for ax in axs[-1, :]]
[ax.set_ylabel('Y [Mm]', fontsize=8) for ax in axs[:, 0]]

[ax.set_xlim(subframe['x'], subframe['x'] + subframe['w']) for ax in axs.ravel()]
[ax.set_ylim(subframe['y'], subframe['y'] + subframe['h']) for ax in axs.ravel()]

[ax.set_xticklabels([]) for ax in axs[:-1, :].ravel()]
[ax.set_yticklabels([]) for ax in axs[:, 1:].ravel()]

[ax.set_xticks(range(60, 140, 20)) for ax in axs.ravel()]

fig.tight_layout()
fig.savefig(f'{output_path}/subframe.png', transparent=True, dpi=300)
plt.close(fig)


########################################################################################################################
# evaluate difference metrics

def _evaluate(b, b_ref):
    E_n = 1 - np.linalg.norm(b - b_ref, axis=-1).sum((0, 1)) / np.linalg.norm(b_ref, axis=-1).sum((0, 1))
    c_vec = np.sum((b_ref * b).sum(-1), (0, 1)) / np.sqrt(
        (b_ref ** 2).sum(-1).sum((0, 1)) * (b ** 2).sum(-1).sum((0, 1)))
    b_abs = np.linalg.norm(b - b_ref, axis=-1).mean((0, 1))
    #
    return {'E_n': E_n, 'c_vec': c_vec, 'b_abs': b_abs}


muram_ref = block_reduce(results['MURaM']['b_xyz'], (4, 4, 1), np.mean)

eval_PINNME = _evaluate(results['PINN-ME']['b_xyz'], muram_ref)
eval_PINNME_PSF = _evaluate(results['PINN-ME PSF']['b_xyz'], muram_ref)
eval_PyMilne1D = _evaluate(results['PyMilne 1D']['b_xyz'], muram_ref)
eval_PyMilne2D = _evaluate(results['PyMilne 2D']['b_xyz'], muram_ref)

# save to file, nicely formatted
with open(f'{output_path}/metrics.txt', 'w') as f:
    f.write(f'{"":<20}{"E_n":<10}{"c_vec":<10}{"b_abs":<10}\n')
    f.write(f'{"PINN-ME":<20}{eval_PINNME["E_n"]:<10.4f}{eval_PINNME["c_vec"]:<10.4f}{eval_PINNME["b_abs"]:<10.4f}\n')
    f.write(f'{"PINN-ME PSF":<20}{eval_PINNME_PSF["E_n"]:<10.4f}{eval_PINNME_PSF["c_vec"]:<10.4f}{eval_PINNME_PSF["b_abs"]:<10.4f}\n')
    f.write(f'{"PyMilne 1D":<20}{eval_PyMilne1D["E_n"]:<10.4f}{eval_PyMilne1D["c_vec"]:<10.4f}{eval_PyMilne1D["b_abs"]:<10.4f}\n')
    f.write(f'{"PyMilne 2D":<20}{eval_PyMilne2D["E_n"]:<10.4f}{eval_PyMilne2D["c_vec"]:<10.4f}{eval_PyMilne2D["b_abs"]:<10.4f}\n')