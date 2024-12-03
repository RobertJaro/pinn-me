import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
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
    'PINN-ME': os.path.join(input_path, 'muram_sunspot_v01.npz'),
    'PINN-ME PSF': os.path.join(input_path, 'muram_sunspot_psf_v01.npz')
}

results = {}
for k, path in paths.items():
    data = np.load(path)
    b = data['b_field'][0, :, :, 0]
    theta = data['theta'][0, :, :, 0]
    chi = data['chi'][0, :, :, 0]
    b_xyz = to_cartesian(b, theta, chi % np.pi)
    b_los = b * np.cos(theta)
    b_trv = b * np.sin(theta)
    azi = chi % np.pi
    results[k] = {'b_xyz': b_xyz, 'b_los': b_los, 'b_trv': b_trv, 'azi': azi}

########################################################################################################################
# plot example images

Mm_per_pix = 0.192
b_ref = results['PINN-ME']['b_los']
extent = [0, b_ref.shape[1] * Mm_per_pix, 0, b_ref.shape[0] * Mm_per_pix]

plot_kwargs = {'extent': extent, 'origin': 'lower'}

fig, axs = plt.subplots(2, 3, figsize=(10, 2))

b_max = 2.5e3#np.max(np.abs(results['PINN-ME']['b_los']))
im_los = axs[0, 0].imshow(results['PINN-ME']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[0, 1].imshow(results['PINN-ME']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[0, 2].imshow(np.rad2deg(results['PINN-ME']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

im_los = axs[1, 0].imshow(results['PINN-ME PSF']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[1, 1].imshow(results['PINN-ME PSF']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[1, 2].imshow(np.rad2deg(results['PINN-ME PSF']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

plt.colorbar(im_los, ax=axs[0, 0], label='$B_{LOS}$ [G]', location='top')
plt.colorbar(im_trv, ax=axs[0, 1], label='$B_{TRV}$ [G]', location='top')
plt.colorbar(im_azi, ax=axs[0, 2], label='Azimuth [deg]', location='top')

# divider = make_axes_locatable(axs[0, 0])
# cax = divider.append_axes('top', size='15%', pad=0.05)
# fig.colorbar(im_los, cax=cax, orientation='horizontal', label='$B_{LOS}$ [G]')
# cax.xaxis.set_ticks_position('top')
# cax.xaxis.set_label_position('top')
#
# divider = make_axes_locatable(axs[0, 1])
# cax = divider.append_axes('top', size='15%', pad=0.05)
# fig.colorbar(im_trv, cax=cax, orientation='horizontal', label='$B_{TRV}$ [G]')
# cax.xaxis.set_ticks_position('top')
# cax.xaxis.set_label_position('top')
#
# divider = make_axes_locatable(axs[0, 2])
# cax = divider.append_axes('top', size='15%', pad=0.05)
# fig.colorbar(im_azi, cax=cax, orientation='horizontal', label='Azimuth [deg]')
# cax.xaxis.set_ticks_position('top')
# cax.xaxis.set_label_position('top')

axs[0, 0].set_ylabel('Y [Mm]', fontsize=8)
axs[1, 0].set_ylabel('Y [Mm]', fontsize=8)
axs[1, 0].set_xlabel('X [Mm]', fontsize=8)
axs[1, 1].set_xlabel('X [Mm]', fontsize=8)
axs[1, 2].set_xlabel('X [Mm]', fontsize=8)

# [ax.set_xlim(0, 10) for ax in axs.ravel()]

[ax.set_xticklabels([]) for ax in axs[0, :]]
[ax.set_yticklabels([]) for ax in axs[:, 1:].ravel()]

# add rectangles to indicate the subframe
for ax in axs.ravel():
    ax.add_patch(plt.Rectangle((20, 1), 10, 10, linewidth=1, edgecolor='black', facecolor='none'))

fig.tight_layout()
fig.savefig(f'{output_path}/comparison.png', transparent=True, dpi=300)
plt.close(fig)


########################################################################################################################
# plot subframe

plot_kwargs = {'extent': extent, 'origin': 'lower'}

fig, axs = plt.subplots(2, 3, figsize=(4, 3))

im_los = axs[0, 0].imshow(results['PINN-ME']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[0, 1].imshow(results['PINN-ME']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[0, 2].imshow(np.rad2deg(results['PINN-ME']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

im_los = axs[1, 0].imshow(results['PINN-ME PSF']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
im_trv = axs[1, 1].imshow(results['PINN-ME PSF']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
im_azi = axs[1, 2].imshow(np.rad2deg(results['PINN-ME PSF']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

axs[0, 0].set_ylabel('Y [Mm]', fontsize=8)
axs[1, 0].set_ylabel('Y [Mm]', fontsize=8)
axs[1, 0].set_xlabel('X [Mm]', fontsize=8)
axs[1, 1].set_xlabel('X [Mm]', fontsize=8)
axs[1, 2].set_xlabel('X [Mm]', fontsize=8)

[ax.set_xlim(20, 30) for ax in axs.ravel()]
[ax.set_ylim(1, 11) for ax in axs.ravel()]

[ax.set_xticklabels([]) for ax in axs[0, :]]
[ax.set_yticklabels([]) for ax in axs[:, 1:].ravel()]

fig.tight_layout()
fig.savefig(f'{output_path}/subframe.png', transparent=True, dpi=300)
plt.close(fig)