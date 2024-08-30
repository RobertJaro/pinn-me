import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import block_reduce

from pme.evaluation.loader import PINNMEOutput, to_cartesian

file = "/glade/work/rce/for_momo/modelador_final_plage.fits"
pme_file_positional = '/glade/work/rjarolim/pinn_me/muram/muram_positional_v01/inversion.pme'
pme_file_gaussian = '/glade/work/rjarolim/pinn_me/muram/muram_gaussian_v01/inversion.pme'

result_dir = '/glade/work/rjarolim/pinn_me/muram/evaluation'
os.makedirs(result_dir, exist_ok=True)

with fits.open(file) as hdu:
    muram_params = hdu[0].data

height_index = np.argmin(np.abs(muram_params[0, :, 100, 100] + 1.5))

b_field = muram_params[4, height_index, :, :].T
b_incl = np.deg2rad(muram_params[6, height_index, :, :]).T
b_azi = np.deg2rad(muram_params[7, height_index, :, :]).T

# bin data
b_field = block_reduce(b_field, (4, 4), np.mean)
b_incl = block_reduce(b_incl, (4, 4), np.mean)
b_azi = block_reduce(b_azi, (4, 4), np.mean)

disamb = b_azi > np.pi

pme_positional = PINNMEOutput(pme_file_positional)
parameters_positional = pme_positional.load_cube()

pme_gaussian = PINNMEOutput(pme_file_gaussian)
parameters_gaussian = pme_gaussian.load_cube()

b_gt = to_cartesian(b_field, b_incl, b_azi)
b_positional = to_cartesian(parameters_positional['b_field'][0, :, :, 0],
                            parameters_positional['theta'][0, :, :, 0],
                            parameters_positional['chi'][0, :, :, 0] % np.pi, disamb=disamb)
b_gaussian = to_cartesian(parameters_gaussian['b_field'][0, :, :, 0],
                          parameters_gaussian['theta'][0, :, :, 0],
                          parameters_gaussian['chi'][0, :, :, 0] % np.pi, disamb=disamb)


fig, axs = plt.subplots(3, 4, figsize=(12, 8))

def _plot_b(row, b, b_ref, cbar=True):
    row[0].imshow(b[..., 0], vmin=-500, vmax=500, cmap='gray')
    row[1].imshow(b[..., 1], vmin=-500, vmax=500, cmap='gray')
    #
    im = row[2].imshow(b[..., 2], vmin=-500, vmax=500, cmap='gray')
    divider = make_axes_locatable(row[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label='[G]')
    #
    im = row[3].imshow(np.linalg.norm(b - b_ref, axis=-1), vmin=0, vmax=500, cmap='viridis')
    if cbar:
        divider = make_axes_locatable(row[3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, label='[G]')


_plot_b(axs[0], b_gt, b_gt, cbar=False)
_plot_b(axs[1], b_positional, b_gt)
_plot_b(axs[2], b_gaussian, b_gt)

axs[0, 3].set_visible(False)

axs[0, 0].set_title('Bx')
axs[0, 1].set_title('By')
axs[0, 2].set_title('Bz')
axs[1, 3].set_title(r'$|| \Delta B ||$')

axs[0, 0].set_ylabel('GT')
axs[1, 0].set_ylabel('Positional')
axs[2, 0].set_ylabel('Gaussian')

fig.tight_layout()
fig.savefig(os.path.join(result_dir, 'b_field.jpg'), dpi=300, transparent=True)
plt.close(fig)

# 2D histogram

bx_gt = b_gt[..., 0].ravel()
by_gt = b_gt[..., 1].ravel()
bz_gt = b_gt[..., 2].ravel()

bins = [np.linspace(-500, 500, 100), np.linspace(-500, 500, 100)]

fig, axs = plt.subplots(2, 4, figsize=(12, 5))

ax = axs[0, 0]
ax.hist2d(bx_gt, b_positional[..., 0].ravel(), bins=bins, cmap='cividis', norm=LogNorm())

ax = axs[0, 1]
ax.hist2d(by_gt, b_positional[..., 1].ravel(), bins=bins, cmap='cividis', norm=LogNorm())

ax = axs[0, 2]
ax.hist2d(bz_gt, b_positional[..., 2].ravel(), bins=bins, cmap='cividis', norm=LogNorm())

ax = axs[0, 3]
im = ax.hist2d(np.linalg.norm(b_gt, axis=-1).ravel(), np.linalg.norm(b_positional, axis=-1).ravel(),
               bins=[np.linspace(0, 500, 100), np.linspace(0, 500, 100)],
               cmap='cividis', norm=LogNorm())
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im[3], cax=cax, label='Counts')

ax = axs[1, 0]
ax.hist2d(bx_gt, b_gaussian[..., 0].ravel(), bins=bins, cmap='cividis', norm=LogNorm())

ax = axs[1, 1]
ax.hist2d(by_gt, b_gaussian[..., 1].ravel(), bins=bins, cmap='cividis', norm=LogNorm())

ax = axs[1, 2]
ax.hist2d(bz_gt, b_gaussian[..., 2].ravel(), bins=bins, cmap='cividis', norm=LogNorm())

ax = axs[1, 3]
im = ax.hist2d(np.linalg.norm(b_gt, axis=-1).ravel(), np.linalg.norm(b_gaussian, axis=-1).ravel(),
               bins=[np.linspace(0, 500, 100), np.linspace(0, 500, 100)],
               cmap='cividis', norm=LogNorm())
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im[3], cax=cax, label='Counts')

axs[0, 0].set_title('Bx')
axs[0, 1].set_title('By')
axs[0, 2].set_title('Bz')

[ax.set_xlabel('True [G]') for ax in axs[1]]
[ax.set_ylabel('Positional [G]') for ax in axs[0]]
[ax.set_ylabel('Gaussian [G]') for ax in axs[1]]

[ax.plot([-500, 500], [-500, 500], 'r--') for ax in axs.ravel()]
[ax.set_aspect('equal') for ax in axs.ravel()]

# axs[0, 0].set_ylabel('Positional')
# axs[1, 0].set_ylabel('Gaussian')

fig.tight_layout()
fig.savefig(os.path.join(result_dir, 'b_field_hist.jpg'), dpi=300, transparent=True)
plt.close(fig)

print('=============== Difference ===============')
print(f'Positional: {np.linalg.norm(b_positional - b_gt, axis=-1).mean()}')
print(f'Gaussian: {np.linalg.norm(b_gaussian - b_gt, axis=-1).mean()}')
