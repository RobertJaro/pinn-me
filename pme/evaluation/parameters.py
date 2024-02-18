import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pme.evaluation.loader import PINNMEOutput, to_cartesian

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file')
parser.add_argument('--reference', type=str, help='the path to the reference file')

args = parser.parse_args()

pinnme = PINNMEOutput(args.input)

parameters_pred = pinnme.load_cube()
parameters_true = np.load(args.reference)

chi_pred = parameters_pred['chi'] % np.pi
chi_true = parameters_true['chi'] % np.pi

b_true = to_cartesian(parameters_true['BField'], parameters_true['theta'], parameters_true['chi'] % np.pi)
b_pred = to_cartesian(parameters_pred['b_field'], parameters_pred['theta'], parameters_pred['chi'] % np.pi)

title_font = {'fontsize': 20}


def _plot_hist2d(x, y, ax, label):
    global _, im, divider, cax
    _, _, _, im = ax.hist2d(x, y.flatten(), bins=n_bins, cmap='cividis', norm=LogNorm())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    min_lim = min(x.min(), y.min())
    max_lim = max(x.max(), y.max())
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    coef = np.polyfit(x, y, 1)

    ax.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)],
            linewidth=2, linestyle='--', color='red', alpha=0.7)

    ax.set_title(f'{label}', fontdict=title_font)
    ax.plot(np.nan, np.nan, '', color='none', label=f'lf: {coef[0]:.2f}')
    ax.plot(np.nan, np.nan, '', color='none', label=f'cc: {np.corrcoef(x, y)[0, 1]:.2f}')
    ax.legend(loc='upper left', fancybox=True, framealpha=0.7, fontsize=15, handlelength=0)

# 2d histogram of parameters
# 1: row b, theta, chi
# 2: Bx, By, Bz
# 3: row B0, B1, vmac
# 4: row damping, mu, kl
fig, axs = plt.subplots(4, 3, figsize=(15, 15), dpi=300)

######################################### Angles #########################################
n_bins = 500

_plot_hist2d(parameters_pred['b_field'].flatten(),
             parameters_true['BField'].flatten(),
             axs[0, 0], 'B field')

_plot_hist2d(parameters_pred['theta'].flatten(),
             parameters_true['theta'].flatten(),
             axs[0, 1], 'Theta')

_plot_hist2d(chi_pred.flatten(),
             chi_true.flatten(),
             axs[0, 2], 'Chi')

######################################### B Vector #########################################
_plot_hist2d(b_pred[..., 0].flatten(),
             b_true[..., 0].flatten(),
             axs[1, 0], r'$\vec{B}_x$')

_plot_hist2d(b_pred[..., 1].flatten(),
             b_true[..., 1].flatten(),
             axs[1, 1], r'$\vec{B}_y$')

_plot_hist2d(b_pred[..., 2].flatten(),
             b_true[..., 2].flatten(),
             axs[1, 2], r'$\vec{B}_z$')

######################################### Parameters B0 B1 #########################################
_plot_hist2d(parameters_pred['b0'].flatten(),
             parameters_true['B0'].flatten(),
             axs[2, 0], 'B0')

_plot_hist2d((parameters_pred['b1'] * parameters_pred['mu']).flatten(),
             (parameters_true['B1'] * parameters_true['mu']).flatten(),
             axs[2, 1], 'B1 $\cdot$ mu')

axs[2, 2].axis('off')

######################################### Parameters vmac damping #########################################
axs[3, 0].hist(parameters_pred['vmac'].flatten(), bins=n_bins, color='b')
axs[3, 0].axvline(parameters_true['vmac'].mean(), color='r', linestyle='dashed', linewidth=2, label='True')
axs[3, 0].legend(loc='upper left', fontsize=15, title_fontsize='15', fancybox=True, framealpha=0.7)
axs[3, 0].semilogy()
diff = np.abs(parameters_pred['vmac'].flatten() - parameters_true['vmac'].flatten()).mean()
axs[3, 0].set_title(f'vmac - $\overline{{\Delta}}$: {diff:.03f}', fontdict=title_font)

axs[3, 1].hist(parameters_pred['damping'].flatten(), bins=n_bins, color='b')
axs[3, 1].axvline(parameters_true['damping'].mean(), color='r', linestyle='dashed', linewidth=2, label='True')
axs[3, 1].legend(loc='upper left', fontsize=15, title_fontsize='15', fancybox=True, framealpha=0.7)
axs[3, 1].semilogy()
diff = np.abs(parameters_pred['damping'].flatten() - parameters_true['damping'].flatten()).mean()
axs[3, 1].set_title(rf'damping - $\overline{{\Delta}}$: {diff:.03f}', fontdict=title_font)

axs[3, 2].axis('off')

for ax in np.ravel(axs):
    ax.set_xlabel('PINN-ME', fontdict={'fontsize': 15})
    ax.set_ylabel('True', fontdict={'fontsize': 15})

axs[3, 0].set_xlabel('PINN-ME', fontdict={'fontsize': 15})
axs[3, 0].set_ylabel('Counts', fontdict={'fontsize': 15})

axs[3, 1].set_xlabel('PINN-ME', fontdict={'fontsize': 15})
axs[3, 1].set_ylabel('Counts', fontdict={'fontsize': 15})

fig.tight_layout()
fig.savefig(args.output)
plt.close(fig)
