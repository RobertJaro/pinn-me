import argparse
import io
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pme.evaluation.loader import PINNMEOutput, to_cartesian, to_spherical

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file')
parser.add_argument('--reference', type=str, help='the path to the reference file')

args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

pme_paths = {'1e-2': '/glade/work/rjarolim/pinn_me/test_data/test_psf_noise_1e-2/inversion.pme',
         '1e-3': '/glade/work/rjarolim/pinn_me/test_data/test_psf_noise_1e-3/inversion.pme',
         '1e-4': '/glade/work/rjarolim/pinn_me/test_data/test_psf_noise_1e-4/inversion.pme',
         '1e-5': '/glade/work/rjarolim/pinn_me/test_data/test_psf_noise_1e-5/inversion.pme',
         'PSF': '/glade/work/rjarolim/pinn_me/test_data/test_psf/inversion.pme',
         'Clear': None
             }

pmes_paths = {'1e-2': '/glade/work/rjarolim/pinn_me/test_data/test_simple_noise_1e-2/inversion.pme',
            '1e-3': '/glade/work/rjarolim/pinn_me/test_data/test_simple_noise_1e-3/inversion.pme',
            '1e-4': '/glade/work/rjarolim/pinn_me/test_data/test_simple_noise_1e-4/inversion.pme',
            '1e-5': '/glade/work/rjarolim/pinn_me/test_data/test_simple_noise_1e-5/inversion.pme',
            'PSF': '/glade/work/rjarolim/pinn_me/test_data/test_simple_psf/inversion.pme',
            'Clear': '/glade/work/rjarolim/pinn_me/test_data/test_clear/inversion.pme'
                }

parameters_true = np.load(args.reference)
b_true = to_cartesian(parameters_true['BField'], parameters_true['theta'], parameters_true['chi'] % np.pi)


def _load_PINNME(paths):
    pme_b = {}
    for idx, path in paths.items():
        if path is None:
            pme_b[idx] = np.zeros_like(b_true) * np.nan
            continue
        pinnme = PINNMEOutput(path)

        parameters_pred = pinnme.load_cube()
        b_pred = to_cartesian(parameters_pred['b_field'], parameters_pred['theta'], parameters_pred['chi'] % np.pi)
        pme_b[idx] = b_pred
    return pme_b


pme_b = _load_PINNME(pme_paths)
pmes_b = _load_PINNME(pmes_paths)





title_font = {'fontsize': 20}

def _angle_diff(b_pred, b_true):
    dot = (b_pred * b_true).sum(-1)
    norm = np.linalg.norm(b_pred, axis=-1) * np.linalg.norm(b_true, axis=-1) + 1e-6
    arg = dot / norm
    angle = np.arccos(arg)
    angle[(arg < -1) | (arg > 1)] = 0
    angle = np.rad2deg(angle)
    return angle

def _evaluate(pme_b):
    pme_angle_diff = [_angle_diff(pme_b[idx], b_true).mean() for idx in pme_b.keys()]
    pme_differences = [np.linalg.norm(pme_b[idx] - b_true, axis=-1).mean() for idx in pme_b.keys()]
    pme_difference_percent = [np.linalg.norm(pme_b[idx] - b_true, axis=-1).mean() / np.linalg.norm(b_true, axis=-1).max() for idx in pme_b.keys()]
    return pme_differences, pme_difference_percent, pme_angle_diff

pme_differences, pme_difference_percent, pme_angle_diff = _evaluate(pme_b)
pmes_differences, pmes_difference_percent, pmes_angle_diff = _evaluate(pmes_b)

fig, axs = plt.subplots(3, 1, figsize=(6, 6))

ax = axs[0]
ax.plot(list(pme_b.keys()), pme_differences, marker='o', label='PINN ME PSF')
ax.plot(list(pmes_b.keys()), pmes_differences, marker='o', label='PINN ME')
ax.set_xlabel('Noise level')
ax.set_ylabel(r'$\Vert$B$\Vert$ [Gauss]')
ax.semilogy()

ax = axs[1]
ax.plot(list(pme_b.keys()), pme_difference_percent, marker='o', label='PINN ME PSF')
ax.plot(list(pmes_b.keys()), pmes_difference_percent, marker='o', label='PINN ME')
ax.set_xlabel('Noise level')
ax.set_ylabel(r'$\Vert$B$\Vert$ [%]')

ax = axs[2]
ax.plot(list(pme_b.keys()), pme_angle_diff, marker='o', label='PINN ME PSF')
ax.plot(list(pmes_b.keys()), pmes_angle_diff, marker='o', label='PINN ME')
ax.set_xlabel('Noise level')
ax.set_ylabel('$\sigma$ [deg]')

[ax.legend(loc='upper right', fancybox=True, framealpha=0.7, fontsize=13) for ax in axs]

fig.tight_layout()
fig.savefig(f'{args.output}/noise_comparison.png', transparent=True)
plt.close(fig)