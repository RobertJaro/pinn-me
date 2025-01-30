import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pme.evaluation.loader import to_cartesian

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file')
parser.add_argument('--reference_file', type=str, help='the path to the reference file')

args = parser.parse_args()

input_path = args.input
output_path = args.output
ref_file = args.reference_file

os.makedirs(output_path, exist_ok=True)


no_psf_paths = {
    'Clear': os.path.join(input_path, 'clear_v01.npz'),
    'PSF': os.path.join(input_path, 'no_psf_0.0_v01.npz'),
    '1e-4': os.path.join(input_path, 'no_psf_1.0e-4_v01.npz'),
    '1e-3': os.path.join(input_path, 'no_psf_1.0e-3_v01.npz'),
    '1e-2': os.path.join(input_path, 'no_psf_1.0e-2_v01.npz'),
}

psf_paths = {
    'Clear': None,
    'PSF': os.path.join(input_path, 'psf_0.0_v01.npz'),
    '1e-4': os.path.join(input_path, 'psf_1.0e-4_v01.npz'),
    '1e-3': os.path.join(input_path, 'psf_1.0e-3_v01.npz'),
    '1e-2': os.path.join(input_path, 'psf_1.0e-2_v01.npz'),
}

static_paths = {
    'Clear': os.path.join(input_path, 'static_clear_v01.npz'),
    'PSF': os.path.join(input_path, 'static_psf_0.0_v01.npz'),
    '1e-4': os.path.join(input_path, 'static_psf_1.0e-4_v01.npz'),
    '1e-3': os.path.join(input_path, 'static_psf_1.0e-3_v01.npz'),
    '1e-2': os.path.join(input_path, 'static_psf_1.0e-2_v01.npz'),
}

pymilne_1D = {
    'Clear': os.path.join(input_path, 'pymilne_testset_1D_clear_v01.npz'),
    'PSF': os.path.join(input_path, 'pymilne_testset_1D_1.0e-8_v01.npz'),
    '1e-4': os.path.join(input_path, 'pymilne_testset_1D_1.0e-4_v01.npz'),
    '1e-3': os.path.join(input_path, 'pymilne_testset_1D_1.0e-3_v01.npz'),
    '1e-2': os.path.join(input_path, 'pymilne_testset_1D_1.0e-2_v01.npz'),
}

pymilne_2D = {
    'Clear': None,
    'PSF': os.path.join(input_path,   'pymilne_testset_2D_1e-8_v01.npz'),
    '1e-4': os.path.join(input_path,  'pymilne_testset_2D_1e-4_v01.npz'),
    '1e-3': os.path.join(input_path,  'pymilne_testset_2D_1e-3_v01.npz'),
    '1e-2': os.path.join(input_path,  'pymilne_testset_2D_1e-2_v01.npz'),
}

parameters_true = np.load(ref_file)
b_true = to_cartesian(parameters_true['b_field'], parameters_true['theta'], parameters_true['chi'] % np.pi)

b_los = parameters_true['b_field'] * np.cos(parameters_true['theta'])
b_trv = parameters_true['b_field'] * np.sin(parameters_true['theta'])
azi = parameters_true['chi'] % np.pi

true_data = {'b_xyz': b_true, 'b_los': b_los, 'b_trv': b_trv, 'azi': azi}

title_font = {'fontsize': 20}


def _angle_diff(b_pred, b_true):
    dot = (b_pred * b_true).sum(-1)
    norm = np.linalg.norm(b_pred, axis=-1) * np.linalg.norm(b_true, axis=-1) + 1e-6
    arg = dot / norm
    angle = np.arccos(arg)
    angle[(arg < -1) | (arg > 1)] = 0
    angle = np.rad2deg(angle)
    return angle


def _evaluate(b, b_ref):
    E_n = 1 - np.linalg.norm(b - b_ref, axis=-1).sum((0, 1)) / np.linalg.norm(b_ref, axis=-1).sum((0, 1))
    c_vec = np.sum((b_ref * b).sum(-1), (0, 1)) / np.sqrt(
        (b_ref ** 2).sum(-1).sum((0, 1)) * (b ** 2).sum(-1).sum((0, 1)))

    return {'E_n': E_n, 'c_vec': c_vec}


no_psf_b = {}
for key, path in no_psf_paths.items():
    if path:
        data = np.load(path)
        b_xyz = to_cartesian(data['b_field'], data['theta'], data['chi'] % np.pi)[9, :, :, 0]
        b_los = data['b_field'] * np.cos(data['theta'])
        b_trv = data['b_field'] * np.sin(data['theta'])
        azi = data['chi'] % np.pi
        metrics = _evaluate(b_xyz, b_true)
        no_psf_b[key] = {'b_xyz': b_xyz, 'b_los': b_los, 'b_trv': b_trv, 'azi': azi, **metrics}

psf_b = {}
for key, path in psf_paths.items():
    if path:
        data = np.load(path)
        b_xyz = to_cartesian(data['b_field'], data['theta'], data['chi'] % np.pi)[9, :, :, 0]
        b_los = data['b_field'] * np.cos(data['theta'])
        b_trv = data['b_field'] * np.sin(data['theta'])
        azi = data['chi'] % np.pi
        metrics = _evaluate(b_xyz, b_true)
        psf_b[key] = {'b_xyz': b_xyz, 'b_los': b_los, 'b_trv': b_trv, 'azi': azi, **metrics}

static_b = {}
for key, path in static_paths.items():
    if path:
        data = np.load(path)
        b_xyz = to_cartesian(data['b_field'], data['theta'], data['chi'] % np.pi)[0, :, :, 0]
        b_los = data['b_field'] * np.cos(data['theta'])
        b_trv = data['b_field'] * np.sin(data['theta'])
        azi = data['chi'] % np.pi
        metrics = _evaluate(b_xyz, b_true)
        static_b[key] = {'b_xyz': b_xyz, 'b_los': b_los, 'b_trv': b_trv, 'azi': azi, **metrics}

pymilne_1D_b = {}
for key, path in pymilne_1D.items():
    if path:
        data = np.load(path)
        b_xyz = to_cartesian(data['b_field'], data['theta'], data['chi'] % np.pi)
        b_los = data['b_field'] * np.cos(data['theta'])
        b_trv = data['b_field'] * np.sin(data['theta'])
        azi = data['chi'] % np.pi
        metrics = _evaluate(b_xyz, b_true)
        pymilne_1D_b[key] = {'b_xyz': b_xyz, 'b_los': b_los, 'b_trv': b_trv, 'azi': azi, **metrics}

pymilne_2D_b = {}
for key, path in pymilne_2D.items():
    if path:
        data = np.load(path)
        b_field = data['b_field'].T
        theta = data['theta'].T
        chi = data['chi'].T
        b_xyz = to_cartesian(b_field, theta, chi % np.pi)
        b_los = b_field * np.cos(theta)
        b_trv = b_field * np.sin(theta)
        azi = chi % np.pi
        metrics = _evaluate(b_xyz, b_true)
        pymilne_2D_b[key] = {'b_xyz': b_xyz, 'b_los': b_los, 'b_trv': b_trv, 'azi': azi, **metrics}


########################################################################################################################
# plot noise comparison

x_labels = no_psf_paths.keys()

fig, axs = plt.subplots(2, 1, figsize=(6, 4.5))

ax = axs[0]
ax.plot(range(0, 5), [val['E_n'] for val in static_b.values()], marker='o', label='PINN ME Static', alpha=0.5)
ax.plot(range(0, 5), [val['E_n'] for val in no_psf_b.values()], marker='o', label='PINN ME', alpha=0.5)
ax.plot(range(1, 5), [val['E_n'] for val in psf_b.values()], marker='o', label='PINN ME PSF', alpha=0.5)
ax.plot(range(0, 5), [val['E_n'] for val in pymilne_1D_b.values()], marker='o', label='PyMilne', alpha=0.5)
ax.plot(range(1, 5), [val['E_n'] for val in pymilne_2D_b.values()], marker='o', label='PyMilne PSF', alpha=0.5)

ax.set_ylabel(r'$E_\text{n}$', fontsize=18)

ax = axs[1]
ax.plot(range(0, 5), [val['c_vec'] for val in static_b.values()], marker='o', label='PINN ME Static', alpha=0.5)
ax.plot(range(0, 5), [val['c_vec'] for val in no_psf_b.values()], marker='o', label='PINN ME', alpha=0.5)
ax.plot(range(1, 5), [val['c_vec'] for val in psf_b.values()], marker='o', label='PINN ME PSF', alpha=0.5)
ax.plot(range(0, 5), [val['c_vec'] for val in pymilne_1D_b.values()], marker='o', label='PyMilne', alpha=0.5)
ax.plot(range(1, 5), [val['c_vec'] for val in pymilne_2D_b.values()], marker='o', label='PyMilne PSF', alpha=0.5)

ax.set_xlabel('Degradation', fontsize=14)
ax.set_ylabel(r'$C_\text{vec}$', fontsize=18)

axs[0].legend(loc='lower left', fancybox=True, framealpha=0.7, fontsize=13)
[ax.set_xticks(range(0, 5)) for ax in axs]
[ax.set_xticklabels(x_labels) for ax in axs]

fig.tight_layout()
fig.savefig(f'{output_path}/noise_comparison.png', transparent=True, dpi=300)
plt.close(fig)

########################################################################################################################
# plot example images for 1e-3
target_noise = '1e-2'

fig, axs = plt.subplots(6, 3, figsize=(6, 12.5))

im_los = axs[0, 0].imshow(true_data['b_los'], cmap='RdBu_r', vmin=-1000, vmax=1000)
im_trv = axs[0, 1].imshow(true_data['b_trv'], cmap='cividis', vmin=0, vmax=1000)
im_azi = axs[0, 2].imshow(np.rad2deg(true_data['azi']), cmap='twilight', vmin=0, vmax=180)

im_los = axs[1, 0].imshow(static_b[target_noise]['b_los'][0, :, :, 0], cmap='RdBu_r', vmin=-1000, vmax=1000)
im_trv = axs[1, 1].imshow(static_b[target_noise]['b_trv'][0, :, :, 0], cmap='cividis', vmin=0, vmax=1000)
im_azi = axs[1, 2].imshow(np.rad2deg(static_b[target_noise]['azi'][0, :, :, 0]), cmap='twilight', vmin=0, vmax=180)

im_los = axs[2, 0].imshow(no_psf_b[target_noise]['b_los'][9, :, :, 0], cmap='RdBu_r', vmin=-1000, vmax=1000)
im_trv = axs[2, 1].imshow(no_psf_b[target_noise]['b_trv'][9, :, :, 0], cmap='cividis', vmin=0, vmax=1000)
im_azi = axs[2, 2].imshow(np.rad2deg(no_psf_b[target_noise]['azi'][9, :, :, 0]), cmap='twilight', vmin=0, vmax=180)

im_los = axs[3, 0].imshow(psf_b[target_noise]['b_los'][9, :, :, 0], cmap='RdBu_r', vmin=-1000, vmax=1000)
im_trv = axs[3, 1].imshow(psf_b[target_noise]['b_trv'][9, :, :, 0], cmap='cividis', vmin=0, vmax=1000)
im_azi = axs[3, 2].imshow(np.rad2deg(psf_b[target_noise]['azi'][9, :, :, 0]), cmap='twilight', vmin=0, vmax=180)

im_los = axs[4, 0].imshow(pymilne_1D_b[target_noise]['b_los'], cmap='RdBu_r', vmin=-1000, vmax=1000)
im_trv = axs[4, 1].imshow(pymilne_1D_b[target_noise]['b_trv'], cmap='cividis', vmin=0, vmax=1000)
im_azi = axs[4, 2].imshow(np.rad2deg(pymilne_1D_b[target_noise]['azi']), cmap='twilight', vmin=0, vmax=180)

im_los = axs[5, 0].imshow(pymilne_2D_b[target_noise]['b_los'], cmap='RdBu_r', vmin=-1000, vmax=1000)
im_trv = axs[5, 1].imshow(pymilne_2D_b[target_noise]['b_trv'], cmap='cividis', vmin=0, vmax=1000)
im_azi = axs[5, 2].imshow(np.rad2deg(pymilne_2D_b[target_noise]['azi']), cmap='twilight', vmin=0, vmax=180)

divider = make_axes_locatable(axs[-1, 0])
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im_los, cax=cax, orientation='horizontal', label='$B_{LOS}$ [G]')

divider = make_axes_locatable(axs[-1, 1])
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im_trv, cax=cax, orientation='horizontal', label='$B_{TRV}$ [G]')

divider = make_axes_locatable(axs[-1, 2])
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im_azi, cax=cax, orientation='horizontal', label='Azimuth [deg]')

[ax.set_xticks([]) for ax in axs.ravel()]
[ax.set_yticks([]) for ax in axs.ravel()]

axs[0, 0].set_ylabel('Ground-Truth', fontsize=12)
axs[1, 0].set_ylabel('PINN ME Static', fontsize=12)
axs[2, 0].set_ylabel('PINN ME', fontsize=12)
axs[3, 0].set_ylabel('PINN ME PSF', fontsize=12)
axs[4, 0].set_ylabel('PyMilne', fontsize=12)
axs[5, 0].set_ylabel('PyMilne PSF', fontsize=12)

fig.tight_layout()
fig.savefig(f'{output_path}/comparison.png', transparent=True, dpi=300)
plt.close(fig)
