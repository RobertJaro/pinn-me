import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pme.evaluation.loader import to_cartesian

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file')

args = parser.parse_args()

input_path = args.input
output_path = args.output

os.makedirs(output_path, exist_ok=True)

ref_b = fits.getdata('/glade/campaign/hao/radmhd/rjarolim/PINN-ME/hinode/20070105_235907.fits', 1)
ref_inc = fits.getdata('/glade/campaign/hao/radmhd/rjarolim/PINN-ME/hinode/20070105_235907.fits', 2)
ref_azi = fits.getdata('/glade/campaign/hao/radmhd/rjarolim/PINN-ME/hinode/20070105_235907.fits', 3)
ref_stray_light = fits.getdata('/glade/campaign/hao/radmhd/rjarolim/PINN-ME/hinode/20070105_235907.fits', 12)

ref_b_los = ref_b * np.cos(np.deg2rad(ref_inc)) * ref_stray_light
ref_b_trv = ref_b * np.sin(np.deg2rad(ref_inc)) * np.sqrt(ref_stray_light)
ref_b_azi = np.deg2rad(ref_azi)

paths = {
    # 'PINN-ME': os.path.join(input_path, '20070105_v01.npz'),
    'PINN-ME PSF': os.path.join(input_path, '20070105_psf_v01.npz')
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
    I = np.abs(data['I']).sum(axis=-1)[0]
    Q = np.abs(data['Q']).sum(axis=-1)[0]
    U = np.abs(data['U']).sum(axis=-1)[0]
    V = np.abs(data['V']).sum(axis=-1)[0]
    results[k] = {'b_xyz': b_xyz, 'b_los': b_los, 'b_trv': b_trv, 'azi': azi, 'b': b, 'theta': theta, 'chi': chi,
                  'I': I, 'Q': Q, 'U': U, 'V': V}

########################################################################################################################
# plot example images

subframe_1 = {'x': 34, 'y': 20, 'w': 5, 'h': 5, 'color': 'm'}
subframe_2 = {'x': 35, 'y': 11, 'w': 8, 'h': 8, 'color': 'g'}

Mm_per_pix = 0.07
b_ref = results['PINN-ME PSF']['b_los']
b_max_los = np.abs(b_ref).max()
b_max_trv = np.abs(results['PINN-ME PSF']['b_trv']).max()
extent = [0, b_ref.shape[1] * Mm_per_pix, 0, b_ref.shape[0] * Mm_per_pix]

plot_kwargs = {'extent': extent, 'origin': 'lower'}

fig, axs = plt.subplots(2, 3, figsize=(10, 4.5))

im_los = axs[0, 0].imshow(results['PINN-ME PSF']['b_los'], cmap='RdBu_r', vmin=-b_max_los, vmax=b_max_los, **plot_kwargs)
im_trv = axs[0, 1].imshow(results['PINN-ME PSF']['b_trv'], cmap='cividis', vmin=0, vmax=b_max_trv, **plot_kwargs)
im_azi = axs[0, 2].imshow(np.rad2deg(results['PINN-ME PSF']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

im_los = axs[1, 0].imshow(ref_b_los, cmap='RdBu_r', vmin=-b_max_los, vmax=b_max_los, **plot_kwargs)
im_trv = axs[1, 1].imshow(ref_b_trv, cmap='cividis', vmin=0, vmax=b_max_trv, **plot_kwargs)
im_azi = axs[1, 2].imshow(np.rad2deg(ref_b_azi), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

divider = make_axes_locatable(axs[0, 0])
cax = divider.append_axes('top', size='5%', pad=0.05)
fig.colorbar(im_los, cax=cax, orientation='horizontal', label=r'$B_\text{LOS}$ [G]')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

divider = make_axes_locatable(axs[0, 1])
cax = divider.append_axes('top', size='5%', pad=0.05)
fig.colorbar(im_trv, cax=cax, orientation='horizontal', label=r'$B_\text{TRV}$ [G]')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

divider = make_axes_locatable(axs[0, 2])
cax = divider.append_axes('top', size='5%', pad=0.05)
fig.colorbar(im_azi, cax=cax, orientation='horizontal', label=r'$\phi$ [deg]')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

[ax.set_xticklabels([]) for ax in axs[0, :]]
[ax.set_yticklabels([]) for ax in axs[:, 1:].ravel()]

# add axis labels
[ax.set_xlabel('X [Mm]') for ax in axs[-1, :]]
[ax.set_ylabel('Y [Mm]') for ax in axs[:, 0]]

# add rectangles to indicate the subframe
for ax in axs.ravel():
    ax.add_patch(plt.Rectangle((subframe_1['x'], subframe_1['y']), subframe_1['w'], subframe_1['h'],
                               edgecolor=subframe_1['color'], facecolor='none'))
    ax.add_patch(plt.Rectangle((subframe_2['x'], subframe_2['y']), subframe_2['w'], subframe_2['h'],
                               edgecolor=subframe_2['color'], facecolor='none'))

fig.tight_layout()
fig.savefig(f'{output_path}/comparison.png', transparent=True, dpi=300)
plt.close(fig)

########################################################################################################################
# plot subframe

def _change_label_format(cax):
    labels = [item.get_text() for item in cax.get_xticklabels()]
    labels = [label.replace('−', '-') for label in labels]
    labels = [f'{int(label) / 1000:.0f}e3' if abs(float(label)) >= 1e3 else label for label in labels]
    cax.set_xticklabels(labels)

def _plot_subframe(subframe, name, ax_color, b_max=2.0e3, azi_diff_range = 90):
    plot_kwargs = {'extent': extent, 'origin': 'lower'}

    fig, axs = plt.subplots(3, 3, figsize=(5, 6))

    im_los = axs[0, 0].imshow(results['PINN-ME PSF']['b_los'], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
    im_trv = axs[0, 1].imshow(results['PINN-ME PSF']['b_trv'], cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
    im_azi = axs[0, 2].imshow(np.rad2deg(results['PINN-ME PSF']['azi']), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

    im_los = axs[1, 0].imshow(ref_b_los, cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
    im_trv = axs[1, 1].imshow(ref_b_trv, cmap='cividis', vmin=0, vmax=b_max, **plot_kwargs)
    im_azi = axs[1, 2].imshow(np.rad2deg(ref_b_azi), cmap='twilight', vmin=0, vmax=180, **plot_kwargs)

    im_los_diff = axs[2, 0].imshow(np.abs(results['PINN-ME PSF']['b_los']) - np.abs(ref_b_los), cmap='seismic', vmin=-b_max, vmax=b_max, **plot_kwargs)
    im_trv_diff = axs[2, 1].imshow(results['PINN-ME PSF']['b_trv'] - ref_b_trv, cmap='seismic', vmin=-b_max, vmax=b_max, **plot_kwargs)
    im_azi_diff = axs[2, 2].imshow(np.rad2deg(results['PINN-ME PSF']['azi']) - np.rad2deg(ref_b_azi), cmap='PiYG', vmin=-azi_diff_range, vmax=azi_diff_range, **plot_kwargs)


    [ax.set_xlim(subframe['x'], subframe['x'] + subframe['w']) for ax in axs.ravel()]
    [ax.set_ylim(subframe['y'], subframe['y'] + subframe['h']) for ax in axs.ravel()]

    [ax.set_xticklabels([]) for ax in axs.ravel()]
    [ax.set_yticklabels([]) for ax in axs[:, 1:].ravel()]

    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(im_los, cax=cax, orientation='horizontal', label=r'$B_\text{LOS}$ [G]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    _change_label_format(cax)

    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(im_trv, cax=cax, orientation='horizontal', label=r'$B_\text{TRV}$ [G]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    _change_label_format(cax)

    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(im_azi, cax=cax, orientation='horizontal', label=r'$\phi$ [deg]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    divider = make_axes_locatable(axs[2, 0])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(im_los_diff, cax=cax, orientation='horizontal', label=r'$\Delta |B_\text{LOS}|$ [G]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    _change_label_format(cax)

    divider = make_axes_locatable(axs[2, 1])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(im_trv_diff, cax=cax, orientation='horizontal', label=r'$\Delta B_\text{TRV}$ [G]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    _change_label_format(cax)

    divider = make_axes_locatable(axs[2, 2])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(im_azi_diff, cax=cax, orientation='horizontal', label=r'$\Delta \phi$ [deg]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # add axis labels
    [ax.set_xlabel('X [Mm]', color=ax_color) for ax in axs[-1, :]]
    [ax.set_ylabel('Y [Mm]', color=ax_color) for ax in axs[:, 0]]

    # change axis color
    for ax in axs.ravel():
        ax.spines['bottom'].set_color(ax_color)
        ax.spines['top'].set_color(ax_color)
        ax.spines['right'].set_color(ax_color)
        ax.spines['left'].set_color(ax_color)
        ax.tick_params(axis='x', colors=ax_color)
        ax.tick_params(axis='y', colors=ax_color)

    fig.tight_layout()
    fig.savefig(f'{output_path}/{name}.png', transparent=True, dpi=300)
    plt.close(fig)

_plot_subframe(subframe_1, 'subframe_1', subframe_1['color'], b_max=1.2e3, azi_diff_range = 45)
_plot_subframe(subframe_2, 'subframe_2', subframe_2['color'], b_max=2.4e3, azi_diff_range = 45)

########################################################################################################################
# save individual images

b = results['PINN-ME PSF']['b'][:400, 380:780]
theta = results['PINN-ME PSF']['theta'][:400, 380:780]
chi = results['PINN-ME PSF']['chi'][:400, 380:780]

I = results['PINN-ME']['I'][:400, 380:780]
Q = results['PINN-ME']['Q'][:400, 380:780]
U = results['PINN-ME']['U'][:400, 380:780]
V = results['PINN-ME']['V'][:400, 380:780]

print('Value ranges:')
print(f'I: {I.min()} - {I.max()}')
print(f'Q: {Q.min()} - {Q.max()}')
print(f'U: {U.min()} - {U.max()}')
print(f'V: {V.min()} - {V.max()}')

plt.imsave(f'{output_path}/b.png', b, cmap='viridis', vmin=0, vmax=b_max_los)
plt.imsave(f'{output_path}/theta.png', theta, cmap='seismic', vmin=0, vmax=np.pi)
plt.imsave(f'{output_path}/chi.png', chi, cmap='twilight', vmin=-np.pi, vmax=np.pi)

i_max = np.max(I)
q_min_max = np.max(np.abs(Q))
u_min_max = np.max(np.abs(U))
v_min_max = np.max(np.abs(V))

plt.imsave(f'{output_path}/I.png', I, cmap='plasma', vmin=0, vmax=i_max)
plt.imsave(f'{output_path}/Q.png', Q, cmap='plasma', vmin=0, vmax=q_min_max)
plt.imsave(f'{output_path}/U.png', U, cmap='plasma', vmin=0, vmax=u_min_max)
plt.imsave(f'{output_path}/V.png', V, cmap='plasma', vmin=0, vmax=v_min_max)

observation_files = sorted(glob.glob("/glade/campaign/hao/radmhd/rjarolim/PINN-ME/hinode/20070105_235907/*.fits"))
stokes_vector = np.stack([fits.getdata(f) for f in observation_files], 2, dtype=np.float32)
stokes_vector = stokes_vector[..., -56:]
max_I = stokes_vector[0].max()
stokes_vector /= max_I * 1.1

I = np.abs(stokes_vector[0]).sum(-1)[:400, 380:780]
Q = np.abs(stokes_vector[1]).sum(-1)[:400, 380:780]
U = np.abs(stokes_vector[2]).sum(-1)[:400, 380:780]
V = np.abs(stokes_vector[3]).sum(-1)[:400, 380:780]

# TODO remove and use same value range
# i_max = np.max(I)
# q_min_max = np.max(np.abs(Q))
# u_min_max = np.max(np.abs(U))
# v_min_max = np.max(np.abs(V))

print('Value ranges:')
print(f'I: {I.min()} - {I.max()}')
print(f'Q: {Q.min()} - {Q.max()}')
print(f'U: {U.min()} - {U.max()}')
print(f'V: {V.min()} - {V.max()}')

plt.imsave(f'{output_path}/I_obs.png', I, cmap='plasma', vmin=0, vmax=i_max)
plt.imsave(f'{output_path}/Q_obs.png', Q, cmap='plasma', vmin=0, vmax=q_min_max)
plt.imsave(f'{output_path}/U_obs.png', U, cmap='plasma', vmin=0, vmax=u_min_max)
plt.imsave(f'{output_path}/V_obs.png', V, cmap='plasma', vmin=0, vmax=v_min_max)
