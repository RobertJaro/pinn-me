import argparse
import io
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from pme.evaluation.loader import PINNMEOutput, to_cartesian, to_spherical

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file')
parser.add_argument('--reference', type=str, help='the path to the reference file')

args = parser.parse_args()

pinnme = PINNMEOutput(args.input)

parameter_cube = pinnme.load_cube()

reference_parameters = np.load(args.reference)

b_ref = reference_parameters['BField'][..., 0]
theta_ref = reference_parameters['theta'][..., 0]
chi_ref = reference_parameters['chi'][..., 0] % (np.pi)

b = parameter_cube['b_field'][..., 0]
theta = parameter_cube['theta'][..., 0]
chi = parameter_cube['chi'][..., 0]

b_norm = np.abs(b).max()

chi = chi % (np.pi)

images = []
for time_step in tqdm(range(0, b.shape[2], 1), desc='Saving images - b, theta, chi'):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), dpi=150)
    im = axs[0, 0].imshow(b[:, :, time_step], cmap='Reds', vmin=0, vmax=b_norm)
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[0, 0].set_title('B field')
    #
    im = axs[0, 1].imshow(theta[:, :, time_step], cmap='RdBu_r', vmin=0, vmax=np.pi)
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[0, 1].set_title('Theta')
    #
    im = axs[0, 2].imshow(chi[:, :, time_step], cmap='twilight', vmin=0, vmax=np.pi)
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[0, 2].set_title('Chi')
    #
    im = axs[1, 0].imshow(b_ref[:, :, time_step], cmap='Reds', vmin=0, vmax=b_norm)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[1, 0].set_title('true B field')
    #
    im = axs[1, 1].imshow(theta_ref[:, :, time_step], cmap='RdBu_r', vmin=0, vmax=np.pi)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[1, 1].set_title('true Theta')
    #
    im = axs[1, 2].imshow(chi_ref[:, :, time_step], cmap='twilight', vmin=0, vmax=np.pi)
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[1, 2].set_title('true Chi')
    #
    fig.suptitle(f'Time step {time_step}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(args.output, f'{time_step:03d}.jpg'))
    # with io.BytesIO() as buff:
    #     fig.savefig(buff, format='png')
    #     buff.seek(0)
    #     im = plt.imread(buff)
    # images += [im]
    plt.close(fig)

b = to_cartesian(b, theta, chi)
b_norm = np.linalg.norm(b, axis=-1).max()

b_ref = to_cartesian(b_ref, theta_ref, chi_ref)

for time_step in tqdm(range(0, b.shape[2], 1), desc='Saving images - cartesian'):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), dpi=150)
    im = axs[0, 0].imshow(b[:, :, time_step, 0], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[0, 0].set_title(r'$B_x$')
    #
    im = axs[0, 1].imshow(b[:, :, time_step, 1], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[0, 1].set_title(r'$B_y$')
    #
    im = axs[0, 2].imshow(b[:, :, time_step, 2], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[0, 2].set_title(r'$B_z$')
    #
    im = axs[1, 0].imshow(b_ref[:, :, time_step, 0], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[1, 0].set_title(r'true $B_x$')
    #
    im = axs[1, 1].imshow(b_ref[:, :, time_step, 1], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[1, 1].set_title(r'true $B_y$')
    #
    im = axs[1, 2].imshow(b_ref[:, :, time_step, 2], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[1, 2].set_title(r'true $B_z$')
    #
    fig.suptitle(f'Time step {time_step}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(args.output, f'cartesian_{time_step:03d}.jpg'))
    # with io.BytesIO() as buff:
    #     fig.savefig(buff, format='png')
    #     buff.seek(0)
    #     im = plt.imread(buff)
    # images += [im]
    plt.close(fig)

# imageio.mimsave(args.output, images, "MP4")
# images = np.random.randn(10, 100, 100, 3)
# writer = imageio.get_writer(os.path.join(args.output, 'video.mp4'))
# for im in images:
#     writer.append_data(im)
#
# writer.close()
