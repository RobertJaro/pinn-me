import argparse
import io
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pme.evaluation.loader import PINNMEOutput, to_cartesian, to_spherical

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file')

args = parser.parse_args()

pinnme = PINNMEOutput(args.input)

parameter_cube = pinnme.load_cube()

b = parameter_cube['b_field'][..., 0]
theta = parameter_cube['theta'][..., 0]
chi = parameter_cube['chi'][..., 0]

b_norm = np.abs(b).max()

b, theta, chi = to_spherical(to_cartesian(b, theta, chi))
chi = chi % (np.pi)

images = []
for time_step in range(0, b.shape[2], 1):
    fig, axs = plt.subplots(1, 3, figsize=(15, 7), dpi=150)
    im = axs[0].imshow(b[:, :, time_step], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[0].set_title('B field')
    im = axs[1].imshow(theta[:, :, time_step], cmap='RdBu_r', vmin=0, vmax=np.pi)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[1].set_title('Theta')
    im = axs[2].imshow(chi[:, :, time_step], cmap='twilight', vmin=0, vmax=np.pi)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[2].set_title('Chi')
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

for time_step in range(0, b.shape[2], 1):
    fig, axs = plt.subplots(1, 3, figsize=(15, 7), dpi=150)
    im = axs[0].imshow(b[:, :, time_step, 0], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[0].set_title('Bx')
    im = axs[1].imshow(b[:, :, time_step, 1], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[1].set_title('By')
    im = axs[2].imshow(b[:, :, time_step, 2], cmap='RdBu_r', vmin=-b_norm, vmax=b_norm)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[2].set_title('Bz')
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
