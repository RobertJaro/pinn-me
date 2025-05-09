import argparse
import os
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import  units as u
from pme.data.test_set_generator import TestSetGenerator, load_profiles, load_parameters

def plot_stokes(profile, save_path):
    """
    Plot all the stokes profiles from an atmosphere

    Input:
        -- atmos, ndarray [4, num_Intensity]
    """
    profile = np.abs(profile)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    print(np.nanmin(profile.sum(axis=-1), (0, 1)), np.nanmax(profile.sum(axis=-1), (0, 1)))

    ax = axs[0]
    im = ax.imshow(profile[..., 0, :].sum(axis=-1), norm=LogNorm())
    ax.set_title("I")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[1]
    im = ax.imshow(profile[..., 1, :].sum(axis=-1), norm=LogNorm())
    ax.set_title("Q")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[2]
    im = ax.imshow(profile[..., 2, :].sum(axis=-1), norm=LogNorm())
    ax.set_title("U")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[3]
    im = ax.imshow(profile[..., 3, :].sum(axis=-1), norm=LogNorm())
    ax.set_title("V")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_parameters(parameters, save_path):
    """
    Plot all the stokes profiles from an atmosphere

    Input:
        -- atmos, ndarray [4, num_Intensity]
    """

    fig, axs = plt.subplots(2, 5, figsize=(16, 4))

    ax = axs[0, 0]
    im = ax.imshow(parameters['b_field'].T, cmap='viridis', vmin=0)
    ax.set_title("B")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[0, 1]
    im = ax.imshow(parameters['inc'].T, cmap='RdBu_r')
    ax.set_title("Inclination")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[0, 2]
    im = ax.imshow(parameters['azi'].T, cmap='twilight')
    ax.set_title("Azimuth")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[0, 3]
    im = ax.imshow(parameters['b0'].T)
    ax.set_title("B0")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[0, 4]
    im = ax.imshow(parameters['b1'].T)
    ax.set_title("B1")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[1, 0]
    im = ax.imshow(parameters['vmac'].T)
    ax.set_title("Vmac")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[1, 1]
    im = ax.imshow(parameters['damping'].T)
    ax.set_title("Damping")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[1, 2]
    im = ax.imshow(parameters['mu'].T)
    ax.set_title("Mu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[1, 3]
    im = ax.imshow(parameters['vdop'].T)
    ax.set_title("Vdop")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[1, 4]
    im = ax.imshow(parameters['kl'].T)
    ax.set_title("Kl")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, required=True, help='base path for the output data')
    parser.add_argument('--resolution', type=int, nargs=2, default=[400, 400], help='resolution of the images')
    parser.add_argument('--n_time_steps', type=int, default=20, help='number of time steps to generate')
    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    lambda0 = 6302.4931 * u.AA
    lambda_step = 0.021743135134784097 * u.AA
    n_lambda = 56
    lambda_range = (n_lambda - 1) * lambda_step
    lambda_grid = np.linspace(-0.5 * lambda_range, 0.5 * lambda_range, n_lambda)

    data_generator = TestSetGenerator(nx=args.resolution[0], ny=args.resolution[1],
                                      lambda_grid=lambda_grid, lambda0=lambda0)

    with Pool(16) as p:
        in_data = [(t, out_path) for t in range(args.n_time_steps)]
        p.starmap(data_generator.create_time_step_file, in_data)


    profiles = load_profiles(os.path.join(out_path, 'profile_*.npz'))
    parameters = load_parameters(os.path.join(out_path, 'parameters_*.npz'))

    os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)
    for i in range(profiles.shape[0]):
        plot_stokes(profiles[i], os.path.join(out_path, 'images', f'stokes_{i:03d}.jpg'))

    for i in range(profiles.shape[0]):
        t_step_parameters = {k: v[i] for k, v in parameters.items()}
        plot_parameters(t_step_parameters, os.path.join(out_path, 'images', f'parameters_{i:03d}.jpg'))