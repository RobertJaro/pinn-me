import argparse
import glob
import os.path
from distutils.command.install_data import install_data
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy import units as u
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pme.train.me_atmosphere import MEAtmosphere


def plot_stokes(profile, save_path):
    """
    Plot all the stokes profiles from an atmosphere

    Input:
        -- atmos, ndarray [4, num_Intensity]
    """
    profile = np.abs(profile)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

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
    im = ax.imshow(parameters['theta'].T, cmap='RdBu_r')
    ax.set_title("Theta")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axs[0, 2]
    im = ax.imshow(parameters['chi'].T, cmap='twilight')
    ax.set_title("Chi")
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


def convert_xy_to_rt(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)

    return r, t


def convert_rt_to_xy(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)

    return int(x), int(y)


class TestSetGenerator():

    def __init__(self, lambda0=6302.4931 * u.AA,
                 j_up=1.0, j_low=0.0, g_up=2.49, g_low=0,
                 lambda_start=6301.989128432432 * u.AA, lambda_step=0.021743135134784097 * u.AA, n_lambda=56,
                 nx=400, ny=400,
                 b_field_0=2000.0, vmac=2.0 * 1e3, damping=0.2, b0=0.8, b1=0.2, mu=1.0, vdop=2.0 * 1e3, kl=25.0):
        self.lambda0 = lambda0
        self.jUp = j_up
        self.jLow = j_low
        self.gUp = g_up
        self.gLow = g_low

        lambda_range = (n_lambda - 1) * lambda_step
        self.lambda_grid = np.linspace(-0.5 * lambda_range, 0.5 * lambda_range, n_lambda)

        # Inputs for the inversion
        self.b_field_0 = b_field_0
        self.vmac = vmac
        self.damping = damping
        self.b0 = b0
        self.b1 = b1
        self.mu = mu
        self.vdop = vdop
        self.kl = kl

        self.nx = nx
        self.ny = ny

    def load_time_step(self, time_step):
        xx, yy = np.meshgrid(np.linspace(-0.5 * self.nx, 0.5 * self.nx, self.nx),
                             np.linspace(-0.5 * self.ny, 0.5 * self.ny, self.ny),
                             indexing='ij')
        r0 = 50 + time_step / 2

        r, t = convert_xy_to_rt(xx, yy)

        # B --> (100, 100); lambda --> (50,); B * lambda --> (100, 100, 50)
        # B[..., None] --> (100, 100, 1); lambda[None, None, :] --> (1, 1, 50)

        b_field = self.b_field_0 * (r0 / (r + r0)) ** 2
        # theta is defined between 0 and pi
        t_arr = ((r % r0) / r0 * np.pi)
        ch_arr = (t + time_step / 180 * np.pi)  # slow down the rotation

        b0_arr = self.b0 * (10 * r0 / (r + 10 * r0)) ** 2
        b1_arr = self.b1 * (10 * r0 / (r + 10 * r0)) ** 2

        atmos = MEAtmosphere(self.lambda0, self.jUp, self.jLow, self.gUp, self.gLow, self.lambda_grid)

        b_field = torch.tensor(b_field, dtype=torch.float32)
        t_arr = torch.tensor(t_arr, dtype=torch.float32)
        ch_arr = torch.tensor(ch_arr, dtype=torch.float32)
        b0_arr = torch.tensor(b0_arr, dtype=torch.float32)
        b1_arr = torch.tensor(b1_arr, dtype=torch.float32)

        vmac_arr = self.vmac * torch.ones_like(b_field)
        damping_arr = self.damping * torch.ones_like(b_field)
        mu_arr = self.mu * torch.ones_like(b_field)
        vdop_arr = self.vdop * torch.ones_like(b_field)
        kl_arr = self.kl * torch.ones_like(b_field)

        # flatten and forward
        I, Q, U, V = atmos.forward(b_field.reshape(-1, 1), t_arr.reshape(-1, 1), ch_arr.reshape(-1, 1),
                                   vmac_arr.reshape(-1, 1), damping_arr.reshape(-1, 1),
                                   b0_arr.reshape(-1, 1), b1_arr.reshape(-1, 1), mu_arr.reshape(-1, 1),
                                   vdop_arr.reshape(-1, 1), kl_arr.reshape(-1, 1))

        stokes_profiles = torch.stack([I, Q, U, V], -2).cpu().numpy()
        # (x, y, n_lambda, n_stokes)
        stokes_profiles = stokes_profiles.reshape(*r.shape, 4, *self.lambda_grid.shape)

        return {'stokes_profiles': stokes_profiles}, {'b_field': b_field, 'theta': t_arr, 'chi': ch_arr,
                                                      'b0': b0_arr, 'b1': b1_arr, 'vmac': vmac_arr,
                                                      'damping': damping_arr, 'mu': mu_arr,
                                                      'vdop': vdop_arr, 'kl': kl_arr}

    def create_time_step_file(self, t_step, base_path):
        profiles, parameters = self.load_time_step(t_step)
        # save to file
        np.savez(os.path.join(base_path, f'profile_{t_step:03d}.npz'), **profiles)
        np.savez(os.path.join(base_path, f'parameters_{t_step:03d}.npz'), **parameters)


def load_profiles(file_path):
    files = sorted(glob.glob(file_path))
    profiles = [np.load(f)['stokes_profiles'] for f in files]
    return np.stack(profiles, axis=0)  # (t, x, y, lambda, stokes)


def load_parameters(file_path):
    files = sorted(glob.glob(file_path))
    parameters = [np.load(f) for f in files]
    out_parameters = {}
    for key in parameters[0].keys():
        out_parameters[key] = np.stack([p[key] for p in parameters], axis=0)
    return out_parameters  # (t, x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, required=True, help='base path for the output data')
    parser.add_argument('--resolution', type=int, nargs=2, default=[400, 400], help='resolution of the images')
    parser.add_argument('--n_time_steps', type=int, default=20, help='number of time steps to generate')
    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    data_generator = TestSetGenerator(nx=args.resolution[0], ny=args.resolution[1])

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
