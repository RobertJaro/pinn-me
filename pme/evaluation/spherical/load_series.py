import argparse
import glob
import os.path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, \
    image_to_spherical_matrix
from pme.evaluation.loader import PINNMEOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
    parser.add_argument('--input', type=str, help='the path to the input file')
    parser.add_argument('--ref_maps', type=str, help='the path to the reference map fld')
    parser.add_argument('--output', type=str, help='the path to the output file', default=None, required=False)
    args = parser.parse_args()

    in_path = args.input

    out_path = args.output
    out_path = out_path if out_path is not None else os.path.join(os.path.dirname(in_path), 'evaluation', 'series')
    os.makedirs(out_path, exist_ok=True)

    # load
    pinnme = PINNMEOutput(in_path)

    # load reference maps
    times = pinnme.times

    # resolution = 0.05 # degrees

    # latitude_range = [-30, -10]
    # longitude_range = [25, 45]
    resolution = 1
    latitude_range = [-90, 90]
    longitude_range = [0, 360]

    latitude = np.deg2rad(np.linspace(latitude_range[0], latitude_range[1], int((latitude_range[1] - latitude_range[0] + 1) // resolution)))
    longitude = np.deg2rad(np.linspace(longitude_range[0], longitude_range[1], int((longitude_range[1] - longitude_range[0] + 1) // resolution)))
    spherical_coords = np.stack(np.meshgrid([1], latitude, longitude, indexing='ij'), -1)  # r, theta, phi
    spherical_coords = spherical_coords.squeeze(0)
    cartesian_coords = spherical_to_cartesian(spherical_coords)

    cartesian_to_spherical_transform = cartesian_to_spherical_matrix(spherical_coords)

    extent = [longitude_range[0], longitude_range[1],
              latitude_range[0], latitude_range[1]]

    for i, target_time in tqdm(enumerate(times), total=len(times)):
        file_path = os.path.join(out_path, f'step{i:03d}.jpg')


        normalized_time = pinnme._normalize_time(target_time)
        time_coords = np.ones((*cartesian_coords.shape[:-1], 1), dtype=np.float32) * normalized_time
        coords = np.concatenate([time_coords, cartesian_coords], axis=-1)

        parameter_cube = pinnme.load_parameters(coords=coords, progress=False)
        b_xyz = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']], axis=-1) * pinnme.gauss_per_dB
        b_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, b_xyz)
        # b_rtp[..., 1] *= -1

        v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']], axis=-1) * 1e-3 * pinnme.meters_per_ds / pinnme.seconds_per_dt
        v_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, v_xyz)
        # v_rtp[..., 1] *= -1

        ########################################################################################################################
        # Plot subframe in B_r, B_theta, B_phi

        b_norm = Normalize(-500, 500) # SymLogNorm(linthresh=1, vmin=-3000, vmax=3000)#
        v_norm = Normalize(vmin=-2, vmax=2)

        fig, axs = plt.subplots(2, 3, figsize=(10, 5))

        ax = axs[0, 0]
        im = ax.imshow(b_rtp[..., 0], cmap='gray', norm=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
        ax.set_title(r'$B_\text{r}$ [G]')

        ax = axs[0, 1]
        im = ax.imshow(b_rtp[..., 1], cmap='gray', norm=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
        ax.set_title(r'$B_\text{t}$ [G]')

        ax = axs[0, 2]
        im = ax.imshow(b_rtp[..., 2], cmap='gray', norm=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
        ax.set_title(r'$B_\text{p}$ [G]')

        ax = axs[1, 0]
        im = ax.imshow(v_rtp[..., 0], cmap='seismic_r', norm=v_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\text{r}$ [km/s]')
        ax.set_title(r'$v_\text{r}$ [km/s]')

        ax = axs[1, 1]
        im = ax.imshow(v_rtp[..., 1], cmap='seismic_r', norm=v_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\text{t}$ [km/s]')
        ax.set_title(r'$v_\text{t}$ [km/s]')

        ax = axs[1, 2]
        im = ax.imshow(v_rtp[..., 2], cmap='seismic_r', norm=v_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\text{p}$ [km/s]')
        ax.set_title(r'$v_\text{p}$ [km/s]')

        for ax in axs.flat:
            ax.set_xlabel(r'Longitude [deg]')
            ax.set_ylabel(r'Latitude [deg]')

        # add subtitle with date
        plt.suptitle(f'{target_time}', fontsize=16)

        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

        fig, axs = plt.subplots(2, 3, figsize=(10, 5))

        ax = axs[0, 0]
        im = ax.imshow(b_xyz[..., 0], cmap='gray', norm=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
        ax.set_title(r'$B_\text{x}$ [G]')

        ax = axs[0, 1]
        im = ax.imshow(b_xyz[..., 1], cmap='gray', norm=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
        ax.set_title(r'$B_\text{y}$ [G]')

        ax = axs[0, 2]
        im = ax.imshow(b_xyz[..., 2], cmap='gray', norm=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
        ax.set_title(r'$B_\text{z}$ [G]')

        ax = axs[1, 0]
        im = ax.imshow(cartesian_coords[..., 0], cmap='viridis', origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$x$ [R$_\odot$]')
        ax.set_title(r'$x$ [R$_\odot$]')

        ax = axs[1, 1]
        im = ax.imshow(cartesian_coords[..., 1], cmap='viridis', origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$y$ [R$_\odot$]')
        ax.set_title(r'$y$ [R$_\odot$]')

        ax = axs[1, 2]
        im = ax.imshow(cartesian_coords[..., 2], cmap='viridis', origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$z$ [R$_\odot$]')
        ax.set_title(r'$z$ [R$_\odot$]')

        for ax in axs.flat:
            ax.set_xlabel(r'Longitude [deg]')
            ax.set_ylabel(r'Latitude [deg]')
        # add subtitle with date
        plt.suptitle(f'{target_time}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f'xyz_step{i:03d}.jpg'), dpi=300)
        plt.close()