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
    parser.add_argument('--output', type=str, help='the path to the output file')
    args = parser.parse_args()

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    # load
    pinnme = PINNMEOutput(args.input)

    # load reference maps
    times = pinnme.times

    resolution = 0.05 # degrees

    latitude_range = [-30, -10]
    longitude_range = [25, 45]

    latitude = np.deg2rad(np.linspace(latitude_range[0], latitude_range[1], int((latitude_range[1] - latitude_range[0] + 1) // resolution)))
    longitude = np.deg2rad(np.linspace(longitude_range[0], longitude_range[1], int((longitude_range[1] - longitude_range[0] + 1) // resolution)))
    spherical_coords = np.stack(np.meshgrid([1], latitude,
                                            longitude, indexing='ij'), -1)  # r, theta, phi
    spherical_coords = spherical_coords.squeeze(0)
    cartesian_coords = spherical_to_cartesian(spherical_coords)

    cartesian_to_spherical_transform = cartesian_to_spherical_matrix(spherical_coords)

    extent = [longitude_range[0], longitude_range[1],
              latitude_range[0], latitude_range[1]]

    for i, target_time in tqdm(enumerate(times), total=len(times)):
        out_path = os.path.join(args.output, f'step{i:03d}.jpg')


        normalized_time = pinnme._normalize_time(target_time)
        time_coords = np.ones((*cartesian_coords.shape[:-1], 1), dtype=np.float32) * normalized_time
        coords = np.concatenate([time_coords, cartesian_coords], axis=-1)

        parameter_cube = pinnme.load_parameters(coords=coords, progress=False)
        b_rtp = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']], axis=-1)
        # b_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, b_xyz)

        v_rtp = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']], axis=-1)
        # v_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, v_xyz)

        ########################################################################################################################
        # Plot subframe in B_r, B_theta, B_phi

        b_norm = Normalize(-500, 500) # SymLogNorm(linthresh=1, vmin=-3000, vmax=3000)#
        v_norm = SymLogNorm(linthresh=10)

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
        plt.savefig(out_path, dpi=300)
        plt.close()
