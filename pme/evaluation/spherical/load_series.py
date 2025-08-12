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
    parser.add_argument('--latitude_range', type=float, nargs=2, default=[-90, 90], help='the latitude range in degrees')
    parser.add_argument('--longitude_range', type=float, nargs=2, default=[0, 360], help='the longitude range in degrees')
    parser.add_argument('--resolution', type=float, default=1, help='the resolution in degrees')
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
    latitude_range = args.latitude_range
    longitude_range = args.longitude_range
    resolution = args.resolution

    latitude = np.deg2rad(np.linspace(latitude_range[0], latitude_range[1], int((latitude_range[1] - latitude_range[0] + 1) // resolution)))
    colatitude = np.pi / 2 - latitude  # convert to colatitude
    longitude = np.deg2rad(np.linspace(longitude_range[0], longitude_range[1], int((longitude_range[1] - longitude_range[0] + 1) // resolution)))
    spherical_coords = np.stack(np.meshgrid([1], colatitude, longitude, indexing='ij'), -1)  # r, theta, phi
    spherical_coords = spherical_coords.squeeze(0)
    cartesian_coords = spherical_to_cartesian(spherical_coords)

    cartesian_to_spherical_transform = cartesian_to_spherical_matrix(spherical_coords)

    extent = [longitude_range[0], longitude_range[1],
              latitude_range[0], latitude_range[1]]

    for i, target_time in tqdm(enumerate(times), total=len(times)):
        file_path = os.path.join(out_path, f'step{i:03d}.jpg')
        # if os.path.exists(file_path):
        #     continue

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

        ##################################################################################################
        # Single panel (spherical): B_z (B_r) with subsampled v_tp arrows
        # - Arrows are red; alpha scales with |B_z|
        # - Arrow scaling is fixed across frames

        # Output path and filename
        out_path_bz = os.path.join(out_path, "bz_vtp")
        os.makedirs(out_path_bz, exist_ok=True)
        file_path_bz = os.path.join(out_path_bz, f"bz_vtp_step{i:03d}.jpg")
        # if os.path.exists(file_path_bz):
        #     continue

        # Fixed norms (consistent across frames)
        b_norm = Normalize(-500, 500)  # for background image
        max_speed_ref = 4.0  # km/s reference for arrow length (kept constant across frames)
        b_alpha_ref = 500.0  # G, sets where alpha ~ 1; match b_norm vmax for simplicity
        alpha_min = 0.08  # small floor so arrows are still faintly visible

        # Lat/lon grids in degrees for plotting and quiver
        lon_deg = np.linspace(longitude_range[0], longitude_range[1], b_rtp.shape[1])
        lat_deg = np.linspace(latitude_range[0], latitude_range[1], b_rtp.shape[0])
        Lon, Lat = np.meshgrid(lon_deg, lat_deg, indexing='xy')

        # Background = B_z in local heliographic sense (vertical) -> B_r
        Bz = b_rtp[..., 0]  # [G]

        # Tangential spherical components mapped to plot axes
        v_lon = v_rtp[..., 2]  # km/s, v_phi (x-axis)
        v_lat = -v_rtp[..., 1]  # km/s, -v_theta (y-axis)

        # Fixed arrow scaling across frames
        width_deg = (longitude_range[1] - longitude_range[0])
        arrow_len_deg = 0.10 * width_deg  # arrows ~10% of width at max_speed_ref
        U = (v_lon / max_speed_ref) * arrow_len_deg
        V = (v_lat / max_speed_ref) * arrow_len_deg

        # --- Simple subsampling for quiver clarity ---
        target = 100  # ~arrows per axis; increase for denser arrows
        H, W = U.shape
        s_lat = max(1, H // target)
        s_lon = max(1, W // target)

        Lon_s = Lon[::s_lat, ::s_lon]
        Lat_s = Lat[::s_lat, ::s_lon]
        U_s = U[::s_lat, ::s_lon]
        V_s = V[::s_lat, ::s_lon]

        # --- Arrow transparency from |Bz| ---
        Bz_s = Bz[::s_lat, ::s_lon]
        alpha = np.clip(np.abs(Bz_s) / b_alpha_ref, 0.0, 1.0)
        alpha = np.maximum(alpha, alpha_min)

        # Build RGBA color array for red with per-arrow alpha
        colors = np.zeros((*alpha.shape, 4), dtype=float)
        colors[..., 0] = 1.0  # R
        colors[..., 3] = alpha
        colors = colors.reshape(-1, 4)  # quiver expects N x 4

        # Plot (background full-res, vectors subsampled)
        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(Bz, cmap='gray', norm=b_norm, origin='lower', extent=extent)
        fig.colorbar(im, ax=ax, orientation='vertical', label=r'$B_z \equiv B_r$ [G]')

        ax.quiver(
            Lon_s, Lat_s, U_s, V_s,
            angles='xy', scale_units='xy', scale=1.0,
            width=0.0015, headwidth=3, headlength=4, pivot='tail',
            color=colors
        )

        ax.set_xlabel(r'Longitude [deg]')
        ax.set_ylabel(r'Latitude [deg]')
        ax.set_title(r'$B_z$ with $\vec{v}_{t\phi}$ arrows (alpha ∝ |B_z|)')

        plt.suptitle(f'{target_time}', fontsize=14)
        plt.tight_layout()
        plt.savefig(file_path_bz, dpi=300)
        plt.close()
