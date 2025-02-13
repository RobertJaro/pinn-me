import argparse
import os.path
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames

from pme.data.util import spherical_to_cartesian, vector_cartesian_to_spherical
from pme.evaluation.loader import PINNMEOutput
from astropy import units as u

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
    parser.add_argument('--input', type=str, help='the path to the input file')
    parser.add_argument('--output', type=str, help='the path to the output file')
    args = parser.parse_args()

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    # load
    pinnme = PINNMEOutput(args.input)

    start_time = pinnme.times[0]
    rotation_time = timedelta(days=25.38)
    times = pd.date_range(start_time, start_time + rotation_time, periods=3600)
    end_time = times[-1].to_pydatetime()

    latitudes = np.linspace(-90, 90, 1800)

    longitudes = []
    for t in times:
        coord = SkyCoord(0 *u.deg, 0 * u.deg, frame=frames.HeliographicStonyhurst, obstime=t, observer='earth')
        longitudes.append(coord.transform_to(frames.HeliographicCarrington).lon.to_value(u.deg))

    normalized_times = np.array([pinnme._normalize_time(t.to_pydatetime()) for t in times])
    latitudes, longitudes = np.deg2rad(latitudes), np.deg2rad(longitudes)

    spherical_coords = np.stack(np.meshgrid( [1], latitudes, longitudes, indexing='ij'),
        axis=-1)
    spherical_coords = spherical_coords[0, :, :]

    cartesian_coords = spherical_to_cartesian(spherical_coords)
    time_coords = np.ones((*cartesian_coords.shape[:-1], 1), dtype=np.float32) *  normalized_times[None, :, None]
    coords = np.concatenate([time_coords, cartesian_coords], axis=-1)

    parameter_cube = pinnme.load_parameters(coords=coords)
    b_xyz = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']], axis=-1)
    b_rtp = vector_cartesian_to_spherical(b_xyz, spherical_coords)

    v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']], axis=-1)
    v_rtp = vector_cartesian_to_spherical(v_xyz, spherical_coords)

    ########################################################################################################################
    # Plot subframe in B_r, B_theta, B_phi
    extent = [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()]
    extent = np.rad2deg(extent)

    v_min_max = np.max(np.abs(b_rtp))
    norm = Normalize(-500, 500)#SymLogNorm(linthresh=1, vmin=-v_min_max, vmax=v_min_max)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ax = axs[0]
    im = ax.imshow(b_rtp[..., 0], cmap='gray', norm=norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('B_r')

    ax = axs[1]
    im = ax.imshow(b_rtp[..., 1], cmap='gray', norm=norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('B_theta')

    ax = axs[2]
    im = ax.imshow(b_rtp[..., 2], cmap='gray', norm=norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('B_phi')

    axs[0].set_ylabel('Latitude [deg]')
    [ax.set_xlabel('Longitude [deg]') for ax in axs]

    # add subtitle with date
    plt.suptitle(f'Carrington map {start_time} -- {end_time}', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'carrington.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # Plot subframe in B, inclination, azimuth

    b = np.linalg.norm(b_rtp, axis=-1)
    inclination = np.arccos(b_rtp[..., 2] / b) % np.pi
    azimuth = np.arctan2(b_rtp[..., 1], b_rtp[..., 0]) % (2 * np.pi)

    extent = [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()]
    extent = np.rad2deg(extent)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ax = axs[0]
    im = ax.imshow(b, cmap='viridis', norm='log', origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('|B| [G]', fontsize=16)

    ax = axs[1]
    im = ax.imshow(np.rad2deg(inclination), cmap='PiYG', origin='lower', extent=extent, vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0, 45, 90, 135, 180])
    cbar.set_label('$\Theta$ [deg]', fontsize=16)

    ax = axs[2]
    im = ax.imshow(np.rad2deg(azimuth), cmap='twilight', origin='lower', extent=extent, vmin=0, vmax=360)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0, 90, 180, 270, 360])
    cbar.set_label('$\phi$ [deg]', fontsize=16)

    axs[0].set_ylabel('Latitude [deg]', fontsize=16)
    [ax.set_xlabel('Longitude [deg]', fontsize=16) for ax in axs]

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'carrington_los_inc_azi.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # Plot velocity

    v_min_max = np.max(np.abs(v_rtp))
    norm = Normalize(vmin=-v_min_max, vmax=v_min_max)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ax = axs[0]
    im = ax.imshow(v_rtp[..., 0], cmap='seismic_r', norm=norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('V_r')

    ax = axs[1]
    im = ax.imshow(v_rtp[..., 1], cmap='seismic_r', norm=norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('V_theta')

    ax = axs[2]
    im = ax.imshow(v_rtp[..., 2], cmap='seismic_r', norm=norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('V_phi')

    axs[0].set_ylabel('Latitudae [deg]')
    [ax.set_xlabel('Longitude [deg]') for ax in axs]

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'carrington_velocity.jpg'), dpi=300)
    plt.close()
