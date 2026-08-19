import argparse
import os.path
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt, dates
from matplotlib.colors import Normalize, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix
from pme.evaluation.loader import PINNMEOutput, SPINNMEOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
    parser.add_argument('--input', type=str, help='the path to the input file')
    parser.add_argument('--output', type=str, help='the path to the output file')
    args = parser.parse_args()

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    # load
    pinnme = SPINNMEOutput(args.input)

    start_time = pinnme.times[0]
    end_time = pinnme.times[-1]
    times = pd.date_range(start_time, end_time, freq=timedelta(hours=.5))

    # latitudes = np.linspace(-90, 90, 1800)
    latitudes = np.linspace(-1, 1, 1800)
    latitudes = np.rad2deg(np.arcsin(latitudes))

    longitudes = []
    for t in times:
        coord = SkyCoord(0 * u.deg, 0 * u.deg, frame=frames.HeliographicStonyhurst, obstime=t, observer='earth')
        longitudes.append(coord.transform_to(frames.HeliographicCarrington).lon.to_value(u.deg))

    normalized_times = np.array([pinnme._normalize_time(t.to_pydatetime()) for t in times])
    latitudes, longitudes = np.deg2rad(latitudes), np.deg2rad(longitudes)

    spherical_coords = np.stack(np.meshgrid([1], np.pi / 2 - latitudes, longitudes, indexing='ij'),
                                axis=-1)
    spherical_coords = spherical_coords[0, :, :]

    cartesian_coords = spherical_to_cartesian(spherical_coords) / pinnme.Rs_per_ds
    time_coords = np.ones((*cartesian_coords.shape[:-1], 1), dtype=np.float32) * normalized_times[None, :, None]
    coords = np.concatenate([time_coords, cartesian_coords], axis=-1)

    parameter_cube = pinnme.load_parameters(coords=coords)
    b_xyz = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']], axis=-1) * pinnme.gauss_per_dB
    transform = cartesian_to_spherical_matrix(spherical_coords)
    b_rtp = np.einsum("...ij,...j->...i", transform, b_xyz)

    v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']], axis=-1) * pinnme.meters_per_ds / pinnme.seconds_per_dt
    transform = cartesian_to_spherical_matrix(spherical_coords)
    v_rtp = np.einsum("...ij,...j->...i", transform, v_xyz)

    ########################################################################################################################
    # Plot subframe in B_r, B_theta, B_phi
    x_min = dates.date2num(start_time)
    x_max = dates.date2num(end_time)
    extent = [x_min, x_max, -1, 1]

    v_min_max = np.max(np.abs(b_rtp))
    norm = SymLogNorm(linthresh=1, vmin=-1000, vmax=1000)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ax = axs[0]
    im = ax.imshow(b_rtp[..., 0], cmap='RdBu', norm=norm, origin='lower', extent=extent, aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('B_r')

    ax = axs[1]
    im = ax.imshow(b_rtp[..., 1], cmap='RdBu', norm=norm, origin='lower', extent=extent, aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('B_theta')

    ax = axs[2]
    im = ax.imshow(b_rtp[..., 2], cmap='RdBu', norm=norm, origin='lower', extent=extent, aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('B_phi')

    axs[0].set_ylabel('Latitude')

    # add subtitle with date
    plt.suptitle(f'Carrington map {start_time} -- {end_time}', fontsize=16)

    # make time axis show dates
    for ax in axs:
        date_format = dates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        ax.set_xlabel('Time [UTC]')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'carrington.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # Plot velocity
    norm = Normalize(vmin=-5000, vmax=5000)

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
