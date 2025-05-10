import argparse
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

from pme.data.create_cartesian_test_set import plot_parameters, plot_stokes
from pme.data.test_set_generator import TestSetGenerator, load_parameters, load_fits_profiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, required=True, help='base path for the output data')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256], help='resolution of the images')
    parser.add_argument('--n_time_steps', type=int, default=100, help='number of time steps to generate')
    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    obs_lon_start = 0 * u.deg
    obs_lat = 0 * u.deg
    observer_distance = 1 * u.AU

    t_start = datetime(2025, 1, 1, )
    t_end = datetime(2025, 1, 15)
    t_range = pd.date_range(t_start, t_end, periods=args.n_time_steps)

    solar_rotation_rate = 27.26
    dlon = (360 * u.deg) / (solar_rotation_rate * u.day)
    obs_lon_end = obs_lon_start + dlon * (t_end - t_start).total_seconds() * u.s
    longitudes = np.linspace(obs_lon_start, obs_lon_end, num=args.n_time_steps)

    lambda_grid = np.array([-0.1695, -0.1017, -0.0339, +0.0339, +0.1017, +0.1695]) / 10 * u.nm  # From Phillip Scherrer
    lambda0 = 617.33433 * u.nm  # From Phillip Scherrer

    data_generator = TestSetGenerator(nx=args.resolution[0], ny=args.resolution[1],
                                      lambda0=lambda0, lambda_grid=lambda_grid, g_up=2.50)

    observers = []
    for time, longitude in zip(t_range, longitudes):
        coord = SkyCoord(lon=longitude, lat=obs_lat, radius=observer_distance,
                         obstime=time, observer="self",
                         frame=frames.HeliographicCarrington)
        observers.append(coord)

    with Pool(16) as p:
        in_data = [(t, out_path, obs) for t, obs in enumerate(observers)]
        p.starmap(data_generator.create_spherical_time_step_file, in_data)

    profiles = load_fits_profiles(out_path)
    parameters = load_parameters(os.path.join(out_path, 'parameters_*.npz'))

    os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)

    with Pool(16) as p:
        in_data = [(profiles[i], os.path.join(out_path, 'images', f'stokes_{i:03d}.jpg'))
                   for i in range(profiles.shape[0])]
        p.starmap(plot_stokes, in_data)

    with Pool(16) as p:
        in_data = [(parameters_dict := {k: v[i] for k, v in parameters.items()},
                    os.path.join(out_path, 'images', f'parameters_{i:03d}.jpg'))
                   for i in range(profiles.shape[0])]
        p.starmap(plot_parameters, in_data)
