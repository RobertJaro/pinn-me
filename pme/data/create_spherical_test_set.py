import argparse
import glob
import os
import os.path
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from sunpy.coordinates import frames
from sunpy.map import make_heliographic_header, Map, make_fitswcs_header, all_coordinates_from_map
from sunpy.sun import constants

from pme.data.create_cartesian_test_set import plot_parameters, plot_stokes, plot_brtp, plot_coords
from pme.data.test_set_generator import TestSetGenerator, load_parameters, load_fits_profiles
from pme.data.util import image_to_spherical_matrix
from pme.data.util import solar_differential_rotation_velocity


class SphericalTestSetGenerator(TestSetGenerator):

    def load_spherical_time_step(self, time_step, obs_coord):
        parameters = self._load_parameters(time_step, resolution=(180, 180))
        # convert to numpy arrays
        parameters = {k: create_collage(v.detach().cpu().numpy()) for k, v in parameters.items()}
        transformed_parameters, dummy_helioprojective_map = self._transform_parameters(parameters, obs_coord)
        # convert back to torch tensors
        transformed_parameters = {k: torch.tensor(v, dtype=torch.float32) for k, v in transformed_parameters.items()}
        input_parameters = {k: v for k, v in transformed_parameters.items() if k not in ['b_rtp', 'v_rtp']}
        stokes_profiles = self.convert_to_profiles(**input_parameters)
        return stokes_profiles, transformed_parameters, dummy_helioprojective_map

    def _transform_parameters(self, input_parameters, obs_coord):
        # setup b_rtp
        b_field = input_parameters['b_field']
        b_inc = input_parameters['inc']
        b_azi = input_parameters['azi']

        # Convert to spherical coordinates
        b_r = b_field * np.cos(b_inc)
        b_theta = b_field * np.sin(b_inc) * np.cos(b_azi)
        b_phi = b_field * np.sin(b_inc) * np.sin(b_azi)
        # add to parameter dict
        input_parameters['b_r'] = b_r
        input_parameters['b_theta'] = b_theta
        input_parameters['b_phi'] = b_phi

        # convert velocity to spherical coordinates
        vdop = input_parameters['vdop']
        v_r = vdop
        v_theta = np.zeros_like(v_r)
        v_phi = np.zeros_like(v_r)
        # add differential rotation
        latitudes = np.linspace(-np.pi / 2, np.pi / 2, v_r.shape[0]) * u.rad
        v_diff = solar_differential_rotation_velocity(latitudes).to_value(u.m / u.s)
        v_phi += v_diff[:, None]
        # add to parameter dict
        input_parameters['v_r'] = v_r
        input_parameters['v_theta'] = v_theta
        input_parameters['v_phi'] = v_phi

        # create carrington map header
        carrington_header = make_heliographic_header(obs_coord.obstime, 'earth',
                                                     b_r.shape, frame='carrington')

        # create helioprojective map header
        solar_semidiameter_rad = np.arcsin(constants.radius / obs_coord.radius)
        angular_radius = Angle(solar_semidiameter_rad.to(u.arcsec))

        scale = angular_radius.to(u.arcsec) / (self.nx // 2 * u.pix), angular_radius.to(u.arcsec) / (
                self.ny // 2 * u.pix)
        dummy_data = np.zeros((self.nx, self.ny), dtype=np.float32)
        reference_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, observer=obs_coord, frame=frames.Helioprojective)
        helioprojective_header = make_fitswcs_header(dummy_data, reference_coord, scale=u.Quantity(scale))

        # transform all parameters to helioprojective frame
        exclude_parameters = ['b_field', 'inc', 'azi', 'vdop']
        input_parameters = {k: v for k, v in input_parameters.items() if k not in exclude_parameters}
        transformed_parameters = {}

        for k, parameter in input_parameters.items():
            carrington_map = Map(parameter, carrington_header)
            helioprojective_map = carrington_map.reproject_to(helioprojective_header)
            transformed_parameters[k] = helioprojective_map.data

        # create dummy helioprojective map
        dummy_carrington = np.zeros_like(b_r)
        carrington_map = Map(dummy_carrington, carrington_header)
        helioprojective_map = carrington_map.reproject_to(helioprojective_header)
        # Compute mu for the helioprojective map
        helioprojective_coords = all_coordinates_from_map(helioprojective_map)
        helioprojective_coords = helioprojective_coords.transform_to(frames.Helioprojective)
        radial_distance = np.sqrt(
            helioprojective_coords.Tx ** 2 + helioprojective_coords.Ty ** 2) / helioprojective_map.rsun_obs
        mu = np.cos(radial_distance.to_value(u.dimensionless_unscaled) * np.pi / 2)
        mu = mu.astype(np.float32)
        mu[mu < 0] = np.nan
        transformed_parameters['mu'] = mu

        # Transform vector quantities -- B, V -- to image coordinates
        # create transformation matrix
        carrington_coords = helioprojective_coords.transform_to(frames.HeliographicCarrington)
        lat, lon = carrington_coords.lat.to_value(u.rad), carrington_coords.lon.to_value(u.rad)
        latc, lonc = helioprojective_map.carrington_latitude.to_value(
            u.rad), helioprojective_map.carrington_longitude.to_value(u.rad)

        # TODO check that CRLT_OBS is in rad units as provided by the map
        pAng = -np.deg2rad(helioprojective_map.meta.get('CROTA2', 0))
        a_matrix = image_to_spherical_matrix(lon, lat, latc, lonc, pAng=pAng)
        rtp_to_img_transform = np.linalg.inv(a_matrix)

        # stack b vector
        b_r = transformed_parameters.pop('b_r')
        b_theta = transformed_parameters.pop('b_theta')
        b_phi = transformed_parameters.pop('b_phi')
        b_rtp = np.stack([b_r, b_theta, b_phi], -1)
        b_rtp[..., 1] *= -1
        # transform b vector to image frame
        b_img = np.einsum("...ij,...j->...i", rtp_to_img_transform, b_rtp)  # in image xyz
        # b_im = (xi, eta, zeta)
        # convert to ME parameters
        b_field = np.linalg.norm(b_img, axis=-1)
        b_inc = np.arccos(b_img[..., 2] / (b_field + 1e-8))
        b_azi = np.arctan2(-b_img[..., 0], b_img[..., 1])

        # stack v vector
        v_r = transformed_parameters.pop('v_r')
        v_theta = transformed_parameters.pop('v_theta')
        v_phi = transformed_parameters.pop('v_phi')
        v_rtp = np.stack([v_r, v_theta, v_phi], -1)
        v_rtp[..., 1] *= -1
        # transform b vector to image frame
        v_img = np.einsum("...ij,...j->...i", rtp_to_img_transform, v_rtp)  # in image xyz
        # convert to ME parameters
        vdop = v_img[..., 2]

        transformed_parameters['b_field'] = b_field
        transformed_parameters['inc'] = b_inc
        transformed_parameters['azi'] = b_azi

        transformed_parameters['vdop'] = vdop

        transformed_parameters['b_rtp'] = b_rtp
        transformed_parameters['v_rtp'] = v_rtp

        return transformed_parameters, helioprojective_map

    def create_spherical_time_step_file(self, t_step, base_path, obs_coords):
        profiles, parameters, dummy_map = self.load_spherical_time_step(t_step, obs_coords)
        header = dummy_map.meta
        header['OBS_VR'] = 0
        header['OBS_VW'] = 0
        header['OBS_VN'] = 0
        for wl_idx in range(profiles.shape[-1]):
            for s_ixd, s_id in enumerate(['I', 'Q', 'U', 'V']):
                p_map = Map(profiles[:, :, s_ixd, wl_idx], header)
                file_path = os.path.join(base_path, f'stokes_{t_step:03d}_{s_id}{wl_idx}.fits')
                p_map.save(file_path, overwrite=True)

        np.savez(os.path.join(base_path, f'parameters_{t_step:03d}.npz'), **parameters)


def create_collage(image_array, rows=3, cols=6):
    """
    Creates an 8x8 collage of a 2D grayscale image.

    Parameters:
    image_array (numpy.ndarray): Input grayscale image (H, W)
    rows (int): Number of rows in the collage
    cols (int): Number of columns in the collage

    Returns:
    numpy.ndarray: Collaged image array (rows*H, cols*W)
    """
    # Create one row with `cols` duplicates
    row = np.concatenate([image_array] * cols, axis=1)

    # Stack `rows` copies of that row to form full collage
    collage = np.concatenate([row] * rows, axis=0)

    return collage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, required=True, help='base path for the output data')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256], help='resolution of the images')
    parser.add_argument('--n_time_steps', type=int, default=100, help='number of time steps to generate')
    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    obs_lon = 0 * u.deg
    obs_lat = 0 * u.deg
    observer_distance = 1 * u.AU

    t_start = datetime(2025, 1, 1, )
    t_end = datetime(2025, 2, 1)
    t_range = pd.date_range(t_start, t_end, periods=args.n_time_steps)

    lambda_grid = np.array([-0.1695, -0.1017, -0.0339, +0.0339, +0.1017, +0.1695]) / 10 * u.nm  # From Phillip Scherrer
    lambda0 = 617.33433 * u.nm  # From Phillip Scherrer

    data_generator = SphericalTestSetGenerator(nx=args.resolution[0], ny=args.resolution[1],
                                      lambda0=lambda0, lambda_grid=lambda_grid, g_up=2.50)

    observers = []
    for time in t_range:
        coord = SkyCoord(lon=obs_lon, lat=obs_lat, radius=observer_distance,
                         obstime=time, observer="self",
                         frame=frames.HeliographicStonyhurst)
        coord = coord.transform_to(frames.HeliographicCarrington)
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
        in_data = [(parameters_dict := {k: v[i] for k, v in parameters.items() if k not in ['b_rtp', 'v_rtp']},
                    os.path.join(out_path, 'images', f'parameters_{i:03d}.jpg'))
                   for i in range(profiles.shape[0])]
        p.starmap(plot_parameters, in_data)

    with Pool(16) as p:
        in_data = [(parameters['b_rtp'][i], os.path.join(out_path, 'images', f'brtp_{i:03d}.jpg'))
                   for i in range(profiles.shape[0])]
        p.starmap(plot_brtp, in_data)

    # plot coordinate grid
    files = sorted(glob.glob(os.path.join(out_path, f'*I0.fits')))
    with Pool(16) as p:
        in_data = [(files[i], os.path.join(out_path, 'images', f'coords_{i:03d}.jpg'))
                   for i in range(len(files))]
        p.starmap(plot_coords, in_data)