# Constants
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
from scipy.special import jvp
from sunpy.coordinates import frames
from sunpy.map import make_heliographic_header, Map, make_fitswcs_header, all_coordinates_from_map
from sunpy.sun import constants

from pme.data.create_cartesian_test_set import plot_parameters, plot_stokes, plot_brtp, plot_coords
from pme.data.differential_rotation import carrington_rotation_velocity
from pme.data.test_set_generator import TestSetGenerator, load_parameters, load_fits_profiles
from pme.data.util import image_to_spherical_matrix, vector_spherical_to_cartesian, vector_cartesian_to_spherical


class SpheromakTestSetGenerator(TestSetGenerator):

    def __init__(self, ref_time, carrington_map_resolution=(180, 360), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_time = ref_time
        self.carrington_map_resolution = carrington_map_resolution

    def _create_spheromak_params(self, time_seconds):
        # Initial conditions
        t0 = 0
        R_solar = 1
        meters_per_Rs = (1 * u.Rsun).to_value(u.m)

        coords_spherical = np.stack(np.meshgrid(R_solar,
                                                np.linspace(0, np.pi, self.carrington_map_resolution[0]),
                                                np.linspace(0, 2 * np.pi, self.carrington_map_resolution[1]),
                                                time_seconds, indexing='ij'), -1)
        R, THETA, PHI, T = (coords_spherical[..., 0],
                            coords_spherical[..., 1],
                            coords_spherical[..., 2],
                            coords_spherical[..., 3])

        # Calculate the spheromak solution

        # Spatial parameters
        R_0 = 0.5
        B_0 = 2000
        n = 1
        m = 0
        gamma = (0.1 * u.km / u.s).to_value(u.Rsun / u.s)
        C_alpha = 4.4934
        alpha_0 = C_alpha / R_0
        #
        alpha = C_alpha * (R_0 + gamma * (T - t0) ** n) ** -1
        dalpha_dt = - C_alpha * (R_0 + gamma * (T - t0) ** n) ** -2 * gamma * n * (T - t0) ** (n - 1)

        j1 = jvp(1, alpha * R, n=0)
        dj1 = jvp(1, alpha * R, n=1)

        B_r = 2 * B_0 * alpha * j1 / (alpha_0 ** 2 * R) * np.cos(THETA)
        B_theta = - B_0 * alpha * (j1 + alpha * R * dj1) / (alpha_0 ** 2 * R) * np.sin(THETA)
        B_phi = B_0 * j1 * (alpha / alpha_0) ** 2 * np.sin(THETA)

        E_r = np.zeros_like(R)
        E_theta = - B_phi * dalpha_dt / alpha * R
        E_phi = B_theta * dalpha_dt / alpha * R
        #
        # c = np.stack([R, THETA, PHI], -1)
        B_rtp = np.stack([B_r, B_theta, B_phi], -1)
        E_rtp = np.stack([E_r, E_theta, E_phi], -1)

        B = vector_spherical_to_cartesian(B_rtp, coords_spherical)
        E = vector_spherical_to_cartesian(E_rtp, coords_spherical)
        V = np.divide(np.cross(E, B), (B ** 2).sum(-1, keepdims=True))

        V_rtp = vector_cartesian_to_spherical(V, coords_spherical)
        V_r = V_rtp[..., 0] * meters_per_Rs
        V_theta = V_rtp[..., 1] * meters_per_Rs
        V_phi = V_rtp[..., 2] * meters_per_Rs

        # reshape for output
        b_r_arr = B_r[0, :, :, 0]  # squeeze out r and t dimensions
        b_theta_arr = B_theta[0, :, :, 0]
        b_phi_arr = B_phi[0, :, :, 0]

        v_r_arr = V_r[0, :, :, 0]
        v_theta_arr = V_theta[0, :, :, 0]
        v_phi_arr = V_phi[0, :, :, 0]

        damping_arr = self.damping * np.ones_like(b_r_arr)
        mu_arr = self.mu * np.ones_like(b_r_arr)
        kl_arr = self.kl * np.ones_like(b_r_arr)
        b0_arr = self.b1 * np.ones_like(b_r_arr)
        b1_arr = self.b0 * np.ones_like(b_r_arr)
        vmac_arr = self.vmac * np.ones_like(b_r_arr)

        return {'b0': b0_arr, 'b1': b1_arr, 'b_r': b_r_arr, 'b_theta': b_theta_arr,
                'damping': damping_arr, 'kl': kl_arr, 'mu': mu_arr, 'b_phi': b_phi_arr,
                'v_r': v_r_arr, 'v_theta': v_theta_arr, 'v_phi': v_phi_arr,
                'vmac': vmac_arr}

    def _transform_parameters(self, input_parameters, obs_coord):
        # create carrington map header
        carrington_header = make_heliographic_header(obs_coord.obstime, 'earth',
                                                     input_parameters['b_r'].shape,
                                                     frame='carrington')

        # create helioprojective map header
        solar_semidiameter_rad = np.arcsin(constants.radius / obs_coord.radius)
        angular_radius = Angle(solar_semidiameter_rad.to(u.arcsec))
        scale = (angular_radius.to(u.arcsec) / (self.nx // 2 * u.pix),
                 angular_radius.to(u.arcsec) / (self.ny // 2 * u.pix))
        dummy_data = np.zeros((self.nx, self.ny), dtype=np.float32)
        reference_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, observer=obs_coord, frame=frames.Helioprojective)
        helioprojective_header = make_fitswcs_header(dummy_data, reference_coord, scale=u.Quantity(scale))

        transformed_parameters = {}
        for k, parameter in input_parameters.items():
            carrington_map = Map(np.array(parameter), carrington_header)
            helioprojective_map = carrington_map.reproject_to(helioprojective_header)
            transformed_parameters[k] = helioprojective_map.data

        # create dummy helioprojective map
        dummy_carrington = np.zeros_like(input_parameters['kl'])
        carrington_map = Map(dummy_carrington, carrington_header)
        helioprojective_map = carrington_map.reproject_to(helioprojective_header)

        # Compute mu for the helioprojective map
        map_coords = all_coordinates_from_map(helioprojective_map)
        projective_coords = map_coords.transform_to(frames.Helioprojective)
        radial_distance = np.sqrt(
            projective_coords.Tx ** 2 +
            projective_coords.Ty ** 2) / helioprojective_map.rsun_obs
        mu = np.sqrt(1 - radial_distance.to_value() ** 2)
        mu = mu.astype(np.float32)
        transformed_parameters['mu'] = mu

        # Convert B and V
        b_r = transformed_parameters.pop('b_r')
        b_theta = transformed_parameters.pop('b_theta')
        b_phi = transformed_parameters.pop('b_phi')
        b_rtp = np.stack([b_r, b_theta, b_phi], -1)

        v_r = transformed_parameters.pop('v_r')
        v_theta = transformed_parameters.pop('v_theta')
        v_phi = transformed_parameters.pop('v_phi')
        v_rtp = np.stack([v_r, v_theta, v_phi], -1)

        # Transform vector quantities -- B, V -- to image coordinates
        # create transformation matrix
        carrington_coords = map_coords.transform_to(frames.HeliographicCarrington)
        lat, lon = carrington_coords.lat.to_value(u.rad), carrington_coords.lon.to_value(u.rad)
        latc, lonc = helioprojective_map.carrington_latitude.to_value(
            u.rad), helioprojective_map.carrington_longitude.to_value(u.rad)

        pAng = -np.deg2rad(helioprojective_map.meta.get('CROTA2', 0))
        a_matrix = image_to_spherical_matrix(lon, lat, lonc, latc, pAng=pAng)
        rtp_to_img_transform = np.linalg.inv(a_matrix)

        # transform b vector to image frame
        b_img = np.einsum("...ij,...j->...i", rtp_to_img_transform, b_rtp)  # in image xyz
        # b_im = (xi, eta, zeta)

        # convert to ME parameters
        # FLD
        b_field = np.linalg.norm(b_img, axis=-1)
        # INC
        sin_inc2 = (b_img[..., 0] ** 2 + b_img[..., 1] ** 2) / (b_field ** 2 + 1e-8)
        cos_inc = b_img[..., 2] / (b_field + 1e-8)
        # AZI
        sin2azi = -2 * b_img[..., 0] * b_img[..., 1] / (b_img[..., 0] ** 2 + b_img[..., 1] ** 2 + 1e-8)
        cos2azi = -(b_img[..., 0] ** 2 - b_img[..., 1] ** 2) / (b_img[..., 0] ** 2 + b_img[..., 1] ** 2 + 1e-8)
        azi = np.arctan2(-b_img[..., 0:1], b_img[..., 1:2])

        # add rotation of carrington frame
        v_rot = carrington_rotation_velocity(lat, radius=(1 * u.Rsun).to_value(u.m), f=np)
        v_rtp[..., 2] += v_rot  # add rotation in phi direction

        # transform b vector to image frame
        v_img = np.einsum("...ij,...j->...i", rtp_to_img_transform, v_rtp)  # in image xyz
        # convert to ME parameters
        vdop = -v_img[..., 2]

        transformed_parameters['b_field'] = b_field
        transformed_parameters['sin_inc2'] = sin_inc2
        transformed_parameters['cos_inc'] = cos_inc
        transformed_parameters['sin2azi'] = sin2azi
        transformed_parameters['cos2azi'] = cos2azi
        transformed_parameters['azi'] = azi
        transformed_parameters['vdop'] = vdop

        transformed_parameters['b_rtp'] = b_rtp
        transformed_parameters['v_rtp'] = v_rtp

        return transformed_parameters, helioprojective_map

    def create_spheromak_time_step(self, time_seconds, obs_coord):
        parameters = self._create_spheromak_params(time_seconds)
        # convert to numpy arrays
        transformed_parameters, dummy_helioprojective_map = self._transform_parameters(parameters, obs_coord)
        # convert back to torch tensors
        transformed_parameters = {k: torch.tensor(v, dtype=torch.float32) for k, v in transformed_parameters.items()}
        input_parameters = {k: v for k, v in transformed_parameters.items() if k not in ['b_rtp', 'v_rtp', 'azi']}
        stokes_profiles = self.convert_to_profiles(**input_parameters)
        return stokes_profiles, transformed_parameters, dummy_helioprojective_map

    def create_spherical_time_step_file(self, t_step, base_path, obs_coords):
        time_seconds = (obs_coords.obstime.to_datetime() - self.ref_time).total_seconds()
        profiles, parameters, dummy_map = self.create_spheromak_time_step(time_seconds, obs_coords)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, required=True, help='base path for the output data')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256], help='resolution of the images')
    parser.add_argument('--n_time_steps', type=int, default=100, help='number of time steps to generate')
    parser.add_argument('--obs_lon', type=float, default=0.0, help='observer longitude in degrees')
    parser.add_argument('--obs_lat', type=float, default=0.0, help='observer latitude in degrees')
    args = parser.parse_args()

    out_path = args.out_path

    os.makedirs(out_path, exist_ok=True)

    n_proc = 16

    obs_lon = args.obs_lon * u.deg
    obs_lat = args.obs_lat * u.deg
    observer_distance = 1 * u.AU

    t_start = datetime(2025, 1, 1, )
    t_end = datetime(2025, 2, 1)
    t_range = pd.date_range(t_start, t_end, periods=args.n_time_steps)

    lambda0 = 6173.3433 * u.AA
    lambda_grid = np.array([-0.1695, -0.1017, -0.0339, +0.0339, +0.1017, +0.1695]) * u.AA

    data_generator = SpheromakTestSetGenerator(nx=args.resolution[0], ny=args.resolution[1],
                                               lambda0=lambda0, lambda_grid=lambda_grid,
                                               g_up=2.50, ref_time=t_start)

    observers = []

    for time in t_range:
        coord = SkyCoord(lon=obs_lon, lat=obs_lat, radius=observer_distance,
                         obstime=time, observer="self",
                         frame=frames.HeliographicStonyhurst)
        coord = coord.transform_to(frames.HeliographicCarrington)
        observers.append(coord)
    #
    with Pool(n_proc) as p:
        in_data = [(t, out_path, obs) for t, obs in enumerate(observers)]
        p.starmap(data_generator.create_spherical_time_step_file, in_data)

    profiles = load_fits_profiles(out_path)
    parameters = load_parameters(os.path.join(out_path, 'parameters_*.npz'))

    os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)

    with Pool(n_proc) as p:
        in_data = [(profiles[i], os.path.join(out_path, 'images', f'stokes_{i:03d}.jpg'))
                   for i in range(profiles.shape[0])]
        p.starmap(plot_stokes, in_data)

    with Pool(n_proc) as p:
        in_data = [(parameters_dict := {k: v[i] for k, v in parameters.items() if k not in ['b_rtp', 'v_rtp']},
                    os.path.join(out_path, 'images', f'parameters_{i:03d}.jpg'))
                   for i in range(profiles.shape[0])]
        p.starmap(plot_parameters, in_data)

    with Pool(n_proc) as p:
        in_data = [(parameters['b_rtp'][i], os.path.join(out_path, 'images', f'brtp_{i:03d}.jpg'))
                   for i in range(profiles.shape[0])]
        p.starmap(plot_brtp, in_data)

    # plot coordinate grid
    files = sorted(glob.glob(os.path.join(out_path, f'*I0.fits')))
    with Pool(n_proc) as p:
        in_data = [(files[i], os.path.join(out_path, 'images', f'coords_{i:03d}.jpg'))
                   for i in range(len(files))]
        p.starmap(plot_coords, in_data)
