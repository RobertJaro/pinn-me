import numpy as np
# Constants
from scipy.special import jvp

import argparse
import glob
import os
import os.path
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sunpy.coordinates import get_earth
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

import matplotlib.pyplot as pl

# from pme.convert.vtk import save_vtk

class SpheromakTestSetGenerator(TestSetGenerator):

    def _create_spheromak_params(self, time_step, t_start = 0, t_end =600):
        '''
        Create the map of parameters for the spheromak test case on a sphere

        '''
        # Initial conditions

        t0 = 0

        R_solar = 1 # 699 Mm

        time_step = 0
        num_longitude_points = self.nx
        num_latitude_points = self.ny

        dlongitude = 2 * np.pi / num_longitude_points
        dlatitude = np.pi / num_latitude_points

        # Spatial parameters
        R_0 = 2

        coords_polar = np.stack(np.meshgrid(R_solar,
                                            np.arange(-np.pi / 2, np.pi / 2, dlatitude),
                                            np.arange(-np.pi, np.pi, dlongitude),
                                            time_step, indexing='ij'), -1)
        # print(f"timestep is {time_step}")
        #
        #  breakpoint()
        R, THETA, PHI, T = (coords_polar[..., 0],
                            coords_polar[..., 1] + np.pi/2,
                            coords_polar[..., 2],
                            coords_polar[..., 3])

        # Calculate the spheromak solution

        B_0 = 2000
        n = 2
        m = 3
        gamma = 2.5e5  # 2.5e5 # Mm/s
        C_alpha = 1.4934
        alpha_0 = C_alpha / R_0
        #
        A_r = np.zeros_like(R)
        A_theta = 0
        A_phi = 0
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
        B = np.stack([B_r, B_theta, B_phi], -1)
        E = np.stack([E_r, E_theta, E_phi], -1)

        V = np.divide(np.cross(E, B), np.array(B) * np.array(B).sum(-1, keepdims=True))

        b_r_arr = torch.tensor(B_r.squeeze(), dtype=torch.float32)
        b_theta_arr = torch.tensor(B_theta.squeeze(), dtype=torch.float32)
        b_phi_arr = torch.tensor(B_phi.squeeze(), dtype=torch.float32)
        b0_arr = self.b1 * torch.ones_like(b_r_arr)
        b1_arr = self.b0 * torch.ones_like(b_r_arr)
        vmac_arr = self.vmac * torch.ones_like(b_r_arr)
        v_r_arr = torch.tensor(V[..., 0].squeeze(), dtype=torch.float32)
        v_theta_arr = torch.tensor(V[..., 1].squeeze(), dtype=torch.float32)
        v_phi_arr = torch.tensor(V[..., 2].squeeze(), dtype=torch.float32)
        damping_arr = self.damping * torch.ones_like(b_r_arr)
        mu_arr = self.mu * torch.ones_like(b_r_arr)
        vdop_arr = self.vdop * torch.ones_like(b_r_arr)
        kl_arr = self.kl * torch.ones_like(b_r_arr)

        return {'b0': b0_arr , 'b1': b1_arr, 'b_r': b_r_arr, 'b_theta': b_theta_arr,
                'damping': damping_arr, 'kl': kl_arr, 'mu': mu_arr, 'b_phi': b_phi_arr,
                'v_r': v_r_arr, 'v_theta': v_theta_arr, 'v_phi':v_phi_arr,
                'vmac': vmac_arr}

    def _transform_parameters(self, input_parameters, obs_coord):

        latitudes = np.linspace(-np.pi / 2, np.pi / 2, self.ny) * u.rad
        # v_diff = solar_differential_rotation_velocity(latitudes).to_value(u.m / u.s)
        # input_parameters['v_phi'] += v_diff[:, None]

        # create carrington map header
        carrington_header = make_heliographic_header(obs_coord.obstime, 'earth',
                                                     input_parameters['b_r'].shape,
                                                     frame='carrington')

        # create helioprojective map header
        solar_semidiameter_rad = np.arcsin(constants.radius / obs_coord.radius)
        angular_radius = Angle(solar_semidiameter_rad.to(u.arcsec))

        scale = angular_radius.to(u.arcsec) / (self.nx // 2 * u.pix), angular_radius.to(u.arcsec) / (
                self.ny // 2 * u.pix)
        dummy_data = np.zeros((self.nx, self.ny), dtype=np.float32)
        reference_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, observer=obs_coord, frame=frames.Helioprojective)
        helioprojective_header = make_fitswcs_header(dummy_data, reference_coord, scale=u.Quantity(scale))
        # transform all parameters to helioprojective frame
        exclude_parameters = ['b_r', 'b_theta', 'b_azi', 'v_r', 'v_phi', 'v_theta']
        # stack b vector
        b_r = input_parameters.pop('b_r')
        b_theta = input_parameters.pop('b_theta')
        b_phi = input_parameters.pop('b_phi')
        b_rtp = np.stack([b_r, b_theta, b_phi], -1)

        v_r = input_parameters.pop('v_r')
        v_theta = input_parameters.pop('v_theta')
        v_phi = input_parameters.pop('v_phi')
        v_rtp = np.stack([v_r, v_theta, v_phi], -1)

        input_parameters = {k: v for k, v in input_parameters.items() if k not in exclude_parameters}
        transformed_parameters = {}
        for k, parameter in input_parameters.items():

            carrington_map = Map(np.array(parameter), carrington_header)
            helioprojective_map = carrington_map.reproject_to(helioprojective_header)
            transformed_parameters[k] = helioprojective_map.data
        print(f"transformed_parameters: {transformed_parameters['vmac'][200:210, 200:210]}")


        # create dummy helioprojective map

        dummy_carrington = np.zeros_like(input_parameters['kl'])
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
        a_matrix = image_to_spherical_matrix(lon, lat, lonc, latc, pAng=pAng)
        rtp_to_img_transform = np.linalg.inv(a_matrix)


        # transform b vector to image frame
        b_img = np.einsum("...ij,...j->...i", rtp_to_img_transform, b_rtp)  # in image xyz
        # b_im = (xi, eta, zeta)
        # convert to ME parameters
        b_field = np.linalg.norm(b_img, axis=-1)
        b_inc = np.arccos(b_img[..., 2] / (b_field + 1e-8))
        b_azi = np.arctan2(-b_img[..., 0], b_img[..., 1])

        # stack v vector

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
        # breakpoint()
        # print(f"transformed_parameters_vmac: {transformed_parameters['vmac']}")
        return transformed_parameters, helioprojective_map

    def create_spheromak_time_step(self, time_step, obs_coord, resolution=(180, 180)):

        parameters = self._create_spheromak_params(time_step, resolution)
        # convert to numpy arrays
        transformed_parameters, dummy_helioprojective_map = self._transform_parameters(parameters, obs_coord)
        # convert back to torch tensors
        transformed_parameters = {k: torch.tensor(v, dtype=torch.float32) for k, v in transformed_parameters.items()}
        input_parameters = {k: v for k, v in transformed_parameters.items() if k not in ['b_rtp', 'v_rtp']}
        stokes_profiles = self.convert_to_profiles(**input_parameters)
        return stokes_profiles, transformed_parameters, dummy_helioprojective_map

    def create_spherical_time_step_file(self, t_step, base_path, obs_coords):
        profiles, parameters, dummy_map = self.create_spheromak_time_step(t_step, obs_coords)
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


def vector_spherical_to_cartesian(v, c):
    vr, vt, vp = v[..., 0], v[..., 1], v[..., 2]
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    vx = vr * sin(t) * cos(p) + vt * cos(t) * cos(p) - vp * sin(p)
    vy = vr * sin(t) * sin(p) + vt * cos(t) * sin(p) + vp * cos(p)
    vz = vr * cos(t) - vt * sin(t)
    #
    return np.stack([vx, vy, vz], -1)

def to_spherical(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.stack([r, theta, phi], -1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_path', type=str, required=True,
                        help='base path for the output data')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256],
                        help='resolution of the images')
    parser.add_argument('--n_time_steps', type=int, default=100,
                        help='number of time steps to generate')

    args = parser.parse_args()

    out_path = args.out_path

    os.makedirs(out_path, exist_ok=True)

    n_proc = 30

    obs_lon = 0 * u.deg
    obs_lat = 0 * u.deg
    observer_distance = 1 * u.AU

    t_start = datetime(2025, 1, 1, )
    t_end = datetime(2025, 2, 1)
    t_range = pd.date_range(t_start, t_end, periods=args.n_time_steps)

    lambda_grid = np.array([-0.1695, -0.1017, -0.0339, +0.0339, +0.1017, +0.1695]) * u.AA  # From Phillip Scherrer
    lambda0 = 617.33433 * u.nm  # From Phillip Scherrer

    data_generator = SpheromakTestSetGenerator(nx=args.resolution[0], ny=args.resolution[1],
                                               lambda0=lambda0, lambda_grid=lambda_grid,
                                               g_up=2.50)

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