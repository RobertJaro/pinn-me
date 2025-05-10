import glob
import os.path

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from sunpy.coordinates import frames
from sunpy.map import make_heliographic_header, Map, make_fitswcs_header, all_coordinates_from_map
from sunpy.sun import constants

from pme.data.util import image_to_spherical_matrix
from pme.data.util import solar_differential_rotation_velocity
from pme.train.me_atmosphere import MEAtmosphere


def convert_xy_to_rt(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)

    return r, t


def convert_rt_to_xy(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)

    return int(x), int(y)

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

class TestSetGenerator():

    def __init__(self, lambda0, lambda_grid,
                 j_up=1.0, j_low=0.0, g_up=2.49, g_low=0,
                 nx=400, ny=400,
                 b_field_0=2000.0, vmac=2.0 * 1e3, damping=0.2, b0=0.8, b1=0.2, mu=1.0, vdop=2.0 * 1e3, kl=25.0):
        self.lambda0 = lambda0
        self.jUp = j_up
        self.jLow = j_low
        self.gUp = g_up
        self.gLow = g_low

        self.lambda_grid = lambda_grid

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
        parameters = self._load_parameters(time_step)
        stokes_profiles = self.convert_to_profiles(**parameters)

        return {'stokes_profiles': stokes_profiles}, parameters

    def load_spherical_time_step(self, time_step, obs_coord):
        parameters = self._load_parameters(time_step, resolution=(180, 180))
        # convert to numpy arrays
        parameters = {k: create_collage(v.detach().cpu().numpy()) for k, v in parameters.items()}
        transformed_parameters, dummy_helioprojective_map = self._transform_parameters(parameters, obs_coord)
        # convert back to torch tensors
        transformed_parameters = {k: torch.tensor(v, dtype=torch.float32) for k, v in transformed_parameters.items()}
        stokes_profiles = self.convert_to_profiles(**transformed_parameters)
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
        v_phi -= v_diff[:, None]
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

        scale = angular_radius.to(u.arcsec) / (self.nx // 2 * u.pix), angular_radius.to(u.arcsec) / (self.ny // 2 * u.pix)
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
        latc, lonc = helioprojective_map.carrington_latitude.to_value(u.rad), helioprojective_map.carrington_longitude.to_value(u.rad)

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

        return transformed_parameters, helioprojective_map

    def convert_to_profiles(self, b0, b1, b_field, azi, damping, kl, mu, inc, vdop, vmac):
        atmos = MEAtmosphere(self.lambda0, self.jUp, self.jLow, self.gUp, self.gLow, self.lambda_grid)
        # flatten and forward
        I, Q, U, V = atmos.forward(b_field.reshape(-1, 1), inc.reshape(-1, 1), azi.reshape(-1, 1),
                                   vmac.reshape(-1, 1), damping.reshape(-1, 1),
                                   b0.reshape(-1, 1), b1.reshape(-1, 1), mu.reshape(-1, 1),
                                   vdop.reshape(-1, 1), kl.reshape(-1, 1))
        stokes_profiles = torch.stack([I, Q, U, V], -2).cpu().numpy()
        # (x, y, n_lambda, n_stokes)
        stokes_profiles = stokes_profiles.reshape(*b_field.shape, 4, *self.lambda_grid.shape)
        return stokes_profiles

    def _load_parameters(self, time_step, resolution=None):
        nx = self.nx if resolution is None else resolution[0]
        ny = self.ny if resolution is None else resolution[1]
        xx, yy = np.meshgrid(np.linspace(-0.5 * nx, 0.5 * nx, nx),
                             np.linspace(-0.5 * ny, 0.5 * ny, ny),
                             indexing='ij')
        r0 = 50 + time_step / 2
        r, t = convert_xy_to_rt(xx, yy)
        # B --> (100, 100); lambda --> (50,); B * lambda --> (100, 100, 50)
        # B[..., None] --> (100, 100, 1); lambda[None, None, :] --> (1, 1, 50)
        b_field = self.b_field_0 * (r0 / (r + r0)) ** 2
        # theta is defined between 0 and pi
        incl_arr = ((r % r0) / r0 * np.pi)
        azi_arr = (t + time_step / 180 * np.pi)  # slow down the rotation
        b0_arr = self.b0 * (10 * r0 / (r + 10 * r0)) ** 2
        b1_arr = self.b1 * (10 * r0 / (r + 10 * r0)) ** 2
        b_field = torch.tensor(b_field, dtype=torch.float32)
        incl_arr = torch.tensor(incl_arr, dtype=torch.float32)
        azi_arr = torch.tensor(azi_arr, dtype=torch.float32)
        b0_arr = torch.tensor(b0_arr, dtype=torch.float32)
        b1_arr = torch.tensor(b1_arr, dtype=torch.float32)
        vmac_arr = self.vmac * torch.ones_like(b_field)
        damping_arr = self.damping * torch.ones_like(b_field)
        mu_arr = self.mu * torch.ones_like(b_field)
        vdop_arr = self.vdop * torch.ones_like(b_field)
        kl_arr = self.kl * torch.ones_like(b_field)
        # return b0_arr, b1_arr, b_field, azi_arr, damping_arr, kl_arr, mu_arr, r, incl_arr, vdop_arr, vmac_arr
        return {'b0': b0_arr, 'b1': b1_arr, 'b_field': b_field, 'azi': azi_arr,
                'damping': damping_arr, 'kl': kl_arr, 'mu': mu_arr, 'inc': incl_arr,
                'vdop': vdop_arr, 'vmac': vmac_arr}

    def create_time_step_file(self, t_step, base_path):
        profiles, parameters = self.load_time_step(t_step)
        # save to file
        np.savez(os.path.join(base_path, f'profile_{t_step:03d}.npz'), **profiles)
        np.savez(os.path.join(base_path, f'parameters_{t_step:03d}.npz'), **parameters)

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


def load_profiles(file_path):
    files = sorted(glob.glob(file_path))
    profiles = [np.load(f)['stokes_profiles'] for f in files]
    return np.stack(profiles, axis=0)  # (t, x, y, lambda, stokes)

def load_fits_profiles(file_path):
    stokes_profiles = {'I': [], 'Q':[], 'U': [], 'V': []}
    for stokes_id in stokes_profiles.keys():
        for wl_idx in range(6):
            # find all time steps
            files = sorted(glob.glob(os.path.join(file_path, f'*{stokes_id}{wl_idx}.fits')))
            profile = np.stack([fits.getdata(f) for f in files], 0)
            stokes_profiles[stokes_id].append(profile)
    I_profiles = np.stack(stokes_profiles['I'], -1)
    Q_profiles = np.stack(stokes_profiles['Q'], -1)
    U_profiles = np.stack(stokes_profiles['U'], -1)
    V_profiles = np.stack(stokes_profiles['V'], -1)
    profiles = np.stack([I_profiles, Q_profiles, U_profiles, V_profiles], -2)
    return profiles  # (t, x, y, stokes, lambda)

def load_parameters(file_path):
    files = sorted(glob.glob(file_path))
    parameters = [np.load(f) for f in files]
    out_parameters = {}
    for key in parameters[0].keys():
        out_parameters[key] = np.stack([p[key] for p in parameters], axis=0)
    return out_parameters  # (t, x, y)
