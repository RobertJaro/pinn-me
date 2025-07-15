import glob
import os.path

import numpy as np
import torch
from astropy.io import fits

from pme.train.me_atmosphere import MEAtmosphere


def convert_xy_to_rt(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)

    return r, t


def convert_rt_to_xy(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)

    return int(x), int(y)


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

    def convert_to_profiles(self, b0, b1, b_field, cos2azi, sin2azi, sin_inc2, cos_inc, damping, kl, mu, vdop, vmac):
        atmos = MEAtmosphere(self.lambda0, self.jUp, self.jLow, self.gUp, self.gLow, self.lambda_grid)
        # flatten and forward
        I, Q, U, V = atmos.forward(b_field.reshape(-1, 1),
                                   cos2azi.reshape(-1, 1), sin2azi.reshape(-1, 1),
                                   sin_inc2.reshape(-1, 1), cos_inc.reshape(-1, 1),
                                   vmac.reshape(-1, 1), damping.reshape(-1, 1),
                                   b0.reshape(-1, 1), b1.reshape(-1, 1), mu.reshape(-1, 1),
                                   vdop.reshape(-1, 1), kl.reshape(-1, 1))
        stokes_profiles = torch.stack([I, Q, U, V], -2).cpu().numpy()
        # (x, y, n_lambda, n_stokes)
        stokes_profiles = stokes_profiles.reshape(*b_field.shape, 4, *self.lambda_grid.shape)
        return stokes_profiles

    def _load_parameters(self, time_step, resolution=None):
        time_step = time_step / 3  # scale temporal evolution

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


def load_profiles(file_path):
    files = sorted(glob.glob(file_path))
    profiles = [np.load(f)['stokes_profiles'] for f in files]
    return np.stack(profiles, axis=0)  # (t, x, y, lambda, stokes)


def load_fits_profiles(file_path):
    stokes_profiles = {'I': [], 'Q': [], 'U': [], 'V': []}
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
