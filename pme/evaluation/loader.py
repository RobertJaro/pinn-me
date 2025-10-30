from datetime import datetime

import numpy as np
import torch
from astropy import units as u
from torch import nn
from tqdm import tqdm

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix
from pme.model import jacobian


class PINNMEOutput:

    def __init__(self, model_path, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state = torch.load(model_path, map_location=self.device)

        self.parameter_model = state['parameter_model']
        self.parameter_model = nn.DataParallel(self.parameter_model)
        self.parameter_model.eval()

        self.cube_shape = state['cube_shape']
        self.lambda_config = state['lambda_config']
        self.data_range = state['data_range']

        lambda_config = list(self.lambda_config.values())[0]  # use first instrument as default
        # self.forward_model = MEAtmosphere(**lambda_config).to(self.device)
        # self.forward_model = nn.DataParallel(self.forward_model)
        # self.forward_model.eval()

        self.times = state['times']
        self.ref_time = state['ref_time']
        self.seconds_per_dt = state['seconds_per_dt']
        self.Rs_per_ds = state['Rs_per_ds']
        self.meters_per_ds = self.Rs_per_ds * (1 * u.Rsun).to_value(u.m)
        self.gauss_per_dB = state['gauss_per_dB']

    def load(self, coords, batch_size=int(2 ** 13), mu=None, progress=True, compute_jacobian=False):
        batch_size = batch_size * torch.cuda.device_count() if torch.cuda.is_available() else batch_size
        coords_shape = coords.shape
        coords_tensor = torch.tensor(coords, dtype=torch.float32).reshape(-1, coords.shape[-1])

        mu = torch.ones(*coords_tensor.shape[:-1], 1, dtype=torch.float32) if mu is None \
            else torch.tensor(mu, dtype=torch.float32).reshape(-1, 1)
        parameters = {}

        n_batches = int(np.ceil(coords_tensor.shape[0] / batch_size))
        iter_ = tqdm(range(n_batches)) if progress else range(n_batches)
        for i in iter_:
            batch = coords_tensor[i * batch_size:(i + 1) * batch_size].to(self.device)
            mu_batch = mu[i * batch_size:(i + 1) * batch_size].to(self.device)
            batch.requires_grad = True

            pred = self.parameter_model(batch)
            # workaround to compute jacobian for all parameters
            profile_keys = ['b_field', 'theta', 'chi', 'vmac', 'damping', 'b0', 'b1', 'vdop', 'kl']
            input_tensor = torch.cat([pred[key] for key in profile_keys], dim=-1)
            I, Q, U, V = self.forward_model(**{k: input_tensor[..., i:i + 1] for i, k in enumerate(profile_keys)},
                                            mu=mu_batch)

            stokes = torch.stack([I, Q, U, V], dim=-2)

            if compute_jacobian:
                flat_stokes = stokes.reshape((*stokes.shape[:-2], -1))
                jac_params = [jacobian(flat_stokes[..., i:i + 1], input_tensor).detach().cpu() for i in
                              range(flat_stokes.shape[-1])]
                jac_params = torch.cat(jac_params, -2).reshape(*stokes.shape, jac_params[0].shape[-1])
                for i, key in enumerate(profile_keys):
                    pred[f'jacobian_{key}'] = jac_params[..., i]

            pred['I'] = I
            pred['Q'] = Q
            pred['U'] = U
            pred['V'] = V

            for key, value in pred.items():
                if key not in parameters:
                    parameters[key] = []
                value = value.detach().cpu().numpy()
                parameters[key].append(value)

        parameters = {key: np.concatenate(value).reshape(*coords_shape[:-1], *value[0].shape[1:])
                      for key, value in parameters.items()}

        # reproject magnetic field vector
        b = parameters['b_field']
        theta = parameters['theta']
        chi = parameters['chi']
        b_xyz = to_cartesian(b, theta, chi)
        parameters['b_xyz'] = b_xyz
        b, theta, chi = to_spherical(b_xyz)
        parameters['b_field'] = b
        parameters['theta'] = theta
        parameters['chi'] = chi

        return parameters

    def load_parameters(self, coords, batch_size=int(2 ** 13), progress=True):
        batch_size = batch_size * torch.cuda.device_count() if torch.cuda.is_available() else batch_size
        coords_shape = coords.shape
        coords_tensor = torch.tensor(coords, dtype=torch.float32).reshape(-1, coords.shape[-1])

        parameters = {}

        n_batches = int(np.ceil(coords_tensor.shape[0] / batch_size))
        iter_ = tqdm(range(n_batches)) if progress else range(n_batches)
        for i in iter_:
            batch = coords_tensor[i * batch_size:(i + 1) * batch_size].to(self.device)
            batch.requires_grad = True

            pred = self.parameter_model(batch)

            for key, value in pred.items():
                if value is None:
                    continue
                if key not in parameters:
                    parameters[key] = []
                value = value.detach().cpu().numpy()
                parameters[key].append(value)

        parameters = {key: np.concatenate(value).reshape(*coords_shape[:-1], *value[0].shape[1:])
                      for key, value in parameters.items()}
        return parameters

    def load_profiles(self, parameters):
        tensors = {key: torch.tensor(value, dtype=torch.float32).to(self.device) for key, value in parameters.items()}
        I, Q, U, V = self.forward_model(**tensors)
        return {'I': I.cpu().numpy(), 'Q': Q.cpu().numpy(), 'U': U.cpu().numpy(), 'V': V.cpu().numpy()}

    def load_cube(self, **kwargs):
        coords = np.meshgrid(
            np.linspace(self.data_range[0][0], self.data_range[0][1], self.cube_shape[0], dtype=np.float32),
            np.linspace(self.data_range[1][0], self.data_range[1][1], self.cube_shape[1], dtype=np.float32),
            np.linspace(self.data_range[2][0], self.data_range[2][1], self.cube_shape[2], dtype=np.float32),
            indexing='ij')
        coords = np.stack(coords, axis=-1)

        return self.load(coords, **kwargs)

    def load_time(self, time, **kwargs):
        coords = np.meshgrid(
            np.ones(1, dtype=np.float32) * self._normalize_time(time),
            np.linspace(self.data_range[1][0], self.data_range[1][1], self.cube_shape[1], dtype=np.float32),
            np.linspace(self.data_range[2][0], self.data_range[2][1], self.cube_shape[2], dtype=np.float32),
            indexing='ij')
        coords = np.stack(coords, axis=-1)

        return self.load(coords, **kwargs)

    def _normalize_time(self, time):
        return (time - self.ref_time).total_seconds() / self.seconds_per_dt


class SPINNMEOutput(PINNMEOutput):

    def load_coords(self, spherical_coords, times):
        #
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        if isinstance(times, datetime):
            time_coords = np.full((*spherical_coords.shape[:-1], 1), times)
        normalized_time = self._normalize_time(times)

        coords = np.concatenate([normalized_time, cartesian_coords], axis=-1)
        cartesian_to_spherical_transform = cartesian_to_spherical_matrix(spherical_coords)

        parameter_cube = self.load_parameters(coords=coords)
        b_xyz = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']],
                               axis=-1) * self.gauss_per_dB
        b_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, b_xyz)

        v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']],
                               axis=-1) * self.meters_per_ds / self.seconds_per_dt
        v_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, v_xyz)

        return {'b_rtp': b_rtp, 'v_rtp': v_rtp}


def to_cartesian(b, inc, azi, disamb=None):
    if disamb is not None:
        azi[disamb] += np.pi
    b_x = b * np.sin(inc) * np.cos(azi)
    b_y = b * np.sin(inc) * np.sin(azi)
    b_z = b * np.cos(inc)
    return np.stack([b_x, b_y, b_z], axis=-1)


def to_spherical(b):
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    #
    b_field = np.linalg.norm(b, axis=-1)
    inc = np.arccos(bz / (b_field + 1e-10))
    azi = np.arctan2(by, bx)
    return b_field, inc, azi
