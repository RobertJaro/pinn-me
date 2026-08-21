from datetime import datetime

import numpy as np
import torch
from astropy import units as u
from torch import nn
from tqdm import tqdm

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix
from pme.model import jacobian
from pme.train.spherical_synthesis import (
    correct_spherical_limb_effects,
    field_free_forward_parameters,
    mix_magnetic_filling_factor,
    scale_spherical_forward_parameters,
    transform_spherical_parameters,
)


class PINNMEOutput:

    def __init__(self, model_path, device=None, instrument_id=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state = torch.load(model_path, map_location=self.device, weights_only=False)

        self.parameter_model = state['parameter_model'].to(self.device)
        self.parameter_model = nn.DataParallel(self.parameter_model)
        self.parameter_model.eval()

        self.cube_shape = state['cube_shape']
        self.wavelength_config = state['wavelength_config']
        self.data_range = state['data_range']

        self.forward_models = state.get('forward_models')
        self.forward_model = None
        if self.forward_models is not None:
            self.forward_models = self.forward_models.to(self.device)
            self.forward_models.eval()
            available_instruments = list(self.forward_models.keys())
            if instrument_id is None:
                if len(available_instruments) != 1:
                    raise ValueError(
                        'instrument_id is required when an inversion contains multiple instruments: '
                        f'{available_instruments}'
                    )
                instrument_id = available_instruments[0]
            if instrument_id not in self.forward_models:
                raise KeyError(
                    f'Unknown instrument {instrument_id!r}; available instruments: {available_instruments}'
                )
            self.forward_model = self.forward_models[instrument_id]
            self.forward_model.eval()
        self.instrument_id = instrument_id

        def restore_module_dict(name):
            modules = state.get(name, nn.ModuleDict())
            modules = nn.ModuleDict() if modules is None else modules
            modules = modules.to(self.device)
            modules.eval()
            return modules

        self.velocity_correction_models = restore_module_dict('velocity_correction_models')
        self.limb_correction_models = restore_module_dict('limb_correction_models')
        self.artifact_correction_models = restore_module_dict('artifact_correction_models')
        self.spectral_response_files = state.get('spectral_response_files', {})
        self.spectral_response_by_acquisition = state.get('spectral_response_by_acquisition', {})
        self.stokes_normalization_by_instrument = state.get('stokes_normalization_by_instrument', {})
        self.stokes_loss_config = state.get('stokes_loss_config', {'type': 'mse'})
        self.stokes_weight_config = state.get('stokes_weight_config')
        self.physics_config = state.get('physics_config')

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
            if self.forward_model is None:
                raise RuntimeError('This inversion does not contain a saved spectral forward model.')
            stokes_components = self.forward_model(
                **{key: input_tensor[..., j:j + 1] for j, key in enumerate(profile_keys)},
                mu=mu_batch,
            )

            stokes = torch.stack(stokes_components, dim=-2)

            if compute_jacobian:
                flat_stokes = stokes.reshape((*stokes.shape[:-2], -1))
                jac_params = [jacobian(flat_stokes[..., i:i + 1], input_tensor).detach().cpu() for i in
                              range(flat_stokes.shape[-1])]
                jac_params = torch.cat(jac_params, -2).reshape(*stokes.shape, jac_params[0].shape[-1])
                for i, key in enumerate(profile_keys):
                    pred[f'jacobian_{key}'] = jac_params[..., i]

            for name, component in zip(('I', 'Q', 'U', 'V'), stokes_components):
                pred[name] = component

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
        if self.forward_model is None:
            raise RuntimeError('This inversion does not contain a saved spectral forward model.')
        tensors = {key: torch.tensor(value, dtype=torch.float32).to(self.device) for key, value in parameters.items()}
        components = self.forward_model(**tensors)
        return {
            name: component.detach().cpu().numpy()
            for name, component in zip(('I', 'Q', 'U', 'V'), components)
        }

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

    def _instrument(self, instrument_id=None):
        instrument_id = self.instrument_id if instrument_id is None else instrument_id
        if self.forward_models is None:
            raise RuntimeError('This inversion does not contain saved spectral forward models.')
        if instrument_id not in self.forward_models:
            raise KeyError(
                f'Unknown instrument {instrument_id!r}; available instruments: {list(self.forward_models.keys())}'
            )
        return instrument_id, self.forward_models[instrument_id]

    @staticmethod
    def _flat_tensor(value, trailing_shape, name):
        value = torch.as_tensor(value, dtype=torch.float32)
        if tuple(value.shape[-len(trailing_shape):]) != tuple(trailing_shape):
            raise ValueError(f'{name} must end in shape {trailing_shape}, got {tuple(value.shape)}.')
        return value.reshape(-1, *trailing_shape)

    def synthesize_observation(self, coords, cartesian_to_spherical_transform,
                               rtp_to_img_transform, v_obs_los, mu,
                               wavelength_grid=None, instrument_id=None,
                               spectral_offsets=None, spectral_weights=None,
                               continuum_weights=None, detector_coords=None,
                               spectral_response_file=None, pix=None,
                               batch_size=int(2 ** 13), progress=True,
                               denormalize_stokes=False):
        """Evaluate a spherical inversion and synthesize observer-frame Stokes profiles.

        Geometry and velocities follow the identical transformation used during
        training. HMI response maps are sampled per detector pixel and per batch,
        so a full-disk evaluation does not materialize the dense response cube.
        Wavelengths and response offsets are in meters; velocities are in m/s.
        """
        instrument_id, forward_model = self._instrument(instrument_id)
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        if denormalize_stokes and instrument_id not in self.stokes_normalization_by_instrument:
            raise KeyError(
                f'No saved Stokes normalization for {instrument_id!r}; this is likely an older inversion artifact.'
            )
        coords = torch.as_tensor(coords, dtype=torch.float32)
        if coords.shape[-1] != 4:
            raise ValueError(f'coords must end in [time, x, y, z], got {tuple(coords.shape)}.')
        output_shape = tuple(coords.shape[:-1])
        n_points = int(np.prod(output_shape))
        coords = coords.reshape(-1, 4)
        c2s = self._flat_tensor(cartesian_to_spherical_transform, (3, 3),
                                'cartesian_to_spherical_transform')
        rtp2img = self._flat_tensor(rtp_to_img_transform, (3, 3), 'rtp_to_img_transform')
        v_obs = self._flat_tensor(v_obs_los, (1,), 'v_obs_los')
        mu = self._flat_tensor(mu, (1,), 'mu')
        for name, value in (('cartesian_to_spherical_transform', c2s),
                            ('rtp_to_img_transform', rtp2img), ('v_obs_los', v_obs), ('mu', mu)):
            if value.shape[0] != n_points:
                raise ValueError(f'{name} has {value.shape[0]} samples, expected {n_points}.')

        if wavelength_grid is None:
            wavelength_grid = self.wavelength_config[instrument_id]['wavelength_grid'].to_value(u.m)
        wavelength_grid = torch.as_tensor(wavelength_grid, dtype=torch.float32)
        if wavelength_grid.ndim == 1:
            wavelength_grid = wavelength_grid.unsqueeze(0).expand(n_points, -1)
        else:
            wavelength_grid = wavelength_grid.reshape(-1, wavelength_grid.shape[-1])
            if wavelength_grid.shape[0] == 1:
                wavelength_grid = wavelength_grid.expand(n_points, -1)
        if wavelength_grid.shape[0] != n_points:
            raise ValueError(f'wavelength_grid has {wavelength_grid.shape[0]} samples, expected {n_points}.')

        supplied_response = (spectral_offsets, spectral_weights, continuum_weights)
        if any(value is not None for value in supplied_response) and not all(
                value is not None for value in supplied_response):
            raise ValueError('spectral_offsets, spectral_weights, and continuum_weights are required together.')
        if spectral_response_file is not None and any(value is not None for value in supplied_response):
            raise ValueError('Provide either a response file or sampled response arrays, not both.')
        if spectral_response_file is not None and detector_coords is None:
            raise ValueError('detector_coords are required when spectral_response_file is provided.')
        if detector_coords is not None:
            detector_coords = self._flat_tensor(detector_coords, (2,), 'detector_coords')
            if detector_coords.shape[0] != n_points:
                raise ValueError(f'detector_coords has {detector_coords.shape[0]} samples, expected {n_points}.')
        if all(value is not None for value in supplied_response):
            spectral_offsets = torch.as_tensor(spectral_offsets, dtype=torch.float32).reshape(
                n_points, wavelength_grid.shape[-1], -1)
            spectral_weights = torch.as_tensor(spectral_weights, dtype=torch.float32).reshape_as(spectral_offsets)
            continuum_weights = torch.as_tensor(continuum_weights, dtype=torch.float32).reshape(
                n_points, wavelength_grid.shape[-1])
        is_hmi = forward_model.__class__.__name__ == 'HMIMEAtmosphere'
        if is_hmi and spectral_response_file is None \
                and spectral_offsets is None:
            raise ValueError('HMI synthesis requires a transmission response file or sampled response arrays.')

        if instrument_id in self.artifact_correction_models:
            if pix is None:
                raise ValueError(f'Artifact correction for {instrument_id!r} requires normalized pix coordinates.')
            pix = self._flat_tensor(pix, (2,), 'pix')
            if pix.shape[0] != n_points:
                raise ValueError(f'pix has {pix.shape[0]} samples, expected {n_points}.')

        finite = (torch.isfinite(coords).all(-1) & torch.isfinite(c2s).all((-2, -1)) &
                  torch.isfinite(rtp2img).all((-2, -1)) & torch.isfinite(v_obs).all(-1) &
                  torch.isfinite(mu).all(-1) & torch.isfinite(wavelength_grid).all(-1))
        if detector_coords is not None:
            finite &= torch.isfinite(detector_coords).all(-1)
        if pix is not None:
            finite &= torch.isfinite(pix).all(-1)
        if spectral_offsets is not None:
            finite &= (torch.isfinite(spectral_offsets).all((-2, -1)) &
                       torch.isfinite(spectral_weights).all((-2, -1)) &
                       torch.isfinite(continuum_weights).all(-1))
        valid_indices = torch.nonzero(finite, as_tuple=False).flatten()
        collected = {}

        def append(name, value):
            collected.setdefault(name, []).append(value.detach().cpu())

        iterator = range(0, valid_indices.numel(), batch_size)
        iterator = tqdm(iterator, total=int(np.ceil(valid_indices.numel() / batch_size)),
                        disable=not progress)
        vector_potential = bool(getattr(self.parameter_model.module, 'vector_potential', False))
        for start in iterator:
            index = valid_indices[start:start + batch_size]
            batch_coords = coords[index].to(self.device)
            if vector_potential:
                batch_coords.requires_grad_(True)
            with torch.set_grad_enabled(vector_potential):
                raw = self.parameter_model(batch_coords)
                transformed = transform_spherical_parameters(
                    raw, batch_coords, c2s[index].to(self.device), rtp2img[index].to(self.device),
                    self.meters_per_ds, self.seconds_per_dt,
                )
                forward_params = scale_spherical_forward_parameters(
                    raw, transformed, v_obs[index].to(self.device), self.gauss_per_dB,
                    self.meters_per_ds, self.seconds_per_dt,
                )
                if instrument_id in self.velocity_correction_models:
                    forward_params['vdop'] = forward_params['vdop'] + self.velocity_correction_models[
                        instrument_id](batch_coords[..., 0:1])
                limb_correction = None
                if instrument_id in self.limb_correction_models:
                    limb_correction = self.limb_correction_models[instrument_id](mu[index].to(self.device))
                    forward_params = correct_spherical_limb_effects(forward_params, limb_correction)

                response = {}
                if spectral_response_file is not None:
                    from pme.loader.spherical import sample_hmi_spectral_response
                    response = sample_hmi_spectral_response(
                        spectral_response_file, detector_coords[index]
                    )
                    response = {key: value.to(self.device) for key, value in response.items()}
                elif spectral_offsets is not None:
                    response = {
                        'spectral_offsets': spectral_offsets[index].to(self.device),
                        'spectral_weights': spectral_weights[index].to(self.device),
                        'continuum_weights': continuum_weights[index].to(self.device),
                    }
                components = forward_model(
                    **forward_params, mu=mu[index].to(self.device),
                    wavelength_grid=wavelength_grid[index].to(self.device), **response,
                )
                stokes = torch.stack(components, dim=-2)
                if limb_correction is not None:
                    field_free_components = forward_model(
                        **field_free_forward_parameters(forward_params),
                        mu=mu[index].to(self.device),
                        wavelength_grid=wavelength_grid[index].to(self.device),
                        **response,
                    )
                    field_free_stokes = torch.stack(field_free_components, dim=-2)
                    stokes = mix_magnetic_filling_factor(
                        stokes,
                        field_free_stokes,
                        limb_correction['filling_factor'],
                    )
                artifact_correction = None
                if instrument_id in self.artifact_correction_models:
                    artifact_correction = self.artifact_correction_models[instrument_id](
                        pix[index].to(self.device), stokes
                    )
                    stokes = artifact_correction['stokes_corr']

            append('_index', index)
            for key, value in raw.items():
                if value is not None:
                    append(f'raw_{key}' if key in forward_params else key, value)
            for key, value in forward_params.items():
                append(key, value)
            for key in ('b_rtp', 'b_img'):
                append(key, transformed[key] * self.gauss_per_dB)
            for key in ('v_rtp', 'v_img'):
                append(key, transformed[key] * self.meters_per_ds / self.seconds_per_dt)
            append('stokes', stokes)
            for name, component in zip(('I', 'Q', 'U', 'V'), stokes.unbind(dim=-2)):
                append(name, component)
            if limb_correction is not None:
                for key, value in limb_correction.items():
                    output_key = 'limb_filling_factor' if key == 'filling_factor' else key
                    append(output_key, value)
            if artifact_correction is not None:
                append('artifact_correction_params', artifact_correction['correction_params'])
                if 'filling_factor' in artifact_correction:
                    append('artifact_filling_factor', artifact_correction['filling_factor'])

        if not collected:
            raise ValueError('No finite coordinates were provided for synthesis.')
        indices = torch.cat(collected.pop('_index')).long()
        result = {}
        for key, chunks in collected.items():
            values = torch.cat(chunks)
            full = torch.full((n_points, *values.shape[1:]), torch.nan, dtype=values.dtype)
            full[indices] = values
            result[key] = full.numpy().reshape(*output_shape, *values.shape[1:])
        if denormalize_stokes:
            scale = self.stokes_normalization_by_instrument[instrument_id]
            for key in ('stokes', 'I', 'Q', 'U', 'V'):
                result[key] *= scale
        return result

    def load_hmi_observation(self, reference_file, spectral_response_file=None,
                             instrument_id=None, mu_limit=0.1,
                             batch_size=int(2 ** 13), progress=True,
                             denormalize_stokes=False):
        """Synthesize the geometry and transmission profile of one HMI FITS observation."""
        from sunpy.map import Map
        from pme.data.hmi_transmission import read_hmi_fits_acquisition
        from pme.data.util import image_to_spherical_matrix
        from pme.loader.spherical import load_hmi_detector_coords, load_map_data

        instrument_id, _ = self._instrument(instrument_id)
        source_map = Map(reference_file)
        map_data = load_map_data(source_map)
        spherical_coords = map_data['spherical_coords'].astype(np.float32)
        invalid = ~np.isfinite(map_data['mu']) | (map_data['mu'] < mu_limit)
        spherical_coords[invalid] = np.nan
        cartesian = spherical_to_cartesian(spherical_coords) / self.Rs_per_ds
        time = np.full((*cartesian.shape[:-1], 1), self._normalize_time(map_data['time']), dtype=np.float32)
        coords = np.concatenate([time, cartesian], axis=-1)
        c2s = cartesian_to_spherical_matrix(spherical_coords).astype(np.float32)
        latitude = np.pi / 2 - spherical_coords[..., 1]
        img_to_rtp = image_to_spherical_matrix(
            spherical_coords[..., 2], latitude, map_data['obs_lon'].to_value(u.rad),
            map_data['obs_lat'].to_value(u.rad), map_data['pAng'].to_value(u.rad),
        )
        rtp2img = np.swapaxes(img_to_rtp, -1, -2).astype(np.float32)
        detector_coords = load_hmi_detector_coords(source_map)
        pix = map_data['pix'].astype(np.float32)
        pix /= max((pix.shape[0] - 1) / 2, (pix.shape[1] - 1) / 2, 1.0)
        pix[invalid] = np.nan

        if spectral_response_file is None:
            acquisition = read_hmi_fits_acquisition(reference_file)
            try:
                spectral_response_file = self.spectral_response_by_acquisition[instrument_id][
                    acquisition['acquisition_key']]
            except KeyError as error:
                raise KeyError(
                    f"No saved HMI response for {acquisition['acquisition_key']} and instrument "
                    f"{instrument_id!r}; pass spectral_response_file explicitly."
                ) from error
        return self.synthesize_observation(
            coords, c2s, rtp2img, map_data['v_obs_los'][..., None], map_data['mu'][..., None],
            instrument_id=instrument_id, detector_coords=detector_coords,
            spectral_response_file=spectral_response_file, pix=pix,
            batch_size=batch_size, progress=progress,
            denormalize_stokes=denormalize_stokes,
        )

    def load_coords(self, spherical_coords, times):
        #
        cartesian_coords = spherical_to_cartesian(spherical_coords) / self.Rs_per_ds
        if isinstance(times, datetime):
            normalized_time = self._normalize_time(times)
            time_coords = np.full((*spherical_coords.shape[:-1], 1), normalized_time, dtype=np.float32)
        else:
            normalized_time = self._normalize_time(times)
            normalized_time = np.asarray(normalized_time, dtype=np.float32)
            time_coords = normalized_time[..., None]

        coords = np.concatenate([time_coords, cartesian_coords], axis=-1)
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
