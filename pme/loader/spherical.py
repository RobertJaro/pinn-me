import glob
import os
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool

import numpy as np
import torch
import wandb
from astropy import units as u
from astropy.io import fits
from astropy.nddata import block_reduce
from dateutil.parser import parse
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import all_coordinates_from_map, Map
from torch.utils.data import DataLoader

from pme.data.phi_util import load_fix_phi_header
from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, image_to_spherical_matrix
from pme.train.data_loader import TensorsDataset, CombinedDataset


class SphericalDataModule(LightningDataModule):

    def __init__(self, train_configs, valid_config, work_directory,
                 seconds_per_dt=24 * 60 * 60, Rs_per_ds=1, gauss_per_dB=1e3,
                 stokes_normalization=83696.0,
                 ref_time=datetime(2010, 5, 1, 18, 58),
                 batch_size=65536, dataset_batch_size=4096,
                 num_workers=None):
        super().__init__()

        # train parameters
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.batch_size = batch_size * n_gpus
        self.dataset_batch_size = dataset_batch_size * n_gpus
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        print('Using {} GPUs'.format(n_gpus))
        print('Using {} CPUs'.format(self.num_workers))

        ref_time = parse(ref_time) if isinstance(ref_time, str) else ref_time
        train_configs = train_configs if isinstance(train_configs, list) else [train_configs]
        train_datasets = []
        for i, train_config in enumerate(train_configs):
            ds_type = train_config['type'].lower()
            if ds_type == 'hmi':
                train_files = self._load_files(train_config['data_path'])
                ds_class = HMISphericalDataset
            elif ds_type == 'phi-hrt':
                train_files = sorted(glob.glob(train_config['data_path']))
                ds_class = PHIHRTSphericalDataset
            elif ds_type == 'phi-fdt':
                train_files = sorted(glob.glob(train_config['data_path']))
                ds_class = PHIFDTSphericalDataset
            else:
                raise ValueError(f'Unknown dataset type: {ds_type}')

            if 'sample_idx' in train_config:  # use a single sample for debugging
                sample_idx = train_config['sample_idx']
                train_files = train_files[sample_idx:sample_idx + 1]
            if 'n_samples' in train_config:  # apply subsampling for debugging
                n_samples = train_config['n_samples']
                sampling = len(train_files) // n_samples
                train_files = train_files[::sampling]
            if 'first_n_samples' in train_config:  # apply subsampling for debugging
                first_n_samples = train_config['first_n_samples']
                train_files = train_files[:first_n_samples]
            with Pool(num_workers) as p:
                print(f'Processing {len(train_files)} training files with {num_workers} workers...')
                ds_ids = [f'train_{i:02d}_{j:03d}' for j in range(len(train_files))]
                ds_normalization = train_config.get('stokes_normalization', stokes_normalization)
                instrument_id = train_config['instrument_id']
                oversample_factor = train_config.get('oversample_factor', 1)
                args = zip(train_files, ds_ids, repeat(instrument_id), repeat(seconds_per_dt), repeat(Rs_per_ds),
                           repeat(ref_time),
                           repeat(ds_normalization), repeat(self.dataset_batch_size), repeat(work_directory),
                           repeat(oversample_factor))
                tds = p.starmap(ds_class, args)
            train_datasets += tds  # append all datasets

        self.train_datasets = train_datasets

        # add lambda configuration for each instrument
        # only use the first dataset for each instrument to define the lambda configuration
        # assumes that lambda0 is the same for all datasets of the same instrument
        lambda_config = {}
        for ds in train_datasets:
            instrument_id = ds.instrument_id
            if instrument_id not in lambda_config:
                lambda_config[instrument_id] = ds.lambda_config
        self.lambda_config = lambda_config

        max_samples = 10
        sample_step = max(1, len(train_datasets) // max_samples)
        for ds in train_datasets[::sample_step]:
            self.plot_dataset(ds)

        valid_ds_type = valid_config['type'].lower()
        if valid_ds_type == 'hmi':
            valid_files = self._load_files(valid_config['data_path'])
            sample_idx = valid_config.get('sample_idx', len(valid_files) // 2)
            ds_normalization = valid_config.get('stokes_normalization', stokes_normalization)
            self.valid_dataset = HMISphericalDataset(valid_files[sample_idx],
                                                     ds_id='valid', instrument_id=valid_config['instrument_id'],
                                                     seconds_per_dt=seconds_per_dt,
                                                     Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                     stokes_normalization=ds_normalization,
                                                     batch_size=self.batch_size, work_directory=work_directory,
                                                     filter_nans=False, shuffle=False)
        elif valid_ds_type == 'phi-hrt':
            valid_files = sorted(glob.glob(valid_config['data_path']))
            sample_idx = valid_config.get('sample_idx', len(valid_files) // 2)
            self.valid_dataset = PHIHRTSphericalDataset(valid_files[sample_idx],
                                                        ds_id='valid', instrument_id=valid_config['instrument_id'],
                                                        seconds_per_dt=seconds_per_dt,
                                                        Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                        stokes_normalization=stokes_normalization,
                                                        batch_size=self.batch_size, work_directory=work_directory,
                                                        filter_nans=False, shuffle=False)
        elif valid_ds_type == 'phi-fdt':
            valid_files = sorted(glob.glob(valid_config['data_path']))
            sample_idx = valid_config.get('sample_idx', len(valid_files) // 2)
            self.valid_dataset = PHIFDTSphericalDataset(valid_files[sample_idx],
                                                        ds_id='valid', instrument_id=valid_config['instrument_id'],
                                                        seconds_per_dt=seconds_per_dt,
                                                        Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                        stokes_normalization=stokes_normalization,
                                                        batch_size=self.batch_size, work_directory=work_directory,
                                                        filter_nans=False, shuffle=False)
        else:
            raise ValueError(f'Unknown validation dataset type: {valid_ds_type}')

        self.ref_time = ref_time
        self.times = [d.time for d in self.train_datasets]
        self.seconds_per_dt = seconds_per_dt
        self.Rs_per_ds = Rs_per_ds
        self.gauss_per_dB = gauss_per_dB
        self.image_shape = self.valid_dataset.image_shape
        self.value_range = self.valid_dataset.value_range
        self.data_range = self.valid_dataset.data_range

    def plot_dataset(self, ds):
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))

        im = axs[0].imshow(ds.integrated_V, cmap='gray', origin='lower', norm='log')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Integrated V')

        im = axs[1].imshow(ds.latitude, origin='lower', cmap='seismic', vmin=-np.pi / 2, vmax=np.pi / 2)
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Latitude [rad]')

        im = axs[2].imshow(ds.longitude % (2 * np.pi), origin='lower', cmap='twilight', vmin=0, vmax=2 * np.pi)
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Longitude [rad]')

        v_min_max = np.nanmax(np.abs(ds.v_obs_los))
        im = axs[3].imshow(ds.v_obs_los, origin='lower', cmap='coolwarm_r', vmin=-v_min_max, vmax=v_min_max)
        divider = make_axes_locatable(axs[3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Observer LOS Velocity [m/s]')

        axs[0].set_title(
            f'Time: {ds.time.isoformat(" ", timespec="hours")} - lat:{ds.obs_lat.to_value(u.deg):.1f}°, lon:{ds.obs_lon.to_value(u.deg):.1f}°')

        fig.tight_layout()
        wandb.log({'Data Overview': wandb.Image(fig)})
        plt.close(fig)

    def _load_files(self, data_path, num_wl=6):
        # load maps
        I = np.stack([sorted(glob.glob(os.path.join(data_path, f'*I{int(i)}.fits')))
                      for i in range(num_wl)], -1)  # t, wl
        Q = np.stack([sorted(glob.glob(os.path.join(data_path, f'*Q{int(i)}.fits')))
                      for i in range(num_wl)], -1)  # t, wl
        U = np.stack([sorted(glob.glob(os.path.join(data_path, f'*U{int(i)}.fits')))
                      for i in range(num_wl)], -1)  # t, wl
        V = np.stack([sorted(glob.glob(os.path.join(data_path, f'*V{int(i)}.fits')))
                      for i in range(num_wl)], -1)  # t, wl
        files = np.stack([I, Q, U, V], 1)  # t, stokes, wl
        return files

    def train_dataloader(self):
        # shuffle asynchronously
        datasets = self.train_datasets
        # shuffle_async(datasets, self.num_workers)
        # update batch size
        for ds in datasets:
            ds.batch_size = self.dataset_batch_size
        # data loader with iterations based on the largest dataset
        combined_dataset = CombinedDataset(datasets, self.batch_size // self.dataset_batch_size)
        loader = DataLoader(combined_dataset, batch_size=None, num_workers=self.num_workers,
                            pin_memory=True, shuffle=True, prefetch_factor=5, persistent_workers=True)
        return loader

    def val_dataloader(self):
        self.valid_dataset.batch_size = self.batch_size
        data_loader = DataLoader(self.valid_dataset, batch_size=None, num_workers=self.num_workers,
                                 pin_memory=True, shuffle=False)
        return data_loader


class SphericalDataset(TensorsDataset):

    def __init__(self, stokes, map_data, lambda_config, ds_id, instrument_id, seconds_per_dt, Rs_per_ds, ref_time,
                 stokes_normalization, batch_size, work_directory, oversample_factor=1, **kwargs):
        self.ds_id = ds_id
        self.instrument_id = instrument_id
        self.oversample_factor = oversample_factor

        self.lambda_config = lambda_config  # lambda grid and reference wavelength

        # load coordinates
        self.num_wl = stokes.shape[-1]
        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.ref_time = ref_time

        stokes = stokes  # x, y, stokes, wl
        coords = map_data['cartesian_coords']  # x, y, 3
        coords /= Rs_per_ds  # normalize to solar radius
        # append time
        time = map_data['time']  # datetime
        normalized_time = (time - ref_time).total_seconds() / seconds_per_dt
        normalized_time = np.ones((*coords.shape[:-1], 1), dtype=np.float32) * normalized_time
        coords = np.concatenate([normalized_time, coords], -1)

        cartesian_to_spherical_transform = map_data['cartesian_to_spherical_transform']  # x, y, 3, 3
        rtp_to_img_transform = map_data['rtp_to_img_transform']  # x, y, 3, 3
        mu = map_data['mu']  # x, y
        v_obs_los = map_data['v_obs_los']  # x, y
        carrington_coords = map_data['carrington_coords']  # x, y, 3

        # apply mask filter - coordinates + stokes for normalization
        coords[(mu < 1e-2) | np.isnan(mu)] = np.nan
        stokes[(mu < 1e-2) | np.isnan(mu)] = np.nan

        # Plot Data Overview
        self.integrated_V = block_reduce(np.abs(stokes[:, :, -1]).sum(-1), (8, 8), np.mean)
        self.latitude = block_reduce(carrington_coords[..., 1], (8, 8), np.mean)
        self.longitude = block_reduce(carrington_coords[..., 2], (8, 8), np.mean)
        self.v_obs_los = block_reduce(v_obs_los, (8, 8), np.mean)
        self.obs_lat = map_data['obs_lat']
        self.obs_lon = map_data['obs_lon']

        # normalize stokes vector
        stokes /= stokes_normalization

        self.time = map_data['time']
        self.value_range = np.stack([np.nanmin(stokes, (0, 1, -1)), np.nanmax(stokes, (0, 1, -1))], -1)
        self.image_shape = stokes.shape[:2]  # x, y
        self.wcs = map_data['wcs']

        self.data_range = np.array(
            [[np.nanmin(carrington_coords[..., i]), np.nanmax(carrington_coords[..., i])] for i in range(3)])

        lambda_grid = lambda_config['lambda_grid'].to_value(u.m)
        lambda_grid = np.ones_like(coords[..., 0:1]) * lambda_grid.reshape((1, 1, -1))  # x, y, wl

        tensors = {'stokes': stokes.reshape((-1, *stokes.shape[2:])),
                   'coords': coords.reshape((-1, *coords.shape[2:])),
                   'cartesian_to_spherical_transform': cartesian_to_spherical_transform.reshape(
                       (-1, *cartesian_to_spherical_transform.shape[2:])),
                   'rtp_to_img_transform': rtp_to_img_transform.reshape((-1, *rtp_to_img_transform.shape[2:])),
                   'mu': mu.reshape((-1, 1)), 'v_obs_los': v_obs_los.reshape((-1, 1)),
                   'lambda_grid': lambda_grid.reshape((-1, *lambda_grid.shape[2:])),
                   }

        super().__init__(tensors=tensors, batch_size=batch_size, work_directory=work_directory, **kwargs)

    def __getitem__(self, *args):
        out = super().__getitem__(*args)
        out['instrument_id'] = self.instrument_id  # add instrument id to output
        return out


def load_v_observer_LOS(s_map):
    hgc_out = all_coordinates_from_map(s_map)
    hpc_out = hgc_out.transform_to(frame=frames.Helioprojective)

    # Components of the satellite velocity
    v_sdo_r = s_map.meta['OBS_VR']
    v_sdo_w = s_map.meta['OBS_VW']
    v_sdo_n = s_map.meta['OBS_VN']

    theta_x = hpc_out.Tx.to_value(u.rad)
    theta_y = hpc_out.Ty.to_value(u.rad)

    theta_p = np.arctan2(np.sqrt(np.cos(theta_y) ** 2 * np.sin(theta_x) ** 2 + np.sin(theta_y) ** 2),
                         np.cos(theta_y) * np.cos(theta_x))
    psi = np.arctan2(-np.cos(theta_y) * np.sin(theta_x), np.sin(theta_y))

    # satellite motion
    v_obs_los = (v_sdo_w * np.sin(theta_p) * np.sin(psi) -
                 v_sdo_n * np.sin(theta_p) * np.cos(psi) +
                 v_sdo_r * np.cos(theta_p))
    return v_obs_los


class HMISphericalDataset(SphericalDataset):

    def __init__(self, files, *args, **kwargs):
        lambda0 = 6173.3433 * u.AA
        lambda_grid = np.array([-0.1695, -0.1017, -0.0339, +0.0339, +0.1017, +0.1695]) * u.AA  # From Phillip Scherrer
        lambda_config = {'lambda_grid': lambda_grid, 'lambda0': lambda0}

        I, Q, U, V = files
        ref_file = I[0]
        s_map = Map(ref_file)
        map_data = load_map_data(s_map)
        stokes = load_stokes_data(files)

        super().__init__(stokes, map_data, lambda_config, *args, **kwargs)


class PHIHRTSphericalDataset(SphericalDataset):

    def __init__(self, file, *args, **kwargs):
        # wave_axis, voltagesData, tunning_constant, cpos, ref_wavelength = fits_get_sampling(file)
        # wave_axis = wave_axis * u.AA  # convert to angstroms
        # ref_wavelength = ref_wavelength * u.AA  # convert to angstroms
        # lambda_center = ref_wavelength
        # lambda_grid = wave_axis - lambda_center
        header = fits.getheader(file)
        lambda_center = header['WAVELNTH'] * u.AA  # reference wavelength from header
        lambda_grid = np.array([header[f'WAVELN{i + 1:02d}'] for i in range(6)]) * u.AA
        lambda_grid = lambda_grid - lambda_center  # center the grid at the reference wavelength
        lambda_config = {'lambda_grid': lambda_grid, 'lambda0': lambda_center}

        # set Earth corrected observation time
        header['DATE-OBS'] = header['DATE_EAR']

        # (wl, stokes, x, y)
        data = fits.getdata(file)
        # --> (x, y, stokes, wl)
        stokes = np.transpose(data, (2, 3, 1, 0))
        stokes[stokes[:, :, 0, :].sum(-1) <= 1e-3] = np.nan  # mask out stokes I < 1e-3
        ref_map = Map(data, header)  # use the first wavelength as reference map
        map_data = load_map_data(ref_map)

        super().__init__(stokes, map_data, lambda_config, *args, **kwargs)


class PHIFDTSphericalDataset(SphericalDataset):

    def __init__(self, file, *args, **kwargs):
        # wave_axis, voltagesData, tunning_constant, cpos, ref_wavelength = fits_get_sampling(file)
        # wave_axis = wave_axis * u.AA  # convert to angstroms
        # ref_wavelength = ref_wavelength * u.AA  # convert to angstroms
        # lambda_center = ref_wavelength
        # lambda_grid = wave_axis - lambda_center
        header = load_fix_phi_header(file)
        lambda_center = header['WAVELNTH'] * u.AA  # reference wavelength from header
        lambda_grid = np.array([header[f'WAVELN{i + 1:02d}'] for i in range(6)]) * u.AA
        lambda_grid = lambda_grid - lambda_center  # center the grid at the reference wavelength
        lambda_config = {'lambda_grid': lambda_grid, 'lambda0': lambda_center}

        # set Earth corrected observation time
        header['DATE-OBS'] = header['DATE_EAR']

        # (wl, stokes, x, y)
        data = fits.getdata(file)
        # --> (x, y, stokes, wl)
        stokes = np.transpose(data, (2, 3, 1, 0))
        ref_map = Map(data, header)
        map_data = load_map_data(ref_map)

        super().__init__(stokes, map_data, lambda_config, *args, **kwargs)


def load_stokes_data(files):
    I, Q, U, V = files
    num_wl = len(I)

    I_profile = np.stack([fits.getdata(I[j]) for j in range(num_wl)], -1)
    Q_profile = np.stack([fits.getdata(Q[j]) for j in range(num_wl)], -1)
    U_profile = np.stack([fits.getdata(U[j]) for j in range(num_wl)], -1)
    V_profile = np.stack([fits.getdata(V[j]) for j in range(num_wl)], -1)

    stokes = np.stack([I_profile, Q_profile, U_profile, V_profile], -2)

    return stokes


def load_map_data(s_map):
    time = s_map.date.to_datetime()

    # convert world coordinates to cartesian
    spherical_coords = all_coordinates_from_map(s_map)

    projective_coords = spherical_coords.transform_to(frames.Helioprojective)
    radial_distance = np.sqrt(projective_coords.Tx ** 2 + projective_coords.Ty ** 2) / s_map.rsun_obs
    mu = np.sqrt(1 - radial_distance ** 2)
    mu = mu.astype(np.float32)

    carrington_coords = spherical_coords.transform_to(frames.HeliographicCarrington)
    lat, lon = carrington_coords.lat.to_value(u.rad), carrington_coords.lon.to_value(u.rad)
    r = np.ones_like(lon)  # carrington_coords.radius
    # r = r * u.solRad if r.unit == u.dimensionless_unscaled else r
    carrington_coords = np.stack([r, lat, lon], -1)

    # create rtp transform
    cartesian_to_spherical_transform = cartesian_to_spherical_matrix(carrington_coords)

    cartesian_coords = spherical_to_cartesian(carrington_coords)

    # create observer transform
    # latc, lonc = np.deg2rad(s_map.meta['CRLT_OBS']), np.deg2rad(s_map.meta['CRLN_OBS'])
    pAng = s_map.meta.get('CROTA2', s_map.meta.get('CROTA', 0)) * u.deg
    pAng *= -1  # negative pAng
    pAng = pAng.to_value(u.rad)
    obs_lat = s_map.carrington_latitude
    obs_lon = s_map.carrington_longitude
    latc, lonc = obs_lat.to_value(u.rad), obs_lon.to_value(u.rad)
    img_to_rtp_transform = image_to_spherical_matrix(lon, lat, lonc, latc, pAng=pAng)
    rtp_to_img_transform = np.linalg.inv(img_to_rtp_transform)

    # load observer velocity
    v_obs_los = load_v_observer_LOS(s_map).astype(np.float32)

    return {'cartesian_to_spherical_transform': cartesian_to_spherical_transform,
            'rtp_to_img_transform': rtp_to_img_transform,
            'mu': mu, 'v_obs_los': v_obs_los,
            'time': time,
            'obs_lat': obs_lat, 'obs_lon': obs_lon,
            'cartesian_coords': cartesian_coords,
            'carrington_coords': carrington_coords, 'wcs': s_map.wcs}
