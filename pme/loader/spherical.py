import copy
import glob
import os
from datetime import datetime
from multiprocessing import Pool
from typing import Iterable, Optional

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
from tqdm import tqdm

from pme.data.phi_util import load_fix_phi_header
from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, image_to_spherical_matrix
from pme.loader.util import MultiprocessingWrapper
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
        self.batch_size = batch_size
        self.dataset_batch_size = dataset_batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        ref_time = parse(ref_time) if isinstance(ref_time, str) else ref_time
        train_configs = train_configs if isinstance(train_configs, list) else [train_configs]
        train_datasets = []
        pix_normalization = 1024
        for i, train_config in enumerate(train_configs):
            train_config = copy.deepcopy(train_config)
            ds_type = train_config.pop('type').lower()
            if ds_type == 'hmi':
                train_files = self._load_IQUV_files(train_config.pop('data_path'))
                ds_class = HMISphericalDataset
            elif ds_type == 'phi-hrt':
                train_files = self._load_all_files(train_config.pop('data_path'))
                ds_class = PHIHRTSphericalDataset
            elif ds_type == 'phi-fdt':
                train_files = self._load_all_files(train_config.pop('data_path'))
                ds_class = PHIFDTSphericalDataset
            elif ds_type == 'test':
                train_files = self._load_IQUV_files(train_config.pop('data_path'))
                ds_class = TestSphericalDataset
            else:
                raise ValueError(f'Unknown dataset type: {ds_type}')

            if 'sample_idx' in train_config:  # use a single sample for debugging
                sample_idx = train_config.pop('sample_idx')
                train_files = train_files[sample_idx:sample_idx + 1]
            if 'n_samples' in train_config:  # apply subsampling for debugging
                n_samples = train_config.pop('n_samples')
                sampling = len(train_files) // n_samples
                train_files = train_files[::sampling]
            if 'first_n_samples' in train_config:  # apply subsampling for debugging
                first_n_samples = train_config.pop('first_n_samples')
                train_files = train_files[:first_n_samples]
            with Pool(num_workers) as p:
                print(f'Processing {len(train_files)} training files with {num_workers} workers...')
                ds_ids = [f'train_{i:02d}_{j:03d}' for j in range(len(train_files))]
                ds_normalization = train_config.pop('stokes_normalization', stokes_normalization)
                instrument_id = train_config.pop('instrument_id')
                oversample_factor = train_config.pop('oversample_factor', 1)
                process_wrapper = MultiprocessingWrapper(ds_class,
                                                         instrument_id=instrument_id,
                                                         seconds_per_dt=seconds_per_dt,
                                                         Rs_per_ds=Rs_per_ds,
                                                         ref_time=ref_time,
                                                         batch_size=self.dataset_batch_size,
                                                         work_directory=work_directory,
                                                         oversample_factor=oversample_factor,
                                                         stokes_normalization=ds_normalization,
                                                         pix_normalization=pix_normalization,
                                                         **train_config)
                max_samples = 10
                sample_step = max(1, len(train_files) // max_samples)
                args = [{'data': tf, 'ds_id': ds_id, 'store_plot_image': (i % sample_step) == 0}
                        for i, (tf, ds_id) in tqdm(enumerate(zip(train_files, ds_ids)))]
                desc = f'Loading training datasets ({i + 1:02d}/{len(train_configs):02d})'
                tds = [r for r in tqdm(p.imap(process_wrapper.run, args), total=len(args), desc=desc)]
            train_datasets += tds  # append all datasets

        self.train_datasets = train_datasets

        # add lambda configuration for each instrument
        # only use the first dataset for each instrument to define the lambda configuration
        # assumes that wavelength_center is the same for all datasets of the same instrument
        wavelength_config = {}
        for ds in train_datasets:
            instrument_id = ds.instrument_id
            if instrument_id not in wavelength_config:
                wavelength_config[instrument_id] = ds.wavelength_config
        self.wavelength_config = wavelength_config

        for ds in train_datasets:
            if ds.store_plot_image:
                self.plot_dataset(ds)

        # log times in hours
        times = [(d.time - ref_time).total_seconds() / 3600 for d in train_datasets]
        table = wandb.Table(columns=["map_index", "time"])
        for i, t in enumerate(times):
            table.add_data(i, t)
        wandb.log({"map_times": table})

        ds_normalization = valid_config.get('stokes_normalization', stokes_normalization)
        valid_ds_type = valid_config['type'].lower()
        if valid_ds_type == 'hmi':
            valid_files = self._load_IQUV_files(valid_config['data_path'])
            sample_idx = valid_config.get('sample_idx', len(valid_files) // 2)
            resolution = valid_config.get('resolution', None)
            self.valid_dataset = HMISphericalDataset(valid_files[sample_idx],
                                                     ds_id='valid', instrument_id=valid_config['instrument_id'],
                                                     seconds_per_dt=seconds_per_dt,
                                                     Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                     stokes_normalization=ds_normalization,
                                                     pix_normalization=pix_normalization,
                                                     batch_size=self.batch_size, work_directory=work_directory,
                                                     filter_nans=False, shuffle=False, resolution=resolution)
        elif valid_ds_type == 'phi-hrt':
            valid_files = sorted(glob.glob(valid_config['data_path']))
            sample_idx = valid_config.get('sample_idx', len(valid_files) // 2)
            self.valid_dataset = PHIHRTSphericalDataset(valid_files[sample_idx],
                                                        ds_id='valid', instrument_id=valid_config['instrument_id'],
                                                        seconds_per_dt=seconds_per_dt,
                                                        Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                        stokes_normalization=ds_normalization,
                                                        pix_normalization=pix_normalization,
                                                        batch_size=self.batch_size, work_directory=work_directory,
                                                        filter_nans=False, shuffle=False)
        elif valid_ds_type == 'phi-fdt':
            valid_files = sorted(glob.glob(valid_config['data_path']))
            sample_idx = valid_config.get('sample_idx', len(valid_files) // 2)
            self.valid_dataset = PHIFDTSphericalDataset(valid_files[sample_idx],
                                                        ds_id='valid', instrument_id=valid_config['instrument_id'],
                                                        seconds_per_dt=seconds_per_dt,
                                                        Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                        stokes_normalization=ds_normalization,
                                                        pix_normalization=pix_normalization,
                                                        fix_header=valid_config.pop('fix_header', True),
                                                        batch_size=self.batch_size, work_directory=work_directory,
                                                        filter_nans=False, shuffle=False)
        elif valid_ds_type == 'test':
            valid_files = self._load_IQUV_files(valid_config['data_path'])
            sample_idx = valid_config.get('sample_idx', len(valid_files) // 2)
            self.valid_dataset = TestSphericalDataset(valid_files[sample_idx], noise=valid_config.get('noise', 0),
                                                      ds_id='valid', instrument_id=valid_config['instrument_id'],
                                                      seconds_per_dt=seconds_per_dt,
                                                      Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                      stokes_normalization=ds_normalization,
                                                      pix_normalization=pix_normalization,
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

        im = axs[1].imshow(ds.latitude, origin='lower', cmap='seismic', vmin=0, vmax=np.pi)
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
            f'Time: {ds.time.isoformat(" ", timespec="hours")} - lat:{np.rad2deg(ds.obs_lat):.1f}°, lon:{np.rad2deg(ds.obs_lon):.1f}°')

        fig.tight_layout()
        wandb.log({'Data Overview': wandb.Image(fig)})
        plt.close(fig)

    def _load_IQUV_files(self, data_path, num_wl=6):
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

    def _load_all_files(self, data_path):
        if isinstance(data_path, str):
            return sorted(glob.glob(data_path))
        elif isinstance(data_path, Iterable):
            files = [f for d in data_path for f in glob.glob(d)]
            return sorted(files)
        else:
            raise ValueError(f'Unknown data path type: {type(data_path)}. Expected str or Iterable[str].')

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
                            pin_memory=False, shuffle=True, prefetch_factor=5, persistent_workers=True)
        return loader

    def val_dataloader(self):
        self.valid_dataset.batch_size = self.batch_size
        data_loader = DataLoader(self.valid_dataset, batch_size=None, num_workers=self.num_workers,
                                 pin_memory=False, shuffle=False)
        return data_loader


class SphericalDataset(TensorsDataset):

    def __init__(self, stokes, map_data, wavelength_config, ds_id, instrument_id, seconds_per_dt, Rs_per_ds, ref_time,
                 stokes_normalization, pix_normalization, work_directory, oversample_factor=1,
                 store_plot_image=False, mu_limit=1e-3,
                 **kwargs):
        self.ds_id = ds_id
        self.instrument_id = instrument_id
        self.oversample_factor = oversample_factor

        self.wavelength_config = wavelength_config  # lambda grid and reference wavelength

        # load coordinates
        self.num_wl = stokes.shape[-1]
        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.ref_time = ref_time

        # normalize time
        time = map_data['time']  # datetime
        self.normalized_time = (time - ref_time).total_seconds() / seconds_per_dt

        # observer info
        self.obs_lat = map_data['obs_lat']
        self.obs_lon = map_data['obs_lon']
        self.pAng = map_data['pAng']
        self.time = map_data['time']
        self.wcs = map_data['wcs']

        # primary saved data
        mu = map_data['mu']  # x, y
        v_obs_los = map_data['v_obs_los']  # x, y
        spherical_coords = map_data['spherical_coords']  # x, y, 3
        pix = map_data['pix']

        # normalize stokes vector
        stokes /= stokes_normalization

        # normalize pix coordinates
        pix /= pix_normalization

        # remove off limb pixels
        spherical_coords[(mu < mu_limit) | np.isnan(mu)] = np.nan
        stokes[(mu < mu_limit) | np.isnan(mu)] = np.nan
        pix[(mu < mu_limit) | np.isnan(mu)] = np.nan

        # Plot Data Overview
        self.store_plot_image = store_plot_image
        if store_plot_image:
            self.integrated_V = block_reduce(np.abs(stokes[:, :, -1]).sum(-1), (8, 8), np.mean)
            self.latitude = block_reduce(spherical_coords[..., 1], (8, 8), np.mean)
            self.longitude = block_reduce(spherical_coords[..., 2], (8, 8), np.mean)
            self.v_obs_los = block_reduce(v_obs_los, (8, 8), np.mean)

        # image info
        self.value_range = np.stack([np.nanmin(stokes, (0, 1, -1)), np.nanmax(stokes, (0, 1, -1))], -1)
        self.image_shape = stokes.shape[:2]  # x, y
        self.data_range = np.array(
            [[np.nanmin(spherical_coords[..., i]), np.nanmax(spherical_coords[..., i])] for i in range(3)])

        # lambda grid
        self.wavelength_grid = torch.tensor(wavelength_config['wavelength_grid'].to_value(u.m),
                                            dtype=torch.float32).reshape(
            (1, -1))

        tensors = {'stokes': stokes.reshape((-1, *stokes.shape[2:])),
                   'spherical_coords': spherical_coords.reshape((-1, *spherical_coords.shape[2:])),
                   'mu': mu.reshape((-1, 1)), 'v_obs_los': v_obs_los.reshape((-1, 1)),
                   'pix': pix.reshape((-1, 2))
                   }

        super().__init__(tensors=tensors, work_directory=work_directory, **kwargs)

    def __getitem__(self, *args):
        out = super().__getitem__(*args)

        out['instrument_id'] = self.instrument_id  # add instrument id to output

        # load spherical coordinates
        spherical_coords = out.pop('spherical_coords')
        co_lat = spherical_coords[..., 1]
        lat = np.pi / 2 - co_lat
        lon = spherical_coords[..., 2]

        # convert to cartesian coordinates in models units
        coords = spherical_to_cartesian(spherical_coords, torch)
        coords /= self.Rs_per_ds  # normalize to solar radius

        # append time
        normalized_time = torch.ones((*coords.shape[:-1], 1), dtype=torch.float32) * self.normalized_time
        coords = torch.cat([normalized_time, coords], -1)
        out['coords'] = coords

        # add lambda grid
        wavelength_grid = torch.ones((*coords.shape[:-1], 1), dtype=torch.float32) * self.wavelength_grid
        out['wavelength_grid'] = wavelength_grid

        cartesian_to_spherical_transform = cartesian_to_spherical_matrix(spherical_coords)
        out['cartesian_to_spherical_transform'] = torch.tensor(cartesian_to_spherical_transform, dtype=torch.float32)

        img_to_rtp_transform = image_to_spherical_matrix(lon, lat,
                                                         self.obs_lon.to_value(u.rad), self.obs_lat.to_value(u.rad),
                                                         self.pAng.to_value(u.rad))
        rtp_to_img_transform = np.transpose(img_to_rtp_transform, (0, 2, 1))
        out['rtp_to_img_transform'] = torch.tensor(rtp_to_img_transform, dtype=torch.float32)

        return out


def load_v_observer_LOS(s_map):
    hgc_out = all_coordinates_from_map(s_map)
    hpc_out = hgc_out.transform_to(frame=frames.Helioprojective)

    # Components of the satellite velocity
    v_sdo_r = s_map.meta['OBS_VR']  # positive is away from Sun
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

    def __init__(self, data, resolution=None, *args, **kwargs):
        wavelength_center = 6173.3433 * u.AA
        wavelength_grid = np.array(
            [-0.1695, -0.1017, -0.0339, +0.0339, +0.1017, +0.1695]) * u.AA  # From Phillip Scherrer
        wavelength_config = {'wavelength_grid': wavelength_grid, 'wavelength_center': wavelength_center}
        I, Q, U, V = data
        ref_file = I[0]
        s_map = Map(ref_file)
        if resolution is not None:
            s_map = s_map.resample(resolution * u.pix)
        map_data = load_map_data(s_map)
        stokes = load_stokes_data(data, resolution)

        super().__init__(stokes, map_data, wavelength_config, *args, **kwargs)


class PHIHRTSphericalDataset(SphericalDataset):

    def __init__(self, data, *args, **kwargs):
        # wave_axis, voltagesData, tunning_constant, cpos, ref_wavelength = fits_get_sampling(file)
        # wave_axis = wave_axis * u.AA  # convert to angstroms
        # ref_wavelength = ref_wavelength * u.AA  # convert to angstroms
        # wavelength_center = ref_wavelength
        # wavelength_grid = wave_axis - wavelength_center
        header = fits.getheader(data)
        wavelength_center = header['WAVELNTH'] * u.AA  # reference wavelength from header
        wavelength_grid = np.array([header[f'WAVELN{i + 1:02d}'] for i in range(6)]) * u.AA
        wavelength_grid = wavelength_grid - wavelength_center  # center the grid at the reference wavelength
        wavelength_config = {'wavelength_grid': wavelength_grid, 'wavelength_center': wavelength_center}

        # set Earth corrected observation time
        header['DATE-OBS'] = header['DATE_EAR']

        # add CROTA2 to header if not present
        if 'CROTA2' not in header:
            header['CROTA2'] = header['CROTA']

        # (wl, stokes, x, y)
        stokes_data = fits.getdata(data)
        # --> (x, y, stokes, wl)
        stokes = np.transpose(stokes_data, (2, 3, 1, 0))
        stokes[stokes[:, :, 0, :].sum(-1) <= 1e-3] = np.nan  # mask out stokes I < 1e-3
        ref_map = Map(stokes_data, header)  # use the first wavelength as reference map
        map_data = load_map_data(ref_map)

        super().__init__(stokes, map_data, wavelength_config, *args, **kwargs)


class PHIFDTSphericalDataset(SphericalDataset):

    def __init__(self, data, fix_header=True, *args, **kwargs):
        # wave_axis, voltagesData, tunning_constant, cpos, ref_wavelength = fits_get_sampling(file)
        # wave_axis = wave_axis * u.AA  # convert to angstroms
        # ref_wavelength = ref_wavelength * u.AA  # convert to angstroms
        # wavelength_center = ref_wavelength
        # wavelength_grid = wave_axis - wavelength_center
        header = load_fix_phi_header(data) if fix_header else fits.getheader(data)
        wavelength_center = header['WAVELNTH'] * u.AA  # reference wavelength from header
        wavelength_grid = np.array([header[f'WAVELN{i + 1:02d}'] for i in range(6)]) * u.AA
        wavelength_grid = wavelength_grid - wavelength_center  # center the grid at the reference wavelength
        wavelength_config = {'wavelength_grid': wavelength_grid, 'wavelength_center': wavelength_center}

        # set Earth corrected observation time
        header['DATE-OBS'] = header['DATE_EAR']

        # (wl, stokes, x, y)
        stokes_data = fits.getdata(data)
        # --> (x, y, stokes, wl)
        stokes = np.transpose(stokes_data, (2, 3, 1, 0))
        ref_map = Map(stokes_data, header)
        map_data = load_map_data(ref_map)

        super().__init__(stokes, map_data, wavelength_config, *args, **kwargs)


class TestSphericalDataset(SphericalDataset):

    def __init__(self, data, noise=0, *args, **kwargs):
        wavelength_center = 6173.3433 * u.AA
        wavelength_grid = np.array(
            [-0.1695, -0.1017, -0.0339, +0.0339, +0.1017, +0.1695]) * u.AA  # From Phillip Scherrer
        wavelength_config = {'wavelength_grid': wavelength_grid, 'wavelength_center': wavelength_center}

        I, Q, U, V = data
        ref_file = I[0]
        s_map = Map(ref_file)
        map_data = load_map_data(s_map)
        stokes = load_stokes_data(data)

        # add noise
        normal_noise = np.random.normal(size=stokes.shape, scale=noise)
        stokes += normal_noise

        super().__init__(stokes, map_data, wavelength_config, *args, **kwargs)


def load_stokes_data(files, resolution: Optional[tuple[int, int]] = None):
    I, Q, U, V = files
    num_wl = len(I)
    if resolution is None:  # default load with astropy fits
        I_profile = np.stack([fits.getdata(I[j]) for j in range(num_wl)], -1)
        Q_profile = np.stack([fits.getdata(Q[j]) for j in range(num_wl)], -1)
        U_profile = np.stack([fits.getdata(U[j]) for j in range(num_wl)], -1)
        V_profile = np.stack([fits.getdata(V[j]) for j in range(num_wl)], -1)
    else:
        I_profile = np.stack([Map(I[j]).resample(resolution * u.pix).data for j in range(num_wl)], -1)
        Q_profile = np.stack([Map(Q[j]).resample(resolution * u.pix).data for j in range(num_wl)], -1)
        U_profile = np.stack([Map(U[j]).resample(resolution * u.pix).data for j in range(num_wl)], -1)
        V_profile = np.stack([Map(V[j]).resample(resolution * u.pix).data for j in range(num_wl)], -1)

    stokes = np.stack([I_profile, Q_profile, U_profile, V_profile], -2)

    return stokes


def load_map_data(s_map):
    time = s_map.date.to_datetime()

    # convert world coordinates to cartesian
    map_coords = all_coordinates_from_map(s_map)

    projective_coords = map_coords.transform_to(frames.Helioprojective)
    radial_distance = np.sqrt(projective_coords.Tx ** 2 + projective_coords.Ty ** 2) / s_map.rsun_obs
    radial_distance = radial_distance.to_value(u.dimensionless_unscaled)  # convert to dimensionless
    with np.errstate(invalid='ignore'):  # ignore invalid values for off-limb
        mu = np.sqrt(1 - radial_distance ** 2)
        mu = mu.astype(np.float32)

    carrington_coords = map_coords.transform_to(frames.HeliographicCarrington)
    lat, lon = carrington_coords.lat.to_value(u.rad), carrington_coords.lon.to_value(u.rad)
    r = np.ones_like(lon)  # carrington_coords.radius
    # r = r * u.solRad if r.unit == u.dimensionless_unscaled else r
    # convert latitude to colatitude
    spherical_coords = np.stack([r, np.pi / 2 - lat, lon], -1)

    # create observer transform
    pAng = s_map.meta.get('CROTA2', s_map.meta.get('CROTA', 0)) * u.deg
    pAng *= -1  # negative pAng
    pAng = pAng.to(u.rad)
    obs_lat = s_map.carrington_latitude.to(u.rad)
    obs_lon = s_map.carrington_longitude.to(u.rad)

    # load observer velocity
    v_obs_los = load_v_observer_LOS(s_map).astype(np.float32)

    # load observer HPC
    pix_coords = np.stack(np.mgrid[0:s_map.data.shape[0], 0:s_map.data.shape[1]], -1)  # x, y
    pix_coords = pix_coords.astype(np.float32)

    return {'mu': mu, 'v_obs_los': v_obs_los, 'pix': pix_coords,
            'obs_lat': obs_lat, 'obs_lon': obs_lon, 'pAng': pAng,
            'time': time, 'spherical_coords': spherical_coords, 'wcs': s_map.wcs,
            'instrument': s_map.instrument}
