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
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import all_coordinates_from_map, Map
from torch.utils.data import DataLoader, RandomSampler

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, image_to_spherical_matrix
from pme.train.data_loader import TensorsDataset, shuffle_async


class SphericalDataModule(LightningDataModule):

    def __init__(self, train_config, valid_config, work_directory, seconds_per_dt=36000, Rs_per_ds=1,
                 stokes_normalization=83696.0,
                 ref_time=datetime(2010, 5, 1, 18, 58), batch_size=4096, num_workers=None):
        super().__init__()

        # train parameters
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.batch_size = batch_size * n_gpus
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        train_files = self._load_files(train_config['data_path'])
        train_files = train_files
        data_set_batch_size = np.ceil(self.batch_size / len(train_files)).astype(int)
        with Pool(num_workers) as p:
            args = zip(train_files, repeat(seconds_per_dt), repeat(Rs_per_ds), repeat(ref_time),
                       repeat(stokes_normalization), repeat(data_set_batch_size), repeat(work_directory))
            train_datasets = p.starmap(HMISphericalDataset, args)

        self.train_datasets = {f'hmi_{i:02d}': ds for i, ds in enumerate(train_datasets)}

        for ds in train_datasets:
            self.plot_dataset(ds)

        valid_files = self._load_files(valid_config['data_path'])
        sample_idx = valid_config.get('sample_idx', len(valid_files) // 2)
        self.valid_dataset = HMISphericalDataset(valid_files[sample_idx],
                                                 seconds_per_dt=seconds_per_dt,
                                                 Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                 stokes_normalization=stokes_normalization,
                                                 batch_size=self.batch_size, work_directory=work_directory,
                                                 filter_nans=False, shuffle=False)

        self.ref_time = ref_time
        self.times = [d.time for d in self.train_datasets.values()]
        self.seconds_per_dt = seconds_per_dt
        self.Rs_per_ds = Rs_per_ds
        self.image_shape = self.valid_dataset.image_shape
        self.value_range = self.valid_dataset.value_range
        self.data_range = self.valid_dataset.data_range

        # centered at lambda0
        self.lambda_grid = self.valid_dataset.lambda_grid
        self.lambda_config = self.valid_dataset.lambda_config

    def plot_dataset(self, ds):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        im = axs[0].imshow(ds.integrated_V, cmap='gray', origin='lower', norm='log')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Integrated V')

        im = axs[1].imshow(ds.latitude, origin='lower', cmap='seismic')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Latitude [rad]')

        im = axs[2].imshow(ds.longitude, origin='lower', cmap='twilight')
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Longitude [rad]')

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
        shuffle_async(datasets, self.num_workers)
        # data loader with iterations based on the largest dataset
        max_ds_len = max([len(ds) for ds in datasets.values()])
        n_ds_workers = max(self.num_workers // len(datasets), 2)
        loaders = {}
        for i, (name, dataset) in enumerate(datasets.items()):
            sampler = RandomSampler(dataset, replacement=True, num_samples=int(max_ds_len))
            loaders[name] = DataLoader(dataset, batch_size=None, num_workers=n_ds_workers,
                                       pin_memory=True, sampler=sampler)
        return loaders

    def val_dataloader(self):
        data_loader = DataLoader(self.valid_dataset, batch_size=None, num_workers=self.num_workers,
                                 pin_memory=True, shuffle=False)
        return data_loader


class HMISphericalDataset(TensorsDataset):

    def __init__(self, files, seconds_per_dt, Rs_per_ds, ref_time, stokes_normalization, batch_size, work_directory,
                 **kwargs):
        lambda_shifts = np.array([-0.1695, -0.1017, -0.0339, +0.0339, +0.1017, +0.1695])  # From Phillip Scherrer
        lambda_center = 6173.3433  # From Phillip Scherrer

        self.lambda_grid = lambda_shifts
        self.lambda_config = {'lambda0': lambda_center * u.AA, 'lambda_grid': self.lambda_grid * u.AA,
                              'j_up': 1.0, 'j_low': 0.0, 'g_up': 2.50, 'g_low': 0.0}

        # load coordinates
        self.num_wl = files.shape[1]
        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.ref_time = ref_time

        data = self._load_data(files)

        stokes = data['stokes']  # x, y, stokes, wl
        coords = data['coords']  # x, y, 4
        cartesian_to_spherical_transform = data['cartesian_to_spherical_transform']  # x, y, 3, 3
        rtp_to_img_transform = data['rtp_to_img_transform']  # x, y, 3, 3
        mu = data['mu']  # x, y
        carrington_coords = data['carrington_coords']  # x, y, 3

        # Plot Data Overview
        self.integrated_V = np.abs(stokes[:, :, -1]).sum(-1)
        self.latitude = carrington_coords[..., 1]
        self.longitude = carrington_coords[..., 2]
        self.obs_lat = data['obs_lat']
        self.obs_lon = data['obs_lon']

        # normalize stokes vector
        stokes /= stokes_normalization

        self.time = data['time']
        self.value_range = np.stack([np.nanmin(stokes, (0, 1, -1)), np.nanmax(stokes, (0, 1, -1))], -1)
        self.image_shape = stokes.shape[:2]  # x, y
        self.wcs = data['wcs']

        self.data_range = np.array(
            [[np.nanmin(carrington_coords[..., i]), np.nanmax(carrington_coords[..., i])] for i in range(3)])

        tensors = {'stokes': stokes.reshape((-1, *stokes.shape[2:])),
                   'coords': coords.reshape((-1, *coords.shape[2:])),
                   'cartesian_to_spherical_transform': cartesian_to_spherical_transform.reshape(
                       (-1, *cartesian_to_spherical_transform.shape[2:])),
                   'rtp_to_img_transform': rtp_to_img_transform.reshape((-1, *rtp_to_img_transform.shape[2:])),
                   'mu': mu.reshape((-1, 1))}

        super().__init__(tensors=tensors, batch_size=batch_size, work_directory=work_directory, **kwargs)

    def _load_data(self, files):
        I, Q, U, V = files
        ref_file = I[0]

        s_map = Map(ref_file)
        print(f'Loading data: {s_map.date.to_datetime().isoformat(" ")}')

        # convert world coordinates to cartesian
        spherical_coords = all_coordinates_from_map(s_map)

        projective_coords = spherical_coords.transform_to(frames.Helioprojective)
        radial_distance = np.sqrt(projective_coords.Tx ** 2 + projective_coords.Ty ** 2) / s_map.rsun_obs
        mu = np.cos(radial_distance.to_value(u.dimensionless_unscaled) * np.pi / 2)
        mu = mu.astype(np.float32)

        carrington_coords = spherical_coords.transform_to(frames.HeliographicCarrington)
        lat, lon = carrington_coords.lat.to(u.rad).value, carrington_coords.lon.to(u.rad).value
        r = carrington_coords.radius
        #
        r = r * u.solRad if r.unit == u.dimensionless_unscaled else r
        carrington_coords = np.stack([r.to_value(u.solRad), lat, lon], -1)
        cartesian_coords = spherical_to_cartesian(carrington_coords) / self.Rs_per_ds

        # append time
        normalized_time = (s_map.date.to_datetime() - self.ref_time).total_seconds() / self.seconds_per_dt
        time = np.ones((*cartesian_coords.shape[:-1], 1), dtype=np.float32) * normalized_time
        cartesian_coords = np.concatenate([time, cartesian_coords], -1)

        # create rtp transform
        cartesian_to_spherical_transform = cartesian_to_spherical_matrix(carrington_coords)

        # create observer transform
        latc, lonc = np.deg2rad(s_map.meta['CRLT_OBS']), np.deg2rad(s_map.meta['CRLN_OBS'])
        pAng = -np.deg2rad(s_map.meta['CROTA2'])
        a_matrix = image_to_spherical_matrix(lon, lat, latc, lonc, pAng=pAng)
        rtp_to_img_transform = np.linalg.inv(a_matrix)

        # alternative hmi_b2ptr
        # stonyhurst_coords = spherical_coords.transform_to(frames.HeliographicStonyhurst)
        # phi, lam = stonyhurst_coords.lon.to(u.rad).value, stonyhurst_coords.lat.to(u.rad).value
        # pAng = -np.deg2rad(s_map.meta['CROTA2'])
        # b = np.deg2rad(s_map.meta['CRLT_OBS'])
        # a_matrix = image_to_spherical_matrix(phi=phi, lam=lam, b=b, pAng=pAng)
        # rtp_to_img_transform = np.linalg.inv(a_matrix)

        I_profile = np.stack([fits.getdata(I[j]) for j in range(self.num_wl)], -1)
        Q_profile = np.stack([fits.getdata(Q[j]) for j in range(self.num_wl)], -1)
        U_profile = np.stack([fits.getdata(U[j]) for j in range(self.num_wl)], -1)
        V_profile = np.stack([fits.getdata(V[j]) for j in range(self.num_wl)], -1)

        stokes = np.stack([I_profile, Q_profile, U_profile, V_profile], -2)

        # apply mask filter - coordinates + stokes for normalization
        radius_mask = radial_distance > 0.99
        cartesian_coords[radius_mask] = np.nan
        stokes[radius_mask] = np.nan

        return {'stokes': stokes,
                'coords': cartesian_coords,
                'cartesian_to_spherical_transform': cartesian_to_spherical_transform,
                'rtp_to_img_transform': rtp_to_img_transform,
                'mu': mu,
                'time': s_map.date.to_datetime(), 'obs_lat': s_map.carrington_latitude,
                'obs_lon': s_map.carrington_longitude,
                'carrington_coords': carrington_coords, 'wcs': s_map.wcs}
