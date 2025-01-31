import glob
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import torch
from astropy import units as u
from astropy.io import fits
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import all_coordinates_from_map, Map
from torch.utils.data import DataLoader
from tqdm import tqdm

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, image_to_spherical_matrix
from pme.train.data_loader import TensorsDataset


class SphericalDataModule(LightningDataModule):

    def __init__(self, train_config, valid_config, work_directory, seconds_per_dt=36000, Rs_per_ds=1,
                 ref_time=datetime(2010, 1, 1), batch_size=4096, num_workers=None):
        super().__init__()

        # train parameters
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.batch_size = batch_size * n_gpus
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        self.train_dataset = HMISphericalDataset(**train_config, seconds_per_dt=seconds_per_dt, Rs_per_ds=Rs_per_ds,
                                                 ref_time=ref_time, work_directory=work_directory)
        self.valid_dataset = HMISphericalDataset(**valid_config, seconds_per_dt=seconds_per_dt,
                                                 Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                 work_directory = work_directory,
                                                 filter_nans=False, shuffle=False)

        self.ref_time = ref_time
        self.times = self.train_dataset.times
        self.seconds_per_dt = seconds_per_dt
        self.Rs_per_ds = Rs_per_ds
        self.image_shape = self.valid_dataset.image_shape
        self.value_range = self.train_dataset.value_range
        self.data_range = self.train_dataset.data_range

        # centered at lambda0
        self.lambda_grid = self.train_dataset.lambda_grid
        self.lambda_config = self.train_dataset.lambda_config

    def train_dataloader(self):
        # self.train_dataset.shuffle()
        data_loader = DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True)
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(self.valid_dataset, batch_size=None, num_workers=self.num_workers,
                                 pin_memory=True, shuffle=False)
        return data_loader


class HMISphericalDataset(TensorsDataset):

    def __init__(self, data_path, seconds_per_dt, Rs_per_ds, ref_time, sampling_idx=None, **kwargs):
        lambda_shifts = np.array([-0.1695, -0.1017, -0.0339, +0.0339, +0.1017, +0.1695])  # From Phillip Scherrer
        lambda_center = 6173.3433  # From Phillip Scherrer

        self.lambda_grid = lambda_shifts
        self.lambda_config = {'lambda0': lambda_center * u.AA, 'lambda_grid': self.lambda_grid * u.AA,
                              'j_up': 1.0, 'j_low': 0.0, 'g_up': 2.50, 'g_low': 0.0}

        # load maps
        num_wl = 6

        I = np.stack([sorted(glob.glob(os.path.join(data_path, f'*I{int(i)}.fits')))
                      for i in range(num_wl)], -1)  # t, wl
        Q = np.stack([sorted(glob.glob(os.path.join(data_path, f'*Q{int(i)}.fits')))
                      for i in range(num_wl)], -1)  # t, wl
        U = np.stack([sorted(glob.glob(os.path.join(data_path, f'*U{int(i)}.fits')))
                      for i in range(num_wl)], -1)  # t, wl
        V = np.stack([sorted(glob.glob(os.path.join(data_path, f'*V{int(i)}.fits')))
                      for i in range(num_wl)], -1)  # t, wl

        files = np.stack([I, Q, U, V],  1)  # t, stokes, wl
        if sampling_idx is not None:
            files = files[sampling_idx: sampling_idx + 1]

        # load coordinates
        self.n_times = I.shape[0]
        self.num_wl = num_wl
        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.ref_time = ref_time

        with Pool(10) as p:
            data = [v for v in tqdm(p.imap(self._load_data, files), total=self.n_times, desc='Loading data')]

        stokes = np.stack([d['stokes'] for d in data], 0)  # t, x, y, stokes, wl
        coords = np.stack([d['coords'] for d in data], 0)  # t, x, y, 4
        cartesian_to_spherical_transform = np.stack([d['cartesian_to_spherical_transform'] for d in data], 0)  # t, x, y, 3, 3
        rtp_to_img_transform = np.stack([d['rtp_to_img_transform'] for d in data], 0)  # t, x, y, 3, 3
        mu = np.stack([d['mu'] for d in data], 0)  # t, x, y

        # normalize stokes vector
        max_I = np.nanmax(stokes[..., 0, :])
        stokes /= max_I


        self.times = [d['time'] for d in data]
        self.value_range = np.stack([np.nanmin(stokes, (0, 1, 2, -1)), np.nanmax(stokes, (0, 1, 2, -1))], -1)
        self.image_shape = stokes.shape[:3]  # t, x, y
        carrington_coords = np.stack([d['carrington_coords'] for d in data], 0)  # t, x, y, 3
        self.data_range = np.array([[np.nanmin(carrington_coords[..., i]), np.nanmax(carrington_coords[..., i])] for i in range(3)])

        print('------------------')
        print('Data range')
        print(self.data_range)

        print('------------------')
        print('Value range')
        print(self.value_range)

        tensors = {'stokes': stokes.reshape((-1, *stokes.shape[3:])),
                   'coords': coords.reshape((-1, *coords.shape[3:])),
                   'cartesian_to_spherical_transform': cartesian_to_spherical_transform.reshape(
                       (-1, *cartesian_to_spherical_transform.shape[3:])),
                   'rtp_to_img_transform': rtp_to_img_transform.reshape((-1, *rtp_to_img_transform.shape[3:])),
                   'mu': mu.reshape((-1, 1))}

        super().__init__(tensors=tensors, **kwargs)

    def _load_data(self, files):
        I, Q, U, V = files
        ref_file = I[0]

        s_map = Map(ref_file)

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
        carrington_coords = np.stack([r.to_value(u.solRad), np.pi / 2 - lat, lon], -1)
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

        I_profile = np.stack([fits.getdata(I[j]).data for j in range(self.num_wl)], -1)
        Q_profile = np.stack([fits.getdata(Q[j]).data for j in range(self.num_wl)], -1)
        U_profile = np.stack([fits.getdata(U[j]).data for j in range(self.num_wl)], -1)
        V_profile = np.stack([fits.getdata(V[j]).data for j in range(self.num_wl)], -1)

        stokes = np.stack([I_profile, Q_profile, U_profile, V_profile], -2)

        # apply mask filter - coordinates + stokes for normalization
        radius_mask = radial_distance > 0.98
        cartesian_coords[radius_mask] = np.nan
        stokes[radius_mask] = np.nan

        return {'stokes': stokes,
                'coords': cartesian_coords,
                'cartesian_to_spherical_transform': cartesian_to_spherical_transform,
                'rtp_to_img_transform': rtp_to_img_transform,
                'mu': mu,
                'time': s_map.date.to_datetime(),
                'carrington_coords': carrington_coords}
