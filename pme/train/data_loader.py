import glob
import itertools
import os
import uuid
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import torch
import wandb
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import getheader
from astropy.nddata import block_reduce
from dateutil.parser import parse
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from scipy.signal import fftconvolve
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm


class BatchDataset(Dataset):

    def __init__(self, *tensors, batch_size):
        super().__init__()
        self.tensors = tensors
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors), f'Invalid shapes: {[t.shape for t in tensors]}'

        self.batch_size = batch_size
        self.n_batches = np.ceil(tensors[0].shape[0] / batch_size).astype(np.int32)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        return [t[idx * self.batch_size: (idx + 1) * self.batch_size] for t in self.tensors]


def apply_along_space(f, np_array, axes, progress_bar=True):
    # apply the function f on each subspace given by iterating over the axes listed in axes, e.g. axes=(0,2)
    iter = itertools.product(*map(lambda ax: range(np_array.shape[ax]) if ax in axes else [slice(None, None, None)],
                                  range(len(np_array.shape))))
    iter = iter if not progress_bar else tqdm(iter, total=np.prod(np_array.shape))
    for slic in iter:
        np_array[slic] = f(np_array[slic])
    return np_array


class BatchesDataset(Dataset):

    def __init__(self, batches_file_paths, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = int(batch_size)

    def __len__(self):
        ref_file = list(self.batches_file_paths.values())[0]
        n_batches = np.ceil(np.load(ref_file, mmap_mode='r').shape[0] / self.batch_size)
        return n_batches.astype(np.int32)

    def __getitem__(self, idx):
        # lazy load data
        data = {k: np.copy(np.load(bf, mmap_mode='r')[idx * self.batch_size: (idx + 1) * self.batch_size])
                for k, bf in self.batches_file_paths.items()}
        return data

    def clear(self):
        [os.remove(f) for f in self.batches_file_paths.values()]

    def shuffle(self):
        r = np.random.permutation(list(self.batches_file_paths.values())[0].shape[0])
        for bf in self.batches_file_paths.values():
            data = np.load(bf, mmap_mode='r')
            data = data[r]
            np.save(bf, data)


class TensorsDataset(BatchesDataset):

    def __init__(self, tensors, work_directory, filter_nans=True, shuffle=True, ds_name=None, **kwargs):
        # filter nan entries
        nan_mask = np.any([np.any(np.isnan(t), axis=tuple(range(1, t.ndim))) for t in tensors.values()], axis=0)
        if nan_mask.sum() > 0 and filter_nans:
            print(f'Filtering {nan_mask.sum()} nan entries')
            tensors = {k: v[~nan_mask] for k, v in tensors.items()}

        # shuffle data
        if shuffle:
            r = np.random.permutation(list(tensors.values())[0].shape[0])
            tensors = {k: v[r] for k, v in tensors.items()}

        ds_name = uuid.uuid4() if ds_name is None else ds_name
        batches_paths = {}
        for k, v in tensors.items():
            coords_npy_path = os.path.join(work_directory, f'{ds_name}_{k}.npy')
            np.save(coords_npy_path, v.astype(np.float32))
            batches_paths[k] = coords_npy_path

        super().__init__(batches_paths, **kwargs)


class GenericDataModule(LightningDataModule):

    def __init__(self, stokes_vector, mu, times, lambda_config,
                 coordinates=None,
                 seconds_per_dt=3600, pixel_per_ds=1e2,
                 batch_size=4096, num_workers=None):
        super().__init__()
        assert stokes_vector.shape[0] == len(times), \
            f'Times need to match Stokes vector: {stokes_vector.shape[0]} != {len(times)}'

        # train parameters
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.batch_size = batch_size * n_gpus
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        #
        times = np.array(times)
        self.ref_time = np.min(times)
        self.times = times
        self.seconds_per_dt = seconds_per_dt
        self.pixel_per_ds = pixel_per_ds

        # centered at lambda0
        self.lambda_grid = lambda_config['lambda_grid']
        self.lambda_config = {'lambda0': lambda_config['0'],
                              'j_up': lambda_config['j_up'], 'j_low': lambda_config['j_low'],
                              'g_up': lambda_config['g_up'], 'g_low': lambda_config['g_low'],
                              'lambda_grid': self.lambda_grid}

        normalized_times = [(t - self.ref_time).total_seconds() / seconds_per_dt for t in times]
        normalized_times = np.array(normalized_times, dtype=np.float32)

        if coordinates is None:
            coordinates = np.stack(np.meshgrid(
                normalized_times,
                np.mgrid[:stokes_vector.shape[1]].astype(np.float32) - (stokes_vector.shape[1] - 1) / 2,
                np.mgrid[:stokes_vector.shape[2]].astype(np.float32) - (stokes_vector.shape[2] - 1) / 2,
                indexing='ij'), -1, dtype=np.float32)
        else:
            print('Using provided coordinates', coordinates.shape, stokes_vector.shape)
            time_coord = np.ones((*coordinates.shape[:-1], 1), dtype=np.float32)
            time_coord = np.einsum('...i,i->...i', time_coord, normalized_times)
            coordinates = np.concatenate([time_coord, coordinates], -1)

        coordinates[..., 1] /= self.pixel_per_ds  # x
        coordinates[..., 2] /= self.pixel_per_ds  # y

        self.cube_shape = coordinates.shape[:3]
        self.data_range = np.array(
            [[coordinates[..., 0].min(), coordinates[..., 0].max()],
             [coordinates[..., 1].min(), coordinates[..., 1].max()],
             [coordinates[..., 2].min(), coordinates[..., 2].max()]])

        self.value_range = np.stack([stokes_vector.min((0, 1, 2, -1)),
                                     stokes_vector.max((0, 1, 2, -1))], -1)
        print('VALUE RANGE')
        print(f'Stokes-I: {self.value_range[0, 0]:.3f} - {self.value_range[0, 1]:.3f}')
        print(f'Stokes-Q: {self.value_range[1, 0]:.3f} - {self.value_range[1, 1]:.3f}')
        print(f'Stokes-U: {self.value_range[2, 0]:.3f} - {self.value_range[2, 1]:.3f}')
        print(f'Stokes-V: {self.value_range[3, 0]:.3f} - {self.value_range[3, 1]:.3f}')

        print('Coordinate Range')
        print(f'Time: {self.data_range[0, 0]:.2f} - {self.data_range[0, 1]:.2f} ({coordinates.shape[0]})')
        print(f'X: {self.data_range[1, 0]:.2f} - {self.data_range[1, 1]:.2f} ({coordinates.shape[1]})')
        print(f'Y: {self.data_range[2, 0]:.2f} - {self.data_range[2, 1]:.2f} ({coordinates.shape[2]})')

        # plot coordinates
        ref_time = stokes_vector.shape[0] // 2
        fig, axs = plt.subplots(1, 3, figsize=(16, 8), dpi=100)
        im = axs[0].imshow(coordinates[ref_time, :, :, 0], origin='lower')
        fig.colorbar(im)
        axs[0].set_title('t')
        im = axs[1].imshow(coordinates[ref_time, :, :, 1], origin='lower')
        fig.colorbar(im)
        axs[1].set_title('x')
        im = axs[2].imshow(coordinates[ref_time, :, :, 2], origin='lower')
        fig.colorbar(im)
        axs[2].set_title('y')
        fig.tight_layout()
        wandb.log({'Coordinates': fig})
        plt.close('all')

        # plot stokes vector
        stokes_min_max = np.abs(stokes_vector).max((0, 1, 2, -1))
        for l in range(0, stokes_vector.shape[-1], stokes_vector.shape[-1] // 10):
            fig, axs = plt.subplots(1, 4, figsize=(9, 3), dpi=100)
            for i, label in enumerate(['I', 'Q', 'U', 'V']):
                im = axs[i].imshow(stokes_vector[ref_time, :, :, i, l], vmin=-stokes_min_max[i],
                                   vmax=stokes_min_max[i])
                axs[i].set_title(label)
                fig.colorbar(im, ax=axs[i])
            fig.suptitle(f'lambda: {self.lambda_grid[l]:.2f}')
            fig.tight_layout()
            wandb.log({'Stokes vector': fig})
            plt.close('all')

        # plot integrated stokes vector
        integrated_stokes_vector = np.abs(stokes_vector).sum(-1)
        fig, ax = plt.subplots(1, 4, figsize=(16, 8), dpi=100)
        for i, label in enumerate(['I', 'Q', 'U', 'V']):
            im = ax[i].imshow(integrated_stokes_vector[ref_time, :, :, i], origin='lower')
            ax[i].set_title(label)
            fig.colorbar(im, ax=ax[i])
        fig.tight_layout()
        wandb.log({'Integrated Stokes vector': fig})
        plt.close('all')

        # prepare for data loader
        nan_mask = np.any(np.isnan(stokes_vector), (-2, -1))
        # flatten data
        train_coords = coordinates[~nan_mask].astype(np.float32)
        train_stokes = stokes_vector[~nan_mask].astype(np.float32)
        train_mu = mu[~nan_mask].astype(np.float32)

        train_coords = torch.tensor(train_coords, dtype=torch.float32)
        train_stokes = torch.tensor(train_stokes, dtype=torch.float32)
        train_mu = torch.tensor(train_mu, dtype=torch.float32)

        valid_coords = coordinates[ref_time:ref_time + 1]
        valid_stokes = stokes_vector[ref_time:ref_time + 1]
        valid_mu = mu[ref_time:ref_time + 1]

        valid_coords = torch.tensor(valid_coords, dtype=torch.float32).reshape(-1, 3)
        valid_stokes = torch.tensor(valid_stokes, dtype=torch.float32).reshape(-1, 4, valid_stokes.shape[-1])
        valid_mu = torch.tensor(valid_mu, dtype=torch.float32).reshape(-1, 1)

        self.valid_dataset = BatchDataset(valid_coords, valid_mu, valid_stokes, batch_size=batch_size)

        self.coords = train_coords
        self.stokes_profile = train_stokes
        self.mu = train_mu

    def train_dataloader(self):
        # shuffle data
        r = np.random.permutation(self.coords.shape[0])
        coords = self.coords[r]
        stokes_profile = self.stokes_profile[r]
        mu = self.mu[r]

        train_dataset = BatchDataset(coords, mu, stokes_profile, batch_size=self.batch_size)

        data_loader = DataLoader(train_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 sampler=RandomSampler(train_dataset, replacement=True, num_samples=int(1e3)))
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(self.valid_dataset, batch_size=None, num_workers=self.num_workers,
                                 pin_memory=True, shuffle=False)
        return data_loader


class TestDataModule(GenericDataModule):

    def __init__(self, files, psf=None, noise=None, **kwargs):
        lambda_center = 6302.4931
        lambda_step = 0.021743135134784097
        n_lambda = 56

        # lambda_grid = np.array([lambda_start + i * lambda_step for i in range(n_lambda)])
        lambda_range = (n_lambda - 1) * lambda_step
        lambda_grid = np.linspace(-0.5 * lambda_range, 0.5 * lambda_range, n_lambda)

        lambda_config = {'0': lambda_center * u.AA, 'lambda_grid': lambda_grid * u.AA,
                         'j_up': 1.0, 'j_low': 0.0, 'g_up': 2.49, 'g_low': 0}
        files = [files] if not isinstance(files, list) else files
        files = [sorted(glob.glob(f)) for f in files]  # load wildcards
        files = [f for fl in files for f in fl]  # flatten list

        stokes_vector = []
        for f in files:
            stokes_vector.append(np.load(f)['stokes_profiles'])
        stokes_vector = np.stack(stokes_vector, 0)
        stokes_vector = add_synthetic_noise(stokes_vector, noise=noise, psf_file=psf)

        ref_time = datetime(2024, 1, 1)
        times = [ref_time + timedelta(minutes=1 * i) for i in range(stokes_vector.shape[0])]

        mu = np.ones((*stokes_vector.shape[:3], 1), dtype=np.float32)

        super().__init__(stokes_vector, mu, times, lambda_config, **kwargs)


class HinodeDataModule(GenericDataModule):

    def __init__(self, files, lambda_config=None, **kwargs):
        files = [files] if not isinstance(files, list) else files
        files = [sorted(glob.glob(f)) for f in files]  # load wildcards

        stokes_vector = []
        times = []
        mu = []
        for fl in files:
            stokes_f, time_f, mu_f = self._load_fits(fl)
            stokes_vector.append(stokes_f)
            times.append(time_f)
            mu.append(mu_f)

        stokes_vector = np.stack(stokes_vector, 0)
        mu = np.stack(mu, 0)[..., None]

        if lambda_config is None:
            header = getheader(files[0][0])
            lambda_step = -header["CDELT1"]
            lambda_center = header["CRVAL1"]
            pixel_center = header["CRPIX1"]
            offset = 6302.4931 - lambda_center
            n_lambda = stokes_vector.shape[-1]

            pixel_range = np.arange(n_lambda)
            pixel_range = pixel_range - pixel_center
            lambda_range = pixel_range * lambda_step
            lambda_grid = lambda_range - offset

            stokes_vector = stokes_vector[..., -56:]
            lambda_grid = lambda_grid[-56:]

            max_I = stokes_vector[..., 0, :].max()
            stokes_vector /= max_I * 1.1

            lambda_config = {'0': lambda_center * u.AA, 'lambda_grid': lambda_grid * u.AA,
                             'j_up': 1.0, 'j_low': 0.0, 'g_up': 2.49, 'g_low': 0}

        super().__init__(stokes_vector, mu, times, lambda_config, pixel_per_ds=512, **kwargs)

    def _load_fits(self, files):
        stokes_vector = np.stack([fits.getdata(f) for f in files], 0, dtype=np.float32)

        # x, Stokes, y, wl --> y, x, Stokes, wl
        stokes_vector = np.moveaxis(stokes_vector, (0, 1, 2, 3), (1, 2, 0, 3))

        center_idx = len(files) // 2
        ref_file = files[center_idx]
        ref_header = getheader(ref_file)

        # construct reference coordinates
        time = parse(ref_header['DATE_OBS'])
        scale = (ref_header['XSCALE'], ref_header['YSCALE']) * u.arcsec / u.pixel
        center_coord = SkyCoord(Tx=ref_header['XCEN'] * u.arcsec, Ty=ref_header['YCEN'] * u.arcsec,
                                obstime=time, frame=frames.Helioprojective)
        center_pix = (center_idx, ref_header['CRPIX2']) * u.pix

        # create reference map
        ref_data = stokes_vector[:, :, 0, 0]
        map_header = make_fitswcs_header(ref_data, center_coord,
                                         reference_pixel=center_pix, scale=scale,
                                         rotation_angle=ref_header['CROTA2'] * u.rad)
        ref_map = Map(ref_data, map_header)

        # compute mu
        coords = all_coordinates_from_map(ref_map)
        radial_distance = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / ref_map.rsun_obs
        mu = np.cos(radial_distance.to_value(u.dimensionless_unscaled) * np.pi / 2)
        mu = mu.astype(np.float32)

        # NAXIS1 = 112
        # NAXIS2 = 512
        # CDELT1 = 0.0215490000000
        # CDELT2 = 0.317000000000
        # XSCALE = 0.297140002251
        # YSCALE = 0.319979995489
        # data shape = (4, 512, 112) = Stokes, y, wl

        # plot mu
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
        im = ax.imshow(mu, origin='lower')
        fig.colorbar(im)
        ax.set_title('mu')
        fig.tight_layout()
        wandb.log({'mu': fig})

        return stokes_vector, time, mu


class FitsDataModule(GenericDataModule):

    def __init__(self, file, atomic_parameters=None, **kwargs):
        atomic_parameters = {'j_up': 1.0, 'j_low': 0.0, 'g_up': 2.49, 'g_low': 0} if atomic_parameters is None \
            else atomic_parameters

        stokes_vector = fits.getdata(file).astype(np.float32)
        stokes_vector = stokes_vector[None]  # add time dimension (t, x, y, stokes, lambda)

        header = getheader(file)
        times = [parse(header['DATE_OBS'])]
        lambda_step = header["CDELT1"]
        lambda_center = header["CRVAL1"]
        pixel_center = header["CRPIX1"]
        n_lambda = stokes_vector.shape[-1]

        pixel_range = np.arange(n_lambda)
        pixel_range = pixel_range - pixel_center
        lambda_range = pixel_range * lambda_step

        lambda_config = {'0': lambda_center * u.AA, 'lambda_grid': lambda_range * u.AA, **atomic_parameters}

        mu = np.ones((*stokes_vector.shape[:3], 1), dtype=np.float32)

        super().__init__(stokes_vector, mu, times, lambda_config, pixel_per_ds=512, **kwargs)


def add_synthetic_noise(stokes_vector, noise=None, psf_file=None, bin=1):
    # convolve with psf
    if bin > 1:
        block_size = (1, bin, bin, 1, 1) if stokes_vector.ndim == 5 else (bin, bin, 1, 1)
        stokes_vector = block_reduce(stokes_vector, block_size, func=np.mean)
    if psf_file is not None:
        if psf_file.endswith('.npy'):
            psf = np.load(psf_file)
        elif psf_file.endswith('.npz'):
            psf = np.load(psf_file)['PSF']
        elif psf_file.endswith('.fits'):
            psf = fits.getdata(psf_file).astype(np.float32)
        else:
            raise ValueError('Invalid psf file format')
        print('CONVOLVING WITH PSF: ', psf.shape)
        # stokes vector (x, y, lambda); psf (x, y)
        flat_stokes_vector = stokes_vector.reshape(*stokes_vector.shape[:2], -1)
        flat_stokes_vector = np.moveaxis(flat_stokes_vector, -1, 0)
        conv = ParallelConvolution(psf)
        with Pool(32) as p:
            convolved_maps = np.stack([r for r in tqdm(p.imap(conv.conv_f, flat_stokes_vector),
                                                       total=flat_stokes_vector.shape[0])], -1)
        stokes_vector = convolved_maps.reshape(stokes_vector.shape)
    # add noise
    if noise is not None:
        np.random.seed(42)  # assure reproducibility
        stokes_vector += np.random.normal(0, noise, stokes_vector.shape)
    return stokes_vector


class SHARPDataModule(LightningDataModule):

    def __init__(self, files, batch_size=4096, num_workers=None, **kwargs):
        super().__init__()

        # train parameters
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        I_maps = [Map(f) for f in sorted(glob.glob(files['I']))]
        Q_maps = [Map(f) for f in sorted(glob.glob(files['Q']))]
        U_maps = [Map(f) for f in sorted(glob.glob(files['U']))]
        V_maps = [Map(f) for f in sorted(glob.glob(files['V']))]

        self.lambda_grid = np.linspace(6173.3 - 0.68, 6173.3 + 0.68, 6) * 1e-10

        I_vector = np.stack([m.data for m in I_maps], -1)
        Q_vector = np.stack([m.data for m in Q_maps], -1)
        U_vector = np.stack([m.data for m in U_maps], -1)
        V_vector = np.stack([m.data for m in V_maps], -1)
        # w, h, t, 4, lambda
        stokes_vector = np.stack([I_vector, Q_vector, U_vector, V_vector], -2)
        stokes_vector = stokes_vector[:, :, None]  # add time dimension

        print('LOADING STOKES VECTOR: ', stokes_vector.shape)
        coordinates = all_coordinates_from_map(I_maps[0])
        # coordinates = coordinates.transform_to(frames.HeliographicCarrington)

        times = np.zeros_like(coordinates.Tx.value)
        coordinates = np.stack([coordinates.Tx.value / 1e3, coordinates.Ty.value / 1e3, times], -1)
        coordinates = coordinates[:, :, None]  # add time dimension

        self.cube_shape = coordinates.shape[:3]
        self.data_range = [[coordinates[..., 0].min(), coordinates[..., 0].max()],
                           [coordinates[..., 1].min(), coordinates[..., 1].max()],
                           [coordinates[..., 2].min(), coordinates[..., 2].max()]]

        normalized_stokes_vector = np.copy(stokes_vector)
        normalized_stokes_vector[:, :, :, 1:] /= normalized_stokes_vector[:, :, :, 0:1]

        self.value_range = np.stack([normalized_stokes_vector.min((0, 1, 2, -1)),
                                     normalized_stokes_vector.max((0, 1, 2, -1))], -1)

        ref_time = stokes_vector.shape[2] // 2
        # plot coordinates
        fig, axs = plt.subplots(1, 3, figsize=(16, 8), dpi=100)
        im = axs[0].imshow(coordinates[:, :, ref_time, 0].T, origin='lower')
        fig.colorbar(im)
        axs[0].set_title('t')
        im = axs[1].imshow(coordinates[:, :, ref_time, 1].T, origin='lower')
        fig.colorbar(im)
        axs[1].set_title('x')
        im = axs[2].imshow(coordinates[:, :, ref_time, 2].T, origin='lower')
        fig.colorbar(im)
        axs[2].set_title('y')
        fig.tight_layout()
        wandb.log({'Coordinates': fig})
        plt.close('all')

        # plot stokes vector
        stokes_min_max = np.abs(stokes_vector).max((0, 1, 2, -1))
        for l in range(len(self.lambda_grid)):
            fig, axs = plt.subplots(1, 4, figsize=(9, 3), dpi=100)
            for i, label in enumerate(['I', 'Q', 'U', 'V']):
                im = axs[i].imshow(stokes_vector[:, :, ref_time, i, l], vmin=-stokes_min_max[i], vmax=stokes_min_max[i])
                axs[i].set_title(label)
                fig.colorbar(im, ax=axs[i])
            fig.suptitle(f'lambda: {self.lambda_grid[l]:.2f}')
            fig.tight_layout()
            wandb.log({'Stokes vector': fig})
            plt.close('all')

        # plot integrated stokes vector
        integerated_stokes_vector = np.abs(stokes_vector).sum(-1)
        fig, ax = plt.subplots(1, 4, figsize=(16, 8), dpi=100)
        for i, label in enumerate(['I', 'Q', 'U', 'V']):
            im = ax[i].imshow(integerated_stokes_vector[:, :, ref_time, i])
            ax[i].set_title(label)
            fig.colorbar(im, ax=ax[i])
        fig.tight_layout()
        wandb.log({'Integrated Stokes vector': fig})
        plt.close('all')

        nan_mask = np.any(np.isnan(stokes_vector), (-2, -1)) | np.any(np.isnan(coordinates), -1)
        # flatten data
        coords = coordinates[~nan_mask].astype(np.float32)
        stokes_profile = stokes_vector[~nan_mask].astype(np.float32)

        coords = torch.tensor(coords, dtype=torch.float32)
        stokes_profile = torch.tensor(stokes_profile, dtype=torch.float32)

        valid_coords = coordinates[:, :, ref_time:ref_time + 1]
        valid_stokes_profile = stokes_vector[:, :, ref_time:ref_time + 1]

        valid_coords = torch.tensor(valid_coords, dtype=torch.float32).reshape(-1, 3)
        valid_stokes_profile = torch.tensor(valid_stokes_profile, dtype=torch.float32).reshape(-1, 4,
                                                                                               stokes_vector.shape[-1])

        self.valid_dataset = BatchDataset(valid_coords, valid_stokes_profile, batch_size=batch_size)

        self.coords = coords
        self.stokes_profile = stokes_profile

    def train_dataloader(self):
        # shuffle data
        r = np.random.permutation(self.coords.shape[0])
        coords = self.coords[r]
        stokes_profile = self.stokes_profile[r]

        train_dataset = BatchDataset(coords, stokes_profile, batch_size=self.batch_size)

        data_loader = DataLoader(train_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 sampler=RandomSampler(train_dataset, replacement=True, num_samples=int(1e3)))
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(self.valid_dataset, batch_size=None, num_workers=self.num_workers,
                                 pin_memory=True, shuffle=False)
        return data_loader


class BatchesDataset(Dataset):

    def __init__(self, batches_file_paths, batch_size):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(
            np.load(list(self.batches_file_paths.values())[0], mmap_mode='r').shape[0] / self.batch_size).astype(
            np.int32)

    def __getitem__(self, idx):
        # lazy load data
        data = {k: np.copy(np.load(bf, mmap_mode='r')[idx * self.batch_size: (idx + 1) * self.batch_size])
                for k, bf in self.batches_file_paths.items()}
        return data


class ParallelConvolution:

    def __init__(self, psf):
        self.psf = psf

    def conv_f(self, img):
        return fftconvolve(img, self.psf, mode='same')


def load_Hinode_files(data_dir):
    ''' Load in Hinode Level 1 data files and return a
        4D array with the Stokes parameters and properties
        of the spectral scans. Point to the original 
        directory where the files were extracted. 

        Input:
            -- data_dir, str
                Directory with level1 data product from 
                the SOT/SP CSAC data. 

        Output:
            -- sp_raster, np.array [nx, ny, nLambda, 4]
                Stokes array
            -- dwavelength, float
                Sampling wavelength grid resolution, Angstroms
            -- wavelength_center, float
                Central wavelength of wavelength range, Angstroms. 
    '''

    file_list = []

    for root, dirs, files in os.walk(data_dir):
        for filename in sorted(files):
            file_list = np.append(file_list, filename)

    file_list = file_list[1:]  # omit the .DC_Store for a Mac
    num_files = len(file_list)

    with fits.open(data_dir + file_list[0]) as a:
        hdr = a[0].header
        # get everything we need from the header of the data partition
        dwave = hdr["CDELT1"]
        wave_cent = hdr["CRVAL1"]

        N_Stokes = a[0].data.shape[0]
        N_y = a[0].data.shape[1]
        N_lambda = a[0].data.shape[2]
        spec_range = dwave * N_lambda / 2
        wavescale = wave_cent + np.linspace(-1 * spec_range, spec_range, num=N_lambda)

    wavelength_central = wave_cent
    dwavelength = wavescale[1] - wavescale[0]

    sp_raster = np.zeros((num_files, N_y, N_lambda, N_Stokes))

    for el in range(0, int(num_files)):
        # print(el)
        with fits.open(data_dir + file_list[el]) as a:
            sp_raster[el, :, :, :] = np.moveaxis(a[0].data, 0, -1)

    return sp_raster, dwavelength, wavelength_central
