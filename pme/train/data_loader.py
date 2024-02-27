import glob
import itertools
import os
from multiprocessing import Pool

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from scipy.signal import convolve2d
from sunpy.map import Map, all_coordinates_from_map
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


class TestDataModule(LightningDataModule):

    def __init__(self, file, batch_size=4096, num_workers=None, noise=None, psf=None, slit=False, **kwargs):
        super().__init__()

        # train parameters
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        lambdaStart = 6300.8 * 1e-10
        lambdaStep = 0.03 * 1e-10
        nLambda = 50
        lambdaEnd = (lambdaStart + lambdaStep * (-1 + nLambda))
        self.lambda_grid = np.linspace(-.5 * (lambdaEnd - lambdaStart), .5 * (lambdaEnd - lambdaStart), num=nLambda)

        stokes_vector = np.load(file)['Stokes_profiles']  # (4, 100, 400, 400, 50)
        # reshape to (400, 400, 100, 4, 50)
        stokes_vector = np.moveaxis(stokes_vector, [0, 1, 2, 3, 4], [3, 2, 0, 1, 4])

        print('LOADING STOKES VECTOR: ', stokes_vector.shape)
        coordinates = np.stack(np.meshgrid(np.linspace(-1, 1, stokes_vector.shape[0], dtype=np.float32),
                                           np.linspace(-1, 1, stokes_vector.shape[1], dtype=np.float32),
                                           np.linspace(0, 1, stokes_vector.shape[2], dtype=np.float32),
                                           indexing='ij'), -1)

        self.cube_shape = coordinates.shape[:3]
        self.data_range = [[-1, 1], [-1, 1], [0, 1]]

        # add noise
        if noise is not None:
            stokes_vector += np.random.normal(0, noise, stokes_vector.shape)

        # convolve with psf
        if psf is not None:
            psf = np.load(psf)['PSF']
            psf /= psf.sum()  # assure valid psf
            print('CONVOLVING WITH PSF: ', psf.shape)
            # stokes vector (x, y, lambda); psf (x, y)
            flat_stokes_vector = stokes_vector.reshape(*stokes_vector.shape[:2], -1)
            flat_stokes_vector = np.moveaxis(flat_stokes_vector, -1, 0)
            conv = ParallelConvolution(psf)
            with Pool(16) as p:
                convolved_maps = np.stack([r for r in tqdm(p.imap(conv.conv_f, flat_stokes_vector),
                                                           total=flat_stokes_vector.shape[0])], -1)
            stokes_vector = convolved_maps.reshape(stokes_vector.shape)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
            im = ax.imshow(psf, vmin=0)
            fig.colorbar(im)
            fig.tight_layout()
            wandb.log({'Ground-truth PSF': fig})
            plt.close('all')

        normalized_stokes_vector = np.copy(stokes_vector)
        normalized_stokes_vector[:, :, :, 1:] /= normalized_stokes_vector[:, :, :, 0:1]

        self.value_range = np.stack([normalized_stokes_vector.min((0, 1, 2, -1)),
                                     normalized_stokes_vector.max((0, 1, 2, -1))], -1)

        if slit:
            time_spacing = slit
            x_axis = stokes_vector.shape[0]
            slit_width = x_axis // time_spacing

            for t in range(0, stokes_vector.shape[2]):
                i = int(t % time_spacing)
                min_x = i * slit_width
                max_x = (i + 1) * slit_width
                stokes_vector[:min_x, :, t] = np.nan
                stokes_vector[max_x:, :, t] = np.nan

        ref_time = stokes_vector.shape[2] // 2
        # plot coordinates
        fig, axs = plt.subplots(1, 3, figsize=(16, 8), dpi=100)
        im = axs[0].imshow(coordinates[:, :, ref_time, 0].T, origin='lower')
        fig.colorbar(im)
        axs[0].set_title('x')
        im = axs[1].imshow(coordinates[:, :, ref_time, 1].T, origin='lower')
        fig.colorbar(im)
        axs[1].set_title('y')
        im = axs[2].imshow(coordinates[:, :, ref_time, 2].T, origin='lower')
        fig.colorbar(im)
        axs[2].set_title('t')
        fig.tight_layout()
        wandb.log({'Coordinates': fig})
        plt.close('all')

        # plot stokes vector
        stokes_min_max = np.abs(stokes_vector).max((0, 1, 2, -1))
        for l in range(nLambda):
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

        nan_mask = np.any(np.isnan(stokes_vector), (-2, -1))
        # flatten data
        coords = coordinates[~nan_mask].astype(np.float32)
        stokes_profile = stokes_vector[~nan_mask].astype(np.float32)

        coords = torch.tensor(coords, dtype=torch.float32)
        stokes_profile = torch.tensor(stokes_profile, dtype=torch.float32)

        valid_coords = coordinates[:, :, ref_time:ref_time + 1]
        valid_stokes_profile = stokes_vector[:, :, ref_time:ref_time + 1]

        valid_coords = torch.tensor(valid_coords, dtype=torch.float32).reshape(-1, 3)
        valid_stokes_profile = torch.tensor(valid_stokes_profile, dtype=torch.float32).reshape(-1, 4, nLambda)

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
        axs[0].set_title('x')
        im = axs[1].imshow(coordinates[:, :, ref_time, 1].T, origin='lower')
        fig.colorbar(im)
        axs[1].set_title('y')
        im = axs[2].imshow(coordinates[:, :, ref_time, 2].T, origin='lower')
        fig.colorbar(im)
        axs[2].set_title('t')
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
        return convolve2d(img, self.psf, mode='same', boundary='symm')
