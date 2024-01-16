import glob
import os

import numpy as np
from astropy import units as u
from lightning import LightningDataModule
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, Dataset


class HMIDataModule(LightningDataModule):

    def __init__(self, files, work_directory,
                 spatial_norm=1e3 * u.arcsec,
                 batch_size=4096, num_workers=None, **kwargs):
        super().__init__()

        # train parameters
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)

        if isinstance(files, str):
            I_files = sorted(glob.glob(os.path.join(files, '*I?.fits')))
            Q_files = sorted(glob.glob(os.path.join(files, '*Q?.fits')))
            U_files = sorted(glob.glob(os.path.join(files, '*U?.fits')))
            V_files = sorted(glob.glob(os.path.join(files, '*V?.fits')))
            files = {'I': I_files, 'Q': Q_files, 'U': U_files, 'V': V_files}

        def _subframe(m):
            return m.submap(u.Quantity([[-300, 300], [-300, 300]] * u.arcsec))

        maps = {k: [_subframe(Map(file)) for file in v] for k, v in files.items()}

        ref_map = maps['I'][0]
        coordinates = all_coordinates_from_map(ref_map)
        coordinates = np.stack([coordinates.Tx, coordinates.Ty], -1)

        i_data = np.stack([m.data for m in maps['I']], -1)
        q_data = np.stack([m.data for m in maps['Q']], -1)
        u_data = np.stack([m.data for m in maps['U']], -1)
        v_data = np.stack([m.data for m in maps['V']], -1)

        stokes_vector = np.stack([i_data, q_data, u_data, v_data], -2)

        # flatten data
        coords = coordinates.reshape((-1, 3)).astype(np.float32)
        values = stokes_vector.reshape((-1, 3,)).astype(np.float32)

        # filter nan entries
        nan_mask = np.any(np.isnan(values), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            values = values[~nan_mask]

        # normalize data
        values = values
        coords = (coords / spatial_norm).value

        cube_shape = [[coords[:, i].min(), coords[:, i].max()] for i in range(2)]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values)
        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path, }

        # create data loaders
        self.dataset = BatchesDataset(batches_path, batch_size)
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True)
        return data_loader

    def val_dataloader(self):
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        return boundary_loader


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
