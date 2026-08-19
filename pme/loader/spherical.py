import copy
import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
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

from pme.data.phi_util import fits_get_sampling, load_fix_phi_header
from pme.data.hmi_transmission import (
    load_hmi_transmission_profile,
    read_hmi_fits_acquisition,
    resolve_hmi_transmission_profile,
)
from pme.data.hmi_time import load_hmi_date_obs_map
from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, image_to_spherical_matrix
from pme.instrument import hmi_wavelength_config
from pme.loader.util import MultiprocessingWrapper
from pme.train.data_loader import TensorsDataset, CombinedDataset, TimeStratifiedBatchSampler


def _summarize_physics_coordinates(spherical_coords):
    """Return compact, seam-safe bounds for continuous collocation sampling."""
    flat_coords = np.asarray(spherical_coords).reshape(-1, 3)
    flat_coords = flat_coords[np.isfinite(flat_coords).all(axis=-1)]
    if flat_coords.size == 0:
        raise ValueError('Cannot build a physics domain from an empty spherical footprint.')

    latitude = np.pi / 2 - flat_coords[:, 1]
    longitude = flat_coords[:, 2]
    sin_sum = np.sin(longitude).sum(dtype=np.float64)
    cos_sum = np.cos(longitude).sum(dtype=np.float64)
    longitude_center = float(np.arctan2(sin_sum, cos_sum))
    longitude_offset = np.arctan2(
        np.sin(longitude - longitude_center),
        np.cos(longitude - longitude_center),
    )

    return {
        'radius_Rs': (float(flat_coords[:, 0].min()), float(flat_coords[:, 0].max())),
        'latitude_rad': (float(latitude.min()), float(latitude.max())),
        'longitude_center_rad': longitude_center,
        'longitude_offset_rad': (
            float(longitude_offset.min()), float(longitude_offset.max()),
        ),
        'longitude_vector_sum': (float(sin_sum), float(cos_sum)),
        'n_points': int(flat_coords.shape[0]),
    }


def spherical_data_cache_signature(data_config):
    """Fingerprint the data configuration and every currently matched input file."""
    digest = hashlib.sha256()
    digest.update(json.dumps(data_config, sort_keys=True, default=str).encode())

    train_configs = data_config.get('train_configs', [])
    train_configs = train_configs if isinstance(train_configs, list) else [train_configs]
    dataset_configs = [*train_configs, data_config.get('valid_config', {})]

    referenced_paths = set()
    for dataset_config in dataset_configs:
        if not isinstance(dataset_config, dict):
            continue
        data_paths = dataset_config.get('data_path', [])
        data_paths = [data_paths] if isinstance(data_paths, str) else data_paths
        for value in data_paths:
            value = os.fspath(value)
            if os.path.isdir(value):
                referenced_paths.update(glob.glob(os.path.join(value, '*.fits')))
            else:
                referenced_paths.update(glob.glob(value))

        transmission_directory = dataset_config.get('transmission_profile_directory')
        if transmission_directory:
            manifest_path = os.path.join(os.fspath(transmission_directory), 'manifest.json')
            referenced_paths.add(manifest_path)
            if os.path.isfile(manifest_path):
                with open(manifest_path) as manifest_file:
                    manifest = json.load(manifest_file)
                referenced_paths.update(
                    os.path.join(os.fspath(transmission_directory), profile['file'])
                    for profile in manifest.get('profiles', {}).values()
                    if isinstance(profile, dict) and 'file' in profile
                )

    for path in sorted(referenced_paths):
        digest.update(os.path.abspath(path).encode())
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            digest.update(b'\0missing')
        else:
            digest.update(f'\0{stat.st_size}\0{stat.st_mtime_ns}'.encode())
    return digest.hexdigest()


class SphericalDataModule(LightningDataModule):

    CACHE_VERSION = 15

    def __init__(self, train_configs, valid_config, work_directory,
                 seconds_per_dt=24 * 60 * 60, Rs_per_ds=1, gauss_per_dB=1e3,
                 stokes_normalization=83696.0,
                 ref_time=datetime(2010, 5, 1, 18, 58),
                 batch_size=65536, dataset_batch_size=4096,
                 num_workers=None):
        super().__init__()
        self.cache_version = self.CACHE_VERSION
        self.spectral_response_files = {}
        self.spectral_response_by_acquisition = {}

        # train parameters
        self.batch_size = batch_size
        self.dataset_batch_size = dataset_batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self._validate_batch_sizes()

        ref_time = parse(ref_time) if isinstance(ref_time, str) else ref_time
        train_configs = train_configs if isinstance(train_configs, list) else [train_configs]
        train_datasets = []
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

            if 'time_range' in train_config:
                train_files = self._filter_files_by_time_range(
                    train_files, train_config.pop('time_range'), ds_type
                )
            if 'sample_idx' in train_config:  # use a single sample for debugging
                sample_idx = train_config.pop('sample_idx')
                train_files = train_files[sample_idx:sample_idx + 1]
            if 'n_samples' in train_config:  # apply subsampling for debugging
                n_samples = train_config.pop('n_samples')
                train_files = self._select_evenly_spaced_samples(train_files, n_samples)
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
                                                         **train_config)
                max_samples = 10
                sample_step = max(1, len(train_files) // max_samples)
                args = []
                for sample_index, (tf, ds_id) in tqdm(enumerate(zip(train_files, ds_ids))):
                    item = {'data': tf, 'ds_id': ds_id,
                            'store_plot_image': (sample_index % sample_step) == 0}
                    args.append(item)
                desc = f'Loading training datasets ({i + 1:02d}/{len(train_configs):02d})'
                tds = [r for r in tqdm(p.imap(process_wrapper.run, args), total=len(args), desc=desc)]
            train_datasets += tds  # append all datasets
            for dataset in tds:
                self._register_hmi_dataset_response(dataset)

        self.train_datasets = train_datasets

        # add lambda configuration for each instrument
        # only use the first dataset for each instrument to define the lambda configuration
        # assumes that wavelength_center is the same for all datasets of the same instrument
        wavelength_config = {}
        stokes_normalization_by_instrument = {}
        for ds in train_datasets:
            instrument_id = ds.instrument_id
            if instrument_id not in wavelength_config:
                wavelength_config[instrument_id] = ds.wavelength_config
            previous_normalization = stokes_normalization_by_instrument.setdefault(
                instrument_id, ds.stokes_normalization
            )
            if previous_normalization != ds.stokes_normalization:
                raise ValueError(
                    f'All datasets for {instrument_id!r} must use the same stokes_normalization; '
                    f'found {previous_normalization} and {ds.stokes_normalization}.'
                )
        self.wavelength_config = wavelength_config
        self.stokes_normalization_by_instrument = stokes_normalization_by_instrument

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
                                                     batch_size=self.batch_size, work_directory=work_directory,
                                                     filter_nans=False, shuffle=False, resolution=resolution,
                                                     transmission_profile_directory=valid_config[
                                                         'transmission_profile_directory'
                                                     ])
            self._register_hmi_dataset_response(self.valid_dataset)
        elif valid_ds_type == 'phi-hrt':
            valid_files = sorted(glob.glob(valid_config['data_path']))
            sample_idx = valid_config.get('sample_idx', len(valid_files) // 2)
            self.valid_dataset = PHIHRTSphericalDataset(valid_files[sample_idx],
                                                        ds_id='valid', instrument_id=valid_config['instrument_id'],
                                                        seconds_per_dt=seconds_per_dt,
                                                        Rs_per_ds=Rs_per_ds, ref_time=ref_time,
                                                        stokes_normalization=ds_normalization,
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
                                                      batch_size=self.batch_size, work_directory=work_directory,
                                                      filter_nans=False, shuffle=False)
        else:
            raise ValueError(f'Unknown validation dataset type: {valid_ds_type}')

        valid_instrument_id = self.valid_dataset.instrument_id
        expected_normalization = self.stokes_normalization_by_instrument.get(valid_instrument_id)
        if expected_normalization is None:
            raise ValueError(
                f'Validation instrument {valid_instrument_id!r} is absent from the training datasets.'
            )
        if expected_normalization != self.valid_dataset.stokes_normalization:
            raise ValueError(
                f'Validation and training data for {valid_instrument_id!r} must use the same '
                f'stokes_normalization; found {self.valid_dataset.stokes_normalization} and '
                f'{expected_normalization}.'
            )

        self.ref_time = ref_time
        self.times = [d.time for d in self.train_datasets]
        self.seconds_per_dt = seconds_per_dt
        self.Rs_per_ds = Rs_per_ds
        self.gauss_per_dB = gauss_per_dB
        self.image_shape = self.valid_dataset.image_shape
        self.value_range = self.valid_dataset.value_range
        self.data_range = self.valid_dataset.data_range

    def _validate_batch_sizes(self):
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError('batch_size must be a positive integer.')
        if not isinstance(self.dataset_batch_size, int) or self.dataset_batch_size <= 0:
            raise ValueError('dataset_batch_size must be a positive integer.')
        if self.batch_size < self.dataset_batch_size:
            raise ValueError('batch_size must be greater than or equal to dataset_batch_size.')
        if self.batch_size % self.dataset_batch_size:
            raise ValueError(
                f'batch_size ({self.batch_size}) must be divisible by dataset_batch_size '
                f'({self.dataset_batch_size}).'
            )

    @staticmethod
    def _select_evenly_spaced_samples(files, n_samples):
        if not isinstance(n_samples, int) or isinstance(n_samples, bool):
            raise TypeError('n_samples must be an integer.')
        if n_samples <= 0:
            raise ValueError('n_samples must be positive.')
        if n_samples > len(files):
            raise ValueError(
                f'n_samples ({n_samples}) exceeds the number of available acquisitions ({len(files)}).'
            )
        if n_samples == len(files):
            return files
        indices = np.rint(np.linspace(0, len(files) - 1, n_samples)).astype(int)
        if len(np.unique(indices)) != n_samples:
            raise RuntimeError(f'Unable to select {n_samples} unique samples from {len(files)} files.')
        return files[indices] if isinstance(files, np.ndarray) else [files[index] for index in indices]

    def _register_hmi_dataset_response(self, dataset):
        if not isinstance(dataset, HMISphericalDataset):
            return
        instrument_id = dataset.instrument_id
        response_file = dataset.spectral_response_file
        registered_files = self.spectral_response_files.setdefault(instrument_id, [])
        response_shapes = getattr(self, '_spectral_response_shapes', {})
        self._spectral_response_shapes = response_shapes
        if response_file not in registered_files:
            response_shape = load_hmi_transmission_profile(response_file)['offsets'].shape
            expected_shape = response_shapes.setdefault(instrument_id, response_shape)
            if response_shape != expected_shape:
                raise ValueError(
                    f'HMI response files for {instrument_id!r} use incompatible quadrature shapes: '
                    f'{expected_shape} and {response_shape}.'
                )
            registered_files.append(response_file)

        registered_acquisitions = self.spectral_response_by_acquisition.setdefault(instrument_id, {})
        existing = registered_acquisitions.get(dataset.acquisition_key)
        if existing is not None and existing != response_file:
            raise ValueError(
                f'Conflicting HMI responses for {instrument_id!r}, acquisition '
                f'{dataset.acquisition_key!r}: {existing!r} and {response_file!r}.'
            )
        registered_acquisitions[dataset.acquisition_key] = response_file

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
        if isinstance(data_path, str):
            if os.path.isdir(data_path):
                candidate_files = sorted(glob.glob(os.path.join(data_path, '*.fits')))
            else:
                candidate_files = sorted(glob.glob(data_path))
        elif isinstance(data_path, Iterable):
            candidate_files = []
            for path in data_path:
                if os.path.isdir(path):
                    candidate_files.extend(sorted(glob.glob(os.path.join(path, '*.fits'))))
                else:
                    candidate_files.extend(sorted(glob.glob(path)))
        else:
            raise ValueError(f'Unknown data path type: {type(data_path)}. Expected str or Iterable[str].')

        if len(candidate_files) == 0:
            raise ValueError(f'No FITS files matched data_path={data_path!r}.')
        candidate_files = sorted(set(candidate_files))

        # Group by the complete acquisition prefix, rather than sorting each
        # Stokes/wavelength stream independently. This makes a missing segment
        # an explicit error instead of silently pairing different timestamps.
        segment_pattern = re.compile(r'^(?P<prefix>.+)\.(?P<stokes>[IQUV])(?P<wl>\d+)\.fits$')
        acquisitions = {}
        for file in candidate_files:
            match = segment_pattern.match(file)
            if match is None:
                continue
            wavelength_index = int(match.group('wl'))
            if wavelength_index >= num_wl:
                continue
            key = match.group('prefix')
            segment = (match.group('stokes'), wavelength_index)
            if segment in acquisitions.setdefault(key, {}):
                raise ValueError(f'Duplicate segment {segment} for acquisition {key!r}.')
            acquisitions[key][segment] = file

        if not acquisitions:
            raise ValueError(
                f'No I/Q/U/V FITS segments matched data_path={data_path!r}. '
                f'Expected filenames ending in .I0.fits ... .V{num_wl - 1}.fits.'
            )

        expected_segments = {(stokes_id, wavelength_index)
                             for stokes_id in 'IQUV' for wavelength_index in range(num_wl)}
        grouped_files = []
        for acquisition, segments in sorted(acquisitions.items()):
            missing = sorted(expected_segments - segments.keys())
            unexpected = sorted(segments.keys() - expected_segments)
            if missing or unexpected:
                raise ValueError(
                    f'Incomplete I/Q/U/V file set for acquisition {acquisition!r}: '
                    f'missing={missing}, unexpected={unexpected}.'
                )
            grouped_files.append([
                [segments[(stokes_id, wavelength_index)] for wavelength_index in range(num_wl)]
                for stokes_id in 'IQUV'
            ])

        return np.asarray(grouped_files)  # time, stokes, wavelength

    @staticmethod
    def _parse_observation_time(value):
        if isinstance(value, datetime):
            observation_time = value
        elif isinstance(value, str) and value.endswith('_TAI'):
            formats = (
                '%Y.%m.%d_%H:%M:%S.%f_TAI',
                '%Y.%m.%d_%H:%M:%S_TAI',
                '%Y%m%d_%H%M%S.%f_TAI',
                '%Y%m%d_%H%M%S_TAI',
            )
            for time_format in formats:
                try:
                    observation_time = datetime.strptime(value, time_format)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f'Invalid TAI observation time {value!r}; expected '
                    'YYYY.MM.DD_HH:MM:SS[.ffffff]_TAI or YYYYMMDD_HHMMSS[.ffffff]_TAI.'
                )
        else:
            observation_time = parse(value) if isinstance(value, str) else value

        if not isinstance(observation_time, datetime):
            raise TypeError(f'Observation time must be a string or datetime, got {type(value).__name__}.')
        # HMI TAI labels are timezone-free. Normalize any explicitly zoned ISO
        # input to UTC before making the comparison representation timezone-free.
        if observation_time.tzinfo is not None:
            observation_time = observation_time.astimezone(timezone.utc)
        return observation_time.replace(tzinfo=None)

    @classmethod
    def _observation_time_for_file_group(cls, files, ds_type):
        representative_file = np.asarray(files, dtype=object).reshape(-1)[0]
        if ds_type == 'hmi':
            value = cls._hmi_date_obs(representative_file)
        else:
            value = Map(representative_file).date.to_datetime()
        return cls._parse_observation_time(value)

    @classmethod
    def _hmi_date_obs(cls, path):
        with fits.open(path, memmap=False) as hdus:
            header = next((hdu.header for hdu in hdus if 'DATE-OBS' in hdu.header), None)
        if header is None:
            raise ValueError(f'HMI segment {os.fspath(path)!r} is missing the DATE-OBS keyword.')
        return cls._parse_observation_time(header['DATE-OBS'])

    @classmethod
    def _filter_files_by_time_range(cls, files, time_range, ds_type):
        if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
            raise ValueError('time_range must contain exactly [start, end].')
        start_time, end_time = (cls._parse_observation_time(value) for value in time_range)
        if end_time < start_time:
            raise ValueError(f'time_range end {end_time} precedes start {start_time}.')

        selected = [
            file_group for file_group in files
            if start_time <= cls._observation_time_for_file_group(file_group, ds_type) <= end_time
        ]
        if not selected:
            raise ValueError(
                f'No {ds_type} observations fall within the inclusive time range '
                f'[{start_time}, {end_time}].'
            )
        if isinstance(files, np.ndarray):
            return np.asarray(selected, dtype=files.dtype)
        return selected

    def _load_all_files(self, data_path):
        if isinstance(data_path, str):
            return sorted(glob.glob(data_path))
        elif isinstance(data_path, Iterable):
            files = [f for d in data_path for f in glob.glob(d)]
            return sorted(files)
        else:
            raise ValueError(f'Unknown data path type: {type(data_path)}. Expected str or Iterable[str].')

    def train_dataloader(self):
        self._validate_batch_sizes()
        datasets = self.train_datasets
        for ds in datasets:
            ds.batch_size = self.dataset_batch_size
        combined_dataset = CombinedDataset(datasets)
        sampler = TimeStratifiedBatchSampler(
            combined_dataset.sample_indices_by_dataset,
            self.batch_size // self.dataset_batch_size,
        )
        loader_kwargs = {
            'dataset': combined_dataset,
            'batch_size': None,
            'sampler': sampler,
            'num_workers': self.num_workers,
            'pin_memory': False,
        }
        if self.num_workers > 0:
            loader_kwargs.update(prefetch_factor=5, persistent_workers=True)
        loader = DataLoader(**loader_kwargs)
        return loader

    def val_dataloader(self):
        self.valid_dataset.batch_size = self.batch_size
        data_loader = DataLoader(self.valid_dataset, batch_size=None, num_workers=self.num_workers,
                                 pin_memory=False, shuffle=False)
        return data_loader


class SphericalDataset(TensorsDataset):

    def __init__(self, stokes, map_data, wavelength_config, ds_id, instrument_id, seconds_per_dt, Rs_per_ds, ref_time,
                 stokes_normalization, work_directory, oversample_factor=1,
                 store_plot_image=False, mu_limit=1e-3,
                 **kwargs):
        self.ds_id = ds_id
        self.instrument_id = instrument_id
        self.oversample_factor = oversample_factor

        self.wavelength_config = wavelength_config  # lambda grid and reference wavelength
        self.stokes_normalization = float(stokes_normalization)

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
        detector_coords = map_data.get('detector_coords')

        # normalize stokes vector
        stokes /= self.stokes_normalization

        # normalize centered pixel coordinates per frame
        pix /= max((pix.shape[0] - 1) / 2, (pix.shape[1] - 1) / 2, 1.0)

        # remove off limb pixels
        spherical_coords[(mu < mu_limit) | np.isnan(mu)] = np.nan
        stokes[(mu < mu_limit) | np.isnan(mu)] = np.nan
        pix[(mu < mu_limit) | np.isnan(mu)] = np.nan
        if detector_coords is not None:
            detector_coords[(mu < mu_limit) | np.isnan(mu)] = np.nan

        # Match the finite-sample mask used by the tensor training store so the
        # collocation box is not enlarged by coordinates whose Stokes target or
        # auxiliary geometry is invalid.
        physics_valid = np.isfinite(spherical_coords).all(axis=-1)
        physics_valid &= np.isfinite(stokes).all(axis=(-2, -1))
        physics_valid &= np.isfinite(mu)
        physics_valid &= np.isfinite(v_obs_los)
        physics_valid &= np.isfinite(pix).all(axis=-1)
        if detector_coords is not None:
            physics_valid &= np.isfinite(detector_coords).all(axis=-1)

        # Store compact TRAIN-footprint metadata before the coordinates are
        # flattened into the tensor store. The circular longitude bounds avoid
        # turning a small field across +/-pi into an almost full-sphere box.
        self.physics_bounds = _summarize_physics_coordinates(
            spherical_coords[physics_valid]
        )
        self.physics_bounds['normalized_time'] = float(self.normalized_time)

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
        if detector_coords is not None:
            tensors['detector_coords'] = detector_coords.reshape((-1, 2))

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


def load_phi_wavelength_config(data, header=None, num_wl=6):
    """Load a PHI wavelength grid from processed keywords or tuning data."""
    header = fits.getheader(data) if header is None else header
    wavelength_keys = [f'WAVELN{i + 1:02d}' for i in range(num_wl)]

    if 'WAVELNTH' in header and all(key in header for key in wavelength_keys):
        wavelength_center = float(header['WAVELNTH'])
        wavelength_axis = np.array([header[key] for key in wavelength_keys], dtype=np.float64)
    else:
        try:
            sampling = fits_get_sampling(data, num_wl=num_wl, verbose=False)
        except Exception as error:
            raise ValueError(
                f'Unable to determine PHI wavelength sampling for {data!r}: processed '
                f'WAVELNTH/WAVELN01..{num_wl:02d} keywords are absent and tuning reconstruction failed.'
            ) from error
        if len(sampling) != 5:
            raise ValueError(
                f'Unable to determine the PHI reference wavelength for {data!r} from tuning metadata.'
            )
        wavelength_axis, _, _, _, wavelength_center = sampling
        wavelength_axis = np.asarray(wavelength_axis, dtype=np.float64)
        wavelength_center = float(wavelength_center)

    if wavelength_axis.shape != (num_wl,) or not np.all(np.isfinite(wavelength_axis)) \
            or not np.isfinite(wavelength_center):
        raise ValueError(f'Invalid PHI wavelength sampling in {data!r}.')

    wavelength_center = wavelength_center * u.AA
    wavelength_grid = wavelength_axis * u.AA - wavelength_center
    return {'wavelength_grid': wavelength_grid, 'wavelength_center': wavelength_center}


_HMI_RESPONSE_CACHE = {}


def sample_hmi_spectral_response(profile_file, detector_coords):
    """Interpolate one CPU-side HMI response map for the current dataset batch."""
    profile_file = os.path.abspath(os.fspath(profile_file))
    response = _HMI_RESPONSE_CACHE.get(profile_file)
    if response is None:
        arrays = load_hmi_transmission_profile(profile_file)
        response = {
            'spectral_offsets': torch.from_numpy(arrays['offsets'] * 1e-10),
            'spectral_weights': torch.from_numpy(arrays['weights']),
            'continuum_weights': torch.from_numpy(arrays['continuum_weights']),
        }
        _HMI_RESPONSE_CACHE[profile_file] = response

    flat_coords = detector_coords.reshape(-1, 2).to(dtype=torch.float32, device='cpu')
    weights = response['spectral_weights']
    continuum_weights = response['continuum_weights']
    height, width, n_filters, n_samples = weights.shape
    grid = flat_coords.reshape(1, -1, 1, 2) * 2 - 1
    profile_grid = weights.permute(2, 3, 0, 1).reshape(1, n_filters * n_samples, height, width)
    sampled_weights = F.grid_sample(
        profile_grid, grid, mode='bilinear', padding_mode='border', align_corners=True
    ).reshape(n_filters, n_samples, -1).movedim(-1, 0)
    continuum_grid = continuum_weights.permute(2, 0, 1).unsqueeze(0)
    sampled_continuum = F.grid_sample(
        continuum_grid, grid, mode='bilinear', padding_mode='border', align_corners=True
    ).reshape(n_filters, -1).movedim(-1, 0)
    sampled_offsets = response['spectral_offsets'].unsqueeze(0).expand(flat_coords.shape[0], -1, -1)

    spatial_shape = detector_coords.shape[:-1]
    return {
        'spectral_offsets': sampled_offsets.reshape(*spatial_shape, n_filters, n_samples),
        'spectral_weights': sampled_weights.reshape(*spatial_shape, n_filters, n_samples),
        'continuum_weights': sampled_continuum.reshape(*spatial_shape, n_filters),
    }


class HMISphericalDataset(SphericalDataset):

    def __init__(self, data, transmission_profile_directory, resolution=None,
                 require_quality_zero=True, *args, **kwargs):
        wavelength_config = hmi_wavelength_config()
        stokes_i_files, _, _, _ = data
        ref_file = stokes_i_files[0]
        acquisition = read_hmi_fits_acquisition(ref_file)
        self.acquisition_key = acquisition['acquisition_key']
        if require_quality_zero:
            self._validate_quality(data)
        self._validate_segment_alignment(data)
        self.spectral_response_file = os.fspath(
            resolve_hmi_transmission_profile(transmission_profile_directory, ref_file)
        )
        source_map = load_hmi_date_obs_map(ref_file)
        s_map = source_map
        if resolution is not None:
            s_map = s_map.resample(resolution * u.pix)
        map_data = load_map_data(s_map)
        # The map frame and latent coordinate now share the same DATE-OBS time.
        # T_REC remains only the JSOC/calibration record identity.
        map_data['time'] = SphericalDataModule._hmi_date_obs(ref_file)
        map_data['detector_coords'] = load_hmi_detector_coords(
            source_map, output_shape=s_map.data.shape
        )
        stokes = load_stokes_data(data, resolution)

        super().__init__(stokes, map_data, wavelength_config, *args, **kwargs)

    def __getitem__(self, *args):
        out = super().__getitem__(*args)
        detector_coords = out.pop('detector_coords')
        out.update(sample_hmi_spectral_response(self.spectral_response_file, detector_coords))
        return out

    @staticmethod
    def _validate_quality(data):
        bad_segments = []
        missing_quality = []
        for path in np.asarray(data, dtype=object).reshape(-1):
            with fits.open(path, memmap=False) as hdus:
                header = next((hdu.header for hdu in hdus if 'T_REC' in hdu.header), None)
            if header is None or 'QUALITY' not in header:
                missing_quality.append(os.fspath(path))
                continue
            try:
                quality = int(header['QUALITY'])
            except (TypeError, ValueError) as error:
                raise ValueError(f'Invalid HMI QUALITY value in {path!r}: {header["QUALITY"]!r}.') from error
            if quality != 0:
                bad_segments.append((os.fspath(path), quality))
        if missing_quality:
            raise ValueError(f'HMI segments are missing the QUALITY keyword: {missing_quality[:3]}')
        if bad_segments:
            summary = ', '.join(f'{os.path.basename(path)}={quality:#x}' for path, quality in bad_segments[:5])
            raise ValueError(f'Refusing nonzero-quality HMI acquisition: {summary}')

    @staticmethod
    def _validate_segment_alignment(data, tolerance_pixels=1e-3):
        """Reject HMI segments that cannot be stacked on one pixel grid."""
        paths = np.asarray(data, dtype=object).reshape(-1)
        if paths.size == 0:
            raise ValueError('Cannot validate an empty HMI acquisition.')
        if tolerance_pixels < 0:
            raise ValueError('Segment-alignment tolerance must be non-negative.')

        reference_path = paths[0]
        reference_map = load_hmi_date_obs_map(reference_path)
        reference_time = SphericalDataModule._hmi_date_obs(reference_path)

        def detector_origin(s_map, key):
            value = s_map.meta.get(key)
            return None if value is None else float(value)

        reference_origin = tuple(detector_origin(reference_map, key) for key in ('CCD_X0', 'CCD_Y0'))
        errors = []
        for path in paths[1:]:
            segment_map = load_hmi_date_obs_map(path)
            segment_name = os.path.basename(os.fspath(path))
            if segment_map.data.shape != reference_map.data.shape:
                errors.append(
                    f'{segment_name}: shape {segment_map.data.shape} != {reference_map.data.shape}'
                )
                continue

            segment_time = SphericalDataModule._hmi_date_obs(path)
            if segment_time != reference_time:
                errors.append(
                    f'{segment_name}: DATE-OBS {segment_time.isoformat()} != {reference_time.isoformat()}'
                )

            segment_origin = tuple(detector_origin(segment_map, key) for key in ('CCD_X0', 'CCD_Y0'))
            if (None in reference_origin) != (None in segment_origin):
                errors.append(
                    f'{segment_name}: detector origin {segment_origin} != {reference_origin}'
                )
            elif None not in reference_origin and not np.allclose(
                    segment_origin, reference_origin, rtol=0, atol=tolerance_pixels):
                errors.append(
                    f'{segment_name}: detector origin {segment_origin} != {reference_origin}'
                )

        if errors:
            reference_name = os.path.basename(os.fspath(reference_path))
            details = '; '.join(errors[:5])
            if len(errors) > 5:
                details += f'; ... and {len(errors) - 5} more'
            raise ValueError(
                f'Misaligned HMI segments relative to {reference_name}: {details}'
            )



class PHIHRTSphericalDataset(SphericalDataset):

    def __init__(self, data, *args, **kwargs):
        header = fits.getheader(data)
        wavelength_config = load_phi_wavelength_config(data, header)

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
        header = load_fix_phi_header(data) if fix_header else fits.getheader(data)
        wavelength_config = load_phi_wavelength_config(data, header)

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
        wavelength_config = hmi_wavelength_config()

        stokes_i_files, _, _, _ = data
        ref_file = stokes_i_files[0]
        s_map = load_hmi_date_obs_map(ref_file)
        map_data = load_map_data(s_map)
        stokes = load_stokes_data(data)

        # add noise
        normal_noise = np.random.normal(size=stokes.shape, scale=noise)
        stokes += normal_noise

        super().__init__(stokes, map_data, wavelength_config, *args, **kwargs)


def load_stokes_data(files, resolution: Optional[tuple[int, int]] = None):
    stokes_i_files, stokes_q_files, stokes_u_files, stokes_v_files = files
    if resolution is None:  # default load with astropy fits
        i_profile = np.stack([fits.getdata(path) for path in stokes_i_files], -1)
        q_profile = np.stack([fits.getdata(path) for path in stokes_q_files], -1)
        u_profile = np.stack([fits.getdata(path) for path in stokes_u_files], -1)
        v_profile = np.stack([fits.getdata(path) for path in stokes_v_files], -1)
    else:
        i_profile = np.stack([load_hmi_date_obs_map(path).resample(resolution * u.pix).data
                              for path in stokes_i_files], -1)
        q_profile = np.stack([load_hmi_date_obs_map(path).resample(resolution * u.pix).data
                              for path in stokes_q_files], -1)
        u_profile = np.stack([load_hmi_date_obs_map(path).resample(resolution * u.pix).data
                              for path in stokes_u_files], -1)
        v_profile = np.stack([load_hmi_date_obs_map(path).resample(resolution * u.pix).data
                              for path in stokes_v_files], -1)

    stokes = np.stack([i_profile, q_profile, u_profile, v_profile], -2)

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

    # centered pixel coordinates; SphericalDataset applies the final scalar normalization
    pix_coords = np.stack(np.mgrid[0:s_map.data.shape[0], 0:s_map.data.shape[1]], -1)  # y, x
    pix_coords = pix_coords.astype(np.float32)
    pix_coords[..., 0] -= (s_map.data.shape[0] - 1) / 2
    pix_coords[..., 1] -= (s_map.data.shape[1] - 1) / 2

    return {'mu': mu, 'v_obs_los': v_obs_los, 'pix': pix_coords,
            'obs_lat': obs_lat, 'obs_lon': obs_lon, 'pAng': pAng,
            'time': time, 'spherical_coords': spherical_coords, 'wcs': s_map.wcs,
            'instrument': s_map.instrument}


def load_hmi_detector_coords(s_map, output_shape=None, ccd_size=4096):
    """Map an HMI image or cutout onto normalized full-detector coordinates.

    Full-disk arrays use their exact physical pixel indices. Cutouts must carry
    the exact ``CCD_X0/Y0`` origins written by ``subframe_hmi``. Coordinates
    are returned in the ``[x, y]`` order expected by grid sampling.
    """
    if ccd_size < 2:
        raise ValueError('ccd_size must be at least 2.')
    height, width = s_map.data.shape[-2:]
    output_height, output_width = (height, width) if output_shape is None else output_shape[-2:]
    if height == ccd_size and width == ccd_size:
        # A full HMI image is already indexed in physical detector pixels;
        # CRPIX is the moving solar reference point, not the CCD center.
        source_x = np.arange(width, dtype=np.float64)
        source_y = np.arange(height, dtype=np.float64)
    elif 'CCD_X0' in s_map.meta and 'CCD_Y0' in s_map.meta:
        source_x = np.arange(width, dtype=np.float64) + float(s_map.meta['CCD_X0'])
        source_y = np.arange(height, dtype=np.float64) + float(s_map.meta['CCD_Y0'])
    else:
        raise ValueError(
            'HMI cutouts require CCD_X0 and CCD_Y0 metadata. Regenerate the cutout with pme.data.subframe_hmi.'
        )
    detector_x = np.linspace(source_x[0], source_x[-1], output_width)
    detector_y = np.linspace(source_y[0], source_y[-1], output_height)
    x_grid, y_grid = np.meshgrid(detector_x, detector_y)
    coords = np.stack([x_grid, y_grid], axis=-1) / (ccd_size - 1)
    return np.clip(coords, 0, 1).astype(np.float32)
