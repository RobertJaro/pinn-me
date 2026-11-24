"""Reproducible local states from disjoint MURaM column regions."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from prom3theus.muram.atmosphere_builder import NATIVE_ATMOSPHERE_FIELDS
from prom3theus.muram.reader import MURaMSnapshot

from .config import TrainingConfig
from .run_files import logger


def _valid_depths(fields, log_tau):
    tau = np.log10(np.maximum(fields["tau"], 1e-35))
    valid = (tau >= log_tau[0]) & (tau <= log_tau[1])
    valid &= (fields["temperature"] > 0) & (fields["pressure"] > 0)
    for name in NATIVE_ATMOSPHERE_FIELDS:
        valid &= np.isfinite(fields[name])
    return tau, valid


def _jitter_states(states, rng, settings):
    """Perturb physical teacher inputs in place before the EOS and exact LTE evaluation."""
    widths = (settings.log_temperature, settings.log_pressure, settings.microturbulence_m_s,
              settings.velocity_m_s) + (0.0,) * 3
    if not any(widths) and not settings.magnetic_field_fraction:
        return np.zeros(len(states), dtype=bool)
    if any(widths):
        # Keep the original seven-column RNG sequence for atmospheric draws on resume.
        noise = rng.normal(0.0, widths, size=states.shape)
        states[:, :2] *= np.exp(noise[:, :2])
        states[:, 2:4] += noise[:, 2:4]
        states[:, 2] = np.abs(states[:, 2])
    fraction = settings.magnetic_field_fraction
    if fraction:
        states[:, 4:] *= rng.uniform(1 - fraction, 1 + fraction, size=(len(states), 1))
    return np.ones(len(states), dtype=bool)


class SnapshotPoints(Dataset):
    """Each index is a complete, independently seeded point batch."""

    def __init__(
        self, columns, *, split, seed, batch_size, steps, perturbations, log_tau,
    ):
        self.columns = columns.dataset
        self.spacing = columns.ds_m
        self.ds = columns.ds_m[-1]
        nx, ny, _ = self.columns.shape
        # Keep all depths and variants together, with a column buffer on each boundary.
        dimension = max(nx, ny)
        if dimension < 5:
            raise ValueError(
                "Three spatial splits and two buffer columns require at least five horizontal columns"
            )
        a = min(max(2, int(0.7 * dimension)), dimension - 3)
        b = min(max(a + 2, int(0.85 * dimension)), dimension - 1)
        regions = {
            "train": (0, a - 1),
            "validation": (a, b - 1),
            "test": (b, dimension),
        }
        self.split = split
        self.region = regions[split]
        self.split_axis = 0 if nx >= ny else 1
        self.seed = seed
        self.batch_size = batch_size
        self.steps = steps
        self.perturbations = perturbations
        self.log_tau = log_tau

    def __len__(self):
        return self.steps

    def _rng(self, index):
        return np.random.default_rng(
            [self.seed, ("train", "validation", "test").index(self.split), index]
        )

    def _column(self, rng):
        nx, ny, _ = self.columns.shape
        x, y = rng.integers(nx), rng.integers(ny)
        selected = rng.integers(*self.region)
        if self.split_axis == 0:
            x = selected
        else:
            y = selected
        flat = int(x * ny + y)
        column = self.columns[flat, flat, (*NATIVE_ATMOSPHERE_FIELDS, "tau")]["fields"]
        tau, valid = _valid_depths(column, self.log_tau)
        return flat, column, tau, np.flatnonzero(valid)

    @staticmethod
    def _values(column, z, basis):
        velocity = np.stack([column[k][z] for k in ("vx", "vy", "vz")], -1)
        field = np.stack([column[k][z] for k in ("bx", "by", "bz")], -1)
        return np.column_stack(
            (
                column["temperature"][z],
                column["pressure"][z],
                np.zeros(len(z)),
                -(velocity @ basis[-1]),
                field @ basis.T,
            )
        )

    def __getitem__(self, index):
        return self.sample_batch(index)

    def sample_batch(self, index, *, size=None, describe=False):
        """Draw snapshot points and perturbations with optional column/depth provenance."""
        size = self.batch_size if size is None else size
        rng = self._rng(index)
        parts = []
        metadata = []
        count = 0
        attempts = 0
        while count < size:
            attempts += 1
            if attempts > size + 128:
                raise ValueError(
                    "Insufficient finite snapshot states in the requested optical-depth range"
                )
            flat, col, tau, z = self._column(rng)
            if not z.size:
                continue
            field = np.sqrt(
                sum(col[name][z].astype(np.float64) ** 2 for name in ("bx", "by", "bz"))
            )
            groups = np.floor(tau[z]).astype(int) * 4 + np.digitize(
                field, [100.0, 500.0, 1500.0]
            )
            _, inverse, frequencies = np.unique(
                groups, return_inverse=True, return_counts=True
            )
            probabilities = 1.0 / frequencies[inverse]
            n = min(16, size - count)
            sampled = rng.choice(z, n, p=probabilities / probabilities.sum())
            mu, phi = rng.uniform(0, 1), rng.uniform(-np.pi, np.pi)
            direction = np.array(
                [np.sqrt(1 - mu**2) * np.cos(phi), np.sqrt(1 - mu**2) * np.sin(phi), mu]
            )
            first = np.array([-np.sin(phi), np.cos(phi), 0])
            basis = np.stack((first, np.cross(direction, first), direction))
            parts.append(self._values(col, sampled, basis))
            if describe:
                metadata.append((np.full(n, flat), sampled, tau[sampled]))
            count += n
        states = np.concatenate(parts)
        mask = _jitter_states(states, rng, self.perturbations)
        points = torch.from_numpy(states.astype(np.float32))
        if describe:
            column, depth, log_tau = (
                torch.from_numpy(np.concatenate(values)) for values in zip(*metadata)
            )
            return points, dict(
                column=column,
                depth=depth,
                log_tau=log_tau,
                perturbed=torch.from_numpy(mask),
            )
        return points

    def reference_ray(self, index, depths, *, mu=1.0):
        rng = self._rng(index)
        for _ in range(256):
            flat, col, _, z = self._column(rng)
            if len(z) < 2:
                continue
            # Ray order is observer to bottom, with the actual native distances.
            z = z[np.linspace(len(z) - 1, 0, min(len(z), depths)).round().astype(int)]
            vertical_distance = (z[0] - z) * self.ds
            if mu == 1.0:
                states = self._values(col, z, np.eye(3))
            else:
                # Incline along the unrestricted horizontal axis so the entire ray
                # stays in its held-out region. Interpolate on the native grid.
                axis = 1 - self.split_axis
                sine = np.sqrt(1 - mu**2)
                direction = np.array([0.0, 0.0, mu])
                direction[axis] = sine
                first = np.array([-direction[1], direction[0], 0.0]) / sine
                basis = np.stack((first, np.cross(direction, first), direction))
                nx, ny, _ = self.columns.shape
                x0, y0 = divmod(flat, ny)
                horizontal = (x0, y0)[axis] - vertical_distance * sine / (
                    mu * self.spacing[axis]
                )
                if horizontal.min() < 0 or horizontal.max() >= (nx, ny)[axis] - 1:
                    continue
                lower = horizontal.astype(int)
                fraction = horizontal - lower
                endpoints = []
                for offset in (0, 1):
                    indices = (
                        (lower + offset) * ny + y0
                        if axis == 0
                        else x0 * ny + lower + offset
                    )
                    fields = {k: np.empty(len(z)) for k in NATIVE_ATMOSPHERE_FIELDS}
                    for native in np.unique(indices):
                        mask = indices == native
                        values = self.columns[int(native), 0, NATIVE_ATMOSPHERE_FIELDS][
                            "fields"
                        ]
                        for key in fields:
                            fields[key][mask] = values[key][z[mask]]
                    endpoints.append(fields)
                sampled = {
                    key: endpoints[0][key] * (1 - fraction)
                    + endpoints[1][key] * fraction
                    for key in NATIVE_ATMOSPHERE_FIELDS
                }
                states = self._values(sampled, np.arange(len(z)), basis)
            return (
                flat,
                torch.from_numpy(states.astype(np.float32)),
                torch.from_numpy(vertical_distance / mu),
            )
        raise ValueError("No usable reference column in held-out region")


class FreshPointBatches(Dataset):
    """Supply only fresh teacher inputs, including a full first batch to seed replay."""

    def __init__(self, dataset, replay_fraction):
        self.dataset = dataset
        self.replay_per_batch = dataset.batch_size * replay_fraction

    def __len__(self):
        return len(self.dataset)

    def _fresh_before(self, index):
        replayed = int(max(index - 1, 0) * self.replay_per_batch)
        return index * self.dataset.batch_size - replayed

    def __getitem__(self, index):
        first = self._fresh_before(index)
        size = self._fresh_before(index + 1) - first
        if size == 0:
            return torch.empty((0, 7), dtype=torch.float32)
        return self.dataset.sample_batch(index, size=size)


def load_snapshot_splits(config: TrainingConfig):
    """Open one snapshot and partition its columns for training, validation and testing."""
    logger.info(
        "Opening snapshot %s iteration %d", config.source_path, config.iteration
    )
    snapshot = MURaMSnapshot(
        str(config.source_path),
        config.iteration,
        vertical_axis=config.vertical_axis,
        component_order=tuple(config.component_order),
    )
    columns = snapshot.synthesis_columns()
    datasets = {
        split: SnapshotPoints(
            columns,
            split=split,
            seed=config.seed,
            batch_size=config.batch_size,
            steps=config.steps,
            perturbations=config.perturbations,
            log_tau=config.log_tau,
        )
        for split in ("train", "validation", "test")
    }
    return snapshot, datasets
