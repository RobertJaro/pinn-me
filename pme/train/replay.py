"""Bounded, rank-local replay of difficult observed Stokes profiles."""

import math

import torch


def _time_balanced_indices(times, count, scores=None):
    """Round-robin acquisition times, choosing hardest (or random) rows within each."""
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=times.device)
    if count == len(times):
        return torch.arange(count, device=times.device)
    unique_times, groups = torch.unique(times, return_inverse=True)
    # Randomize which acquisition receives a remainder slot.
    group_count = unique_times.numel()
    groups = torch.randperm(group_count, device=times.device)[groups]
    order = (torch.randperm(len(times), device=times.device) if scores is None
             else torch.argsort(scores, descending=True, stable=True))
    order = order[torch.argsort(groups[order], stable=True)]
    sizes = torch.bincount(groups, minlength=group_count)
    offsets = sizes.cumsum(0) - sizes
    rank_in_group = torch.arange(len(times), device=times.device) - offsets[groups[order]]
    return order[torch.argsort(rank_in_group, stable=True)[:count]]


class StokesReplayBuffer:
    """Retain up to fraction * batch_size rows per instrument, without graphs.

    Buffer contents live on the training device and are transient: checkpoints
    resume with a fresh batch. No priorities or extra geometry are stored on disk.
    Replaying changes the effective data weighting; no importance correction is
    applied. Each DDP rank independently maintains its own buffer.
    """

    def __init__(self, fraction=0.25, max_replays=5):
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError('Replay fraction must be in [0, 1).')
        if isinstance(max_replays, bool) or not isinstance(max_replays, int) or max_replays < 1:
            raise ValueError('max_replays must be a positive integer.')
        self.fraction = float(fraction)
        self.max_replays = max_replays
        self._entries = {}

    def configuration(self):
        return {'fraction': self.fraction, 'max_replays': self.max_replays}

    def clear(self):
        self._entries.clear()

    @staticmethod
    def _profile_inputs(inputs, instrument):
        # BatchesDataset emits one lin_idx per disk chunk, not per profile.
        # It is used to place validation images, never to synthesize Stokes.
        if 'lin_idx' in inputs:
            inputs = {key: value for key, value in inputs.items() if key != 'lin_idx'}
        n = len(inputs['coords'])
        for key, value in inputs.items():
            if not torch.is_tensor(value) or value.ndim == 0 or value.shape[0] != n:
                shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                raise ValueError(f'Replay input {instrument}.{key} must have {n} profile rows; '
                                 f'got {shape}. Chunk metadata must not enter the replay buffer.')
        return inputs

    @torch.no_grad()
    def mix(self, fresh):
        """Replace part of each incoming instrument batch with retained inputs."""
        mixed, counts = {}, {}
        for instrument, inputs in fresh.items():
            inputs = self._profile_inputs(inputs, instrument)
            n = len(inputs['coords'])
            entry = self._entries.get(instrument)
            n_replay = 0 if entry is None else min(int(n * self.fraction), len(entry[1]))
            if n_replay == 0:
                mixed[instrument] = inputs
                counts[instrument] = torch.zeros(n, dtype=torch.long, device=inputs['coords'].device)
                continue
            retained, previous_counts = entry
            new_idx = _time_balanced_indices(inputs['coords'][:, 0], n - n_replay)
            if n_replay == len(previous_counts):
                # Normally the entire pool is reused: no grouping, sorting or copy.
                replay_inputs, replay_counts = retained, previous_counts
            else:
                old_idx = _time_balanced_indices(retained['coords'][:, 0], n_replay)
                replay_inputs = {key: value[old_idx] for key, value in retained.items()}
                replay_counts = previous_counts[old_idx]
            mixed[instrument] = {
                key: torch.cat((value[new_idx], replay_inputs[key]))
                for key, value in inputs.items()
            }
            counts[instrument] = torch.cat((previous_counts.new_zeros(n - n_replay),
                                            replay_counts + 1))
        return mixed, counts

    @torch.no_grad()
    def update(self, batch, counts, scores):
        """Re-score all candidates on this visit, dropping easy and expired rows."""
        entries = {}
        for instrument, inputs in batch.items():
            inputs = self._profile_inputs(inputs, instrument)
            score = scores[instrument].detach()
            expected = (len(inputs['coords']),)
            if score.shape != expected or counts[instrument].shape != expected:
                raise ValueError(f'Replay scores and counts for {instrument} must have shape {expected}.')
            eligible = torch.where(torch.isfinite(score) & (counts[instrument] < self.max_replays))[0]
            capacity = min(int(len(score) * self.fraction), len(eligible))
            if capacity == 0:
                continue
            selected = eligible[_time_balanced_indices(
                inputs['coords'][eligible, 0], capacity, score[eligible],
            )]
            # Advanced indexing creates compact copies, without retaining the
            # full batch storage or its autograd graph.
            entries[instrument] = ({key: value.detach()[selected] for key, value in inputs.items()},
                                   counts[instrument][selected].clone())
        self._entries = entries
