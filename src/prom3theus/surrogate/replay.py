"""Device-resident hard-example replay for local LTE coefficient profiles."""

import torch


class HardProfileBuffer:
    def __init__(self, capacity, wavelengths, seed):
        self.capacity = capacity
        self.seed = seed
        self.size = 0
        device = wavelengths.device
        self.points = torch.empty(capacity, 7, dtype=torch.float32, device=device)
        self.targets = torch.empty(capacity, wavelengths.numel(), 7, dtype=torch.float32, device=device)
        self.offsets = wavelengths.new_empty(capacity)
        self.scores = self.points.new_empty(capacity)

    def sample_indices(self, count, step):
        """Sample without replacement with probability weighted by the last measured loss."""
        if count > self.size:
            raise ValueError("Requested more replay profiles than the buffer contains")
        scores = self.scores[:self.size]
        generator = torch.Generator(device=scores.device).manual_seed(self.seed + step)
        # An exponential race samples by weight using a fixed-size top-k on the device.
        noise = torch.empty_like(scores).exponential_(generator=generator)
        return (scores.clamp_min(1e-12) / noise).topk(count, sorted=False).indices

    @torch.no_grad()
    def update(self, points, targets, offset, scores, replay_indices):
        """Refresh sampled priorities, then keep the hardest old and fresh profiles."""
        count = len(points)
        if replay_indices is not None:
            self.scores[replay_indices] = scores[count:]
        available = min(count, self.capacity - self.size)
        if available:
            slots = slice(self.size, self.size + available)
            self.points[slots] = points[:available]
            self.targets[slots] = targets[:available]
            self.offsets[slots] = offset
            self.scores[slots] = scores[:available]
            self.size += available
        remaining = min(count - available, self.capacity)
        if not remaining:
            return

        # Match the highest new scores against the lowest stored scores. Fixed-shape
        # writes avoid a host synchronization for a data-dependent number of winners.
        new_scores, candidates = scores[available:count].topk(remaining)
        candidates = candidates + available
        old_scores, slots = self.scores.topk(remaining, largest=False)
        replace = new_scores > old_scores
        self.points[slots] = torch.where(replace[:, None], points[candidates], self.points[slots])
        self.targets[slots] = torch.where(replace[:, None, None], targets[candidates], self.targets[slots])
        self.offsets[slots] = torch.where(replace, offset, self.offsets[slots])
        self.scores[slots] = torch.maximum(new_scores, old_scores)

    def state_dict(self):
        # Save only initialized rows, even when interrupted during warmup.
        return {name: getattr(self, name)[:self.size].to("cpu", copy=True)
                for name in ("points", "targets", "offsets", "scores")}

    def load_state_dict(self, state):
        size = len(state["scores"])
        if size > self.capacity:
            raise ValueError("Checkpoint replay buffer exceeds configured capacity")
        for name in ("points", "targets", "offsets", "scores"):
            destination = getattr(self, name)[:size]
            if state[name].shape != destination.shape:
                raise ValueError(f"Incompatible replay checkpoint tensor: {name}")
            destination.copy_(state[name])
        self.size = size
