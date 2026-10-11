"""Native Muon and AdamW exposed as one optimizer for Lightning automation."""

import math

import torch


class MuonAdamW(torch.optim.Optimizer):
    """Share groups/state with native optimizers for scheduling and checkpointing.

    DDP synchronizes gradients before this optimizer runs; each rank steps both
    groups locally. Input/output layers and biases belong in ``adamw_params``.
    """

    def __init__(self, muon_params, adamw_params, lr=1e-3, muon_lr=None,
                 weight_decay=0.0, momentum=0.95, ns_steps=5,
                 betas=(0.9, 0.95), eps=1e-8):
        if any(not math.isfinite(value) or value <= 0
               for value in (lr, lr if muon_lr is None else muon_lr)):
            raise ValueError('Learning rates must be finite and positive.')
        if isinstance(ns_steps, bool) or not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError('ns_steps must be a positive integer.')
        if not hasattr(torch.optim, 'Muon'):
            raise RuntimeError('Muon requires PyTorch >= 2.9. Use optimizer_config.type=soap '
                               'or upgrade PyTorch.')
        self._muon = torch.optim.Muon(
            muon_params, lr=lr if muon_lr is None else muon_lr,
            weight_decay=weight_decay, momentum=momentum, nesterov=True,
            ns_steps=ns_steps, adjust_lr_fn='match_rms_adamw',
        )
        self._adamw = torch.optim.AdamW(
            adamw_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
        )
        groups = self._muon.param_groups + self._adamw.param_groups
        groups[0]['algorithm'] = 'muon'
        groups[1]['algorithm'] = 'adamw'
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Rebind each step: loading a checkpoint replaces the outer state and
        # group dictionaries, and Lightning may subsequently move their tensors.
        for optimizer, algorithm in ((self._muon, 'muon'), (self._adamw, 'adamw')):
            optimizer.param_groups = [g for g in self.param_groups if g['algorithm'] == algorithm]
            optimizer.state = self.state
            optimizer.step()
        return loss

    def load_state_dict(self, state_dict):
        if [g.get('algorithm') for g in state_dict['param_groups']] != ['muon', 'adamw']:
            raise ValueError('Cannot resume Muon/AdamW from a different optimizer. '
                             'Start a new run in a fresh base_path.')
        super().load_state_dict(state_dict)
