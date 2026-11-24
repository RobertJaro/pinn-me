"""Optimizer selection for Lightning automatic optimization."""

import torch

from prom3theus.core.nn import MLPModel, SIRENModel


def exponential_lr_factor(completed_steps, initial_lr, final_lr, decay_steps):
    progress = min(max(completed_steps / max(decay_steps, 1), 0.0), 1.0)
    return (final_lr / initial_lr) ** progress


class MuonAdamW(torch.optim.Optimizer):
    """Share state and groups so Lightning clips and schedules both algorithms.

    Updates use native PyTorch implementations and one closure/backward pass.
    """

    def __init__(self, muon_params, adamw_params, *, lr, groups=None):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("Muon requires PyTorch >=2.9; install PROM3THEUS[muon].")
        self._optimizers = []
        candidates = groups if groups is not None else [
            ({"params": list(muon_params), "lr": lr}, "muon"),
            ({"params": list(adamw_params), "lr": lr}, "adamw")]
        for group, kind in candidates:
            params = group["params"]
            if not params:
                continue
            if kind == "muon":
                optimizer = torch.optim.Muon(
                    [group], lr=lr, weight_decay=0.0, momentum=0.95,
                    adjust_lr_fn="match_rms_adamw",
                )
            else:
                optimizer = torch.optim.AdamW([group], lr=lr, weight_decay=0.0)
            optimizer.param_groups[0]["algorithm"] = kind
            self._optimizers.append(optimizer)
        super().__init__([opt.param_groups[0] for opt in self._optimizers], {"lr": lr})
        self._bind_state()

    def _bind_state(self):
        for optimizer, group in zip(self._optimizers, self.param_groups, strict=True):
            optimizer.param_groups = [group]
            optimizer.state = self.state

    def load_state_dict(self, state_dict):
        if [g.get("algorithm") for g in state_dict["param_groups"]] != [
            g["algorithm"] for g in self.param_groups
        ]:
            raise ValueError("Checkpoint optimizer groups do not match Muon/AdamW.")
        super().load_state_dict(state_dict)
        self._bind_state()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self._optimizers:
            optimizer.step()
        return loss


def build_optimizer(model, *, kind, lr, parameter_blocks=None):
    if kind == "adam":
        return torch.optim.Adam(parameter_blocks if parameter_blocks is not None else model.parameters(), lr=lr)
    if kind == "soap":
        from prom3theus.training.soap import SOAP
        parameters = parameter_blocks if parameter_blocks is not None else (
            p for p in model.parameters() if p.requires_grad
        )
        return SOAP(parameters, lr=lr, weight_decay=0., max_precond_dim=2048)
    if kind != "muon":
        raise ValueError(f"Unknown optimizer: {kind!r}")
    # Coordinate-network hidden matrices use Muon. Input/readout layers,
    # biases, calibration parameters, and other auxiliary parameters use AdamW.
    hidden = {
        id(layer.weight)
        for module in model.modules()
        if isinstance(module, (MLPModel, SIRENModel))
        for layer in module.hidden_layers
    }
    if parameter_blocks is not None:
        groups = []
        for block in parameter_blocks:
            for algorithm in ("muon", "adamw"):
                params = [p for p in block["params"] if (id(p) in hidden) == (algorithm == "muon")]
                if params:
                    groups.append(({**block, "params": params}, algorithm))
        return MuonAdamW([], [], lr=lr, groups=groups)
    muon, adamw = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (muon if id(parameter) in hidden else adamw).append(parameter)
    if not muon:
        raise ValueError("Muon requires trainable MLP/SIREN hidden weight matrices.")
    return MuonAdamW(muon, adamw, lr=lr)
