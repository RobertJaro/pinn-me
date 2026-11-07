"""Muon dispatch, native-update parity, and checkpoint/scheduler continuity."""

from copy import deepcopy

import pytest
import torch

from prom3theus.config.joint_schema import JointTrainingConfig
from prom3theus.core.nn import MLPModel, SIRENModel
from prom3theus.training.optimizers import MuonAdamW, build_optimizer

requires_muon = pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"), reason="Native Muon requires PyTorch >=2.9"
)

@requires_muon
@pytest.mark.parametrize("network_type", [MLPModel, SIRENModel])
def test_hidden_weights_use_muon_and_other_parameters_use_adamw(network_type):
    model = network_type(3, 2, dim=8, n_layers=3)
    model.hidden_layers[0].bias.requires_grad_(False)
    optimizer = build_optimizer(model, kind="muon", lr=1e-3)
    groups = {group["algorithm"]: {id(p) for p in group["params"]}
              for group in optimizer.param_groups}
    assert groups["muon"] == {id(layer.weight) for layer in model.hidden_layers}
    assert groups["muon"].isdisjoint(groups["adamw"])
    assert groups["muon"] | groups["adamw"] == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    assert id(model.in_layer.weight) in groups["adamw"]
    assert id(model.out_layer.weight) in groups["adamw"]


@requires_muon
def test_combined_updates_match_native_optimizers_and_resume():
    torch.manual_seed(5)
    parameters = [torch.nn.Parameter(torch.randn(8, 8)), torch.nn.Parameter(torch.randn(8))]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    optimizer = MuonAdamW(parameters[:1], parameters[1:], lr=1e-3)
    native = [
        torch.optim.Muon(reference[:1], lr=1e-3, weight_decay=0.,
                         momentum=.95, adjust_lr_fn="match_rms_adamw"),
        torch.optim.AdamW(reference[1:], lr=1e-3, weight_decay=0.),
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .8**step)
    for step in range(4):
        for p, ref in zip(parameters, reference):
            p.grad = torch.randn_like(p)
            ref.grad = p.grad.clone()
        for opt, group in zip(native, optimizer.param_groups):
            opt.param_groups[0]["lr"] = group["lr"]
            opt.step()
        optimizer.step()
        scheduler.step()
        for p, ref in zip(parameters, reference):
            torch.testing.assert_close(p, ref)
        if step == 1:
            saved_optimizer = deepcopy(optimizer.state_dict())
            saved_scheduler = deepcopy(scheduler.state_dict())
            parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
            optimizer = MuonAdamW(parameters[:1], parameters[1:], lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .8**step)
            scheduler.load_state_dict(saved_scheduler)
            optimizer.load_state_dict(saved_optimizer)


@requires_muon
def test_closure_runs_once_with_gradients_enabled():
    matrix = torch.nn.Parameter(torch.randn(8, 8))
    bias = torch.nn.Parameter(torch.randn(8))
    optimizer = MuonAdamW([matrix], [bias], lr=1e-3)
    calls = []

    def closure():
        calls.append(True)
        optimizer.zero_grad()
        loss = matrix.square().mean() + bias.square().mean()
        loss.backward()
        return loss

    assert optimizer.step(closure).isfinite()
    assert len(calls) == 1
    assert optimizer.state[matrix]
    assert optimizer.state[bias]


def test_configuration_and_legacy_default():
    options = dict(max_steps=10, learning_rate=1e-3, final_learning_rate=1e-4)
    assert JointTrainingConfig(**options).optimizer == "adam"
    assert JointTrainingConfig(**options, optimizer="muon").optimizer == "muon"
    assert JointTrainingConfig(**options, optimizer="soap").optimizer == "soap"
    with pytest.raises(ValueError, match="training.optimizer"):
        JointTrainingConfig(**options, optimizer="other")
    assert isinstance(build_optimizer(torch.nn.Linear(2, 2), kind="adam", lr=1e-3),
                      torch.optim.Adam)


def test_missing_native_muon_has_actionable_error(monkeypatch):
    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    with pytest.raises(RuntimeError, match="PyTorch >=2.9"):
        MuonAdamW([torch.nn.Parameter(torch.ones(2, 2))], [], lr=1e-3)


def test_resume_rejects_changing_optimizer():
    from prom3theus.training.joint import JointInversionModule

    settings = JointTrainingConfig(max_steps=5, learning_rate=1e-3,
                                   final_learning_rate=1e-4, optimizer="muon")
    module = JointInversionModule(torch.nn.Linear(2, 2), settings, {"contract": {}})
    with pytest.raises(ValueError, match="Checkpoint optimizer differs"):
        module.on_load_checkpoint({"formal_transfer_format_version": 1, "prom3theus_joint": {"contract": {}}})


@requires_muon
def test_lightning_automatic_optimization_and_checkpoint(tmp_path):
    from pytorch_lightning import LightningModule, Trainer
    from torch.utils.data import DataLoader
    from prom3theus.training.joint import JointInversionModule

    class TinyInversion(LightningModule):
        configure_optimizers = JointInversionModule.configure_optimizers
        learning_rate_factor = JointInversionModule.learning_rate_factor

        def __init__(self):
            super().__init__()
            self.network = SIRENModel(3, 2, dim=8, n_layers=3)
            self.settings = JointTrainingConfig(
                max_steps=4, learning_rate=1e-3, final_learning_rate=1e-3,
                optimizer="muon",
            )

        def training_step(self, batch, batch_idx):
            return self.network(batch).square().mean()

    loader = DataLoader(torch.ones(8, 3), batch_size=2)
    options = dict(accelerator="cpu", logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, enable_model_summary=False,
                   gradient_clip_val=.1, default_root_dir=tmp_path)
    model = TinyInversion()
    initial = deepcopy(model.state_dict())
    trainer = Trainer(max_steps=2, **options)
    trainer.fit(model, loader)
    assert trainer.global_step == 2
    checkpoint = tmp_path / "muon.ckpt"
    trainer.save_checkpoint(checkpoint)
    resumed = TinyInversion()
    resumed_trainer = Trainer(max_steps=4, **options)
    resumed_trainer.fit(resumed, loader, ckpt_path=checkpoint)
    uninterrupted = TinyInversion()
    uninterrupted.load_state_dict(initial)
    Trainer(max_steps=4, **options).fit(uninterrupted, loader)
    for actual, expected in zip(resumed.parameters(), uninterrupted.parameters()):
        torch.testing.assert_close(actual, expected)
    optimizer = resumed_trainer.optimizers[0]
    assert all(p in optimizer.state for p in resumed.parameters())
