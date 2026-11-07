"""SOAP update parity, precision, closure count, and checkpoint continuity."""

from copy import deepcopy

import pytest
import torch

from prom3theus.training.optimizers import build_optimizer
from prom3theus.training.soap import SOAP


def test_updates_survive_basis_refresh_and_resume():
    torch.manual_seed(5)
    parameters = [torch.nn.Parameter(torch.randn(shape)) for shape in [(8, 4), (8,), ()]]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    optimizer = SOAP(parameters, lr=1e-3, weight_decay=0., max_precond_dim=2048)
    native = SOAP(reference, lr=1e-3, weight_decay=0., max_precond_dim=2048)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .98**step)
    for step in range(14):
        for p, ref in zip(parameters, reference):
            p.grad = torch.randn_like(p)
            ref.grad = p.grad.clone()
        native.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]
        optimizer.step()
        native.step()
        scheduler.step()
        for p, ref in zip(parameters, reference):
            torch.testing.assert_close(p, ref, rtol=0, atol=0)
            torch.testing.assert_close(optimizer.state[p], native.state[ref], rtol=0, atol=0)
        if step == 5:
            saved_optimizer, saved_scheduler = deepcopy(optimizer.state_dict()), deepcopy(scheduler.state_dict())
            parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
            optimizer = SOAP(parameters, lr=1e-3, weight_decay=0., max_precond_dim=2048)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .98**step)
            scheduler.load_state_dict(saved_scheduler)
            optimizer.load_state_dict(saved_optimizer)


def test_one_closure_per_step_and_float32_optimizer_operations():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class Float32Audit(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            assert not any(isinstance(x, torch.Tensor) and x.dtype == torch.float64 for x in tree_leaves(result))
            return result

    model = torch.nn.Linear(4, 2)
    optimizer = build_optimizer(model, kind="soap", lr=1e-3)
    calls = []
    initial = model.weight.detach().clone()

    def closure():
        calls.append(True)
        optimizer.zero_grad()
        loss = model(torch.ones(3, 4)).square().mean()
        loss.backward()
        return loss

    with Float32Audit():
        for _ in range(12):
            assert torch.isfinite(optimizer.step(closure))
    assert len(calls) == 12
    assert not torch.equal(initial, model.weight)


def test_failed_eigendecomposition_never_retries_in_float64(monkeypatch):
    calls = []

    def fail(matrix):
        calls.append(matrix.dtype)
        raise torch.linalg.LinAlgError("test failure")

    monkeypatch.setattr(torch.linalg, "eigh", fail)
    with pytest.raises(torch.linalg.LinAlgError):
        SOAP([torch.nn.Parameter(torch.ones(3))]).get_orthogonal_matrix([torch.eye(3)])
    assert calls == [torch.float32]


def test_soap_preserves_explicit_parameter_groups():
    model = torch.nn.Linear(3, 2)
    optimizer = build_optimizer(model, kind="soap", lr=1e-3, parameter_blocks=[
        {"params": [model.weight], "lr": 1e-3}, {"params": [model.bias], "lr": 2e-3}])
    assert [g["lr"] for g in optimizer.param_groups] == [1e-3, 2e-3]
    with pytest.raises(ValueError, match="float32"):
        build_optimizer(model.double(), kind="soap", lr=1e-3)


def test_lightning_soap_checkpoint_matches_uninterrupted_training(tmp_path):
    from pytorch_lightning import LightningModule, Trainer
    from torch.utils.data import DataLoader
    from prom3theus.config.joint_schema import JointTrainingConfig
    from prom3theus.training.joint import JointInversionModule

    class TinyInversion(LightningModule):
        configure_optimizers = JointInversionModule.configure_optimizers
        learning_rate_factor = JointInversionModule.learning_rate_factor

        def __init__(self):
            super().__init__()
            self.network = torch.nn.Linear(3, 2)
            self.settings = JointTrainingConfig(max_steps=14, learning_rate=1e-3,
                final_learning_rate=1e-4, learning_rate_decay_steps=14, optimizer="soap")

        def training_step(self, batch, batch_idx):
            return self.network(batch).square().mean()

    loader = DataLoader(torch.ones(28, 3), batch_size=2)
    options = dict(accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, gradient_clip_val=.1,
        default_root_dir=tmp_path)
    model = TinyInversion()
    initial = deepcopy(model.state_dict())
    trainer = Trainer(max_steps=6, **options)
    trainer.fit(model, loader)
    checkpoint = tmp_path / "soap.ckpt"
    trainer.save_checkpoint(checkpoint)
    resumed = TinyInversion()
    Trainer(max_steps=14, **options).fit(resumed, loader, ckpt_path=checkpoint)
    uninterrupted = TinyInversion()
    uninterrupted.load_state_dict(initial)
    Trainer(max_steps=14, **options).fit(uninterrupted, loader)
    for actual, expected in zip(resumed.parameters(), uninterrupted.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rejects_previous_external_soap_state():
    optimizer = SOAP([torch.nn.Parameter(torch.ones(3))])
    state = optimizer.state_dict()
    state["param_groups"][0]["max_precondition_dim"] = state["param_groups"][0].pop("max_precond_dim")
    with pytest.raises(ValueError, match="implementation changed"):
        optimizer.load_state_dict(state)
