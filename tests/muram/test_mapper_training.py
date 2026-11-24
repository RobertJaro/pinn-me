"""Checks the standalone trainer's state on interruption and spatial partitioning."""

import json
from copy import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from prom3theus.surrogate import evaluation
from prom3theus.surrogate.cli import main, parse_options
from prom3theus.surrogate.config import Perturbations, TrainingConfig
from prom3theus.surrogate.sampling import SnapshotPoints, _jitter_states, load_snapshot_splits
from prom3theus.muram.reader import MURaMSnapshot
from tests.muram.test_column_loading import _write_snapshot


def test_online_training_matches_resumed_run(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    window = yaml.safe_load((root / "configs/muram/hinode/surrogate.yaml").read_text())
    config_path = tmp_path / "surrogate.yaml"
    detector = root / "configs/muram/hinode/synthesis.yaml"
    (tmp_path / "synthesis.yaml").write_bytes(detector.read_bytes())
    monkeypatch.setattr(evaluation, "save_profile_figure", lambda *args: None)
    source = tmp_path / "snapshot"
    _write_snapshot(source, (20, 9, 8), 2, physical=True)
    columns = MURaMSnapshot(str(source), 10).synthesis_columns()
    regions = []
    for split in ("train", "validation", "test"):
        ds = SnapshotPoints(
            columns,
            split=split,
            seed=42,
            batch_size=8,
            steps=2,
            perturbations=Perturbations(),
            log_tau=(-6, 2),
        )
        regions.append(set(range(*ds.region)))
        torch.testing.assert_close(ds[3], ds[3])
    assert not any(a & b for i, a in enumerate(regions) for b in regions[i + 1 :])

    def train(folder, steps, resume=False):
        out = tmp_path / folder
        config_path.write_text(yaml.safe_dump({
            **window, "knots": 12, "wavelength_count": 16,
            "workers": 0, "batch_size": 8, "steps": steps,
            "learning_rate_decay_steps": 2,
            "replay_buffer_size": 10,
            "source_path": "snapshot", "iteration": 10, "output_dir": folder,
        }))
        main(
            [
                "--config",
                str(config_path),
                "--device",
                "cpu",
                "--width",
                "8",
                "--ray-depths",
                "4",
                "--validate-every",
                "1",
                "--log-every",
                "1",
                *(["--resume"] if resume else []),
            ],
        )
        return torch.load(out / "last.pt", weights_only=True)

    full = train("complete", 4)
    from prom3theus.rt.coefficient_mapper import CoefficientMapper

    selected = CoefficientMapper.load(tmp_path / "complete/best.pt")
    assert sum(isinstance(layer, torch.nn.SiLU) for layer in selected.network) == 4
    assert selected.spec["spectral_frame"] == "rest"
    assert selected.spec["linear_residual"] is True
    assert selected.input_center.numel() == 5
    assert selected.linear_scale[0] == 1
    assert torch.equal(selected.linear_scale[[2, 4]], selected.relative_scale[[2, 5]])
    assert full["contract"]["spec"]["knots"] == 12
    assert full["contract"]["wavelength_count"] == 16
    assert full["contract"]["batch_size"] == 8
    assert full["contract"]["perturbations"]["magnetic_field_fraction"] == 0.10
    assert full["step"] == 4
    assert full["contract"]["optimizer"] == "soap"
    assert full["contract"]["replay"] == {"fraction": 0.5, "capacity": 10}
    assert len(full["replay"]["points"]) == 10
    assert full["contract"]["wavelength_jitter_fraction"] == window["wavelength_jitter_fraction"] > 0
    assert torch.all(full["replay"]["offsets"] != 0)
    assert "max_precond_dim" in full["optimizer"]["param_groups"][0]
    train("restart", 2)
    resumed = train("restart", 4, True)
    for folder in ("complete", "restart"):
        records = (json.loads(line) for line in (tmp_path / folder / "metrics.jsonl").read_text().splitlines())
        rates = [record["learning_rate"] for record in records if record["stage"] == "train"]
        assert rates == pytest.approx([0.001, 0.001 * 0.1**0.5, 0.0001, 0.0001])
    for name in full["mapper"]:
        torch.testing.assert_close(
            full["mapper"][name], resumed["mapper"][name], rtol=0, atol=0
        )
    for name in full["replay"]:
        torch.testing.assert_close(full["replay"][name], resumed["replay"][name], rtol=0, atol=0)
    assert "replay" not in torch.load(tmp_path / "complete/best.pt", weights_only=True)
    assert full["best_loss"] == resumed["best_loss"]
    assert full["selected"]["step"] == resumed["selected"]["step"]

    # A mixed-source checkpoint can contain synthetic points in its replay buffer.
    resumed["contract"]["sobol_sampling"] = {"fraction": 0.5}
    torch.save(resumed, tmp_path / "restart/last.pt")
    with pytest.raises(ValueError, match="identical snapshot, model, sampling"):
        train("restart", 5, True)


def test_training_points_have_snapshot_provenance_with_parallel_workers(tmp_path):
    source = tmp_path / "snapshot"
    _write_snapshot(source, (20, 9, 8), 2, physical=True)
    config = TrainingConfig(
        source_path=source, iteration=10, window_config=Path(__file__).resolve().parents[2]
        / "configs/muram/hinode/surrogate.yaml",
        batch_size=17, steps=4, log_tau=(-6.1, 2.1),
    )
    _, datasets = load_snapshot_splits(config)
    train = datasets["train"]
    native = copy(train)
    native.perturbations = Perturbations()
    loader = torch.utils.data.DataLoader(train, batch_size=None, num_workers=4, timeout=60)
    for step, batch in enumerate(loader):
        described, ids = train.sample_batch(step, describe=True)
        torch.testing.assert_close(batch, described, rtol=0, atol=0)
        assert batch.shape == (config.batch_size, 7)
        assert torch.isfinite(batch).all()
        assert (ids["column"] >= 0).all() and (ids["depth"] >= 0).all()
        assert ids["log_tau"].isfinite().all()
        baseline = native[step]
        assert (batch != baseline).any(dim=0).all()
        assert (batch[:, :2] > 0).all()
        assert torch.equal(batch[:, 2] > 0, ids["perturbed"])
        assert ids["perturbed"].all()


def test_jitter_config_and_per_parameter_gaussian_draws(tmp_path):
    root = Path(__file__).resolve().parents[2] / "configs/muram/hinode"
    window = yaml.safe_load((root / "surrogate.yaml").read_text())
    window["instrument_config"] = str(root / "synthesis.yaml")
    window["jitter"].update(log_temperature=0.05, log_pressure=0.2,
                           microturbulence_m_s=1000)
    config = tmp_path / "surrogate.yaml"
    config.write_text(yaml.safe_dump(window))
    p = parse_options(["--config", str(config), "--velocity-m-s", "80",
                       "--magnetic-field-fraction", "0"]).perturbations
    assert p.velocity_m_s == 80
    native = np.tile([6200., 8000., 0., 300., -150., 80., 900.], (30000, 1))
    states = native.copy()
    mask = _jitter_states(states, np.random.default_rng(31), p)
    assert mask.all()
    delta = states[mask] - native[mask]
    delta[:, :2] = np.log(states[mask, :2] / native[mask, :2])
    np.testing.assert_array_equal(states[:, 4:], native[:, 4:])
    signed = [0, 1, 3]
    widths = [0.05, 0.2, 80.]
    standardized = delta[:, signed] / widths
    np.testing.assert_allclose(standardized.mean(0), 0, atol=0.025)
    np.testing.assert_allclose(standardized.std(0), 1, atol=0.025)
    # The speed at a native value of zero follows a half-normal distribution.
    assert (delta[:, 2] >= 0).all()
    assert delta[:, 2].mean() == pytest.approx(1000 * np.sqrt(2 / np.pi), rel=0.025)


def test_hinode_jitter_scales_with_native_field_magnitude():
    preset = Path(__file__).resolve().parents[2] / "configs/muram/hinode/surrogate.yaml"
    settings = parse_options(["--config", str(preset)]).perturbations
    native_fields = np.array([[0., 0., 0.], [0., -200., 0.], [600., 0., -800.]])
    draws = 30000
    fields = np.repeat(native_fields, draws, axis=0)
    states = np.column_stack((np.zeros((len(fields), 4)), fields))
    _jitter_states(states, np.random.default_rng(42), settings)
    np.testing.assert_array_equal(states[:draws, 4:], 0)
    nonzero = fields[draws:]
    scaled = states[draws:, 4:]
    factors = np.linalg.norm(scaled, axis=1) / np.linalg.norm(nonzero, axis=1)
    np.testing.assert_allclose(scaled, factors[:, None] * nonzero)
    assert np.all((factors >= 0.9) & (factors <= 1.1))
    np.testing.assert_allclose(np.quantile(factors, [0.1, 0.5, 0.9]), [0.92, 1.0, 1.08], atol=0.002)
