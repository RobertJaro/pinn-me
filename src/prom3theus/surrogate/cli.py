"""Train an LTE coefficient surrogate on MURaM atmospheric states."""

import argparse
from dataclasses import fields
from pathlib import Path

from .config import Perturbations, TrainingConfig
from .training import train


def parse_options(argv=None) -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS
    )
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--iteration", type=int)
    parser.add_argument(
        "--config",
        dest="window_config",
        type=Path,
        help="Surrogate run YAML with snapshot, output directory, spectral window and training settings",
    )
    parser.add_argument(
        "--window", help="Window ID when the inversion defines multiple windows"
    )
    parser.add_argument(
        "--instrument-config",
        type=Path,
        help="Optional synthesis YAML with detector response",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--vertical-axis", type=int, choices=(0, 1, 2))
    parser.add_argument("--component-order", nargs=3, type=int)
    parser.add_argument("--device", help="Override automatic Lightning device selection")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--knots", type=int)
    parser.add_argument("--architecture", choices=("silu", "tanh", "siren"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--hidden-layers", type=int)
    parser.add_argument("--wavelength-count", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--ray-depths", type=int)
    parser.add_argument("--log-tau", type=float, nargs=2)
    perturbation_names = [item.name for item in fields(Perturbations)]
    for name in perturbation_names:
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--resume", action="store_true")
    values = vars(parser.parse_args(argv))
    perturbations = {
        name: values.pop(name) for name in perturbation_names if name in values
    }
    try:
        return TrainingConfig(**values, perturbations=perturbations)
    except ValueError as error:
        parser.error(str(error))


def main(argv=None):
    config = parse_options(argv)
    train(config)
