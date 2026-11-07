"""Run the production trainer locally with CSV metrics and parameter-change checks.

PYTHONPATH=src python scripts/diagnostics/run_local_hinode.py CONFIG.yaml
The supplied configuration controls the data, optimizer, and epoch count.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import shutil
from unittest.mock import patch

import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from prom3theus.muram import config as _register_muram_schema
from prom3theus.config import load_config
from prom3theus.application import joint_training


class LocalCSVLogger(CSVLogger):
    def log_image(self, *, key, images, caption=None, step=None):
        # Renderers already persist the images locally; record their locations.
        path = Path(self.log_dir) / 'images.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps(dict(key=key, images=images, caption=caption,
                                         step=step)) + '\n')


class LocalAudit(Callback):
    def on_train_start(self, trainer, module):
        self.initial = {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.model.atmosphere_model.named_parameters()
        }
        for callback in trainer.callbacks:
            if isinstance(callback, joint_training.JointTrainingDiagnostics):
                callback._evaluate(trainer, module)

    def on_train_end(self, trainer, module):
        square_sum, count, changed = 0., 0, 0
        for name, parameter in module.model.atmosphere_model.named_parameters():
            delta = parameter.detach().cpu() - self.initial[name]
            square_sum += float(delta.double().square().sum())
            count += delta.numel()
            changed += int(torch.count_nonzero(delta))
        changes = {
            'atmosphere_model': {
                'parameter_count': count,
                'changed_parameter_count': changed,
                'rms_parameter_change': (square_sum / max(count, 1)) ** 0.5,
            }
        }
        output = Path(trainer.default_root_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'parameter_changes.json').write_text(json.dumps(changes, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_everything(42, workers=True)
    config = load_config(args.configuration)
    output = config.solver.output_directory

    def local_trainer(**kwargs):
        kwargs['callbacks'].append(LocalAudit())
        kwargs['enable_progress_bar'] = False
        kwargs['default_root_dir'] = str(output)
        return Trainer(**kwargs)

    # CSVLogger replaces an existing run directory. Retain earlier segments
    # when the production trainer resumes its checkpoint.
    log_directory = output / 'metrics' / 'local'
    if (log_directory / 'metrics.csv').exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        shutil.copytree(log_directory, output / 'metrics' / 'history' / stamp)
    logger = LocalCSVLogger(str(output), name='metrics', version='local')
    with patch.object(joint_training, 'Trainer', local_trainer), patch.object(
        joint_training, '_online_logger', lambda _: logger
    ):
        joint_training.run_joint_inversion(config)


if __name__ == '__main__':
    main()
