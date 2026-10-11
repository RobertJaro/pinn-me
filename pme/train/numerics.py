"""Failure-only numerical summaries; no retained graphs or per-batch copies."""
import json
from pathlib import Path

import torch


def _tensor_summary(value):
    value = value.detach()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    return {
        'shape': list(value.shape), 'dtype': str(value.dtype),
        'nonfinite': int((~finite).sum().cpu()),
        'nan': int(torch.isnan(value).sum().cpu()),
        'inf': int(torch.isinf(value).sum().cpu()),
        'min': float(finite_values.min().cpu()) if finite_values.numel() else None,
        'max': float(finite_values.max().cpu()) if finite_values.numel() else None,
    }


def _flatten(values, prefix=''):
    for key, value in values.items():
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict):
            yield from _flatten(value, name)
        elif torch.is_tensor(value):
            yield name, value


@torch.no_grad()
def write_numerical_report(module, stage, tensors=None):
    """Write one JSON per failing rank without any distributed collectives.

    A bad data batch may affect only one rank. Never synchronize/report metrics
    here: the other ranks could already be waiting inside backward.
    """
    trainer = getattr(module, '_trainer', None)
    rank = module.global_rank
    step = module.global_step
    report = {'stage': stage, 'rank': rank, 'global_step': step,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'compile_config': getattr(module, 'compile_config', {}),
              'optimizer_config': getattr(module, 'optimizer_config', {}),
              'tensors': {name: _tensor_summary(value) for name, value in _flatten(tensors or {})},
              'parameters': {}, 'gradients': {}}
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            report['parameters'][name] = _tensor_summary(parameter)
            if parameter.grad is not None:
                report['gradients'][name] = _tensor_summary(parameter.grad)
    bad = [f'{group}.{key}' for group in ('tensors', 'parameters', 'gradients')
           for key, summary in report[group].items() if summary['nonfinite']]
    report['nonfinite_tensors'] = bad
    checkpoint = getattr(trainer, 'checkpoint_callback', None)
    root = getattr(checkpoint, 'dirpath', None) or getattr(trainer, 'default_root_dir', '.')
    path = Path(root) / 'numerical_failures' / f'rank{rank}-step{step}-{stage}.json'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False))
        location = f'Report: {path}'
    except OSError as error:
        location = f'Could not save report: {error}'
    diagnosis = ', '.join(bad[:8]) if bad else 'all reported tensors finite; the reduced norm or objective overflowed'
    return f'Numerical failure during {stage}, rank {rank}, step {step}. {diagnosis}. {location}'
