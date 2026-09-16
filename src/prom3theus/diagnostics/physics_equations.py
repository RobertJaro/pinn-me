"""Height–latitude decomposition of enabled volume objectives."""
from pathlib import Path
import math
import json

import numpy as np
import torch

from .providers import DiagnosticOutput


def equation_figure(latitude_deg, height_mm, equation, data, *, longitude_deg, time_hours):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    def color_norm(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        positive = finite[finite > 0.0]
        if not positive.size:
            return Normalize(vmin=0.0, vmax=1.0), values
        vmax = float(np.percentile(positive, 99.5))
        vmax = max(vmax, float(positive.min()))
        vmin = max(float(positive.min()), vmax * 1.0e-6)
        if vmax <= vmin:
            vmax = vmin * 10.0
        return LogNorm(vmin=vmin, vmax=vmax), np.ma.masked_less_equal(values, 0.0)

    raw_terms = list(data['terms'].items())
    normalized_terms = list(data['normalized_terms'].items())
    if len(normalized_terms) != len(raw_terms):
        raise ValueError(
            "Physics diagnostics must provide one normalized value for every "
            "raw term."
        )
    # The normalized diagnostic labels are not required to match the raw
    # labels.  For example, the raw current-free term is ``∇×B_hat`` while
    # its normalized counterpart is currently labelled ``normalized
    # residual``.  Pair terms by insertion order, which is how the constraint
    # recorder constructs both mappings, and use the raw label for the shared
    # column title.
    term_count = len(raw_terms)
    fig, axes = plt.subplots(2, 1 + term_count, figsize=(4.3 * (1 + term_count), 7), squeeze=False, layout='constrained')
    shape = (len(height_mm), len(latitude_deg))
    for row, prefix in enumerate(('', 'normalized_')):
        residual = np.asarray(data[prefix + 'residual'])
        residual_magnitude = (
            np.abs(residual)
            if residual.ndim == 1
            else np.linalg.norm(residual, axis=-1)
        )
        term_items = raw_terms if not prefix else normalized_terms
        panels = [('|residual|', residual_magnitude)]
        for (raw_name, _), (_, term_values_for_row) in zip(raw_terms, term_items):
            values = np.asarray(term_values_for_row)
            magnitude = np.abs(values) if values.ndim == 1 else np.linalg.norm(values, axis=-1)
            panels.append((f'|{raw_name}|', magnitude))
        term_values = np.concatenate([values.ravel() for _, values in panels[1:]])
        for col, (name, values) in enumerate(panels):
            ax = axes[row, col]
            values = values.reshape(shape)
            if not np.isfinite(values).all():
                raise FloatingPointError(f'Nonfinite physics plot values: {equation}/{name}')
            scale_values = values if col == 0 else term_values
            norm, plotted = color_norm(scale_values)
            if col != 0:
                plotted = np.ma.masked_less_equal(values, 0.0)
            mesh = ax.pcolormesh(latitude_deg, height_mm, plotted, shading='auto', cmap='magma', norm=norm)
            unit = 'dimensionless' if row else data['units']
            fig.colorbar(mesh, ax=ax, label=unit)
            ax.set_title(name)
            ax.set_xlabel('Latitude [deg]')
            ax.set_yscale('linear')
            if col == 0:
                base_label = (
                    'Model units'
                    if data['units'].startswith('dimensionless')
                    else 'Unnormalized'
                )
                ax.set_ylabel((base_label if row == 0 else 'Normalized (loss)') + '\nHeight [Mm]')
    title = {'magnetohydrostatic_equilibrium': 'MHS', 'hydrostatic_equilibrium': 'HSE',
             'magnetic_divergence': 'div B'}.get(equation, equation.replace('_', ' '))
    raw_residual = np.asarray(data['residual'])
    normalized_residual = np.asarray(data['normalized_residual'])
    raw_mse = float(np.mean(raw_residual**2))
    normalized_mse = float(np.mean(normalized_residual**2))
    robust_delta = float(data.get("robust_loss_delta", 0.0))
    objective = (
        f"pseudo-Huber δ={robust_delta:g}" if robust_delta > 0.0 else "MSE"
    )
    fig.suptitle(
        f'{title} — longitude {longitude_deg:g}°; t={time_hours:g} h; '
        f'Panels: residual/term magnitudes. Diagnostic normalized MSE: {normalized_mse:g}; '
        f'objective: {objective}; '
        f'model/raw residual MSE: {raw_mse:g} ({data["units"]})².\n'
        'Linear height and logarithmic color scales; color limits use the 99.5th percentile.'
    )
    return fig


def evaluate_equation_slice(model, constraints, domain, *, longitude_deg, latitude_points, height_points, batch_size):
    """Keep complete latitude rows together so height-local scales are exact."""
    parameter = next(model.parameters())
    lat = torch.linspace(*domain.latitude_bounds_rad, latitude_points, device=parameter.device)
    latitude_fraction = (lat.sin() - math.sin(domain.latitude_bounds_rad[0])) / (
        math.sin(domain.latitude_bounds_rad[1]) - math.sin(domain.latitude_bounds_rad[0])
    )
    fraction = torch.linspace(0, 1, height_points, device=parameter.device)
    height = domain.height_bounds_Mm[0] + fraction * (domain.height_bounds_Mm[1] - domain.height_bounds_Mm[0])
    lon_min = domain.longitude_center_rad + domain.longitude_offset_bounds_rad[0]
    lon_width = domain.longitude_offset_bounds_rad[1] - domain.longitude_offset_bounds_rad[0]
    longitude = domain.longitude_center_rad + math.remainder(math.radians(longitude_deg) - domain.longitude_center_rad, 2 * math.pi)
    u = (longitude - lon_min) / lon_width
    if not 0 <= u <= 1:
        raise ValueError('Physics slice longitude is outside the observed domain')
    time = sum(domain.time_bounds_hours) / 2
    chunk_heights = max(1, batch_size // latitude_points)
    collected = {}
    losses = {}
    with torch.inference_mode(False), torch.enable_grad():
        for begin in range(0, height_points, chunk_heights):
            h = height[begin:begin + chunk_heights]
            hh, vv = torch.meshgrid(h, latitude_fraction, indexing='ij')
            positions = domain._positions(hh.flatten(), torch.full_like(hh.flatten(), u), vv.flatten(), torch.full_like(hh.flatten(), .5), device=parameter.device)['position_m']
            result = constraints.volume(model, positions, torch.full((len(positions), 1), time, device=parameter.device), height_group_shape=(len(h), latitude_points), create_graph=False, return_diagnostics=True)
            for name, data in result.equation_diagnostics.items():
                dest = collected.setdefault(name, {'units': data['units'], 'weight': data['weight'], 'robust_loss_delta': data.get('robust_loss_delta', 0.0), 'terms': {}, 'normalized_terms': {}, 'residual': [], 'normalized_residual': []})
                for key in ('terms', 'normalized_terms'):
                    for term, values in data[key].items():
                        dest[key].setdefault(term, []).append(values.cpu().numpy())
                for key in ('residual', 'normalized_residual'):
                    dest[key].append(data[key].cpu().numpy())
                losses[name] = losses.get(name, 0.) + float(result.losses[name].detach()) * len(h) / height_points
    for data in collected.values():
        for key in ('terms', 'normalized_terms'):
            data[key] = {name: np.concatenate(parts) for name, parts in data[key].items()}
        for key in ('residual', 'normalized_residual'):
            data[key] = np.concatenate(data[key])
    return lat.cpu().numpy() * 180 / np.pi, height.cpu().numpy(), collected, losses, time


def _height_statistics(values, height_mm, latitude_count):
    """Summarize one residual/term at every sampled height."""

    values = np.asarray(values)
    magnitudes = (
        np.abs(values)
        if values.ndim == 1
        else np.linalg.norm(values, axis=-1)
    )
    height_count = len(height_mm)
    magnitudes = magnitudes.reshape(height_count, latitude_count)
    return [
        {
            "height_Mm": float(height_mm[index]),
            "mean": float(np.mean(row)),
            "rms": float(np.sqrt(np.mean(row**2))),
            "p50": float(np.percentile(row, 50)),
            "p90": float(np.percentile(row, 90)),
            "p95": float(np.percentile(row, 95)),
            "p99": float(np.percentile(row, 99)),
            "max": float(np.max(row)),
        }
        for index, row in enumerate(magnitudes)
    ]


def render_physics_equations(runtime, trainer, output):
    from prom3theus.inversion.data_terms.regularization import PhysicsConstraintTerm
    options = runtime.config.diagnostics.visualization
    if not (options.enabled and options.render_atmosphere and options.meridional_slice.enabled):
        return DiagnosticOutput({'enabled': False})
    import matplotlib.pyplot as plt
    paths, report = [], {}
    for name, term in runtime.model.shared_objectives.items():
        if not isinstance(term, PhysicsConstraintTerm) or not term.constraints.volume_active:
            continue
        sampling = options.slice_sampling
        lat, height, equations, losses, time = evaluate_equation_slice(
            runtime.model.atmosphere_model, term.constraints, term.sampling_domain,
            longitude_deg=options.meridional_slice.longitude_deg,
            latitude_points=sampling.latitude_points, height_points=sampling.radial_points,
            batch_size=sampling.batch_size,
        )
        directory = Path(output) / 'physics_equations'
        directory.mkdir(parents=True, exist_ok=True)
        for equation, data in equations.items():
            if equation == 'magnetic_current_free':
                data['weight'] *= term._magnetic_current_free_factor()
                if data['weight'] == 0:
                    continue
            path = directory / f'{name}_{equation}_step{trainer.global_step}.png'
            fig = equation_figure(lat, height, equation, data, longitude_deg=options.meridional_slice.longitude_deg, time_hours=time)
            try:
                fig.savefig(path, dpi=options.dpi)
            finally:
                plt.close(fig)
            paths.append(str(path))
            np.savez_compressed(path.with_suffix('.npz'), latitude_deg=lat, height_Mm=height,
                                residual=data['residual'], normalized_residual=data['normalized_residual'],
                                **{f'raw_{i}': value for i, value in enumerate(data['terms'].values())},
                                **{f'normalized_{i}': value for i, value in enumerate(data['normalized_terms'].values())})
            report[equation] = {
                'path': str(path),
                'unweighted_slice_loss': losses[equation],
                'weighted_slice_loss': losses[equation] * max(float(data['weight']), 0.0),
                'weight': data['weight'],
                'robust_loss_delta': data.get('robust_loss_delta', 0.0),
                'objective': (
                    'pseudo_huber' if float(data.get('robust_loss_delta', 0.0)) > 0.0
                    else 'mse'
                ),
                'terms': list(data['terms']),
                'raw_units': data['units'],
                'normalized_mse': float(
                    np.mean(np.asarray(data['normalized_residual']) ** 2)
                ),
                'weighted_normalized_mse': float(
                    max(float(data['weight']), 0.0)
                    * np.mean(np.asarray(data['normalized_residual']) ** 2)
                ),
                'raw_rms': float(np.sqrt(np.mean(np.asarray(data['residual']) ** 2))),
                'normalized_rms': float(
                    np.sqrt(np.mean(np.asarray(data['normalized_residual']) ** 2))
                ),
                'weighted_rms': float(
                    np.sqrt(
                        max(float(data['weight']), 0.0)
                        * np.mean(np.asarray(data['normalized_residual']) ** 2)
                    )
                ),
                'per_height': {
                    'residual': _height_statistics(data['residual'], height, len(lat)),
                    'normalized_residual': _height_statistics(
                        data['normalized_residual'], height, len(lat)
                    ),
                    'weighted_normalized_residual': _height_statistics(
                        np.asarray(data['normalized_residual'])
                        * np.sqrt(max(float(data['weight']), 0.0)),
                        height,
                        len(lat),
                    ),
                    'terms': {
                        term_name: _height_statistics(values, height, len(lat))
                        for term_name, values in data['terms'].items()
                    },
                    'normalized_terms': {
                        term_name: _height_statistics(values, height, len(lat))
                        for term_name, values in data['normalized_terms'].items()
                    },
                },
            }
        (directory / f'{name}_step{trainer.global_step}.json').write_text(json.dumps(report, indent=2))
    return DiagnosticOutput(report, tuple(paths), media_groups={f'Physics equation {Path(path).stem.split("_step")[0]}': (path,) for path in paths})
