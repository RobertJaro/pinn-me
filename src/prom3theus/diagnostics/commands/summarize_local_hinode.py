"""Plot fixed-sample learning curves from retained local validation reports."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout='constrained')
    summary = {}
    for run in args.runs:
        rows = []
        for path in sorted((run / 'work' / 'validation').glob('step_*/metrics.json')):
            report = json.loads(path.read_text())
            metrics = report['metrics']
            rows.append(dict(step=report['global_step'], **{
                k: metrics[f'streams.photospheric_stokes.stokes_{k}_raw_residual_rms']
                for k in 'IQUV'}))
        if not rows:
            raise ValueError(f'No validation reports in {run}')
        summary[run.name] = {'history': rows}
        for ax, component in zip(axes, 'IQUV'):
            ax.semilogy([r['step'] for r in rows], [r[component] for r in rows],
                        marker='.', label=run.name.removeprefix('local_hinode_'))
            ax.set(title=component, xlabel='Optimizer updates', ylabel='Raw RMS error (atlas-continuum units)')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed-sample checks from the training frame; not held-out data')
    fig.savefig(args.output / 'learning_curves.png', dpi=150)
    plt.close(fig)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
