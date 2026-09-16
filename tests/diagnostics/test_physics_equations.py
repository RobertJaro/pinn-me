import numpy as np
import torch

from prom3theus.diagnostics.physics_equations import evaluate_equation_slice, equation_figure
from prom3theus.inversion.constraints.magnetofluid import MagnetofluidConstraints
from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.rt import StratifiedAtmosphereModel


def test_slice_preserves_height_groups_and_reconstructs_loss_across_batches():
    model = StratifiedAtmosphereModel(shell_height_bounds_Mm=(1.5, -.1),
        scene_geometry_config={'solar_radius_m': 696e6, 'scene_basis': torch.eye(3)},
        model_config={'type': 'mlp', 'dim': 8, 'n_layers': 2, 'activation': 'silu', 'encoding_config': {'type': 'identity'}})
    domain = SphericalShellDomain(0., (-.04, .04), (-.04, .04), (0., 2.), (-.1, 1.5), 696e6)
    constraints = MagnetofluidConstraints({'magnetic_divergence': {'enabled': True, 'weight': .001}}, vector_basis_matches_spatial_coordinates=True)
    outputs = [evaluate_equation_slice(model, constraints, domain, longitude_deg=0., latitude_points=5, height_points=4, batch_size=batch) for batch in (5, 20)]
    for lat, height, data, losses, time in outputs:
        d = data['magnetic_divergence']
        np.testing.assert_allclose(np.mean(d['normalized_residual']**2), losses['magnetic_divergence'], rtol=1e-5)
        assert len(lat) == 5 and len(height) == 4 and time == 1.
        np.testing.assert_allclose(height[[0,-1]], [-.1, 1.5])
        assert all(p.grad is None for p in model.parameters())
    np.testing.assert_allclose(outputs[0][2]['magnetic_divergence']['normalized_residual'], outputs[1][2]['magnetic_divergence']['normalized_residual'], rtol=1e-5, atol=1e-7)


def test_equation_plot_has_total_first_and_two_rows_including_zero_terms():
    import matplotlib.pyplot as plt
    raw = np.zeros((12, 3))
    data = {'weight': .1, 'units': 'Pa/m', 'residual': raw, 'normalized_residual': raw,
            'terms': {'pressure': raw, 'gravity': raw}, 'normalized_terms': {'pressure': raw, 'gravity': raw}}
    fig = equation_figure(np.linspace(-10,10,3), np.linspace(0,50,4), 'MHS', data, longitude_deg=0, time_hours=0)
    try:
        axes = fig.axes[:6]
        assert [a.get_title() for a in axes] == ['|residual|','|pressure|','|gravity|'] * 2
        assert 'Unnormalized' in axes[0].get_ylabel()
        assert 'Normalized' in axes[3].get_ylabel()
        assert axes[0].collections[0].colorbar.ax.get_ylabel() == 'Pa/m'
        assert axes[3].collections[0].colorbar.ax.get_ylabel() == 'dimensionless'
    finally:
        plt.close(fig)


def test_equation_plot_pairs_raw_and_normalized_terms_by_order():
    import matplotlib.pyplot as plt

    raw = np.zeros((12, 3))
    data = {
        'weight': .1,
        'units': 'G/m',
        'residual': raw,
        'normalized_residual': raw,
        'terms': {'∇×B_hat': raw},
        'normalized_terms': {'normalized residual': raw},
    }
    fig = equation_figure(
        np.linspace(-10, 10, 3),
        np.linspace(0, 50, 4),
        'magnetic_current_free',
        data,
        longitude_deg=0,
        time_hours=0,
    )
    try:
        axes = fig.axes[:4]
        assert [axis.get_title() for axis in axes] == [
            '|residual|',
            '|∇×B_hat|',
            '|residual|',
            '|∇×B_hat|',
        ]
    finally:
        plt.close(fig)
