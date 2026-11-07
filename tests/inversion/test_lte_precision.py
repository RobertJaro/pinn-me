"""Production precision from atmospheric sampling through detector profiles."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from prom3theus.inversion.data_terms.stokes import StokesObservationTerm
from prom3theus.inversion.lte import LTEWindow
from prom3theus.rt import StratifiedAtmosphereModel


def make_precision_system(*, dtype=torch.float32, magnetic_representation="direct", depth_count=33, model_config=None, coarse_to_fine=None):
    atmosphere = StratifiedAtmosphereModel(
        shell_height_bounds_megameter=(1.5, -1.),
        magnetic_representation=magnetic_representation,
        spatial_coordinate_scale_megameter=(10., 10.),
        scene_geometry_config={"solar_radius_m": 695700000., "scene_basis": torch.eye(3)},
        model_config=model_config or {"type": "siren", "dim": 8, "n_layers": 1,
                      "first_omega_0": 1., "hidden_omega_0": 1., "radial_weighting_config": None},
    ).to(dtype=dtype)
    window = LTEWindow(atmosphere, configuration={
        "id": "fe", "line_ids": ["FeI_6302.4932"], "radiance_scale": 1e13,
        "wavelength_bounds_vacuum_angstrom": [6300., 6307.],
        "depth_count": depth_count, "coarse_to_fine": coarse_to_fine, "jitter_fraction": 1., "wavelength_chunk_size": 8,
        "transfer_precision": "float32" if dtype == torch.float32 else "float64",
    })
    term = StokesObservationTerm(
        atmosphere_model=atmosphere, synthesis_window=window, synthesis_window_id="fe",
        observation_id="precision", wavelength_angstrom=torch.linspace(6302.2, 6302.8, 7, dtype=torch.float64),
        continuum_indices=[0, 6], atlas_continuum_radiance_w_m3_sr=1e13,
        velocity_synthesis_mode="carrington_registered_relative",
        instrument_config={"type": "hinode_sp", "spectral_psf": {
            "type": "gaussian", "fwhm_angstrom": .025, "oversample": 1, "truncate_sigma": 4.}},
        objective_config={"type": "mse", "continuum_normalization": False},
        weight_config={"I": 1., "Q": 1., "U": 1., "V": 1.},
        optimize_instrument_line_of_sight_velocity_correction=True,
    ).to(dtype=dtype)
    batch = {
        "coordinates": torch.tensor([[0., 0., 0.], [1., -2., .5]], dtype=dtype),
        "ray_direction": torch.tensor([[0., 0., -1.]], dtype=dtype).expand(2, -1),
        "stokes_basis": torch.eye(3, dtype=dtype).expand(2, -1, -1),
        "removed_solar_los_velocity_m_per_s": torch.tensor([-1000., 2000.], dtype=dtype),
        "stokes": torch.zeros(2, 4, 7, dtype=dtype),
    }
    return atmosphere, window, term, batch


class Float64Operations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = set()

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if any(isinstance(value, torch.Tensor) and value.dtype == torch.float64
               for value in tree_leaves(result)):
            self.operations.add(str(function))
        return result


@pytest.mark.parametrize("magnetic_representation", ["direct", "vector_potential"])
def test_native_float32_forward_and_backward_have_no_float64_arithmetic(magnetic_representation):
    torch.manual_seed(42)
    atmosphere, _, term, batch = make_precision_system(magnetic_representation=magnetic_representation)
    audit = Float64Operations()
    with audit:
        result = term.evaluate_training_batch(batch)
        result.likelihood_loss.backward()
    assert not audit.operations
    gradients = [parameter.grad for parameter in atmosphere.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    correction_gradient = term.instrument_line_of_sight_velocity_correction_normalized.grad
    assert correction_gradient is not None and torch.isfinite(correction_gradient)


def test_full_float32_pipeline_matches_double_profiles_and_training_gradients():
    torch.manual_seed(42)
    atmosphere, window, term, batch = make_precision_system(depth_count=257)
    with torch.no_grad():
        # Resolve all polarization components and a nonzero atmospheric velocity.
        atmosphere.network.out_layer.weight[1:7].zero_()
        atmosphere.network.out_layer.bias[1:4].copy_(torch.tensor([.2, -.3, .9]))
        atmosphere.network.out_layer.bias[4:7].copy_(torch.tensor([6., -8., 12.]))
    double_atmosphere, double_window, double_term, double_batch = make_precision_system(
        dtype=torch.float64, depth_count=257)
    double_atmosphere.network.load_state_dict(atmosphere.network.state_dict())
    values, gradients = [], []
    for model, synthesis, observation, inputs in (
        (atmosphere, window, term, batch),
        (double_atmosphere, double_window, double_term, double_batch),
    ):
        model.eval()
        synthesis.eval()
        observation.eval()
        result = observation.predict(inputs)
        parameters = (*model.network.parameters(),
                      observation.instrument_line_of_sight_velocity_correction_normalized)
        weights = result.new_tensor([1., 1000., 1000., 100.])[None, :, None]
        gradient = torch.autograd.grad((result.square() * weights).mean(), parameters)
        values.append(result.detach().double())
        gradients.append(torch.cat([value.reshape(-1) for value in gradient]).double())
    actual, expected = values
    assert torch.all(expected.abs().amax(dim=(0, 2)) > .01)
    component_errors = (actual - expected).norm(dim=(0, 2)) / expected.norm(dim=(0, 2))
    assert torch.all(component_errors < 3e-5)
    actual_gradient, expected_gradient = gradients
    assert (actual_gradient - expected_gradient).norm() < 3e-5 * expected_gradient.norm()
    # Wavelength calibration is especially sensitive to loss of line offsets.
    assert (actual_gradient[-1] - expected_gradient[-1]).abs() < 1e-4 * expected_gradient[-1].abs()


@pytest.mark.parametrize("magnetic_representation", ["direct", "vector_potential"])
@pytest.mark.parametrize("optimizer_kind", ["adam", "soap"])
@pytest.mark.parametrize("refine", [False, True])
def test_isotropic_stokes_training_uses_one_network_pass_in_float32(magnetic_representation, optimizer_kind, refine):
    from prom3theus.training.optimizers import build_optimizer
    torch.manual_seed(42)
    atmosphere, window, term, batch = make_precision_system(
        magnetic_representation=magnetic_representation, depth_count=32 if refine else 257,
        coarse_to_fine={"enabled": refine, "fine_sample_count": 32, "uniform_weight_floor": .05},
        model_config={"type": "mlp", "dim": 16, "n_layers": 2, "activation": "silu",
                      "encoding_config": {"type": "positional_isotropic", "num_frequencies": [32],
                                          "max_frequencies": [8.], "seed": 0}})
    from prom3theus.inversion.objective import StokesObjective
    term.objective = StokesObjective(continuum_normalization=True,
        stokes_scales={"I": 1., "Q": .1, "U": .1, "V": .1})
    batch["stokes"][:, 0] = .8
    calls = []
    handle = atmosphere.network.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape))
    optimizer = build_optimizer(atmosphere, kind=optimizer_kind, lr=1e-4)
    before = atmosphere.network.out_layer.weight.detach().clone()
    audit = Float64Operations()
    try:
        with audit:
            for _ in range(2):  # SOAP initializes its basis on the first step.
                optimizer.zero_grad(set_to_none=True)
                loss = term.evaluate_training_batch(batch).likelihood_loss
                loss.backward()
                optimizer.step()
    finally:
        handle.remove()
    assert torch.isfinite(loss)
    assert len(calls) == (4 if refine else 2)
    assert not audit.operations
    gradients = [p.grad for p in atmosphere.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max() > 0 for g in gradients)
    assert not torch.equal(before, atmosphere.network.out_layer.weight)

