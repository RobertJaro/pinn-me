"""End-to-end atmosphere gradients through polarized formal transfer."""
import pytest
import torch
from prom3theus.rt.polarized_exponential import propagation_matrix
from prom3theus.rt import StratifiedAtmosphereModel
from prom3theus.inversion.lte import LTEWindow
from prom3theus.inversion.data_terms.stokes import StokesObservationTerm
from prom3theus.inversion.joint import JointForwardModel

def make_system(magnetic_representation="direct", *, dtype=torch.float32, magnetic_reference_height=None, **window_options):
    atmosphere = StratifiedAtmosphereModel(
        shell_height_bounds_megameter=(1.5, -1.),
        magnetic_representation=magnetic_representation,
        magnetic_reference_height_megameter=magnetic_reference_height,
        spatial_coordinate_scale_megameter=(10., 10.),
        scene_geometry_config={"solar_radius_m": 695700000., "scene_basis": torch.eye(3)},
        model_config={"type": "siren", "dim": 8, "n_layers": 1,
                      "first_omega_0": 1., "hidden_omega_0": 1., "radial_weighting_config": None},
    ).to(dtype=dtype)
    config = dict(id="fe", line_ids=["FeI_6302.4932"],
                  wavelength_bounds_vacuum_angstrom=[6300.,6307.], radiance_scale=1e13,
                  depth_count=33)
    config.update(window_options)
    window = LTEWindow(atmosphere, configuration=config)
    term = StokesObservationTerm(
        atmosphere_model=atmosphere, synthesis_window=window, synthesis_window_id="fe",
        observation_id="test", wavelength_angstrom=torch.linspace(6302.2,6302.8,7,dtype=torch.float64),
        continuum_indices=[0,6], atlas_continuum_radiance_w_m3_sr=1e13,
        velocity_synthesis_mode="carrington_registered_relative",
        instrument_config={"type":"hinode_sp", "spectral_psf":{"type":"gaussian","fwhm_angstrom":.025,"oversample":1,"truncate_sigma":4.}},
        objective_config={"type":"mse", "continuum_normalization":False},
        weight_config={"I":1.,"Q":1.,"U":1.,"V":1.})
    batch={"coordinates":torch.zeros(2,3), "ray_direction":torch.tensor([[0.,0.,-1.]]).expand(2,-1),
           "stokes_basis":torch.eye(3).expand(2,-1,-1),
           "removed_solar_los_velocity_m_per_s":torch.zeros(2), "stokes":torch.zeros(2,4,7)}
    return atmosphere,window,term,batch


@pytest.mark.parametrize("magnetic_representation", ["direct", "vector_potential"])
@pytest.mark.parametrize("jitter_fraction", [0., 1.])
def test_observation_loss_directly_trains_atmosphere_without_radiation_parameters(magnetic_representation, jitter_fraction):
    atmosphere, window, term, batch = make_system(magnetic_representation, jitter_fraction=jitter_fraction)
    model = JointForwardModel(atmosphere, {"stokes": term}, {"stokes": 1.},
                              synthesis_windows={"fe": window})
    assert not list(window.parameters())
    assert not any("terms.stokes.synthesis_window" in key for key in model.state_dict())
    result = model.evaluate_batches({"stokes": batch})
    assert result.streams["stokes"].diagnostics["prediction"].shape == (2, 4, 7)
    assert not result.shared_terms
    assert result.loss.isfinite()
    result.loss.backward()
    gradients = [p.grad for p in atmosphere.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max() > 0 for g in gradients)


def test_batched_spectral_queries_agree_with_individual_rays():
    _, window, term, batch = make_system()
    window.eval()
    batch["removed_solar_los_velocity_m_per_s"] = torch.tensor([-1000., 1000.])
    with torch.no_grad():
        actual = term.predict(batch)
        expected = torch.cat([term.predict({k: v[i:i+1] for k, v in batch.items()}) for i in range(2)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_detector_spin_two_rotation():
    _, _, term, batch = make_system()
    with torch.no_grad():
        original = term.predict(batch)
        angle = .37
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        basis = torch.tensor([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]).expand(2, -1, -1)
        rotated = term.predict({**batch, "stokes_basis": basis})
    c2, s2 = torch.cos(torch.tensor(2*angle)), torch.sin(torch.tensor(2*angle))
    torch.testing.assert_close(rotated[:, 0], original[:, 0])
    torch.testing.assert_close(rotated[:, 3], original[:, 3])
    torch.testing.assert_close(rotated[:, 1], c2*original[:, 1]+s2*original[:, 2])
    torch.testing.assert_close(rotated[:, 2], -s2*original[:, 1]+c2*original[:, 2])


def test_one_optimizer_step_updates_atmosphere(monkeypatch):
    from prom3theus.config.joint_schema import JointTrainingConfig
    from prom3theus.training.joint import JointInversionModule
    atmosphere, window, term, batch = make_system(
        coarse_to_fine=dict(enabled=True, fine_sample_count=5, uniform_weight_floor=.05))
    from prom3theus.inversion.objective import StokesObjective
    term.objective = StokesObjective(stokes_scales={"I": 1., "Q": .1, "U": .1, "V": .1})
    model = JointForwardModel(atmosphere, {"stokes": term}, {"stokes": 1.}, synthesis_windows={"fe": window})
    module = JointInversionModule(model, JointTrainingConfig(max_steps=2, learning_rate=1e-3, final_learning_rate=1e-3, learning_rate_decay_steps=2), {})
    module.log = lambda *args, **kwargs: None
    before = [p.detach().clone() for p in atmosphere.parameters()]
    optimizer = module.configure_optimizers()["optimizer"]
    def reject_host_read(*args, **kwargs):
        pytest.fail("Training read a tensor on the host")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", reject_host_read)
        patch.setattr(torch.Tensor, "item", reject_host_read)
        patch.setattr(torch.Tensor, "__bool__", reject_host_read)
        loss = module.training_step({"stokes": batch}, 0)
        loss.backward()
        module.on_after_backward()
    optimizer.step()
    assert any(not torch.equal(p, old) for p, old in zip(atmosphere.parameters(), before))


def test_differentiable_instrument_wavelength_calibration():
    _, _, term, batch = make_system()
    term.instrument_line_of_sight_velocity_correction_normalized.requires_grad_(True)
    term.predict(batch).square().sum().backward()
    grad = term.instrument_line_of_sight_velocity_correction_normalized.grad
    assert grad is not None and torch.isfinite(grad) and grad != 0


@pytest.mark.parametrize("fine_count", [0, 5])
@pytest.mark.parametrize("fixed_magnetic_height", [False, True])
def test_full_ray_batch_and_spectrum_per_sampling_stage(monkeypatch, fine_count, fixed_magnetic_height):
    from unittest.mock import patch
    from prom3theus.rt.formal_transfer import formal_stokes

    refinement = dict(enabled=True, fine_sample_count=fine_count, uniform_weight_floor=.05)
    atmosphere, window, _, batch = make_system(
        depth_count=8, coarse_to_fine=refinement if fine_count else None,
        magnetic_reference_height=.15 if fixed_magnetic_height else None)
    geometry = window.boundary_queries(batch["coordinates"], batch["ray_direction"])
    ray_count = len(batch["coordinates"])
    wave = torch.linspace(6301., 6303., 71)
    network_points, coefficient_shapes = [], []
    evaluate = window.coefficients.evaluate_spectrum

    def record_spectrum(*args, **kwargs):
        local = evaluate(*args, **kwargs)
        coefficient_shapes.append(local.source.shape)
        return local

    monkeypatch.setattr(window.coefficients, "evaluate_spectrum", record_spectrum)
    handle = atmosphere.network.register_forward_pre_hook(
        lambda module, args: network_points.append(args[0].shape[0]))
    try:
        with patch("prom3theus.inversion.lte.formal_stokes", wraps=formal_stokes) as transfer:
            output = window(geometry, wave)
    finally:
        handle.remove()
    depths = [window.depth_count, fine_count] if fine_count else [window.depth_count]
    points_per_depth = ray_count * (2 if fixed_magnetic_height else 1)
    assert network_points == [points_per_depth * depth for depth in depths]
    assert coefficient_shapes == [(ray_count, depth, len(wave), 4) for depth in depths]
    transfer.assert_called_once()
    assert transfer.call_args.args[0].shape == (ray_count, sum(depths), len(wave), 7)
    assert output.shape == (ray_count, 4, len(wave))


@pytest.mark.parametrize("per_ray", [False, True])
def test_full_spectrum_prepares_plasma_once_without_wavelength_duplication(monkeypatch, per_ray):
    _, window, term, batch = make_system()
    if per_ray:
        batch["removed_solar_los_velocity_m_per_s"] = torch.tensor([-1000., 1000.])
    provider = window.coefficients.plasma_provider
    original = provider.prepare_plasma_state
    points = []
    def prepare(temperature, pressure):
        points.append(temperature.numel())
        return original(temperature, pressure)
    monkeypatch.setattr(provider, "prepare_plasma_state", prepare)
    line_points, spectral_queries = [], []
    line_prepare = window.coefficients.line_opacity.prepare
    spectrum_prepare = window.coefficients.prepare_spectrum
    def prepare_lines(temperature, *args, **kwargs):
        line_points.append(temperature.numel())
        return line_prepare(temperature, *args, **kwargs)
    def prepare_spectrum(wave):
        from prom3theus.rt.wavelength import CenteredWavelength
        spectral_queries.append(wave.offset_angstrom.shape if isinstance(wave, CenteredWavelength) else wave.shape)
        return spectrum_prepare(wave)
    monkeypatch.setattr(window.coefficients.line_opacity, "prepare", prepare_lines)
    monkeypatch.setattr(window.coefficients, "prepare_spectrum", prepare_spectrum)
    term.predict(batch).square().mean().backward()
    assert points == [len(batch["coordinates"]) * window.depth_count]
    assert line_points == points
    assert len(spectral_queries) == 1


@pytest.mark.parametrize("incoming_temperature", [None, 5500.])
@pytest.mark.parametrize("jitter_fraction", [0., 1.])
def test_physical_spectrum_matches_independent_serial_solver(monkeypatch, incoming_temperature, jitter_fraction):
    from prom3theus.validation.reference_transfer import PolarizedFormalSolver, RayDistancePath
    _, window, term, batch = make_system(jitter_fraction=jitter_fraction, transfer_precision="float64")
    window.double()
    window.configuration["incoming_temperature_k"] = incoming_temperature
    batch["removed_solar_los_velocity_m_per_s"] = torch.tensor([-1000., 1000.])
    rng = torch.get_rng_state()
    with torch.no_grad():
        actual = term.predict(batch)

    def serial(k, source, distance, *, bottom_boundary=None, precision, polarized, intervals_m):
        k = propagation_matrix(k)
        grid = torch.arange(k.shape[-4], dtype=torch.float64, device=k.device)
        return PolarizedFormalSolver()(k.double(), source.double(), grid,
            path=RayDistancePath(distance.double()),
            bottom_boundary=None if bottom_boundary is None else bottom_boundary.double()).to(k.dtype)

    monkeypatch.setattr("prom3theus.inversion.lte.formal_stokes", serial)
    torch.set_rng_state(rng)
    with torch.no_grad():
        expected = term.predict(batch)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_direct_constructor_rejects_learned_solver_options():
    with pytest.raises(ValueError, match="Unknown LTE synthesis options"):
        LTEWindow(None, configuration={"hidden_dimension": 128})


@pytest.mark.parametrize("jitter_fraction", [.5, 1.])
def test_jittered_samples_follow_oblique_shell_rays_and_preserve_boundaries(jitter_fraction):
    _, window, _, batch = make_system(jitter_fraction=jitter_fraction)
    batch["ray_direction"] = torch.tensor([[0., 0., -1.], [.6, 0., -.8]])
    geometry = window.boundary_queries(batch["coordinates"], batch["ray_direction"])
    samples = window.sample_rays(geometry)
    distance = samples.distance_m
    assert distance.dtype == torch.float32
    assert torch.equal(distance[:, 0], torch.zeros_like(distance[:, 0]))
    assert torch.equal(distance[:, -1], geometry["path_length_m"])
    assert (distance.diff(dim=-1) > 0).all()
    torch.testing.assert_close(samples.position_m[:, 0], geometry["outgoing_position_m"], rtol=0, atol=0)
    # Heights come from local offsets, never norm(global float32 position)-R.
    torch.testing.assert_close(samples.height_m[:, 0], window.height_bounds_m[1].expand(2), rtol=0, atol=.5)
    torch.testing.assert_close(samples.height_m[:, -1], window.height_bounds_m[0].expand(2), rtol=0, atol=.5)
    assert (samples.height_m >= window.height_bounds_m[0] - .5).all()
    assert (samples.height_m <= window.height_bounds_m[1] + .5).all()
    assert (samples.intervals_m > 0).all()
    torch.testing.assert_close(samples.intervals_m.sum(-1), geometry["path_length_m"])
    torch.testing.assert_close(samples.intervals_m, distance.diff(dim=-1), rtol=0, atol=.5)
    fraction = distance / geometry["path_length_m"][:, None]
    nominal = torch.arange(window.depth_count, dtype=distance.dtype) / (window.depth_count - 1)
    assert ((fraction - nominal).abs() < jitter_fraction / (2 * (window.depth_count - 1))).all()
    # Stratification is in physical ray distance, not radial height.
    assert geometry["path_length_m"][1] > geometry["path_length_m"][0]


def test_ray_jitter_is_independent_reproducible_and_redrawn_each_forward():
    _, window, _, batch = make_system(jitter_fraction=1.)
    geometry = window.boundary_queries(batch["coordinates"], batch["ray_direction"])
    rng = torch.get_rng_state()
    first = window.sample_rays(geometry).distance_m
    second = window.sample_rays(geometry).distance_m
    assert not torch.equal(first, second)
    assert not torch.equal(first[0], first[1])  # Duplicate rays have independent draws.
    torch.set_rng_state(rng)
    assert torch.equal(first, window.sample_rays(geometry).distance_m)


def test_midpoint_epsilon_separates_extreme_jitter_and_integrates_true_ray_lengths(monkeypatch):
    from prom3theus.rt.formal_transfer import formal_stokes

    _, window, _, batch = make_system(jitter_fraction=1., depth_count=257)
    batch["ray_direction"] = torch.tensor([[0., 0., -1.], [.6, 0., -.8]])
    geometry = window.boundary_queries(batch["coordinates"], batch["ray_direction"])

    def extremes(shape, *, device, dtype):
        return (torch.arange(shape[-1], device=device) % 2).to(dtype).expand(shape)

    monkeypatch.setattr(torch, "rand", extremes)
    samples = window.sample_rays(geometry)
    assert (samples.distance_m.diff(dim=-1) > 0).all()
    assert (samples.intervals_m > 0).all()
    torch.testing.assert_close(samples.intervals_m.sum(-1), geometry["path_length_m"])
    # A homogeneous slab must have the correct optical thickness on each LOS,
    # including the longer oblique ray, regardless of irregular node spacing.
    absorption = 1e-6
    propagation = absorption * torch.eye(4).expand(2, window.depth_count, 1, 4, 4)
    source = torch.zeros(2, window.depth_count, 1, 4)
    source[..., 0] = 1
    result = formal_stokes(propagation, source, samples.distance_m,
        intervals_m=samples.intervals_m, bottom_boundary=torch.zeros(2, 1, 4), polarized=True)
    expected = -torch.expm1(-absorption * geometry["path_length_m"])
    torch.testing.assert_close(result[:, 0, 0], expected, rtol=2e-6, atol=1e-7)
    assert torch.equal(result[..., 1:], torch.zeros_like(result[..., 1:]))


@pytest.mark.parametrize("training,jitter_fraction,depth_count", [(False, 1., 33), (True, 0., 33), (True, 1., 2)])
def test_deterministic_ray_sampling_does_not_consume_rng(training, jitter_fraction, depth_count):
    _, window, _, batch = make_system(jitter_fraction=jitter_fraction, depth_count=depth_count)
    window.train(training)
    geometry = window.boundary_queries(batch["coordinates"], batch["ray_direction"])
    rng = torch.get_rng_state()
    first = window.sample_rays(geometry).distance_m
    second = window.sample_rays(geometry).distance_m
    assert torch.equal(rng, torch.get_rng_state())
    expected = geometry["path_length_m"][:, None] * (torch.arange(depth_count, dtype=first.dtype) / (depth_count - 1))
    assert torch.equal(first, expected)
    assert torch.equal(first, second)


@pytest.mark.parametrize("jitter_fraction", [-.01, 1.01, float("nan"), float("inf"), True])
def test_direct_constructor_rejects_invalid_ray_jitter(jitter_fraction):
    with pytest.raises(ValueError, match="jitter_fraction"):
        make_system(jitter_fraction=jitter_fraction)


@pytest.mark.parametrize("azimuth_offset", [0., 37., 90.])
def test_detector_profiles_match_independent_lte_synthesis(azimuth_offset):
    """Check coefficient assembly, LOS projection, Q/U convention, and radiance.

    The reference synthesizes directly in the detector magnetic basis, rather
    than rotating the production Stokes result or reusing its local coefficients.
    """
    from dataclasses import replace
    from prom3theus.rt import StratifiedAtmosphere, project_vectors_to_stokes
    from prom3theus.rt.ray_geometry import canonical_basis
    from prom3theus.validation.reference_synthesis import LTESynthesizer
    from prom3theus.validation.reference_transfer import RayDistancePath

    torch.manual_seed(42)
    atmosphere, window, term, batch = make_system(dtype=torch.float64, transfer_precision="float64")
    atmosphere.double().eval()
    window.double().eval()
    term.double().eval()
    window.depth_count = 65
    with torch.no_grad():
        # A resolved polarized line, including a nonzero material velocity.
        atmosphere.network.out_layer.weight[1:7].zero_()
        atmosphere.network.out_layer.bias[1:4].copy_(torch.tensor([.2, -.3, .9]))
        atmosphere.network.out_layer.bias[4:7].copy_(torch.tensor([6., -8., 12.]))
    term.instrument.polarization_convention = replace(
        term.instrument.polarization_convention, offset_deg=azimuth_offset)
    batch = {key: value.double() for key, value in batch.items()}
    direction = torch.tensor([[0., 0., 1.], [.4, 0., .84**.5]], dtype=torch.float64)
    batch["ray_direction"] = -direction
    basis = canonical_basis(direction)
    # Evaluate the reference in a detector chart distinct from the solver chart.
    angle = direction.new_tensor(.37)
    c, s = angle.cos(), angle.sin()
    batch["stokes_basis"] = torch.stack((c*basis[:, 0]+s*basis[:, 1],
        -s*basis[:, 0]+c*basis[:, 1], direction), dim=-2)
    with torch.no_grad():
        actual = term.predict(batch)
        geometry = window.boundary_queries(batch["coordinates"], batch["ray_direction"])
        distance = geometry["path_length_m"][:, None] * torch.linspace(0, 1, window.depth_count, dtype=torch.float64)
        position = geometry["outgoing_position_m"][:, None] - distance[..., None] * direction[:, None]
        state = window.evaluate_atmosphere(position, geometry["time_hours"][:, None])
        magnetic = project_vectors_to_stokes(state["magnetic_field"], batch["stokes_basis"])
        sampled = StratifiedAtmosphere(
            depth_coordinate=torch.arange(window.depth_count, dtype=torch.float64),
            temperature=state["temperature"], gas_pressure=state["gas_pressure"],
            microturbulence=state["microturbulence"],
            velocity_field=project_vectors_to_stokes(state["velocity_field"], batch["stokes_basis"]),
            magnetic_field=term.instrument.polarization_convention.to_synthesis_frame(magnetic),
        )
        reference = LTESynthesizer(line_ids=["FeI_6302.4932"]).double()
        raw = reference(sampled, term.synthesis_wavelength_angstrom,
            path=RayDistancePath(distance), radiance_scale=term.atlas_continuum_radiance_w_m3_sr)
        expected = term.instrument(raw, term.synthesis_wavelength_angstrom, term.wavelength_angstrom)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-9)
    assert actual[:, 1:3].abs().max() > 1e-3
    assert actual[:, 3].abs().max() > 1e-3
    assert torch.all(actual[:, 0] >= torch.linalg.vector_norm(actual[:, 1:], dim=1))


@pytest.mark.parametrize("randomize", [False, True])
def test_refined_rays_keep_64_unique_shell_points_and_actual_intervals(monkeypatch, randomize):
    _, window, _, batch = make_system(depth_count=32, jitter_fraction=1.,
        coarse_to_fine={"enabled": True, "fine_sample_count": 32, "uniform_weight_floor": 0.})
    window.train(randomize)
    batch["ray_direction"] = torch.tensor([[0., 0., -1.], [.6, 0., -.8]])
    geometry = window.boundary_queries(batch["coordinates"], batch["ray_direction"])

    def extremes(shape, *, device, dtype):
        # Adjacent coarse nodes and fine quantiles approach each other.
        return (1 - torch.arange(shape[-1], device=device) % 2).to(dtype).expand(shape)

    monkeypatch.setattr(torch, "rand", extremes)
    coarse = window.sample_rays(geometry)
    proposal = torch.zeros_like(coarse.intervals_m)
    proposal.scatter_(-1, coarse.intervals_m.argmin(-1, keepdim=True), 1.)
    fine, distance, intervals, order = window.refine_rays(geometry, coarse, proposal)
    assert distance.shape == (2, 64)
    assert torch.all(intervals > 0)
    assert torch.equal(distance[:, 0], coarse.distance_m[:, 0])
    assert torch.equal(distance[:, -1], coarse.distance_m[:, -1])
    assert not distance.requires_grad
    torch.testing.assert_close(intervals, distance.diff(dim=-1), rtol=0, atol=0)
    torch.testing.assert_close(intervals.sum(-1), geometry["path_length_m"])
    from prom3theus.rt.ray_geometry import sample_shell_characteristic
    chart, height, _ = sample_shell_characteristic(geometry, fine.distance_m, solar_radius_m=window.solar_radius_m)
    torch.testing.assert_close(chart, fine.chart_megameter)
    torch.testing.assert_close(height, fine.height_m)
    assert (height <= window.height_bounds_m[1]).all()
    assert (height >= window.height_bounds_m[0]).all()
    from prom3theus.rt.formal_transfer import formal_stokes
    k = 1e-6 * torch.eye(4).expand(2, 64, 1, 4, 4)
    source = torch.zeros(2, 64, 1, 4)
    source[..., 0] = 1
    result = formal_stokes(k, source, distance, intervals_m=intervals,
        bottom_boundary=torch.zeros(2, 1, 4), polarized=True)
    torch.testing.assert_close(result[:, 0, 0], -torch.expm1(-1e-6 * geometry["path_length_m"]), rtol=2e-6, atol=1e-7)


def test_refinement_evaluates_only_new_points_and_reuses_coarse_physics(monkeypatch):
    atmosphere, window, term, batch = make_system(depth_count=32, jitter_fraction=1.,
        coarse_to_fine={"enabled": True, "fine_sample_count": 32, "uniform_weight_floor": .05})
    network_outputs, point_counts = [], []
    def record(module, args, output):
        output.retain_grad()
        network_outputs.append(output)
    handle = atmosphere.network.register_forward_hook(record)
    original = window.coefficients.prepare_atmosphere
    def prepare(state, **kwargs):
        point_counts.append(state["temperature"].shape)
        return original(state, **kwargs)
    monkeypatch.setattr(window.coefficients, "prepare_atmosphere", prepare)
    try:
        prediction = term.predict(batch)
        prediction.square().mean().backward()
    finally:
        handle.remove()
    assert point_counts == [(2, 32), (2, 32)]
    assert len(network_outputs) == 2
    for output in network_outputs:
        assert output.grad is not None and torch.isfinite(output.grad).all()
        assert output.grad.abs().max() > 0


def test_refined_spectrum_and_gradients_are_deterministic_in_eval():
    torch.manual_seed(42)
    atmosphere, window, term, batch = make_system(depth_count=32, jitter_fraction=1.,
        coarse_to_fine={"enabled": True, "fine_sample_count": 32, "uniform_weight_floor": .05})
    window.eval()
    batch["removed_solar_los_velocity_m_per_s"] = torch.tensor([-1000., 1000.])
    rng = torch.get_rng_state()
    actual = term.predict(batch)
    actual.square().sum().backward()
    gradients = [p.grad.clone() for p in atmosphere.parameters() if p.grad is not None]
    assert torch.equal(rng, torch.get_rng_state())
    atmosphere.zero_grad(set_to_none=True)
    expected = term.predict(batch)
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for reference, p in zip(gradients, [p for p in atmosphere.parameters() if p.grad is not None]):
        torch.testing.assert_close(p.grad, reference, rtol=5e-4, atol=2e-5)


@pytest.mark.parametrize("refinement", [
    {"enabled": True}, {"enabled": 1}, {"enabled": True, "fine_sample_count": 0, "uniform_weight_floor": .05},
    {"enabled": True, "fine_sample_count": 32, "uniform_weight_floor": 2.}, {"unknown": 1},
])
def test_invalid_direct_refinement_configuration(refinement):
    with pytest.raises((ValueError, TypeError)):
        make_system(coarse_to_fine=refinement)


def test_refined_physical_spectrum_matches_independent_serial_solver(monkeypatch):
    from prom3theus.validation.reference_transfer import PolarizedFormalSolver, RayDistancePath
    _, window, term, batch = make_system(dtype=torch.float64, depth_count=32,
        transfer_precision="float64", coarse_to_fine={"enabled": True, "fine_sample_count": 32,
                                                     "uniform_weight_floor": .05})
    window.eval()
    with torch.no_grad():
        actual = term.predict(batch)
    def serial(k, source, distance, *, bottom_boundary=None, precision, polarized, intervals_m):
        k = propagation_matrix(k)
        assert distance.shape[-1] == 64
        torch.testing.assert_close(intervals_m, distance.diff(dim=-1))
        grid = torch.arange(k.shape[-4], dtype=torch.float64)
        return PolarizedFormalSolver()(k, source, grid, path=RayDistancePath(distance),
                                       bottom_boundary=bottom_boundary)
    monkeypatch.setattr("prom3theus.inversion.lte.formal_stokes", serial)
    with torch.no_grad():
        expected = term.predict(batch)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
