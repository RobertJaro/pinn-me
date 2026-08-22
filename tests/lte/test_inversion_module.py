import math

import pytest
import torch
from torch import nn
from types import SimpleNamespace

import pme.train.lte_physics as physics_source
from pme.inversion_lte import (
    _inject_height_gauge_quadrature,
    _inject_spatial_coordinate_affine,
    _inject_stic_thermodynamic_bounds,
    _resolve_log_tau500,
    _resolve_physics_activation,
    _resolve_physics_reference,
)


def test_compact_depth_grid_config_resolves_exact_endpoints():
    grid = _resolve_log_tau500({"min": -5.0, "max": 1.0, "count": 51})
    assert len(grid) == 51
    assert grid[0] == -5.0
    assert grid[-1] == 1.0


def test_raster_coordinate_affine_is_injected_and_conflicts_are_rejected():
    metadata = {
        "coordinates": {
            "network_affine": {
                "center_mm": [-12.5, 4.0],
                "scale_mm": [56.0, 56.0],
            }
        }
    }
    config = _inject_spatial_coordinate_affine({}, metadata)
    assert config["spatial_coordinate_center_mm"] == [-12.5, 4.0]
    assert config["spatial_coordinate_scale_mm"] == [56.0, 56.0]

    matching = _inject_spatial_coordinate_affine(
        {
            "spatial_coordinate_center_mm": [-12.5, 4.0],
            "spatial_coordinate_scale_mm": [56.0, 56.0],
        },
        metadata,
    )
    assert matching == config
    with pytest.raises(ValueError, match="conflicts"):
        _inject_spatial_coordinate_affine(
            {"spatial_coordinate_center_mm": [0.0, 0.0]}, metadata
        )


def test_stic_lookup_bounds_are_injected_exactly_and_conflicts_are_rejected():
    resources = {
        "stic_lookup_bounds": {
            "temperature_log10_k": [3.4, 4.0],
            "gas_pressure_log10_pa": [-1.5, 6.0],
        }
    }
    resolved = _inject_stic_thermodynamic_bounds({}, resources)
    assert resolved["temperature_log10_bounds"] == [3.4, 4.0]
    assert resolved["gas_pressure_log10_bounds"] == [-1.5, 6.0]
    with pytest.raises(ValueError, match="must match the verified STiC table"):
        _inject_stic_thermodynamic_bounds(
            {"gas_pressure_log10_bounds": [-1.25, 5.75]}, resources
        )


def test_height_gauge_quadrature_is_fixed_and_uses_only_valid_raster_pixels():
    slit, scan = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    coords = torch.stack(
        (
            torch.zeros_like(slit),
            scan,
            slit,
        ),
        dim=-1,
    ).float()
    valid = torch.ones((3, 4), dtype=torch.bool)
    valid[1, 2] = False
    raster = SimpleNamespace(coords=coords, valid_mask=valid)
    config = {
        "height_mapping_config": {
            "gauge_quadrature_points": 7,
            "gauge_log_tau500": 0.0,
        }
    }

    resolved = _inject_height_gauge_quadrature(config, raster)
    reference = torch.tensor(
        resolved["height_mapping_config"]["gauge_reference_coords"]
    )
    assert reference.shape == (7, 3)
    assert not torch.any(torch.all(reference == coords[1, 2], dim=-1))
    assert "gauge_quadrature_points" not in resolved["height_mapping_config"]


def test_hse_top_pressure_comes_from_verified_resource_metadata():
    resources = {
        "falc_top_boundary": {
            "target_log10_tau500": -5.0,
            "gas_pressure_pa": 0.137,
            "gravity_cm_s2": 27542.28703338169,
        }
    }
    base = {
        "equations": {
            "hse": {"enabled": True},
            "pressure_boundary": {"enabled": True},
        }
    }
    resolved = _resolve_physics_reference(base, [-5.0, 1.0], resources)
    assert resolved["top_pressure_pa"] == 0.137
    assert resolved["gravity_m_per_s2"] == pytest.approx(275.4228703338169)
    with pytest.raises(ValueError, match="conflicts"):
        _resolve_physics_reference(
            {**base, "top_pressure_pa": 0.3}, [-5.0, 1.0], resources
        )
    with pytest.raises(ValueError, match="gravity_m_per_s2 conflicts"):
        _resolve_physics_reference(
            {**base, "gravity_m_per_s2": 274.0},
            [-5.0, 1.0],
            resources,
        )
    with pytest.raises(ValueError, match="atmosphere top"):
        _resolve_physics_reference(base, [-4.0, 1.0], resources)
    gravity_only = _resolve_physics_reference(
        {"equations": {"hse": {"enabled": True}}},
        [-4.0, 1.0],
        resources,
    )
    assert gravity_only["gravity_m_per_s2"] == pytest.approx(275.4228703338169)
    assert "top_pressure_pa" not in gravity_only


def test_entrypoint_requires_identifiable_tau_to_height_mapping():
    assert _resolve_physics_activation(
        {
            "equations": {
                "hse": {"enabled": True},
                "tau_mapping": {"enabled": True},
                "pressure_boundary": {"enabled": True},
            }
        }
    ) == (True, True)
    assert _resolve_physics_activation(
        {"equations": {"tau_mapping": {"enabled": True}}}
    ) == (True, False)
    with pytest.raises(ValueError, match="tau_mapping.enabled=true"):
        _resolve_physics_activation({})
    with pytest.raises(ValueError, match="tau_mapping.enabled=true"):
        _resolve_physics_activation({"equations": {"hse": {"enabled": True}}})


def test_entrypoint_direct_log_tau_accepts_hse_and_rejects_geometric_physics():
    assert _resolve_physics_activation(
        {
            "equations": {
                "hse": {"enabled": True},
                "pressure_boundary": {"enabled": True},
            }
        },
        "log_tau",
    ) == (True, True)
    with pytest.raises(ValueError, match="Direct log_tau.*geometric"):
        _resolve_physics_activation(
            {"equations": {"tau_mapping": {"enabled": True}}},
            "log_tau",
        )


class _StubSynthesizer(nn.Module):
    """Cheap differentiable stand-in isolating inversion-module integration."""

    def __init__(self, log_tau500, **kwargs):
        super().__init__()
        del kwargs
        grid = (
            torch.empty(0)
            if log_tau500 is None
            else torch.as_tensor(log_tau500, dtype=torch.float32)
        )
        self.register_buffer("log_tau500", grid)
        self.continuum_opacity = _StubOpacity()

    def forward(
        self,
        atmosphere,
        wavelength,
        mu=1.0,
        return_diagnostics=False,
        radiance_scale=None,
    ):
        wavelength = torch.as_tensor(
            wavelength,
            dtype=atmosphere.temperature.dtype,
            device=atmosphere.temperature.device,
        )
        mu = torch.as_tensor(mu, dtype=wavelength.dtype, device=wavelength.device)
        if mu.ndim and mu.shape[-1] == 1:
            mu = mu.squeeze(-1)
        batch_shape = atmosphere.temperature.shape[:-1]
        mu = torch.broadcast_to(mu, batch_shape)
        spectral_coordinate = (wavelength - 6302.0) / 2.0
        spectral_coordinate = spectral_coordinate.reshape(*([1] * len(batch_shape)), -1)
        temperature = atmosphere.temperature.mean(-1, keepdim=True) / 6000.0
        velocity = atmosphere.v_los.mean(-1, keepdim=True) / 3000.0
        magnetic = atmosphere.magnetic_field.mean(-2) / 2000.0
        microturbulence = atmosphere.microturbulence.mean(-1, keepdim=True) / 1000.0
        intensity = temperature * (1.0 + 0.02 * spectral_coordinate) * mu[..., None]
        q = magnetic[..., 0:1] * torch.exp(-spectral_coordinate.square())
        u = magnetic[..., 1:2] * torch.exp(-spectral_coordinate.square())
        v = (
            (magnetic[..., 2:3] + 0.05 * velocity + 0.01 * microturbulence)
            * spectral_coordinate
            * torch.exp(-spectral_coordinate.square())
        )
        # Match the order of magnitude of the production physical-radiance
        # synthesizer. LTEModule expresses transfer directly in fixed atlas-Ic
        # units when radiance_scale is supplied.
        stokes = 1.0e13 * torch.stack((intensity, q, u, v), dim=-2)
        if radiance_scale is not None:
            stokes = stokes / torch.as_tensor(radiance_scale).to(stokes)
        if not return_diagnostics:
            return stokes
        diagnostics = SimpleNamespace(
            eos_state=SimpleNamespace(
                mass_density=torch.full_like(atmosphere.temperature, 1.0e-4)
            )
        )
        return stokes, diagnostics


class _StubOpacity(nn.Module):
    reference_top_pressure_pa = 0.3
    reference_gravity_m_per_s2 = 274.0

    def __init__(self):
        super().__init__()
        self.reference_mass_density_calls = 0

    def reference_mass_density(self, temperature, gas_pressure):
        self.reference_mass_density_calls += 1
        return gas_pressure / (temperature * 3.0e7)

    def volume_extinction_at_5000(self, temperature, gas_pressure):
        del temperature
        return torch.ones_like(gas_pressure)

    @staticmethod
    def validate_physical_height_contract(log_tau500_top, top_pressure_pa):
        if float(log_tau500_top) != -5.0 or float(top_pressure_pa) != 0.3:
            raise RuntimeError("stub FALC/STiC top pair mismatch")


def test_lte_module_forward_objectives_and_checkpoint(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    depth_grid = torch.linspace(-4.0, 1.0, 9)
    observed_wavelength = torch.linspace(6301.2, 6302.8, 17)
    module = lte_module.LTEModule(
        log_tau500=depth_grid,
        wavelength_angstrom=observed_wavelength,
        atmosphere_config={
            "model_config": {
                "dim": 12,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        instrument_config={"oversample": 2},
        weight_config={"I": 1.0, "Q": 2.0, "U": 3.0, "V": 4.0},
        checkpoint_metadata={"fixture": "generated"},
    )
    coords = torch.tensor([[0.0, -0.1, 0.2], [0.3, 0.2, -0.4]])
    mu = torch.tensor([[1.0], [0.75]])

    result = module.synthesize(coords, mu)
    prediction = result["stokes"]
    assert prediction.shape == (2, 4, 17)
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(module(coords, mu), prediction)
    assert "smoothness_config" not in module.hparams
    assert not hasattr(module, "smoothness_weight")

    target = torch.zeros_like(prediction)
    components, objective = module._stokes_objective(prediction, target)
    expected_components = prediction.square().mean(dim=(0, 2))
    torch.testing.assert_close(components, expected_components)
    torch.testing.assert_close(
        objective,
        torch.dot(expected_components, module.stokes_weights),
    )

    objective.backward()
    assert module.atmosphere_model.network.out_layer.weight.grad is not None
    magnetic_gradients = module.atmosphere_model.network.out_layer.weight.grad[4:7]
    # The weak oblique reference field must break the exact B=0 symmetry so
    # every Cartesian magnetic component can move on the first optimizer step.
    assert torch.all(torch.count_nonzero(magnetic_gradients, dim=1) > 0)

    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    metadata = checkpoint["lte_metadata"]
    assert metadata["schema_version"] == 28
    assert "velocity_field" in metadata["units"]
    assert metadata["physics"]["iterative_forward_solve"] is False
    assert metadata["units"]["geometric_height"].startswith("m, increasing upward")
    assert metadata["atmosphere_parameterization"]["network_inputs"] == ["x", "y", "z"]
    height_mapping = metadata["atmosphere_parameterization"]["height_mapping"]
    assert height_mapping["monotonicity"] == (
        "positive linear base metric plus perturbation constrained by the "
        "configured tau-mapping physics loss"
    )
    assert height_mapping["gauge"] == "mean_FOV[z(log_tau500=0.0)] = 0"
    assert height_mapping["runtime_iterations"] == 0
    mapping_physics = metadata["physics"]["optical_depth_mapping"]
    assert mapping_physics["equation"] == (
        "alpha500*dz/dlog10(tau500) + ln(10)*tau500 = 0"
    )
    assert metadata["physics"]["equation_definitions"]["hse"] == (
        "[dP/dlog10(tau500) - rho*g*(-dz/dlog10(tau500))] / mean_xy(P) = 0"
    )
    assert metadata["units"]["wavelength"] == (
        "standard-air angstrom at data/instrument boundary"
    )
    assert metadata["units"]["wavelength_internal"] == (
        "vacuum angstrom for frequency, Planck, and opacity"
    )
    assert metadata["data"] == {"fixture": "generated"}
    assert metadata["depth_sampling"]["sample_count"] == 9
    assert metadata["depth_sampling"]["integration"] == (
        "actual geometric-height line elements from the learned "
        "Z(x,y,log_tau500) mapping at every realized nonuniform "
        "optical-depth sample"
    )
    assert metadata["stokes_objective"] == {
        "loss": {"type": "mse"},
        "normalization": {"asinh_alphas": None},
        "weight": {
            name: {"type": "fixed", "start": value, "end": value}
            for name, value in {"I": 1.0, "Q": 2.0, "U": 3.0, "V": 4.0}.items()
        },
    }
    assert metadata["instrument"]["fwhm_angstrom"] == 0.025
    assert (
        metadata["instrument"]["provenance"]["sources"]["LITES_ET_AL_2013"][
            "measured_fwhm_milliangstrom"
        ]
        == "approximately 25"
    )

    module.on_load_checkpoint(checkpoint)
    checkpoint["lte_metadata"]["continuum_normalization"]["learned_gain"] = True
    module.on_load_checkpoint(checkpoint)
    checkpoint["lte_metadata"]["instrument"]["fwhm_angstrom"] = 0.123
    module.on_load_checkpoint(checkpoint)

    module.double()
    double_checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(double_checkpoint)
    module.float()
    module.on_load_checkpoint(double_checkpoint)
    assert next(module.parameters()).dtype == torch.float32

    legacy_checkpoint = {
        "state_dict": {
            "log_radiometric_gain": torch.zeros(()),
            "synthesized_radiance_reference": torch.tensor(2.0e13),
        }
    }
    module.on_load_checkpoint(legacy_checkpoint)
    torch.testing.assert_close(
        legacy_checkpoint["state_dict"]["atlas_continuum_radiance_w_m3_sr"],
        module.atlas_continuum_radiance_w_m3_sr.detach().cpu(),
    )
    assert "log_radiometric_gain" not in legacy_checkpoint["state_dict"]
    assert "synthesized_radiance_reference" not in legacy_checkpoint["state_dict"]


def test_hse_is_an_optimized_physics_loss_without_iterative_forward_solve(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 17),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 13),
        atmosphere_config={
            "model_config": {
                "dim": 12,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        physics_config={
            "top_pressure_pa": 0.3,
            "gravity_m_per_s2": 274.0,
            "equations": {
                "hse": {
                    "enabled": True,
                    "weight": {
                        "type": "linear",
                        "start": 0.14,
                        "end": 0.7,
                        "iterations": 5,
                    },
                },
                "tau_mapping": {"enabled": True, "weight": 1.0},
                "pressure_boundary": {"enabled": True, "weight": 0.35},
            },
        },
    )
    coords = torch.tensor(((0.0, 0.1, -0.2),))
    volume = module.physics.volume(
        module.atmosphere_model,
        module.synthesizer.continuum_opacity,
        coords,
        torch.tensor((-2.0,)),
        global_step=0,
    )
    boundary_result = module.physics.pressure_boundary(
        module.atmosphere_model,
        coords,
        torch.tensor((-5.0,)),
        torch.tensor((0.3,)),
        global_step=0,
    )
    equilibrium = volume.losses["hse"]
    tau_mapping = volume.losses["tau_mapping"]
    boundary = boundary_result.losses["pressure_boundary"]
    assert all(torch.isfinite(value) for value in (equilibrium, tau_mapping, boundary))
    assert equilibrium >= 0 and tau_mapping >= 0 and boundary >= 0
    (equilibrium + tau_mapping + boundary).backward()
    pressure_row = module.atmosphere_model.network.out_layer.weight.grad[8]
    assert torch.isfinite(pressure_row).all()
    assert torch.count_nonzero(pressure_row) > 0
    height_gradients = [
        parameter.grad
        for parameter in module.atmosphere_model.height_mapping.parameters()
    ]
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for gradient in height_gradients
    )
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in height_gradients
    )
    assert module.physics.weights(0)["hse"] == pytest.approx(0.14)


def test_training_step_consumes_independent_hse_and_boundary_batches(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 9),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 13),
        atmosphere_config={
            "model_config": {
                "dim": 10,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        physics_config={
            "top_pressure_pa": 0.3,
            "gravity_m_per_s2": 274.0,
            "equations": {
                "hse": {"enabled": True, "weight": 0.7},
                "tau_mapping": {"enabled": True, "weight": 0.4},
                "pressure_boundary": {"enabled": True, "weight": 0.14},
            },
        },
    )
    logged = {}
    monkeypatch.setattr(
        module,
        "log_dict",
        lambda metrics, *args, **kwargs: logged.update(metrics),
    )
    stokes_coords = torch.tensor(((0.0, -0.2, 0.1), (0.0, 0.3, -0.1)))
    mu = torch.ones(2, 1)
    target = module(stokes_coords, mu).detach()
    batch = {
        "stokes": {
            "coords": stokes_coords,
            "mu": mu,
            "stokes": target,
            "pixel_index": torch.tensor(((0, 0), (0, 1))),
        },
        "physics_volume": {
            "coords": torch.tensor(
                ((0.0, -0.4, 0.2), (0.0, 0.2, 0.4), (0.0, 0.1, -0.3))
            ),
            "log_tau500": torch.tensor((-4.7, -2.1, 0.4)),
        },
        "pressure_boundary": {
            "coords": torch.tensor(((0.0, -0.35, -0.2), (0.0, 0.4, 0.3))),
            "log_tau500": torch.full((2,), -5.0),
            "gas_pressure_pa": torch.full((2,), 0.3),
        },
    }
    mapping_calls = []
    height_mapping = module.atmosphere_model.height_mapping
    original_height_and_metric = height_mapping.height_and_metric

    def record_height_and_metric(coords, log_tau500, **kwargs):
        height, metric = original_height_and_metric(coords, log_tau500, **kwargs)
        mapping_calls.append(
            (
                log_tau500.detach().clone(),
                height.detach().clone(),
                metric.detach().clone(),
            )
        )
        return height, metric

    monkeypatch.setattr(height_mapping, "height_and_metric", record_height_and_metric)
    loss = module.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(module.atmosphere_model.network.out_layer.weight.grad).all()
    assert torch.isfinite(
        module.atmosphere_model.height_mapping.network.out_layer.weight.grad
    ).all()

    volume_call = next(call for call in mapping_calls if call[0].shape == (3, 1))
    torch.testing.assert_close(
        volume_call[0][..., 0], batch["physics_volume"]["log_tau500"]
    )
    assert torch.isfinite(volume_call[1]).all()
    assert torch.isfinite(volume_call[2]).all()
    assert "train.tau_mapping" in logged and torch.isfinite(logged["train.tau_mapping"])
    expected_loss = (
        logged["train.stokes_loss"]
        + 0.7 * logged["train.hse"]
        + 0.14 * logged["train.pressure_boundary"]
        + 0.4 * logged["train.tau_mapping"]
    )
    torch.testing.assert_close(loss, expected_loss)
    assert "train.radiometric_gain" not in logged
    assert "train.tau_mapping_weight" not in logged
    assert "train.monotonic" not in logged
    assert "monotonic" not in module.hparams["physics_config"]["equations"]

    validation = module.validation_step(batch["stokes"], 0)
    assert torch.isfinite(validation["loss"])
    assert "valid.tau_mapping" in logged
    assert torch.isfinite(logged["valid.tau_mapping"])
    assert "valid.monotonic" not in logged


def test_tau_mapping_schedule_is_independent_of_hse(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 9),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 13),
        atmosphere_config={
            "model_config": {
                "dim": 10,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        physics_config={
            "equations": {
                "hse": {"enabled": False, "weight": 3.0},
                "tau_mapping": {
                    "enabled": True,
                    "weight": {
                        "type": "linear",
                        "start": 0.2,
                        "end": 0.8,
                        "iterations": 3,
                    },
                },
            },
        },
    )
    logged = {}
    monkeypatch.setattr(
        module,
        "log_dict",
        lambda metrics, *args, **kwargs: logged.update(metrics),
    )
    coords = torch.tensor(((0.0, -0.2, 0.1), (0.0, 0.3, -0.1)))
    mu = torch.ones(2, 1)
    target = module(coords, mu).detach()
    batch = {
        "coords": coords,
        "mu": mu,
        "stokes": target,
        "pixel_index": torch.tensor(((0, 0), (0, 1))),
    }

    assert module.physics.weights(0)["hse"] == 0.0
    assert module.physics.weights(0)["tau_mapping"] == pytest.approx(0.2)
    loss = module.training_step(batch, 0)
    expected = logged["train.stokes_loss"] + 0.2 * logged["train.tau_mapping"]
    torch.testing.assert_close(loss, expected)
    assert "train.hse" not in logged
    assert "train.hse_weight" not in logged
    assert "train.tau_mapping_weight" not in logged

    assert module.physics.weights(3)["hse"] == 0.0
    assert module.physics.weights(3)["tau_mapping"] == pytest.approx(0.8)


def test_real_tabulated_eos_hse_residual_has_finite_atmosphere_gradients():
    from pme.train.lte_module import LTEModule

    module = LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 11),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 13),
        atmosphere_config={
            "model_config": {
                "dim": 10,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            },
        },
        instrument_config={"oversample": 1},
        physics_config={
            "equations": {
                "hse": {"enabled": True, "weight": 1.0},
                "tau_mapping": {"enabled": True, "weight": 1.0},
                "pressure_boundary": {"enabled": True, "weight": 0.1},
            }
        },
    )
    continuum = module.synthesizer.continuum_opacity
    log_top_pressure = math.log10(module.top_pressure_pa)
    stic_pressure_bounds = (
        float(continuum.stic_log_pressure[0]),
        float(continuum.stic_log_pressure[-1]),
    )
    assert module.top_pressure_pa == pytest.approx(continuum.reference_top_pressure_pa)
    assert stic_pressure_bounds[0] < log_top_pressure < stic_pressure_bounds[1]
    coords = torch.tensor(((0.0, -0.2, 0.3),))
    q = torch.tensor((-5.0, -4.31, -3.72, -2.84, -1.93, -0.77, 0.22, 1.0))
    volume = module.physics.volume(
        module.atmosphere_model,
        continuum,
        coords.expand(q.numel(), -1),
        q,
        global_step=0,
    )
    boundary_result = module.physics.pressure_boundary(
        module.atmosphere_model,
        coords,
        torch.tensor((-5.0,)),
        torch.tensor((module.top_pressure_pa,)),
        global_step=0,
    )
    equilibrium = volume.losses["hse"]
    tau_mapping = volume.losses["tau_mapping"]
    boundary = boundary_result.losses["pressure_boundary"]
    (equilibrium + tau_mapping + 0.1 * boundary).backward()
    gradient = module.atmosphere_model.network.out_layer.weight.grad
    assert all(torch.isfinite(value) for value in (equilibrium, tau_mapping, boundary))
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[0]) > 0  # temperature affects kappa500
    assert torch.count_nonzero(gradient[8]) > 0  # inferred gas pressure
    height_gradient = (
        module.atmosphere_model.height_mapping.network.out_layer.weight.grad
    )
    assert height_gradient is not None and torch.isfinite(height_gradient).all()
    assert torch.count_nonzero(height_gradient) > 0

    module.zero_grad(set_to_none=True)
    second_volume = module.physics.volume(
        module.atmosphere_model,
        continuum,
        torch.tensor(((0.0, -0.3, 0.1), (0.0, 0.25, -0.2))),
        torch.tensor((-3.7, -0.4)),
        global_step=0,
    )
    second_boundary = module.physics.pressure_boundary(
        module.atmosphere_model,
        torch.tensor(((0.0, -0.4, -0.1), (0.0, 0.35, 0.2))),
        torch.full((2,), -5.0),
        torch.full((2,), module.top_pressure_pa),
        global_step=0,
    )
    volume_loss = second_volume.losses["hse"]
    volume_mapping_loss = second_volume.losses["tau_mapping"]
    boundary_loss = second_boundary.losses["pressure_boundary"]
    (volume_loss + volume_mapping_loss + boundary_loss).backward()
    assert all(
        torch.isfinite(value)
        for value in (volume_loss, volume_mapping_loss, boundary_loss)
    )
    assert torch.isfinite(module.atmosphere_model.network.out_layer.weight.grad).all()
    assert torch.isfinite(
        module.atmosphere_model.height_mapping.network.out_layer.weight.grad
    ).all()


def test_random_tau_mapping_and_geometric_hse_match_governing_equations(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 9),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 13),
        atmosphere_config={
            "height_mapping_config": {
                "model_config": {
                    "dim": 8,
                    "n_layers": 2,
                    "encoding_config": {"type": "identity"},
                },
            },
            "model_config": {
                "dim": 10,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            },
        },
        physics_config={
            "equations": {
                "hse": {"enabled": True, "weight": 1.0},
                "tau_mapping": {"enabled": True, "weight": 1.0},
            }
        },
    )
    coords = torch.tensor(((0.0, -0.3, 0.2), (0.0, 0.25, -0.15)))
    paired_q = torch.tensor(((-4.7, -2.2, 0.4), (-4.1, -0.8, 0.8)), requires_grad=True)

    flat_coords = coords[:, None, :].expand(2, 3, 3).reshape(-1, 3)
    flat_q = paired_q.reshape(-1)
    result = module.physics.volume(
        module.atmosphere_model,
        module.synthesizer.continuum_opacity,
        flat_coords,
        flat_q,
        global_step=0,
        return_state=True,
    )
    actual_hse = result.losses["hse"]
    actual_mapping = result.losses["tau_mapping"]
    state = result.state
    surface_pressure = physics_source._tau_surface_mean(
        state.gas_pressure, flat_q
    )
    pressure_derivative_q = (
        -state.derivative("pressure")[:, 0, 2]
        * state.metric_m_per_log_tau
    )
    required_derivative_q = (
        state.mass_density
        * module.gravity_m_per_s2
        * state.metric_m_per_log_tau
    )
    expected_hse = (
        (pressure_derivative_q - required_derivative_q)
        / surface_pressure.detach()
    ).square().mean()
    target = math.log(10.0) * torch.pow(flat_q.new_tensor(10.0), flat_q)
    signed_ratio = state.alpha500 * state.metric_m_per_log_tau / target
    expected_mapping = (
        torch.asinh(signed_ratio) - torch.asinh(torch.ones_like(signed_ratio))
    ).square().mean()

    torch.testing.assert_close(actual_hse, expected_hse)
    torch.testing.assert_close(actual_mapping, expected_mapping)

    pressure_parameter = module.atmosphere_model.network.out_layer.weight
    actual_hse_gradient = torch.autograd.grad(
        actual_hse, pressure_parameter, retain_graph=True
    )[0]
    expected_hse_gradient = torch.autograd.grad(
        expected_hse, pressure_parameter, retain_graph=True
    )[0]
    torch.testing.assert_close(actual_hse_gradient, expected_hse_gradient)

    (actual_hse + actual_mapping).backward()
    for network in (
        module.atmosphere_model.height_mapping.network,
        module.atmosphere_model.network,
    ):
        gradients = [parameter.grad for parameter in network.parameters()]
        assert any(
            gradient is not None and torch.count_nonzero(gradient) > 0
            for gradient in gradients
        )
        assert all(
            gradient is None or torch.isfinite(gradient).all() for gradient in gradients
        )


def test_geometric_hse_uses_stic_density_lookup(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 9),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 13),
        atmosphere_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        physics_config={
            "equations": {
                "hse": {"enabled": True, "weight": 1.0},
                "tau_mapping": {"enabled": True, "weight": 1.0},
            }
        },
    )

    opacity = module.synthesizer.continuum_opacity
    opacity.reference_mass_density_calls = 0
    physics_result = module.physics.volume(
        module.atmosphere_model,
        opacity,
        torch.tensor(((0.0, -0.2, 0.1), (0.0, 0.25, -0.15))),
        torch.tensor((-4.6, 0.3)),
        global_step=0,
    )
    hse_loss = physics_result.losses["hse"]
    mapping_loss = physics_result.losses["tau_mapping"]

    assert opacity.reference_mass_density_calls == 1
    assert torch.isfinite(hse_loss)
    assert torch.isfinite(mapping_loss)


def test_synthesized_stokes_use_fixed_atlas_continuum_unit(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    common = {
        "log_tau500": torch.linspace(-4.0, 1.0, 7),
        "wavelength_angstrom": torch.linspace(6301.2, 6302.8, 13),
        "atmosphere_config": {
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        "instrument_config": {"oversample": 2},
    }
    module = lte_module.LTEModule(**common)
    coords = torch.tensor(((0.0, -0.2, 0.3), (0.1, 0.4, -0.1)))
    mu = torch.tensor(((1.0,), (0.8,)))
    result = module.synthesize(coords, mu)
    stokes = result["stokes"]
    output_continuum = stokes[..., 0, :].index_select(
        -1, module.continuum_indices
    ).mean()
    torch.testing.assert_close(output_continuum, result["predicted_continuum"])
    assert 0.0 < result["predicted_continuum"] < 10.0
    assert not hasattr(module, "log_radiometric_gain")
    assert not module.atlas_continuum_radiance_w_m3_sr.requires_grad

    target = torch.zeros_like(stokes)
    _, loss = module._stokes_objective(stokes, target)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    module.zero_grad(set_to_none=True)

    with torch.no_grad():
        module.atlas_continuum_radiance_w_m3_sr.mul_(2.0)
    rescaled_stokes = module(coords, mu)
    torch.testing.assert_close(rescaled_stokes, 0.5 * stokes)
    rescaled_continuum = rescaled_stokes[..., 0, :].index_select(
        -1, module.continuum_indices
    ).mean()
    torch.testing.assert_close(rescaled_continuum, 0.5 * output_continuum)

    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    calibration = checkpoint["lte_metadata"]["continuum_normalization"]
    assert calibration["learned_gain"] is None
    assert calibration["raster_continuum_normalization"] is False
    assert calibration["type"] == (
        "absolute atlas radiometry in one fixed numerical unit"
    )


def test_training_depth_grid_is_new_stratified_sample_for_every_batch(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    base = torch.linspace(-5.0, 1.0, 25)
    module = lte_module.LTEModule(
        log_tau500=base,
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 17),
        atmosphere_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        depth_sampling_config={},
    )

    first = module.sample_depth_grid(randomize=True)
    second = module.sample_depth_grid(randomize=True)
    torch.testing.assert_close(first[[0, -1]], base[[0, -1]])
    torch.testing.assert_close(second[[0, -1]], base[[0, -1]])
    assert torch.all(first[1:] > first[:-1])
    assert torch.all(second[1:] > second[:-1])
    assert not torch.equal(first[1:-1], second[1:-1])

    left_midpoint = 0.5 * (base[:-2] + base[1:-1])
    right_midpoint = 0.5 * (base[1:-1] + base[2:])
    assert torch.all((first[1:-1] >= left_midpoint) & (first[1:-1] < right_midpoint))

    controlled_draws = torch.linspace(0.0, 0.9, base.numel() - 2)
    with monkeypatch.context() as controlled:
        controlled.setattr(
            lte_module.torch,
            "rand_like",
            lambda values: controlled_draws.to(values),
        )
        controlled_grid = module.sample_depth_grid(randomize=True)
    maximum_shift = 0.5 * (base[1] - base[0])
    expected_interior = base[1:-1] + (2.0 * controlled_draws - 1.0) * maximum_shift
    torch.testing.assert_close(controlled_grid[1:-1], expected_interior)

    coords = torch.tensor([[0.0, 0.0, 0.0]])
    mu = torch.ones(1, 1)
    sampled_result = module.synthesize(coords, mu, randomize_depth=True)
    sampled = sampled_result["atmosphere"]
    deterministic = module.synthesize(coords, mu)["atmosphere"]
    assert not torch.equal(sampled.log_tau500[1:-1], base[1:-1])
    torch.testing.assert_close(deterministic.log_tau500, base)
    torch.testing.assert_close(
        sampled.geometric_height_m,
        module.atmosphere_model.height_mapping(coords, sampled.log_tau500),
    )
    gauge_height = module.atmosphere_model.height_mapping(coords, torch.tensor([0.0]))
    torch.testing.assert_close(
        gauge_height,
        torch.zeros_like(gauge_height),
        rtol=0,
        atol=1.0e-3,
    )
    assert torch.isfinite(sampled.geometric_height_m).all()
    assert torch.var(sampled.geometric_height_m) > 0
    assert torch.isfinite(sampled.temperature).all()
    assert torch.isfinite(sampled.v_los).all()
    assert torch.isfinite(sampled.magnetic_field).all()
    assert torch.isfinite(sampled.microturbulence).all()

    sampled_result["stokes"].square().mean().backward()
    mapping_gradients = [
        parameter.grad
        for parameter in module.atmosphere_model.height_mapping.parameters()
    ]
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for gradient in mapping_gradients
    )
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in mapping_gradients
    )


def test_training_uses_one_fixed_depth_sample_count(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    reference = torch.linspace(-5.0, 1.0, 51)
    module = lte_module.LTEModule(
        log_tau500=reference,
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 17),
        atmosphere_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        depth_sampling_config={
            "sample_count": 37,
        },
    )
    parameter_count = sum(parameter.numel() for parameter in module.parameters())

    assert module.sample_depth_grid().numel() == 37
    torch.testing.assert_close(module.sample_depth_grid(randomize=False), reference)
    assert (
        sum(parameter.numel() for parameter in module.parameters()) == parameter_count
    )


def test_validation_physics_uses_simple_linear_log_tau_grid(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    reference = torch.linspace(-5.0, 1.0, 9)
    component_count = 17
    module = lte_module.LTEModule(
        log_tau500=reference,
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 13),
        atmosphere_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        physics_config={
            "validation_depth_points": component_count,
            "equations": {
                "hse": {"enabled": True, "weight": 1.0},
                "tau_mapping": {"enabled": True, "weight": 1.0},
            },
        },
    )
    grid = module.physics_validation_grid()
    torch.testing.assert_close(grid[[0, -1]], reference[[0, -1]], rtol=0, atol=0)
    assert torch.all(grid[1:] > grid[:-1])
    assert grid.numel() == component_count
    torch.testing.assert_close(
        grid,
        torch.linspace(reference[0], reference[-1], component_count),
    )

    seen_physics_grids = []
    original_volume = module.physics.volume

    def record_physics_grid(*args, **kwargs):
        seen_physics_grids.append(torch.unique(args[3].detach(), sorted=True))
        return original_volume(*args, **kwargs)

    monkeypatch.setattr(module.physics, "volume", record_physics_grid)
    monkeypatch.setattr(module, "log_dict", lambda *args, **kwargs: None)
    coords = torch.tensor(((0.0, -0.2, 0.1), (0.0, 0.3, -0.1)))
    mu = torch.ones(2, 1)
    batch = {
        "coords": coords,
        "mu": mu,
        "stokes": module(coords, mu).detach(),
        "pixel_index": torch.tensor(((0, 0), (0, 1))),
    }
    result = module.validation_step(batch, 0)
    assert torch.isfinite(result["loss"])
    assert len(seen_physics_grids) == 1
    torch.testing.assert_close(seen_physics_grids[0], grid)


def test_validation_logs_only_primary_losses(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 9),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 13),
        atmosphere_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
    )
    logged = {}
    monkeypatch.setattr(
        module, "log_dict", lambda metrics, *args, **kwargs: logged.update(metrics)
    )
    coords = torch.tensor(((0.0, -0.2, 0.1), (0.0, 0.3, -0.1)))
    mu = torch.ones(2, 1)
    target = module(coords, mu).detach()
    module.validation_step(
        {
            "coords": coords,
            "mu": mu,
            "stokes": target,
            "pixel_index": torch.tensor(((0, 0), (0, 1))),
        },
        0,
    )
    assert set(logged) == {
        "valid.I",
        "valid.Q",
        "valid.U",
        "valid.V",
        "valid.loss",
        "valid.stokes_loss",
    }


def test_auto_learning_rate_schedule_uses_actual_trainer_steps(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 9),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 17),
        atmosphere_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        lr_params={"start": 1.0e-3, "end": 1.0e-4, "iterations": "auto"},
    )
    module._trainer = SimpleNamespace(estimated_stepping_batches=40)
    configured = module.configure_optimizers()
    scheduler = configured["lr_scheduler"]["scheduler"]
    assert module.resolved_lr_iterations == 40
    assert scheduler.gamma**40 == pytest.approx(0.1)


def test_resume_hook_does_not_apply_custom_checkpoint_checks(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 9),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 17),
        atmosphere_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
    )
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    checkpoint["optimizer_states"] = [
        {"state": {0: {"exp_avg": torch.tensor(float("nan"))}}}
    ]
    module.on_load_checkpoint(checkpoint)


def test_failed_backward_stops_before_optimizer_step(
    monkeypatch,
):
    import pme.train.lte_module as lte_module

    class _FiniteForwardNaNBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, torch.nan)

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-5.0, 1.0, 9),
        wavelength_angstrom=torch.linspace(6301.2, 6302.8, 17),
        atmosphere_config={
            "coordinate_mode": "log_tau",
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            },
        },
    )
    original_forward = module.synthesizer.forward

    def unstable_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if isinstance(result, tuple):
            stokes, diagnostics = result
            return _FiniteForwardNaNBackward.apply(stokes), diagnostics
        return _FiniteForwardNaNBackward.apply(result)

    monkeypatch.setattr(module.synthesizer, "forward", unstable_forward)
    monkeypatch.setattr(module, "log_dict", lambda *args, **kwargs: None)
    batch = {
        "coords": torch.tensor(((0.0, -0.2, 0.1), (0.0, 0.3, -0.1))),
        "mu": torch.ones(2, 1),
        "stokes": torch.zeros(2, 4, 17),
    }
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()

    with pytest.raises(FloatingPointError) as captured:
        module.on_after_backward()
    message = str(captured.value)
    assert "Non-finite LTE gradients before the optimizer step" in message
    assert "atmosphere_model.network" in message


def test_lte_objective_matches_hmi_asinh_scaling_and_equal_component_weights(
    monkeypatch,
):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-4.0, 1.0, 5),
        wavelength_angstrom=torch.linspace(6302.0, 6302.2, 3),
        atmosphere_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        normalization_config={"asinh_alphas": {"Q": 1.0e-2, "U": 1.0e-2, "V": 1.0e-2}},
        stokes_loss_config={"type": "mse"},
        weight_config={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0},
    )
    prediction = torch.tensor(
        [[[1.0, 1.0, 1.0], [0.01, 0.01, 0.01], [0.10, 0.10, 0.10], [1.0, 1.0, 1.0]]]
    )
    target = torch.zeros_like(prediction)

    components, total = module._stokes_objective(prediction, target)
    alpha = prediction.new_tensor(0.01)
    denominator = torch.asinh(1.0 / alpha)
    expected = torch.stack(
        (
            prediction.new_tensor(1.0),
            (torch.asinh(prediction.new_tensor(0.01) / alpha) / denominator).square(),
            (torch.asinh(prediction.new_tensor(0.10) / alpha) / denominator).square(),
            (torch.asinh(prediction.new_tensor(1.0) / alpha) / denominator).square(),
        )
    )
    torch.testing.assert_close(components, expected)
    torch.testing.assert_close(total, expected.sum())
    torch.testing.assert_close(module.stokes_weights, torch.ones(4))


def test_lte_module_wavelength_mask_excludes_unsupported_blend(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    wavelength = torch.tensor((6302.00, 6302.05, 6302.10, 6302.15, 6302.20))
    module = lte_module.LTEModule(
        log_tau500=torch.linspace(-4.0, 1.0, 5),
        wavelength_angstrom=wavelength,
        atmosphere_config={
            "model_config": {
                "dim": 8,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        wavelength_exclude_windows_angstrom=((6302.04, 6302.14),),
    )
    assert module.wavelength_weights.tolist() == [1.0, 0.0, 0.0, 1.0, 1.0]

    prediction = torch.zeros(2, 4, 5)
    target = torch.zeros_like(prediction)
    # Arbitrarily large discrepancies inside the excluded molecular window
    # must not affect the objective.
    target[..., 1:3] = 1.0e6
    per_component = torch.tensor((1.0, 2.0, 3.0, 4.0))
    target[:, :, 0] = per_component
    components, objective = module._stokes_objective(prediction, target)
    expected = per_component.square() / 3.0
    torch.testing.assert_close(components, expected)
    torch.testing.assert_close(objective, expected.sum())

    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["lte_metadata"]["wavelength_objective"][
        "exclude_windows_angstrom"
    ] == [[6302.04, 6302.14]]


def test_lte_module_rejects_an_objective_with_no_active_wavelengths(monkeypatch):
    import pme.train.lte_module as lte_module

    monkeypatch.setattr(lte_module, "LTESynthesizer", _StubSynthesizer)
    with pytest.raises(ValueError, match="At least one wavelength"):
        lte_module.LTEModule(
            log_tau500=torch.linspace(-4.0, 1.0, 5),
            wavelength_angstrom=torch.linspace(6302.0, 6302.2, 5),
            atmosphere_config={
                "model_config": {
                    "dim": 8,
                    "n_layers": 2,
                    "encoding_config": {"type": "identity"},
                }
            },
            wavelength_exclude_windows_angstrom=((6301.0, 6303.0),),
        )
