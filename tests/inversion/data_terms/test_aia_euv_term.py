import math

import torch
from torch import nn

from prom3theus.inversion.data_terms import AIAEUVObservationTerm
from prom3theus.observations import SceneContract


RADIUS_M = 6.957e8


class _EOS:
    def electron_density(self, temperature, pressure):
        del temperature
        return pressure


class _Atmosphere(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0e6), dtype=torch.float64)
        )
        self.log_pressure = nn.Parameter(
            torch.tensor(math.log(1.0e15), dtype=torch.float64)
        )
        self.magnetic_parameter = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        self.register_buffer(
            "solar_radius_m", torch.tensor(RADIUS_M, dtype=torch.float64)
        )
        self.solar_radius_value_m = RADIUS_M
        self.shell_height_bounds_Mm = (1.0, -0.1)
        self.thermodynamic_eos = _EOS()

    def evaluate_chart_height_points(self, coordinates, height):
        del coordinates
        return {
            "temperature": self.log_temperature.exp().expand_as(height),
            "gas_pressure": self.log_pressure.exp().expand_as(height),
        }


def _scene():
    return SceneContract(
        scene_basis=torch.eye(3, dtype=torch.float64),
        solar_radius_m=RADIUS_M,
        reference_time_tai_seconds=1_700_000_000.0,
        spatial_coordinate_center_mm=(0.0, 0.0),
        spatial_coordinate_scale_mm=(10.0, 10.0),
        time_coordinate_center_hours=0.0,
        time_coordinate_scale_hours=1.0,
        height_bounds_m=(-1.0e5, 1.0e6),
    )


def _batch():
    count = 3
    surface = torch.zeros(count, 3, dtype=torch.float64)
    surface[:, 2] = RADIUS_M
    ray = torch.zeros_like(surface)
    ray[:, 2] = -1.0
    return {
        "intensity": torch.tensor([120.0, 250.0, 180.0], dtype=torch.float64),
        "uncertainty": torch.tensor([3.0, 4.0, 3.5], dtype=torch.float64),
        "ray_direction": ray,
        "surface_position_m": surface,
        "absolute_tai_seconds": torch.full(
            (count,), 1_700_000_000.0, dtype=torch.float64
        ),
        "channel_angstrom": torch.tensor([171, 193, 211]),
        "channel_index": torch.tensor([0, 1, 2]),
        "image_index": torch.tensor([0, 1, 2]),
        "pixel_index": torch.tensor([[0, 0], [0, 0], [0, 0]]),
    }


def _term(atmosphere, *, intensity_scales=None):
    return AIAEUVObservationTerm(
        atmosphere_model=atmosphere,
        scene=_scene(),
        observation_id="aia-fixture",
        channels_angstrom=(171, 193, 211),
        response_resource="aia_euv_v1:aia_temperature_response",
        ray_samples=16,
        height_sampling_power=2.0,
        asinh_scales=(20.0, 20.0, 20.0),
        channel_weights=(1.0, 2.0, 1.0),
        intensity_scales=intensity_scales,
    ).to(torch.float64)


def test_aia_term_forward_loss_diagnostics_and_unweighted_nuisance_prior():
    atmosphere = _Atmosphere()
    term = _term(atmosphere)
    result = term.evaluate_batch(_batch())
    assert torch.isfinite(result.likelihood_loss)
    assert set(result.component_losses) == {
        "channel_171",
        "channel_193",
        "channel_211",
    }
    assert result.sample_count == 3
    assert result.diagnostics["prediction_raw"].shape == (3,)
    assert result.diagnostics["contribution"].shape == (3, 16)
    assert term.nuisance_losses()["calibration_prior"].item() == 0.0
    torch.testing.assert_close(
        result.diagnostics["prediction_raw"],
        result.diagnostics["prediction_calibrated"],
    )


def test_channel_max_normalization_matches_physical_reference_max_and_gradients():
    physical_atmosphere = _Atmosphere().float()
    normalized_atmosphere = _Atmosphere().float()
    physical = _term(physical_atmosphere).float()
    maxima = torch.tensor([1.0e4, 2.0e4, 3.0e4])
    normalized = _term(normalized_atmosphere, intensity_scales=maxima.tolist()).float()
    with torch.no_grad():
        physical.calibration.common_log_gain.fill_(0.2)
        normalized.calibration.common_log_gain.fill_(0.2)
    batch = {
        name: value.float()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for name, value in _batch().items()
    }
    target = batch["intensity"]
    original = physical.evaluate_batch(batch)
    scaled = normalized.evaluate_batch({**batch, "intensity": target / maxima})
    assert scaled.diagnostics["target"].dtype == torch.float32
    assert scaled.diagnostics["target"].abs().max() <= 1
    torch.testing.assert_close(
        scaled.diagnostics["prediction_raw"],
        original.diagnostics["prediction_raw"] / maxima,
    )
    # In physical count-rate units, the unit anchor is the channel maximum,
    # not one DN/s. This is equivalent to f(x)=asinh(x/a)/asinh(1/a) after scaling.
    physical_scale = physical.objective.asinh_scales.to(target)
    expected_residual = (
        torch.asinh(original.diagnostics["prediction_calibrated"] / physical_scale)
        - torch.asinh(target / physical_scale)
    ) / torch.asinh(maxima / physical_scale)
    expected_loss = (
        expected_residual.square() * physical.objective.channel_weights.to(target)
    ).sum()
    torch.testing.assert_close(scaled.diagnostics["asinh_residual"], expected_residual)
    torch.testing.assert_close(scaled.likelihood_loss, expected_loss)
    old_gradients = torch.autograd.grad(
        expected_loss,
        (*physical_atmosphere.parameters(), *physical.parameters()),
        allow_unused=True,
    )
    new_gradients = torch.autograd.grad(
        scaled.likelihood_loss,
        (*normalized_atmosphere.parameters(), *normalized.parameters()),
        allow_unused=True,
    )
    for old, new in zip(old_gradients, new_gradients, strict=True):
        if old is None:
            assert new is None
        else:
            assert torch.isfinite(new).all()
            torch.testing.assert_close(new, old, atol=1e-6, rtol=1e-4)


def test_normalization_does_not_clip_synthesis(monkeypatch):
    from dataclasses import replace

    maxima = torch.tensor([120.0, 250.0, 180.0], dtype=torch.float64)
    term = _term(_Atmosphere(), intensity_scales=maxima.tolist())
    batch = _batch()
    result = term.synthesize(batch)
    monkeypatch.setattr(
        term,
        "synthesize",
        lambda batch, **kwargs: replace(result, raw_prediction=maxima * 4.0),
    )
    normalized = term.evaluate_batch(
        {**batch, "intensity": batch["intensity"] / maxima}
    )
    torch.testing.assert_close(
        normalized.diagnostics["prediction_calibrated"], torch.full_like(maxima, 4.0)
    )


def test_aia_term_gradients_reach_shared_temperature_and_density_but_not_b_directly():
    atmosphere = _Atmosphere()
    term = _term(atmosphere)
    loss = term.evaluate_batch(_batch()).likelihood_loss
    gradients = torch.autograd.grad(
        loss,
        (
            atmosphere.log_temperature,
            atmosphere.log_pressure,
            atmosphere.magnetic_parameter,
        ),
        allow_unused=True,
    )
    assert gradients[0] is not None and torch.isfinite(gradients[0])
    assert gradients[1] is not None and torch.isfinite(gradients[1])
    assert gradients[0].abs() > 0
    assert gradients[1].abs() > 0
    assert gradients[2] is None


def test_aia_term_rejects_channel_index_wavelength_disagreement():
    term = _term(_Atmosphere())
    batch = _batch()
    batch["channel_angstrom"] = torch.tensor([193, 171, 211])
    try:
        term.evaluate_batch(batch)
    except ValueError as error:
        assert "channel_angstrom" in str(error)
    else:  # pragma: no cover - assertion is clearer than a fixture dependency
        raise AssertionError("mismatched channel metadata was accepted")


def test_aia_term_streams_single_channel_only_through_diagnostic_boundary():
    term = _term(_Atmosphere())
    batch = {
        name: value[:1] if isinstance(value, torch.Tensor) else value
        for name, value in _batch().items()
    }

    with torch.no_grad():
        result = term.evaluate_diagnostic_batch(batch)

    assert result.sample_count == 1
    assert set(result.component_losses) == {"channel_171"}
    assert set(result.metrics) == {
        "171_gain",
        "asinh_rmse",
    }
    assert torch.isfinite(result.likelihood_loss)


def test_disabled_calibration_emits_no_nuisance_loss():
    term = _term(_Atmosphere())
    term.calibration.enabled = False
    assert term.nuisance_losses() == {}
