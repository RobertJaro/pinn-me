"""Precision of calibrated spectral setup and the complete training graph."""

from contextlib import nullcontext
import copy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from prom3theus.validation.reference_assembly import build_forward_assembly
from prom3theus.validation.reference_configuration import DepthSamplingSettings
from prom3theus.validation.reference_forward import ForwardRuntime
from prom3theus.muram.hmi_filter import build_idealized_hmi_filter
from prom3theus.rt import StratifiedAtmosphereModel
from prom3theus.rt.opacity import planck_lambda


class Float32Only(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value, torch.Tensor):
                assert value.dtype not in (torch.float64, torch.complex128), func
        return result


def model(representation="direct"):
    torch.manual_seed(71)
    return StratifiedAtmosphereModel(
        shell_height_bounds_megameter=(1.5, -0.1),
        magnetic_representation=representation,
        scene_geometry_config={
            "solar_radius_m": 695700000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "mlp",
            "dim": 8,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    )


@pytest.mark.parametrize(
    "representation", ["direct", "vector_potential", "potential_delta"]
)
def test_local_ray_heights_and_fields_match_double(representation):
    low = model(representation)
    high = copy.deepcopy(low).double()
    coords = torch.tensor([[0.0, 0.0, 0.0], [35.0, 22.0, 0.0]])
    rays = torch.tensor([[0.0, 0.0, -1.0], [0.6, 0.0, -0.8]])
    heights = torch.linspace(1.5e6, -1e5, 41)
    a, trace = low.trace_rays(coords, rays, heights)
    b, reference = high.trace_rays(coords.double(), rays.double(), heights.double())
    # Previously 91 m and 87 m respectively, including the disk-centre ray.
    torch.testing.assert_close(
        trace.geometric_height_m.double(),
        reference.geometric_height_m,
        atol=0.6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        trace.distance_m.double(), reference.distance_m, atol=0.6, rtol=1e-6
    )
    for name in ("temperature", "gas_pressure", "magnetic_field", "velocity_field"):
        torch.testing.assert_close(
            getattr(a, name).double(), getattr(b, name), rtol=2e-5, atol=2e-4
        )


@pytest.mark.parametrize("instrument_type", ["hmi", "hinode"])
@pytest.mark.parametrize(
    "representation", ["direct", "vector_potential", "potential_delta"]
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_training_profiles_and_parameter_gradients_match_double(
    instrument_type, representation, device
):
    atmosphere = model(representation)
    if instrument_type == "hmi":
        op, observed, weights, continuum = build_idealized_hmi_filter(n_quadrature=81)
        config = {
            "type": "hmi_filter_profiles",
            "quadrature_wavelength_angstrom": op.quadrature_wavelength_angstrom.tolist(),
            "inner_half_width_angstrom": 0.65,
            "magnetic_azimuth_offset_deg": 90.0,
        }
        line_ids = ["FeI_6173.3352"]
        response = {
            "spectral_weights": weights[None],
            "continuum_weights": continuum[None],
        }
    else:
        observed = torch.linspace(6301.1, 6302.8, 61, dtype=torch.float64)
        config = {
            "type": "hinode_sp",
            "spectral_psf": {
                "type": "gaussian",
                "fwhm_angstrom": 0.03,
                "oversample": 2,
                "truncate_sigma": 4,
            },
        }
        line_ids = ["FeI_6301.5008", "FeI_6302.4932"]
        response = {}
    results = []
    for dtype in (torch.float64, torch.float32):
        current = copy.deepcopy(atmosphere).to(device=device, dtype=dtype)
        assembly = build_forward_assembly(
            atmosphere_model=current,
            synthesizer_config={"line_ids": line_ids},
            instrument_config=config,
            velocity_synthesis_mode="carrington_registered_relative",
            depth_sampling=DepthSamplingSettings(25, True, 16, 0.05),
            wavelength_angstrom=observed,
        )
        runtime = ForwardRuntime(
            torch.linspace(-5, 1, 25, device=device, dtype=dtype),
            observed.to(device=device, dtype=dtype),
            assembly.synthesis_wavelength_angstrom.to(device=device, dtype=dtype),
            1e13,
            0.0,
            0.0,
        )
        response_tensors = {
            k: v.to(device=device, dtype=dtype) for k, v in response.items()
        }
        # Only static calibration preparation may use float64. The entire
        # atmosphere -> refined ray -> LTE -> instrument -> backward stays f32.
        with Float32Only() if dtype == torch.float32 else nullcontext():
            result = assembly.composition.synthesize(
                torch.tensor([[12.0, 8.0, 0.0]], device=device, dtype=dtype),
                runtime=runtime,
                ray_direction=torch.tensor(
                    [[0.0, 0.0, -1.0]], device=device, dtype=dtype
                ),
                stokes_basis=torch.eye(3, device=device, dtype=dtype)[None],
                removed_solar_los_velocity_m_per_s=torch.zeros(
                    1, device=device, dtype=dtype
                ),
                instrument_response=response_tensors,
            )["stokes"]
            loss = result.square().mean()
            gradients = torch.autograd.grad(loss, tuple(current.network.parameters()))
        results.append(
            (
                result.detach().double(),
                torch.cat([g.flatten() for g in gradients]).double(),
            )
        )
    reference, actual = results
    assert ((actual[0] - reference[0]).abs() / reference[0][:, :1].abs()).max() < 1e-4
    # Normalize the gradient error by the vector norm: relative elementwise
    # errors are meaningless for parameters whose true sensitivity is zero.
    assert torch.isfinite(actual[1]).all()
    assert (
        torch.linalg.vector_norm(actual[1] - reference[1])
        / torch.linalg.vector_norm(reference[1])
        < 5e-4
    )


def test_planck_scaled_values_and_gradients_across_solar_temperature_range():
    wave = torch.tensor(
        [940.0, 1710.0, 5000.0, 6173.0, 6302.0, 10830.0], dtype=torch.float64
    )
    temperature = torch.tensor(
        [2500.0, 4500.0, 6000.0, 1e4, 1e5, 1e6, 1e7],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference = planck_lambda(temperature, wave, radiance_scale=1e13)
    expected_grad = torch.autograd.grad(reference.sum(), temperature)[0]
    low = temperature.detach().float().requires_grad_(True)
    actual = planck_lambda(low, wave.float(), radiance_scale=1e13)
    actual_grad = torch.autograd.grad(actual.sum(), low)[0]
    torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-12)
    torch.testing.assert_close(
        actual_grad.double(), expected_grad, rtol=1e-5, atol=1e-12
    )



def test_planck_preserves_small_unscaled_radiances_and_scales():
    temperature = torch.tensor([1000.], dtype=torch.float64)
    wave = torch.tensor([1100.], dtype=torch.float64)
    for scale in (None, 1e-38):
        expected = planck_lambda(temperature, wave, radiance_scale=scale)
        actual = planck_lambda(temperature.float(), wave.float(), radiance_scale=scale)
        assert torch.isfinite(actual).all() and (actual > 0).all()
        torch.testing.assert_close(actual.double(), expected, rtol=5e-5, atol=1e-44)
