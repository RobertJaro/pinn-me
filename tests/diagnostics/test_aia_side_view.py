from types import SimpleNamespace

import torch

from prom3theus.diagnostics.aia_side_view import render_side_view, synthesize_side_view
from prom3theus.inversion.sampling import SphericalShellDomain


class UniformAtmosphere(torch.nn.Module):
    def __init__(self, domain):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.domain = domain
        self.calls = []
        self.thermodynamic_eos = SimpleNamespace(
            electron_density=lambda temperature, pressure: torch.ones_like(temperature)
        )

    def evaluate_position_rsun(self, points, *, time_hours):
        assert not torch.is_grad_enabled()
        assert torch.all(time_hours == 0.5)
        r = points.norm(dim=-1)
        assert torch.all(r >= 1 - 1e-6)
        assert torch.all(r <= 1 + 50e6 / self.domain.solar_radius_m + 1e-6)
        self.calls.append(len(points))
        return {
            "temperature": torch.ones_like(r) * 1e6,
            "gas_pressure": torch.ones_like(r),
        }


def test_side_view_integration_and_chunking(tmp_path):
    domain = SphericalShellDomain(
        0.0, (-0.03, 0.03), (-0.02, 0.02), (0.0, 1.0), (-0.1, 50.0), 6.957e8
    )
    atmosphere = UniformAtmosphere(domain)

    def emission(temperature, density, distance, *, sample_dim):
        integral = torch.trapezoid(density, distance, dim=sample_dim) / 1e6
        return integral[:, None] * torch.tensor([1.0, 2.0, 3.0])

    term = SimpleNamespace(
        ray_samples=192,
        emission_operator=emission,
        calibration=SimpleNamespace(gains=torch.tensor([2.0, 1.0, 1.0])),
        intensity_scales=torch.tensor([2.0, 2.0, 3.0]),
        objective=SimpleNamespace(asinh_scales=torch.ones(3)),
    )
    image, extent = synthesize_side_view(
        atmosphere, term, domain, 0.5, max_pixels=256, batch_size=7
    )
    other, _ = synthesize_side_view(
        atmosphere, term, domain, 0.5, max_pixels=256, batch_size=256
    )
    assert image.shape == (16, 16, 3)
    assert not image.requires_grad
    assert torch.isfinite(image).all()
    assert image.max() > 0
    assert (image == 0).any()  # Empty rays outside the angular wedge.
    torch.testing.assert_close(image, other)
    torch.testing.assert_close(image[..., 0], image[..., 1])
    torch.testing.assert_close(image[..., 0], image[..., 2])
    # Central ray above the limb: uniform density integrates its wedge chord.
    z = extent[2] + (8.5 / 16) * (extent[3] - extent[2])
    expected_length = (
        2 * (domain.solar_radius_m / 1e6 + z) * torch.tan(torch.tensor(0.03))
    )
    torch.testing.assert_close(
        torch.sinh(image[8, 8, 0] * torch.asinh(torch.tensor(1.0))), expected_length, rtol=0.03, atol=0
    )
    path = tmp_path / "side.png"
    render_side_view(
        image,
        extent,
        (171, 193, 211),
        path,
        time_hours=0.5,
        height_markers=(1.5, 2.5),
        dpi=50,
    )
    assert path.stat().st_size > 1000


def test_side_view_wandb_logging(tmp_path):
    from prom3theus.application.joint_training import _log_validation

    folder = tmp_path / "diagnostics"
    folder.mkdir()
    (folder / "coronal_aia_side_view.png").touch()
    logged = []
    logger = SimpleNamespace(
        log_metrics=lambda *args, **kwargs: None,
        log_image=lambda **kwargs: logged.append(kwargs),
    )
    _log_validation(
        logger,
        {
            "metrics": {},
            "rendering": {
                "artifacts": [
                    {
                        "stream_id": "coronal",
                        "path": str(folder / "coronal_aia_side_view.png"),
                    }
                ]
            },
        },
        42,
    )
    assert logged[0]["key"] == "Observation comparison"
    assert logged[0]["step"] == 42


def test_side_view_real_aia_response():
    from prom3theus.instruments.aia_euv.operator import AIAEmissionOperator

    domain = SphericalShellDomain(
        0.0, (-0.03, 0.03), (-0.02, 0.02), (0.0, 1.0), (-0.1, 50.0), 6.957e8
    )
    atmosphere = UniformAtmosphere(domain)
    atmosphere.thermodynamic_eos.electron_density = lambda temperature, pressure: (
        torch.ones_like(temperature) * 1e15
    )
    term = SimpleNamespace(
        ray_samples=192,
        emission_operator=AIAEmissionOperator(
            (171, 193, 211), response_resource="aia_euv_v1:aia_temperature_response"
        ),
        calibration=SimpleNamespace(gains=torch.ones(3)),
        intensity_scales=torch.ones(3) * 1000,
        objective=SimpleNamespace(asinh_scales=torch.ones(3) * 0.01),
    )
    image, _ = synthesize_side_view(
        atmosphere, term, domain, 0.5, max_pixels=64, batch_size=8
    )
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert torch.all(image.amax(dim=(0, 1)) > 0)
