"""A divergence-free horizontal current sheet must become costly as it sharpens."""

import torch

from prom3theus.inversion.constraints.magnetofluid import MagnetofluidConstraints


class RotatingSheet(torch.nn.Module):
    solar_radius_m = 10.0

    def __init__(self, width):
        super().__init__()
        self.width = torch.nn.Parameter(torch.tensor(width))

    def evaluate_position_rsun(self, position_rsun, time_hours=None):
        z = position_rsun[:, 2] * self.solar_radius_m
        angle = 0.5 * torch.pi * torch.tanh((z - 20.0) / self.width)
        zero = 0 * z
        return {
            "temperature": zero + 5000,
            "gas_pressure": zero + 1,
            "magnetic_field": torch.stack((angle.cos(), angle.sin(), zero), -1),
            "velocity_field": zero[:, None].expand(-1, 3),
        }


def test_quadratic_curl_penalty_discourages_sheet_concentration():
    constraints = MagnetofluidConstraints(
        {
            "magnetic_current_free": {"enabled": True},
            "magnetic_divergence": {"enabled": True},
        },
        normalization={"length_m": 1.0, "magnetic_field_floor_gauss": 0.1},
        vector_basis_matches_spatial_coordinates=True,
    )
    z = torch.linspace(10.0, 30.0, 1024)
    position = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z), -1)
    losses = []
    for width in (1.0, 0.25):
        model = RotatingSheet(width)
        result = constraints.volume(
            model, position, torch.zeros(len(z), 1),
            height_group_shape=(len(z), 1),
        )
        assert result.losses["magnetic_divergence"] == 0
        loss = result.losses["magnetic_current_free"]
        loss.backward()
        assert torch.isfinite(model.width.grad)
        assert model.width.grad < 0  # Descent broadens the sheet.
        losses.append(loss.detach())
    # Same field strength and total rotation, four times thinner: ~4x the loss.
    torch.testing.assert_close(losses[1] / losses[0], torch.tensor(4.0), rtol=0.01, atol=0)
