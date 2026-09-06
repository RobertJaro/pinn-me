"""Smoke-test LTE synthesis and objective gradients on a compact atmosphere.

This is deliberately not an end-to-end PINN inversion: gas pressure is fixed
and twelve explicit coefficients replace the Z and F coordinate networks.  It
isolates differentiability and parameter sensitivity of the two-line forward
synthesis and configured Stokes objective.  The operational Hinode tests cover
the loader, instrument, random-depth quadrature, coordinate networks, and
physics collocation separately.
"""

from __future__ import annotations

import torch

from prom3theus.diagnostics.synthetic import synthetic_temperature_profile
from prom3theus.inversion.objective import StokesObjective
from prom3theus.rt import (
    LTESynthesizer,
    OpticalDepthPath,
    StratifiedAtmosphere,
    air_to_vacuum_angstrom,
    planck_lambda,
)


FIELD_NAMES = (
    "temperature_k",
    "v_los_m_per_s",
    "b_x_gauss",
    "b_y_gauss",
    "b_los_gauss",
    "microturbulence_m_per_s",
)
PARAMETER_SCALES = (500.0, 2_000.0, 1_000.0, 1_000.0, 1_000.0, 500.0)


def run_recovery(steps: int = 500, device: str = "cpu") -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(0)
    dtype = torch.float32
    target_device = torch.device(device)
    depth = 11
    log_tau500 = torch.linspace(-4.0, 1.0, depth, dtype=dtype, device=target_device)
    reference = synthetic_temperature_profile(log_tau500).unsqueeze(0)
    gas_pressure = torch.logspace(
        -0.5, 5.0, depth, dtype=dtype, device=target_device
    ).unsqueeze(0)
    wavelength = torch.linspace(6300.9, 6303.05, 81, dtype=dtype, device=target_device)
    synthesizer = LTESynthesizer(
        log_tau500,
        line_ids=("FeI_6301.5008", "FeI_6302.4932"),
    ).to(target_device)

    normalized_depth = (
        2.0 * (log_tau500 - log_tau500[0]) / (log_tau500[-1] - log_tau500[0]) - 1.0
    )
    basis = torch.stack((torch.ones_like(normalized_depth), normalized_depth), dim=0)
    scales = log_tau500.new_tensor(PARAMETER_SCALES)[:, None]
    # Rows are T, v_LOS, Bx, By, B_LOS, xi; columns are constant and slope.
    truth_scaled = log_tau500.new_tensor(
        (
            (0.30, 0.18),
            (0.45, -0.20),
            (0.30, 0.12),
            (-0.20, 0.10),
            (0.60, -0.16),
            (0.40, 0.16),
        )
    )

    def make_atmosphere(parameters_scaled: torch.Tensor) -> StratifiedAtmosphere:
        physical_profiles = (parameters_scaled * scales) @ basis
        temperature = reference + physical_profiles[0].unsqueeze(0)
        velocity = physical_profiles[1].unsqueeze(0)
        magnetic = physical_profiles[2:5].transpose(0, 1).unsqueeze(0)
        microturbulence = 1_000.0 + physical_profiles[5].unsqueeze(0)
        return StratifiedAtmosphere(
            log_tau500=log_tau500,
            temperature=temperature,
            velocity_field=torch.stack(
                (torch.zeros_like(velocity), torch.zeros_like(velocity), -velocity),
                dim=-1,
            ),
            microturbulence=microturbulence,
            magnetic_field=magnetic,
            gas_pressure=gas_pressure,
        )

    with torch.no_grad():
        # Match production synthesis by normalizing Planck radiance inside the
        # logarithmic evaluation. Leaving the ~1e13 SI radiance in the graph
        # makes the float32 MSE backward overflow before Adam can take a step.
        radiance_scale = planck_lambda(
            reference[..., -1], air_to_vacuum_angstrom(wavelength)
        ).median()
        target = synthesizer(
            make_atmosphere(truth_scaled),
            wavelength,
            path=OpticalDepthPath(mu=1.0),
            radiance_scale=radiance_scale,
        )
    inferred_scaled = torch.nn.Parameter(0.05 * torch.randn_like(truth_scaled))
    optimizer = torch.optim.Adam((inferred_scaled,), lr=0.03)
    stokes_loss = StokesObjective(
        type="huber",
        stokes_sigmas={"I": 5.0e-3, "Q": 2.0e-3, "U": 2.0e-3, "V": 2.0e-3},
        huber_delta=1.0,
    ).to(target_device)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        prediction = synthesizer(
            make_atmosphere(inferred_scaled),
            wavelength,
            path=OpticalDepthPath(mu=1.0),
            radiance_scale=radiance_scale,
        )
        # Match the shipped LTE configurations: noise-standardized Huber
        # residuals with equal component preference.
        loss = stokes_loss(prediction, target).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    recovered_scaled = inferred_scaled.detach().clone()
    # Zeeman Q/U encode transverse azimuth modulo pi, so simultaneous reversal
    # of Bx and By is observationally identical. Report the representative on
    # the same azimuth branch as the target before evaluating recovery error.
    transverse = slice(2, 4)
    direct_error = (
        (recovered_scaled[transverse] - truth_scaled[transverse]).square().sum()
    )
    reversed_error = (
        (recovered_scaled[transverse] + truth_scaled[transverse]).square().sum()
    )
    transverse_branch_reversed = bool(reversed_error < direct_error)
    if transverse_branch_reversed:
        recovered_scaled[transverse] = -recovered_scaled[transverse]

    truth = truth_scaled * scales
    recovered = recovered_scaled * scales
    truth_profiles = (truth @ basis).detach()
    recovered_profiles = (recovered @ basis).detach()
    coefficients = {}
    profile_errors = {}
    for index, name in enumerate(FIELD_NAMES):
        coefficients[name] = {
            "truth": {
                "constant": float(truth[index, 0]),
                "slope": float(truth[index, 1]),
            },
            "recovered": {
                "constant": float(recovered[index, 0]),
                "slope": float(recovered[index, 1]),
            },
        }
        profile_errors[name] = float(
            torch.sqrt(
                torch.mean((recovered_profiles[index] - truth_profiles[index]).square())
            )
        )
    return {
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction": losses[0] / max(losses[-1], torch.finfo(dtype).tiny),
        "coefficients": coefficients,
        "profile_rmse": profile_errors,
        "transverse_azimuth_branch_reversed": transverse_branch_reversed,
        "max_scaled_error": float((recovered_scaled - truth_scaled).abs().max()),
    }


__all__ = ["run_recovery"]
