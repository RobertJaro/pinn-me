"""Smoke-test LTE synthesis and objective gradients on a compact atmosphere.

This is deliberately not an end-to-end PINN inversion: gas pressure is fixed
and twelve explicit coefficients replace the Z and F coordinate networks.  It
isolates differentiability and parameter sensitivity of the two-line forward
synthesis and configured Stokes objective.  The operational Hinode tests cover
the loader, instrument, random-depth quadrature, coordinate networks, and
physics collocation separately.
"""

from __future__ import annotations

import argparse
import json

import torch

from pme.lte import LTESynthesizer, StratifiedAtmosphere
from pme.lte.examples import synthetic_temperature_profile
from pme.model import NormalizationModule
from pme.train.stokes_loss import StokesLossModule


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
    wavelength = torch.linspace(
        6300.9, 6303.05, 81, dtype=dtype, device=target_device
    )
    synthesizer = LTESynthesizer(log_tau500).to(target_device)

    normalized_depth = 2.0 * (log_tau500 - log_tau500[0]) \
        / (log_tau500[-1] - log_tau500[0]) - 1.0
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
        target = synthesizer(make_atmosphere(truth_scaled), wavelength)
    inferred_scaled = torch.nn.Parameter(0.05 * torch.randn_like(truth_scaled))
    optimizer = torch.optim.Adam((inferred_scaled,), lr=0.03)
    normalization = NormalizationModule(
        asinh_alphas={"Q": 1.0e-2, "U": 1.0e-2, "V": 1.0e-2}
    ).to(target_device)
    stokes_loss = StokesLossModule(type="mse")
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        prediction = synthesizer(make_atmosphere(inferred_scaled), wavelength)
        # Match config/hmi/hmi_fd_20240323.yaml and the LTE inversion:
        # linear I, asinh-scaled Q/U/V, and equal component weights.
        loss = stokes_loss(prediction, target, normalization).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    truth = truth_scaled * scales
    recovered = inferred_scaled.detach() * scales
    truth_profiles = (truth @ basis).detach()
    recovered_profiles = (recovered @ basis).detach()
    coefficients = {}
    profile_errors = {}
    for index, name in enumerate(FIELD_NAMES):
        coefficients[name] = {
            "truth": {"constant": float(truth[index, 0]), "slope": float(truth[index, 1])},
            "recovered": {
                "constant": float(recovered[index, 0]),
                "slope": float(recovered[index, 1]),
            },
        }
        profile_errors[name] = float(
            torch.sqrt(torch.mean((recovered_profiles[index] - truth_profiles[index]).square()))
        )
    return {
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction": losses[0] / max(losses[-1], torch.finfo(dtype).tiny),
        "coefficients": coefficients,
        "profile_rmse": profile_errors,
        "max_scaled_error": float((inferred_scaled.detach() - truth_scaled).abs().max()),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit unsuccessfully unless the compact atmosphere is recovered",
    )
    args = parser.parse_args(argv)
    result = run_recovery(steps=args.steps, device=args.device)
    print(json.dumps(result, indent=2))
    if args.check and (
        result["final_loss"] >= 5.0e-8
        or result["max_scaled_error"] >= 0.025
    ):
        raise SystemExit("synthetic recovery did not reach the validation tolerance")


if __name__ == "__main__":
    main()
