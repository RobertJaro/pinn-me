"""Deterministic self-consistency control; not a failed-run checkpoint replay.

Fits four fixed FALC-based, strong-field profiles with the production 256x8
SIREN at fixed geometric quadrature (41 nodes by default). Both representations use
the same initialized network and optimizer; spectra use the production HMI
operator with idealized response curves. Optional --target-npz fits saved native
profiles at x-major --pixel-indices, in atlas units by default. This is a
plane-parallel column control; full oblique-ray composition is not exercised.
It does not establish physical correctness of an external target file.
Run from repository root with
PYTHONPATH=src python -m prom3theus.diagnostics.commands.check_stokes_recovery --representation vector_potential --warmup 1000.
"""

from dataclasses import replace
from pathlib import Path
import argparse
import json
import math
import time
import torch
import numpy as np
from prom3theus.inversion.depth_sampling import reference_height_grid
from prom3theus.rt import StratifiedAtmosphereModel
from prom3theus.validation.reference_synthesis import LTESynthesizer
from prom3theus.validation.reference_transfer import GeometricHeightPath
from prom3theus.muram.hmi_filter import build_idealized_hmi_filter
from prom3theus.muram.synthesize import MURAM_HMI_RADIANCE_SCALE_W_M3_SR
from prom3theus.inversion.objective import StokesObjective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/deep_audit_model"),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--depth-nodes", type=int, default=41)
    parser.add_argument(
        "--target-npz", type=Path, help="native_hmi.npz with stokes in 1e13 radiance units"
    )
    parser.add_argument(
        "--pixel-indices",
        type=int,
        nargs="+",
        help="x-major flattened indices; saved image axes are transposed before selection",
    )
    parser.add_argument(
        "--representation", choices=["direct", "vector_potential"], required=True
    )
    parser.add_argument(
        "--target-intensity-units", choices=["atlas", "raw"], default="atlas"
    )
    parser.add_argument("--magnetic-scale", type=float, default=100.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.depth_nodes < 2
        or args.warmup < 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.magnetic_scale)
        or args.magnetic_scale <= 0
    ):
        parser.error(
            "steps, learning-rate and magnetic-scale must be positive; warmup must be nonnegative"
        )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    external_target = None
    external_radiance_factor = 1.0
    coords = torch.tensor(
        [[-20.0, -3.0, 0.0], [-15.0, 4.0, 0.0], [15.0, -4.0, 0.0], [20.0, 3.0, 0.0]]
    )
    if args.target_npz is not None:
        if not args.pixel_indices:
            parser.error("--target-npz requires --pixel-indices")
        with np.load(args.target_npz) as stored:
            fields = {key: stored[key] for key in ("stokes", "coordinates", "wavelength")}
            if args.target_intensity_units == "atlas":
                external_radiance_factor = float(stored["atlas_radiance"]) / 1.0e13
                fields["stokes"] = fields["stokes"] / external_radiance_factor
        if fields["stokes"].ndim != 4 or fields["stokes"].shape[-2:] != (4, 6):
            parser.error("target Stokes must have image shape [ny,nx,4,6]")
        targets = fields["stokes"].transpose(1, 0, 2, 3).reshape(-1, 4, 6)
        positions = fields["coordinates"].transpose(1, 0, 2).reshape(-1, 3)
        if min(args.pixel_indices) < 0 or max(args.pixel_indices) >= len(targets):
            parser.error("pixel indices fall outside the target image")
        external_target = torch.tensor(targets[args.pixel_indices], dtype=torch.float32)
        coords = torch.tensor(positions[args.pixel_indices], dtype=torch.float32)
    model = StratifiedAtmosphereModel(
        shell_height_bounds_megameter=(1.5 if external_target is not None else 2.0, -1.0),
        time_coordinate_center_hours=float(coords[:, 2].mean()),
        time_dependent=True,
        height_input_scale_m=1.0e7,
        uniform_spatial_scaling=True,
        temperature_log_scale=0.1,
        gas_pressure_log_scale=0.5,
        microturbulence_log_scale=0.2,
        magnetic_scale_gauss=args.magnetic_scale,
        magnetic_representation=args.representation,
        scene_geometry_config={"solar_radius_m": 695700000.0, "scene_basis": torch.eye(3)},
        model_config={
            "type": "siren",
            "dim": 256,
            "n_layers": 8,
            "first_omega_0": 5.0,
            "hidden_omega_0": 1.0,
            "radial_weighting_config": None,
        },
    )
    q = torch.linspace(-5, 1, args.depth_nodes)
    height = (
        reference_height_grid(model, q)
        if external_target is not None
        else model.reference_atmosphere.height_from_log_tau(q)
    ).float()
    op, observed, weights, continuum = build_idealized_hmi_filter(dtype=torch.float32)
    wavelength = op.synthesis_grid(observed)
    synth = LTESynthesizer(line_ids=["FeI_6173.3352"])
    synth.prepare_wavelength_grid(wavelength)
    op.float()
    path = GeometricHeightPath(height.expand(len(coords), -1))
    objective = StokesObjective(continuum_normalization=True, continuum_indices=[0, 5])
    stokes_weights = torch.tensor([1.0, 200.0, 200.0, 200.0])


    def profiles(atmosphere):
        rotated = replace(
            atmosphere,
            magnetic_field=op.polarization_convention.to_synthesis_frame(
                atmosphere.magnetic_field
            ),
        )
        raw = synth(
            rotated, None, path=path, radiance_scale=MURAM_HMI_RADIANCE_SCALE_W_M3_SR
        )
        integrated = op(
            raw,
            wavelength,
            observed,
            spectral_weights=weights.expand(len(coords), *weights.shape),
            continuum_weights=continuum.expand(len(coords), *continuum.shape),
        )
        return integrated / external_radiance_factor


    with torch.no_grad():
        base = model(coords, height)
        target_field = torch.tensor(
            [
                [1200.0, 600.0, 1400.0],
                [-1300.0, 800.0, -1200.0],
                [1100.0, -900.0, 1500.0],
                [-1400.0, -600.0, -1100.0],
            ]
        )
        if external_target is None:
            target_atm = replace(
                base,
                temperature=base.temperature
                * torch.tensor([0.92, 0.95, 1.0, 1.03])[:, None],
                magnetic_field=target_field[:, None, :].expand_as(base.magnetic_field),
                velocity_field=torch.zeros_like(base.velocity_field),
                vector_potential=None,
            )
            target = profiles(target_atm).detach()
        if external_target is not None:
            torch.testing.assert_close(observed, torch.tensor(fields["wavelength"]))
            target = external_target
            target_field = None


    def rms(x):
        return x.detach().square().mean((0, 2)).sqrt().tolist()


    def evaluation(step):
        atmosphere = model(coords, height)
        prediction = profiles(atmosphere)
        components = objective(prediction, target).mean((0, 2))
        loss = (components * stokes_weights).sum()
        if step in (0, args.steps):
            magnetic_params = (
                model.network.out_layer.weight,
                model.network.in_layer.weight,
            )
            gradient_rows = []
            for component in components:
                gradient = torch.autograd.grad(
                    component, magnetic_params, retain_graph=True
                )
                gradient_rows.append(
                    {
                        "output_field_rows_norm": float(gradient[0][4:7].norm()),
                        "input_norm": float(gradient[1].norm()),
                    }
                )
        else:
            gradient_rows = None
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "component_mse": components.detach().tolist(),
            "predicted_rms": rms(prediction),
            "field_rms_gauss": atmosphere.magnetic_field.detach()
            .square()
            .mean((0, 1))
            .sqrt()
            .tolist(),
            "field_mean_gauss": atmosphere.magnetic_field.detach().mean(1).tolist(),
            "temperature_range_k": [
                float(atmosphere.temperature.detach().min()),
                float(atmosphere.temperature.detach().max()),
            ],
        }
        if gradient_rows is not None:
            row["component_gradients"] = gradient_rows
        return loss, row


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rows = []
    start = time.monotonic()
    for step in range(args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * (
                max((step + 1) / args.warmup, 0.001) if args.warmup > step else 1.0
            )
        optimizer.zero_grad(set_to_none=True)
        loss, row = evaluation(step)
        if step % 25 == 0 or step == args.steps:
            rows.append(row)
            print(json.dumps(row), flush=True)
        if step < args.steps:
            loss.backward()
            row["preclip_gradient_norm"] = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            )
            optimizer.step()
    zero_component_mse = objective(torch.zeros_like(target), target).mean((0, 2))
    relative_rmse = (torch.as_tensor(rows[-1]["component_mse"]) / zero_component_mse).sqrt()
    result = {
        "zero_prediction_component_mse": zero_component_mse.tolist(),
        "final_component_relative_rmse": relative_rmse.tolist(),
        "description": __doc__,
        "representation": args.representation,
        "magnetic_scale": args.magnetic_scale,
        "steps": args.steps,
        "seconds": time.monotonic() - start,
        "seed": 0,
        "warmup": args.warmup,
        "learning_rate": args.learning_rate,
        "known_target_field_gauss": target_field.tolist()
        if target_field is not None
        else None,
        "target_npz": str(args.target_npz) if args.target_npz is not None else None,
        "pixel_indices_x_major": args.pixel_indices,
        "depth_nodes": args.depth_nodes,
        "target_intensity_units": args.target_intensity_units
        if external_target is not None
        else "raw",
        "geometry_note": "Independent plane-parallel vertical columns; full oblique solar ray geometry is not exercised.",
        "target_rms": rms(target),
        "rows": rows,
        "observed": target.tolist(),
        "predicted": profiles(model(coords, height)).detach().tolist(),
    }
    name = f"{args.representation}_scale{args.magnetic_scale:g}"
    if args.target_npz is not None:
        name += f"_native_{args.target_intensity_units}_depth{args.depth_nodes}"
    if args.warmup:
        name += f"_warmup{args.warmup}"
    if args.learning_rate != 0.001:
        name += f"_lr{args.learning_rate:g}"
    (args.output_directory / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
