"""LTE teacher and held-out ray diagnostics for pointwise coefficient training."""

from __future__ import annotations

from time import perf_counter

import torch

from prom3theus.rt.formal_transfer import formal_stokes

from .teacher import density_inputs, direct_local, targets


def ray_comparison(
    model, teacher, points, distance, wave, observe, radiance_scale, *, gradients
):
    def render(state, surrogate):
        if surrogate:
            density = teacher.plasma_provider.mass_density(state[..., 0], state[..., 1])
            local = model.evaluate_spectrum(
                density_inputs(state, density),
                model.prepare_spectrum(wave),
                radiance_scale=radiance_scale,
            )
        else:
            local = direct_local(teacher, state, wave, radiance_scale)
        stokes = formal_stokes(
            local.propagation_m1, local.source, distance, polarized=True
        ).transpose(-1, -2)
        return observe(stokes)

    # For a common test direction use perturbations of physical input units.
    direction = (
        points.new_tensor([10.0, 0.0, 10.0, 100.0, 10.0, 10.0, 10.0])
        .expand_as(points)
        .clone()
    )
    direction[:, 1] = 0.01 * points[:, 1]

    def evaluate(surrogate):
        if points.is_cuda:
            torch.cuda.synchronize(points.device)
        start = perf_counter()
        with torch.set_grad_enabled(gradients):
            state = points.detach().requires_grad_(gradients)
            profile = render(state, surrogate)
            sensitivity = None
            if gradients:
                # A reproducible signed projection of the full detector vector.
                gen = torch.Generator(device="cpu").manual_seed(314)
                probe = torch.randn(profile.shape, generator=gen).to(profile)
                (sensitivity,) = torch.autograd.grad((profile * probe).sum(), state)
                sensitivity = sensitivity * direction
        if points.is_cuda:
            torch.cuda.synchronize(points.device)
        return profile.detach(), sensitivity, perf_counter() - start

    reference, ref_grad, direct_time = evaluate(False)
    prediction, grad, model_time = evaluate(True)
    error = prediction - reference
    report = dict(
        stokes_rmse=error.square().mean(-1).sqrt().cpu().tolist(),
        stokes_p95=error.abs().quantile(0.95, dim=-1).cpu().tolist(),
        edge_intensity_error=float(error[..., 0, [0, -1]].abs().mean()),
        exact_lte_seconds=direct_time,
        surrogate_seconds=model_time,
    )
    if gradients:
        absolute = (grad - ref_grad).norm(dim=0).tolist()
        reference_norm = ref_grad.norm(dim=0).tolist()
        report["gradient_absolute_l2"] = absolute
        report["gradient_relative_l2"] = [
            error / norm if norm > 0 else None
            for error, norm in zip(absolute, reference_norm)
        ]
        report["gradient_cosine"] = float(
            torch.nn.functional.cosine_similarity(
                grad.flatten(), ref_grad.flatten(), dim=0
            )
        )
    return report, torch.stack((reference, prediction)).cpu()


@torch.no_grad()
def spline_resolution_check(model, teacher, points):
    """Measure interpolation of exact knot labels at successively finer unseen wavelengths."""
    knots = model.spec["knots"]
    grid = torch.linspace(
        *model.spec["bounds"], knots, dtype=torch.float64, device=points.device
    )
    knot_points = points
    if model.spectral_frame == "rest":
        knot_points = points.clone()
        knot_points[:, 3] = 0
    exact_knots = targets(teacher, knot_points, grid)
    results = {}
    for factor in (2, 4):
        fine = torch.linspace(
            *model.spec["bounds"],
            factor * (knots - 1) + 1,
            dtype=grid.dtype,
            device=grid.device,
        )
        left, fraction = model.profile_indices(points[:, 3], model.prepare_spectrum(fine))
        index = left[..., None].expand(len(points), len(fine), exact_knots.shape[-1])
        interpolated = torch.lerp(
            exact_knots.gather(1, index),
            exact_knots.gather(1, index + 1),
            fraction.to(points)[..., None],
        )
        exact = targets(teacher, points, fine)
        results[f"interpolation_loss_{factor}x"] = model.loss_channels(
            interpolated, exact
        ).tolist()
    return results


def save_profile_figure(path, wavelengths, reference, prediction):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(
        4, 1, sharex=True, figsize=(9, 8), constrained_layout=True
    )
    for i, (axis, name) in enumerate(zip(axes, "IQUV")):
        axis.plot(wavelengths, reference[i], label="Exact LTE")
        axis.plot(wavelengths, prediction[i], "--", label="NN surrogate")
        axis.set_ylabel(name)
    axes[0].legend()
    axes[-1].set_xlabel("Wavelength [Å]")
    figure.savefig(path, dpi=180)
    plt.close(figure)
