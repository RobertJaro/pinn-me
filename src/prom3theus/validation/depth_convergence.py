"""Measure Stokes quadrature bias and jitter independently of optimization.

Supply an evaluator ``evaluate(coarse_count, fine_count, randomize)`` that
holds the pixels and atmospheric parameters fixed and returns Stokes profiles
in one common radiance scale, shaped ``[..., 4, wavelength]``. It may wrap the
validation reference formal solver or known tabulated columns. The reference and
every test must use the same geometry, atmosphere, instrument and wavelengths.
The helper deliberately leaves gradient mode unchanged: vector-potential
atmospheres need coordinate derivatives even during a forward diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def measure_depth_convergence(
    evaluate: Callable[[int, int, bool], torch.Tensor],
    *,
    levels: Sequence[tuple[int, int]] = ((25, 16), (64, 32), (128, 64)),
    reference_level: tuple[int, int] = (1537, 0),
    jitter_repeats: int = 8,
    continuum_indices: Sequence[int] = (0, -1),
    continuum_floor: float = 0.1,
    seed: int = 0,
    cuda_devices: Sequence[int] = (),
) -> dict:
    """Return component errors and jitter RMS with fixed reference scaling.

    ``rms_error_per_continuum`` divides each pixel's residual by its reference
    continuum proxy before averaging over pixels and wavelengths. The proxy
    is reference Stokes I averaged over ``continuum_indices``, floored in the
    evaluator's radiance units. ``relative_l2_per_component`` instead divides
    each component's residual norm by that component's reference norm; it is
    ``None`` for an exactly zero reference component. Jitter RMS is the root
    mean variance over repeated forward evaluations (population variance).

    CPU RNG state and the explicitly listed CUDA-device RNG states are restored
    on exit. A dense reference is a convergence comparison, not an independent
    validation of the forward physics; check a second reference resolution.
    """
    levels = tuple(tuple(level) for level in levels)
    reference_level = tuple(reference_level)
    for level in (reference_level, *levels):
        if len(level) != 2 or any(type(value) is not int for value in level):
            raise ValueError("Each depth level must contain integer coarse/fine counts.")
        if level[0] < 2 or level[1] < 0:
            raise ValueError("Depth counts require coarse >= 2 and fine >= 0.")
    if type(jitter_repeats) is not int or jitter_repeats < 2:
        raise ValueError("jitter_repeats must be an integer >= 2.")
    if not continuum_indices or not 0 < continuum_floor < float("inf"):
        raise ValueError("Continuum indices and a finite positive floor are required.")

    def run(level, randomize):
        profiles = evaluate(*level, randomize).detach()
        if profiles.ndim < 2 or profiles.shape[-2] != 4:
            raise ValueError("Evaluator must return [..., 4, wavelength] Stokes profiles.")
        if not profiles.is_floating_point() or not torch.isfinite(profiles).all():
            raise ValueError("Evaluator profiles must be finite floating-point values.")
        return profiles

    def components(value):
        return value.movedim(-2, 0).reshape(4, -1)

    with torch.random.fork_rng(devices=list(cuda_devices)):
        torch.random.default_generator.manual_seed(seed)
        for device in cuda_devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        reference = run(reference_level, False)
        index = torch.as_tensor(continuum_indices, device=reference.device)
        proxy = reference[..., 0, index].mean(-1).clamp_min(continuum_floor)
        scale = proxy[..., None, None]
        reference_norm = torch.linalg.vector_norm(components(reference), dim=-1)
        rows = []
        for level in levels:
            actual = run(level, False)
            if actual.shape != reference.shape:
                raise ValueError("Changing quadrature must preserve the Stokes shape.")
            residual = components(actual - reference)
            normalized = components((actual - reference) / scale)
            relative = torch.linalg.vector_norm(residual, dim=-1) / reference_norm
            jittered = torch.stack([run(level, True) for _ in range(jitter_repeats)])
            if jittered.shape[1:] != reference.shape:
                raise ValueError("Jitter must preserve the reference Stokes shape.")
            jitter_variance = components(jittered.var(dim=0, correction=0) / scale.square())
            rows.append({
                "coarse_count": level[0],
                "fine_count": level[1],
                "rms_error_per_continuum": normalized.square().mean(-1).sqrt().tolist(),
                "relative_l2_per_component": [
                    float(value) if float(norm) > 0 else None
                    for value, norm in zip(relative, reference_norm)
                ],
                "jitter_rms_per_continuum": jitter_variance.mean(-1).sqrt().tolist(),
            })
    return {
        "component_order": ["I", "Q", "U", "V"],
        "reference_level": list(reference_level),
        "continuum_indices": list(continuum_indices),
        "continuum_floor": continuum_floor,
        "jitter_repeats": jitter_repeats,
        "seed": seed,
        "rows": rows,
    }


__all__ = ["measure_depth_convergence"]
