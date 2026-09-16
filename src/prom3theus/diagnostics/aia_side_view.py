"""Synthetic orthographic EUV view, looking east–west through the shell wedge."""

import math

import torch

from prom3theus.core.transforms import normalized_asinh


@torch.no_grad()
def synthesize_side_view(
    atmosphere, term, domain, time_hours, *, max_pixels, batch_size
):
    """Integrate straight parallel rays; never evaluate fields outside the domain.

    The image axes are local north and projected height above the central tangent
    plane. Spherical curvature is retained, so projected height is not radial height.
    """
    parameter = next(atmosphere.parameters())
    device, dtype = parameter.device, parameter.dtype
    lon = domain.longitude_center_rad
    lat = sum(domain.latitude_bounds_rad) / 2
    east = parameter.new_tensor([-math.sin(lon), math.cos(lon), 0])
    north = parameter.new_tensor(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)]
    )
    radial = parameter.new_tensor(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )
    radius = domain.solar_radius_m
    inner = radius + max(0.0, domain.height_bounds_Mm[0]) * 1e6
    outer = radius + domain.height_bounds_Mm[1] * 1e6
    # A conservative cone encloses the angular wedge, including its curved faces.
    angle = min(
        math.pi,
        (domain.latitude_bounds_rad[1] - domain.latitude_bounds_rad[0]) / 2
        + max(abs(v) for v in domain.longitude_offset_bounds_rad),
    )
    half_width = outer * math.sin(min(angle, math.pi / 2))
    bottom = (inner if angle < math.pi / 2 else outer) * math.cos(angle) - radius
    top = outer - radius
    size = max(1, min(64, math.isqrt(max_pixels)))
    horizontal = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size * (
        2 * half_width
    ) - half_width
    height = bottom + (torch.arange(size, device=device, dtype=dtype) + 0.5) / size * (
        top - bottom
    )
    z, y = torch.meshgrid(height, horizontal, indexing="ij")
    centers = y.reshape(-1, 1) * north + (radius + z.reshape(-1, 1)) * radial
    distance = torch.linspace(
        0, 2 * half_width, term.ray_samples, device=device, dtype=dtype
    )
    predictions = []
    for center in centers.split(batch_size):
        position = center[:, None] + (distance - half_width)[None, :, None] * east
        r = torch.linalg.vector_norm(position, dim=-1)
        longitude = torch.atan2(position[..., 1], position[..., 0]) - lon
        longitude = torch.atan2(torch.sin(longitude), torch.cos(longitude))
        latitude = torch.atan2(
            position[..., 2], torch.linalg.vector_norm(position[..., :2], dim=-1)
        )
        valid = (
            (r >= inner)
            & (r <= outer)
            & (longitude >= domain.longitude_offset_bounds_rad[0])
            & (longitude <= domain.longitude_offset_bounds_rad[1])
            & (latitude >= domain.latitude_bounds_rad[0])
            & (latitude <= domain.latitude_bounds_rad[1])
        )
        # The observer is on the +east side. The opaque solar disk hides the
        # far segment of rays that intersect the photosphere.
        impact_squared = center.square().sum(dim=-1)
        occulting = impact_squared < radius**2
        near_surface = (radius**2 - impact_squared).clamp_min(0).sqrt()
        valid &= ~occulting[:, None] | (
            (distance - half_width)[None, :] >= near_surface[:, None]
        )
        temperature = torch.full_like(r, 1e6)
        density = torch.zeros_like(r)
        if valid.any():
            points = position[valid]
            fields = atmosphere.evaluate_position_rsun(
                points / radius,
                time_hours=points.new_full((len(points), 1), time_hours),
            )
            temperature[valid] = fields["temperature"].reshape(-1)
            density[valid] = atmosphere.thermodynamic_eos.electron_density(
                fields["temperature"], fields["gas_pressure"]
            ).reshape(-1)
        raw = term.emission_operator(temperature, density, distance, sample_dim=1)
        normalized = raw * term.calibration.gains / term.intensity_scales
        predictions.append(normalized_asinh(normalized, term.objective.asinh_scales).cpu())
    return torch.cat(predictions).reshape(size, size, -1), (
        -half_width / 1e6,
        half_width / 1e6,
        bottom / 1e6,
        top / 1e6,
    )


def render_side_view(image, extent, channels, path, *, time_hours, height_markers, dpi):
    """One calibrated, training-scaled image and colorbar per channel; no target."""
    import astropy.units as u
    from matplotlib.figure import Figure
    from sunpy.visualization.colormaps.color_tables import aia_color_table

    figure = Figure(figsize=(4 * len(channels), 4), layout="constrained")
    axes = figure.subplots(1, len(channels), squeeze=False)[0]
    for index, (axis, channel) in enumerate(zip(axes, channels, strict=True)):
        artist = axis.imshow(
            image[..., index],
            origin="lower",
            extent=extent,
            interpolation="none",
            aspect="auto",
            vmin=0,
            cmap=aia_color_table(channel * u.angstrom),
        )
        for height in height_markers:
            axis.axhline(height, color="white", linestyle=":", linewidth=0.6, alpha=0.7)
        axis.set(title=f"{channel} Å", xlabel="Local north [Mm]")
        figure.colorbar(artist, ax=axis, label="asinh(I / a) / asinh(1 / a)")
    axes[0].set_ylabel("Projected height [Mm]")
    figure.suptitle(f"Synthetic AIA side view | t={time_hours:.3f} h | east–west LOS")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    figure.clear()
