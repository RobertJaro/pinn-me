"""A reproducible spherical, current-free coronal baseline from an inversion.

The inferred photospheric radial field supplies Neumann boundary data. The
derived corona is a separate alpha=0 force-free model; it is not a measurement
of coronal field or a claim that an unconstrained neural atmosphere extrapolates.
Br is zero outside the source angular bounding rectangle, and its net flux is
retained. The inferred model supplies any unobserved corners inside that rectangle.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.rt.spherical_potential import (
    source_cell_quadrature,
    spherical_photosphere_matrix,
    spherical_potential_matrix,
    spherical_source_cells,
)


class SphericalPotentialField:
    """Evaluate integrated source cells in native heliocentric Cartesian axes."""

    def __init__(self, cells, radial_field_gauss, solar_radius_m):
        self.cells = np.asarray(cells, dtype=np.float64)
        self.radial_field_gauss = np.asarray(radial_field_gauss, dtype=np.float64)
        self.solar_radius_m = float(solar_radius_m)
        if self.radial_field_gauss.shape != (len(self.cells),):
            raise ValueError("One photospheric radial field is required per source cell")
        if not np.isfinite(self.radial_field_gauss).all():
            raise ValueError("Photospheric source field must be finite")
        if not np.isfinite(self.solar_radius_m) or self.solar_radius_m <= 0:
            raise ValueError("Solar radius must be finite and positive")

    def __call__(self, position_m, *, batch_size=128):
        """Return Gauss at exterior [N,3] positions, expressed in metres."""
        positions = np.asarray(position_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[-1] != 3 or batch_size < 1:
            raise ValueError("Expected [N,3] positions and a positive batch size")
        values = []
        for begin in range(0, len(positions), batch_size):
            matrix = spherical_potential_matrix(
                self.cells,
                positions[begin:begin + batch_size] / self.solar_radius_m,
            ).numpy()
            values.append(np.einsum("pic,c->pi", matrix, self.radial_field_gauss))
        return np.concatenate(values) if values else np.empty((0, 3))

    def photosphere(self):
        """Return the exterior surface limit exactly at source-cell centres."""
        operator = spherical_photosphere_matrix(self.cells).numpy()
        return np.einsum("pic,c->pi", operator, self.radial_field_gauss)


@torch.no_grad()
def infer_photospheric_source(atmosphere_model, domain, *, grid_size=12,
                              time_hours=None, batch_size=256,
                              source_quadrature_order=4):
    """Average inferred Br with tensor-product Gauss quadrature in each cell.

    ``source_quadrature_order`` nodes per angular axis resolve variation within
    a source cell before the piecewise-constant boundary field is reconstructed.
    """
    if grid_size < 2 or batch_size < 1:
        raise ValueError("Source grid must have at least two cells per axis")
    if type(source_quadrature_order) is not int:
        raise TypeError("source_quadrature_order must be an integer >= 2")
    if source_quadrature_order < 2:
        raise ValueError("source_quadrature_order must be an integer >= 2")
    bounds = np.asarray(domain.longitude_offset_bounds_rad) + domain.longitude_center_rad
    cells = spherical_source_cells(bounds, domain.latitude_bounds_rad, grid_size)
    directions, weights = source_cell_quadrature(cells, source_quadrature_order)
    flat = directions.reshape(-1, 3)
    time = float(np.mean(domain.time_bounds_hours) if time_hours is None else time_hours)
    if not np.isfinite(time):
        raise ValueError("Source evaluation time must be finite")
    parameter = next(atmosphere_model.parameters())
    chunks = []
    was_training = atmosphere_model.training
    atmosphere_model.eval()
    try:
        for begin in range(0, len(flat), batch_size):
            position = torch.as_tensor(flat[begin:begin + batch_size]).to(parameter)
            times = torch.full((len(position), 1), time).to(parameter)
            value = atmosphere_model.evaluate_position_rsun(position, time_hours=times)
            chunks.append(value["magnetic_field"].detach().double().cpu().numpy())
    finally:
        atmosphere_model.train(was_training)
    magnetic = np.concatenate(chunks).reshape(directions.shape)
    radial = np.sum(np.sum(magnetic * directions, axis=-1) * weights, axis=-1)
    return SphericalPotentialField(cells, radial, domain.solar_radius_m), time


def differential_metrics(field, position_m, *, step_megameter=0.02):
    """Independent centred Cartesian differences of the integrated field.

    Residuals use a fixed 1 Mm length. Unlike a current-weighted angle, these
    remain meaningful for a current-free model, whose exact current is zero.
    Two step sizes expose differentiation/quadrature error in the exported data.
    """
    positions = np.asarray(position_m, dtype=np.float64)
    if len(positions) == 0 or not np.isfinite(step_megameter) or step_megameter <= 0:
        raise ValueError("Differential validation needs probes and a positive step")
    magnetic = field(positions)
    magnitude = np.linalg.norm(magnetic, axis=-1)
    scale = np.maximum(magnitude, 1e-12)
    reports = []
    for step in (float(step_megameter), float(step_megameter) / 2):
        delta = np.eye(3) * step * 1e6
        plus = field((positions[:, None] + delta[None]).reshape(-1, 3)).reshape(-1, 3, 3)
        minus = field((positions[:, None] - delta[None]).reshape(-1, 3)).reshape(-1, 3, 3)
        # [probe, component, coordinate], in Gauss / Mm.
        jacobian = ((plus - minus) / (2 * step)).transpose(0, 2, 1)
        divergence = np.trace(jacobian, axis1=1, axis2=2)
        curl = np.stack((jacobian[:, 2, 1] - jacobian[:, 1, 2],
                         jacobian[:, 0, 2] - jacobian[:, 2, 0],
                         jacobian[:, 1, 0] - jacobian[:, 0, 1]), axis=-1)
        div_error = np.abs(divergence) / scale
        curl_error = np.linalg.norm(curl, axis=-1) / scale
        force_error = np.linalg.norm(np.cross(curl, magnetic), axis=-1) / scale**2
        reports.append({
            "difference_step_megameter": step,
            "max_abs_divergence_gauss_per_megameter": float(np.max(np.abs(divergence))),
            "max_curl_gauss_per_megameter": float(np.max(np.linalg.norm(curl, axis=-1))),
            "max_normalized_divergence": float(np.max(div_error)),
            "max_normalized_curl": float(np.max(curl_error)),
            "max_normalized_lorentz_force": float(np.max(force_error)),
            "median_normalized_divergence": float(np.median(div_error)),
            "median_normalized_curl": float(np.median(curl_error)),
        })
    return {"probe_count": len(positions), "normalization_length_megameter": 1.0,
            "method": "centred finite differences in physical Cartesian coordinates",
            "step_checks": reports}


def trace_field_lines(field, *, count=12, top_height_megameter=20.0,
                      step_megameter=0.25, max_steps=800):
    """Trace outward from dispersed strong photospheric cells with RK4.

    The solution exists beyond the source rectangle; crossing an angular edge does not
    terminate a line. Bottom/top crossing endpoints are truncated to the radial
    boundary. No interpolation of the exported coarse volume is used.
    """
    if count < 1 or max_steps < 1 or step_megameter <= 0 or top_height_megameter <= 0.2:
        raise ValueError("Field-line count, extent and integration step must be positive")
    centres = source_cell_quadrature(field.cells, 1)[0][:, 0]
    order = np.argsort(-np.abs(field.radial_field_gauss))
    # Spatial suppression keeps a cluster of strong cells from consuming all seeds.
    separation = np.sqrt(np.mean((field.cells[:, 1] - field.cells[:, 0]) *
                                 (field.cells[:, 3] - field.cells[:, 2]))) * 1.8
    seeds = []
    for index in order:
        if abs(field.radial_field_gauss[index]) < 1e-10:
            continue
        if not seeds or np.min(np.linalg.norm(centres[seeds] - centres[index], axis=-1)) >= separation:
            seeds.append(int(index))
        if len(seeds) == count:
            break
    if not seeds:
        return [], []
    radius = field.solar_radius_m
    bottom = radius + 0.10e6
    top = radius + top_height_megameter * 1e6
    positions = centres[seeds] * (radius + 0.15e6)
    signs = np.sign(field.radial_field_gauss[seeds])[:, None]
    lines = [[point.copy()] for point in positions]
    statuses = ["maximum_length"] * len(seeds)
    active = np.ones(len(seeds), dtype=bool)
    step = step_megameter * 1e6

    def direction(points, orientation):
        # Intermediate RK stages near the bottom use an exterior guard; the
        # accepted step is stopped on the actual bottom boundary below.
        norms = np.linalg.norm(points, axis=-1, keepdims=True)
        safe = points * np.maximum(norms, bottom) / norms
        vector = field(safe)
        strength = np.linalg.norm(vector, axis=-1, keepdims=True)
        return vector / np.maximum(strength, 1e-30) * orientation

    for _ in range(max_steps):
        selected = np.flatnonzero(active)
        if not len(selected):
            break
        p, orientation = positions[selected], signs[selected]
        k1 = direction(p, orientation)
        k2 = direction(p + 0.5 * step * k1, orientation)
        k3 = direction(p + 0.5 * step * k2, orientation)
        k4 = direction(p + step * k3, orientation)
        new = p + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        norm = np.linalg.norm(new, axis=-1)
        for local, index in enumerate(selected):
            if not np.isfinite(new[local]).all() or np.linalg.norm(k1[local]) < 0.5:
                statuses[index] = "weak_or_invalid_field"
                active[index] = False
                continue
            if norm[local] <= bottom or norm[local] >= top:
                boundary = bottom if norm[local] <= bottom else top
                # Intersect the last integration segment with the boundary sphere.
                segment = new[local] - p[local]
                roots = np.roots([np.dot(segment, segment), 2 * np.dot(p[local], segment),
                                  np.dot(p[local], p[local]) - boundary**2])
                crossing = [float(v.real) for v in roots if abs(v.imag) < 1e-8 and 0 <= v.real <= 1]
                if crossing:
                    new[local] = p[local] + min(crossing) * segment
                statuses[index] = "photosphere" if boundary == bottom else "top"
                active[index] = False
            lines[index].append(new[local].copy())
        positions[selected] = new
    return [np.asarray(line) for line in lines], statuses


def export_potential_corona(atmosphere_model, domain, output_directory, *,
                            time_hours=None, source_grid_size=12, horizontal_points=24,
                            heights_megameter=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
                            trace_count=12, make_figure=True, source_provenance=None,
                            source_quadrature_order=4):
    """Save the derived field, boundary, numerical validation, and connectivity.

    Positive height layers use the requested output grid. The exact r=1 surface
    is stored separately on the source-cell grid, where Br is well defined.
    """
    heights = np.asarray(heights_megameter, dtype=np.float64)
    if (heights.ndim != 1 or len(heights) < 2 or not np.isfinite(heights).all()
            or (heights <= 0).any() or (np.diff(heights) <= 0).any() or heights[-1] <= 2.5):
        raise ValueError("Use increasing positive heights extending above 2.5 Mm")
    if horizontal_points < 2:
        raise ValueError("Output grid requires at least two points per axis")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    field, time = infer_photospheric_source(atmosphere_model, domain,
                                           grid_size=source_grid_size, time_hours=time_hours,
                                           source_quadrature_order=source_quadrature_order)
    source_directions = source_cell_quadrature(field.cells, 1)[0][:, 0]
    surface = field.photosphere()
    surface_br = np.sum(surface * source_directions, axis=-1)
    boundaries = np.asarray(domain.longitude_offset_bounds_rad) + domain.longitude_center_rad
    output_cells = spherical_source_cells(boundaries, domain.latitude_bounds_rad, horizontal_points)
    directions = source_cell_quadrature(output_cells, 1)[0][:, 0]
    position = directions[None] * (domain.solar_radius_m + heights[:, None, None] * 1e6)
    magnetic = field(position.reshape(-1, 3)).reshape(position.shape)
    magnitude = np.linalg.norm(magnetic, axis=-1)
    radial = np.sum(magnetic * directions[None], axis=-1)
    layers = [{"height_megameter": float(h), "mean_field_gauss": float(np.mean(b)),
               "rms_field_gauss": float(np.sqrt(np.mean(b**2))),
               "max_field_gauss": float(np.max(b)), "min_field_gauss": float(np.min(b)),
               "mean_abs_radial_field_gauss": float(np.mean(np.abs(br)))}
              for h, b, br in zip(heights, magnitude, radial)]
    # Cell-centre probes avoid exact source-cell edges and boundary interpolation.
    probe_indices = np.linspace(0, len(directions) - 1, min(9, len(directions)), dtype=int)
    probe_heights = np.unique(np.maximum(heights[[0, len(heights) // 2, -1]], 0.5))
    probes = directions[probe_indices][None] * (domain.solar_radius_m + probe_heights[:, None, None] * 1e6)
    differential = differential_metrics(field, probes.reshape(-1, 3))
    lines, statuses = trace_field_lines(field, count=trace_count, top_height_megameter=float(heights[-1]))
    max_line_heights = [float(np.max(np.linalg.norm(line, axis=-1) - domain.solar_radius_m) / 1e6)
                        for line in lines]
    boundary_error = float(np.max(np.abs(surface_br - field.radial_field_gauss)))
    tolerance = 0.02
    steps = differential["step_checks"]
    residual_names = ("max_normalized_divergence", "max_normalized_curl", "max_normalized_lorentz_force")
    # Float32 geometry operators impose a finite-difference noise floor. Below
    # 1e-4 / Mm, a strict factor-of-four convergence test would measure rounding
    # rather than the field. Above it, halving the step must not worsen residuals
    # by more than 50 percent. Both steps independently meet the main tolerance.
    convergence_floor = 1e-4
    differential["step_refinement_noise_floor"] = convergence_floor
    differential["step_refinement_max_growth_factor"] = 1.5
    convergence = all(steps[1][name] <= max(convergence_floor, 1.5 * steps[0][name])
                      for name in residual_names)
    checks = {
        "finite_field": bool(np.isfinite(magnetic).all() and np.isfinite(surface).all()),
        "nonzero_photospheric_source": bool(np.max(np.abs(field.radial_field_gauss)) > 1e-6),
        "photospheric_radial_boundary_recovered": boundary_error < max(1e-5, float(np.max(np.abs(field.radial_field_gauss))) * 1e-6),
        "nonzero_field_at_top": layers[-1]["rms_field_gauss"] > 1e-6,
        "divergence_residual_below_tolerance": all(s["max_normalized_divergence"] < tolerance for s in steps),
        "curl_residual_below_tolerance": all(s["max_normalized_curl"] < tolerance for s in steps),
        "lorentz_residual_below_tolerance": all(s["max_normalized_lorentz_force"] < tolerance for s in steps),
        "differential_step_refinement_stable": convergence,
        "traced_field_reaches_corona": any(h >= 2.5 for h in max_line_heights),
    }
    area = (field.cells[:, 1] - field.cells[:, 0]) * (field.cells[:, 3] - field.cells[:, 2])
    signed_flux = float(np.sum(field.radial_field_gauss * area) * domain.solar_radius_m**2 * 1e4)
    unsigned_flux = float(np.sum(np.abs(field.radial_field_gauss) * area) * domain.solar_radius_m**2 * 1e4)
    report = {
        "model": "spherical potential field; alpha=0 force-free baseline",
        "photospheric_source": "radial projection of the inverted atmosphere at r=1; Gauss cell means",
        "coronal_model": "exterior spherical Neumann solution, independent of neural upper-atmosphere outputs",
        "exterior_assumption": "Br=0 outside the source angular bounding rectangle; net flux retained; field decays at infinity",
        "source_support_assumption": "the inferred model fills unobserved corners or gaps inside the angular bounding rectangle",
        "coordinate_frame": "heliocentric Carrington Cartesian",
        "time_hours": time, "solar_radius_m": domain.solar_radius_m,
        "source_provenance": source_provenance or {"type": "caller-provided atmosphere model"},
        "source_grid_size": source_grid_size, "horizontal_points": horizontal_points,
        "grid_geometry": {
            "source": "equal-solid-angle longitude/sin(latitude) cells at r=1",
            "source_longitude_bounds_rad": boundaries.tolist(),
            "source_latitude_bounds_rad": list(domain.latitude_bounds_rad),
            "source_cell_mean_quadrature_order": source_quadrature_order,
            "source_samples_per_cell": source_quadrature_order**2,
            "reconstruction": "cell-integrated exterior spherical Neumann Green kernel",
            "integration_quadrature_order": 6,
            "adaptive_cell_distance_ratio": 0.8,
            "operator_storage_dtype": "float32; geometry integration in float64",
            "output_array_axes": ["height", "longitude", "sin(latitude)", "Cartesian component"],
            "photospheric_surface": "separate exterior-limit samples at source-cell centres",
        },
        "surface_max_radial_error_gauss": boundary_error,
        "source_signed_flux_maxwell": signed_flux, "source_unsigned_flux_maxwell": unsigned_flux,
        "source_flux_imbalance_fraction": abs(signed_flux) / max(unsigned_flux, 1e-30),
        "layers": layers, "differential_validation": differential,
        "normalized_residual_tolerance": tolerance,
        "field_lines": {"count": len(lines), "termination": statuses,
                        "maximum_heights_megameter": max_line_heights,
                        "step_megameter": 0.25, "method": "RK4 on integrated spherical field"},
        "checks": checks, "valid": all(checks.values()),
    }
    offsets = np.cumsum([0] + [len(line) for line in lines])
    arrays_path = output / "coronal-potential.npz"
    np.savez_compressed(arrays_path,
        source_cells=field.cells, source_radial_field_gauss=field.radial_field_gauss.reshape(source_grid_size, source_grid_size),
        source_position_m=(source_directions * domain.solar_radius_m).reshape(source_grid_size, source_grid_size, 3),
        photospheric_potential_field_gauss=surface.reshape(source_grid_size, source_grid_size, 3),
        heights_megameter=heights, solar_radius_m=domain.solar_radius_m,
        position_m=position.reshape(len(heights), horizontal_points, horizontal_points, 3),
        magnetic_field_gauss=magnetic.reshape(len(heights), horizontal_points, horizontal_points, 3),
        line_positions_m=np.concatenate(lines) if lines else np.empty((0, 3)), line_offsets=offsets)
    report_path = output / "coronal-validation.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    paths = {"arrays": str(arrays_path.resolve()), "validation": str(report_path.resolve())}
    if make_figure:
        figure = potential_corona_figure(field, domain, heights, position, magnetic, lines, report)
        figure_path = output / "coronal-potential.png"
        figure.savefig(figure_path, dpi=160)
        paths["figure"] = str(figure_path.resolve())
    return {"report": report, "paths": paths}


def potential_corona_figure(field, domain, heights, positions, magnetic, lines, report):
    """Show the inferred boundary, physical field amplitudes, and traced corona."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(13, 9), layout="constrained")
    axes = figure.subplots(2, 2)
    source_size = int(np.sqrt(len(field.cells)))
    longitude = np.rad2deg(np.asarray(domain.longitude_offset_bounds_rad) + domain.longitude_center_rad)
    latitude = np.rad2deg(domain.latitude_bounds_rad)
    extent = (*longitude, *latitude)
    br = field.radial_field_gauss.reshape(source_size, source_size)
    scale = max(float(np.max(np.abs(br))), 1e-6)
    artist = axes[0, 0].imshow(br.T, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    axes[0, 0].set(title="Photospheric radial field from the source model", xlabel="Carrington longitude [deg]", ylabel="Latitude [deg]")
    figure.colorbar(artist, ax=axes[0, 0], label="Br [G]")
    rms = [layer["rms_field_gauss"] for layer in report["layers"]]
    peak = [layer["max_field_gauss"] for layer in report["layers"]]
    axes[0, 1].semilogy(heights, rms, "o-", label="RMS |B|")
    axes[0, 1].semilogy(heights, peak, "s--", label="Maximum |B|")
    axes[0, 1].axvspan(2.5, heights[-1], color="tab:orange", alpha=0.08)
    axes[0, 1].set(title="Spherical potential field into the corona", xlabel="Height [Mm]", ylabel="Field [G]")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)
    index = int(np.argmin(np.abs(heights - 5)))
    direction = positions[index] / np.linalg.norm(positions[index], axis=-1, keepdims=True)
    value = np.sum(magnetic[index] * direction, axis=-1)
    output_size = int(np.sqrt(len(value)))
    scale = max(float(np.max(np.abs(value))), 1e-6)
    artist = axes[1, 0].imshow(value.reshape(output_size, output_size).T, origin="lower", extent=extent,
                              aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    axes[1, 0].set(title=f"Coronal radial field at {heights[index]:g} Mm", xlabel="Carrington longitude [deg]", ylabel="Latitude [deg]")
    figure.colorbar(artist, ax=axes[1, 0], label="Br [G]")
    axes[1, 1].remove()
    trace_axis = figure.add_subplot(2, 2, 4, projection="3d")
    lon = domain.longitude_center_rad + np.mean(domain.longitude_offset_bounds_rad)
    lat = np.mean(domain.latitude_bounds_rad)
    radial = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    west = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.cross(radial, west)
    for line in lines:
        x, y = line @ west / 1e6, line @ north / 1e6
        z = (np.linalg.norm(line, axis=-1) - domain.solar_radius_m) / 1e6
        trace_axis.plot(x, y, z, lw=1.1)
    trace_axis.set(xlabel="West [Mm]", ylabel="North [Mm]", zlabel="Height [Mm]", title="Field lines from the inferred boundary")
    trace_axis.set_zlim(0, heights[-1])
    trace_axis.view_init(elev=23, azim=-65)
    figure.suptitle("Photospheric boundary → spherical α=0 force-free corona\n"
                   "Br=0 outside the source angular rectangle; coronal field is an extrapolation", fontsize=14)
    return figure


def export_p3s_potential_corona(path, output_directory, *, stream_id=None, **options):
    """Load an inversion P3S and export its spherical potential coronal baseline."""
    from prom3theus.artifacts.loader import P3SLoader

    loader = P3SLoader(path, device="cpu", stream_id=stream_id)
    geometry = loader.config.atmosphere.geometry
    domain = SphericalShellDomain.from_observation_bounds(loader.observation.bounds,
        (geometry.outer_height_megameter, geometry.inner_height_megameter))
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    provenance = {"type": "PROM3THEUS P3S inversion", "path": str(Path(path).resolve()),
                  "sha256": digest.hexdigest(), "global_step": loader.global_step,
                  "epoch": loader.epoch, "stream_id": loader.stream_id,
                  "source_signature": loader.observation.source_signature,
                  "magnetic_reference_height_megameter":
                      loader.config.atmosphere.parameters.magnetic_field.reference_height_megameter}
    return export_potential_corona(loader.module.atmosphere_model, domain, output_directory,
                                   source_provenance=provenance, **options)
