"""Scheduled, detached potential targets on the actual spherical boundaries."""

import torch

from prom3theus.rt.spherical_potential import (
    source_cell_quadrature,
    spherical_photosphere_matrix,
    spherical_potential_matrix,
    spherical_source_cells,
)

from .base import SharedObjectiveTerm, SharedTermResult


class ProgressivePotentialBoundary(SharedObjectiveTerm):
    """Detached spherical potential targets, restricted to exterior boundaries.

    Cell-averaged model Br at r=1 specifies the exterior Neumann problem. It
    is zero outside the Stokes footprint; the field decays at infinity. No
    potential field is computed below the photosphere. Volume PDE losses retain
    their independent normalization. Exterior boundary errors are normalized by
    a detached mean target-field strength: one scale over the complete top
    surface and one scale for all side points at each height. The physical
    target buffer is retained only as a detached checkpoint/diagnostic adapter.
    """

    def __init__(
        self,
        options,
        domain,
        source_domain,
        scene_basis,
        *,
        time_dependent=True,
        observation_times_hours=None,
    ):
        super().__init__()
        self.options = dict(options)
        self.step = 0
        self.last_update = -1
        self.update_count = 0
        self.frozen = False
        self.time_dependent = time_dependent
        self.radius = float(domain.solar_radius_m)
        if options["source_height_megameter"] != 0:
            raise ValueError("Potential source must be at the photosphere (height 0)")
        if domain.height_bounds_Mm[1] <= 0:
            raise ValueError(
                "Potential boundaries require a domain above the photosphere"
            )
        self.surface_ranges = {}
        self.surface_shapes = {}
        # Immutable chart bounds reconstruct jittered Cartesian coordinates;
        # the fixed grids and their operator remain cached for reference updates.
        self.surface_domains = {"outer": domain, "photosphere": source_domain}
        surfaces = []

        def centres(n):
            return (torch.arange(n, dtype=torch.float64) + 0.5) / n

        def add(name, height, lon, lat, shape):
            height, lon, lat = torch.broadcast_tensors(height, lon, lat)
            positions = domain._positions(
                height.flatten(),
                lon.flatten(),
                lat.flatten(),
                torch.zeros(height.numel()),
            )["position_m"]
            begin = sum(len(surface) for surface in surfaces)
            self.surface_ranges[name] = (begin, begin + len(positions))
            self.surface_shapes[name] = shape
            surfaces.append(positions)

        n = options["top_grid_size"]
        xy = centres(n)
        add(
            "top",
            torch.tensor(domain.height_bounds_Mm[1], dtype=torch.float64),
            xy[:, None],
            xy[None, :],
            (n, n),
        )
        nh, nz = options["side_horizontal_points"], options["side_height_points"]
        h = centres(nz)[:, None] * domain.height_bounds_Mm[1]
        edge = centres(nh)[None, :]
        for name, lon, lat in (
            ("west", torch.tensor(1.0), edge),
            ("east", torch.tensor(0.0), edge),
            ("north", edge, torch.tensor(1.0)),
            ("south", edge, torch.tensor(0.0)),
        ):
            add(name, h, lon, lat, (nz, nh))
        self.top_count = n * n
        self.outer_count = sum(len(surface) for surface in surfaces)
        # Photospheric queries use source-cell centres, exactly at r=1.
        cells = spherical_source_cells(
            tuple(
                source_domain.longitude_center_rad + v
                for v in source_domain.longitude_offset_bounds_rad
            ),
            source_domain.latitude_bounds_rad,
            options["grid_size"],
        )
        if options["photosphere"]["enabled"]:
            photo = (
                torch.from_numpy(source_cell_quadrature(cells, 1)[0][:, 0])
                * self.radius
            )
            self.surface_ranges["photosphere"] = (
                self.outer_count,
                self.outer_count + len(photo),
            )
            self.surface_shapes["photosphere"] = (options["grid_size"],) * 2
            surfaces.append(photo)
        boundary = torch.cat(surfaces).double()
        self.register_buffer("positions", boundary)
        # One shuffle per surface; batches subsequently read contiguous slices.
        generator = torch.Generator().manual_seed(options["seed"])
        order = torch.cat(
            [
                torch.randperm(end - begin, generator=generator) + begin
                for begin, end in self.surface_ranges.values()
            ]
        )
        self.register_buffer("sample_order", order)
        self.register_buffer("sample_positions", boundary[order])
        if time_dependent:
            if observation_times_hours is None:
                raise ValueError(
                    "Dynamic potential boundaries require observed Stokes times"
                )
            times = torch.as_tensor(observation_times_hours, dtype=torch.float32)
            if times.ndim != 1 or not times.numel() or not torch.isfinite(times).all():
                raise ValueError(
                    "Potential observation times must be a nonempty finite vector"
                )
            endpoints = torch.tensor(domain.time_bounds_hours, dtype=torch.float32)
            if ((times < endpoints[0]) | (times > endpoints[-1])).any():
                raise ValueError(
                    "Potential observation times lie outside the joint time domain"
                )
            # Extra endpoints cover an AIA interval extending beyond Stokes.
            # They are model-inferred anchors, not additional observations.
            times = torch.unique(torch.cat((times, endpoints)), sorted=True)
        else:
            times = torch.tensor([sum(domain.time_bounds_hours) / 2])
        count = len(times)
        self.register_buffer("times", times)
        self.register_buffer("targets", torch.zeros(count, len(boundary), 3))
        self.register_buffer("last_relative_change", torch.tensor(0.0))
        self.register_buffer("source_flux", torch.zeros(count))
        self.register_buffer("normalization", torch.zeros(count, len(boundary)))
        self.register_buffer("sample_targets", torch.zeros(count, len(boundary), 3))
        self.register_buffer("sample_normalization", torch.zeros(count, len(boundary)))
        self._make_source_geometry(source_domain, cells)

    def _photosphere_weight(self):
        o = self.options["photosphere"]
        if (
            not o["enabled"]
            or self.step < o["start_step"]
            or self.step >= o["end_step"]
        ):
            return 0.0
        ramp = (
            min((self.step - o["start_step"]) / o["ramp_steps"], 1.0)
            if o["ramp_steps"]
            else 1.0
        )
        return o["weight"] * ramp

    def _make_source_geometry(self, domain, cells):
        o = self.options
        directions, weights = source_cell_quadrature(cells, o["source_supersampling"])
        directions = torch.from_numpy(directions.reshape(-1, 3))
        self.register_buffer("source_cells", torch.from_numpy(cells))
        self.register_buffer("source_directions", directions)
        self.register_buffer("source_positions", directions * self.radius)
        self.register_buffer("source_weights", torch.from_numpy(weights).float())
        self.register_buffer(
            "source_reference", torch.zeros(len(self.times), len(cells))
        )
        # Cached geometry operator: rebuild lazily after loading a checkpoint,
        # without persisting a large derived matrix in every checkpoint.
        self.register_buffer("potential_operator", torch.empty(0), persistent=False)
        self.geometry_metadata = {
            "type": "spherical_exterior_neumann",
            "source_radius_m": self.radius,
            "reference_times_hours": self.times.tolist(),
            "time_sampling": (
                "all Stokes observation times plus joint interval endpoints"
                if self.time_dependent
                else "static midpoint"
            ),
            "validation_times": "all reference anchors",
            "source_coverage": domain.configuration(),
            "outside_source_footprint": "zero radial field",
            "far_boundary": "decay at infinity, retaining net source flux",
            "source_discretization": "piecewise constant equal-solid-angle cells",
            "source_grid_shape": [o["grid_size"], o["grid_size"]],
            "extraction": "adaptive surface-cell integration of spherical Neumann Green kernel",
            "normalization_scope": (
                "detached mean |B_hat_potential|; one scale over the full top "
                "surface and one scale over all side faces at each height"
            ),
            "exterior_objective": (
                "mean componentwise ((B_hat - B_hat_potential) / "
                "sqrt(Bbar_hat^2 + B_floor_hat^2))^2"
            ),
            "surface_shapes": self.surface_shapes,
            "photosphere_radius_m": self.radius,
            "surface_sampling": "once-shuffled cell chunks with fresh within-cell surface jitter; continuous times",
            "jitter_fraction": o["jitter_fraction"],
            "jitter_reference": "bilinear surface interpolation; exact containing-cell photospheric Br",
            "side_height_sampling": "uniform height cell centres above r=1",
        }

    def get_extra_state(self):
        return {
            "version": 10,
            "step": self.step,
            "last_update": self.last_update,
            "update_count": self.update_count,
            "frozen": self.frozen,
        }

    def set_extra_state(self, state):
        if state.get("version") != 10:
            raise ValueError("Unsupported progressive potential checkpoint state")
        self.potential_operator = self.potential_operator.new_empty(0)
        self.step = int(state["step"])
        self.last_update = int(state["last_update"])
        self.update_count = int(state["update_count"])
        self.frozen = bool(state["frozen"])

    def set_step(self, step):
        if type(step) is not int or step < 0:
            raise ValueError("Potential boundary step must be non-negative")
        self.step = step

    def _weight(self):
        elapsed = self.step - self.options["start_step"]
        if elapsed < 0:
            return 0.0
        ramp = self.options["ramp_steps"]
        return self.options["weight"] * (min(elapsed / ramp, 1.0) if ramp else 1.0)

    @torch.no_grad()
    def _refresh(self, model):
        o = self.options
        if not self.potential_operator.numel():
            cells = self.source_cells.detach().cpu().numpy()
            operator = spherical_potential_matrix(
                cells,
                (self.positions[: self.outer_count] / self.radius)
                .detach()
                .cpu()
                .numpy(),
            )
            if "photosphere" in self.surface_ranges:
                operator = torch.cat((operator, spherical_photosphere_matrix(cells)))
            self.potential_operator = operator.to(device=self.positions.device)
        generated, fluxes = [], []
        for anchor, time in enumerate(self.times):
            radial = []
            for positions, unit in zip(
                self.source_positions.split(o["batch_size"]),
                self.source_directions.split(o["batch_size"]),
                strict=True,
            ):
                field = model.evaluate_position_rsun(positions / self.radius, time)[
                    "magnetic_field"
                ]
                radial.append((field * unit.to(field)).sum(-1))
            radial = torch.cat(radial).reshape(-1, len(self.source_weights))
            source = (radial * self.source_weights).sum(-1)
            areas = (self.source_cells[:, 1] - self.source_cells[:, 0]) * (
                self.source_cells[:, 3] - self.source_cells[:, 2]
            )
            fluxes.append((source * areas).sum() * self.radius**2)
            reference = source.to(self.source_reference)
            if self.last_update >= 0:
                reference = torch.lerp(
                    self.source_reference[anchor], reference, o["blend"]
                )
            self.source_reference[anchor].copy_(reference)
            generated.append(self.potential_operator @ reference)

        proposed = torch.stack(generated)
        if not torch.isfinite(proposed).all():
            raise FloatingPointError(
                "Potential boundary update produced non-finite targets"
            )
        if self.last_update >= 0:
            change = (proposed - self.targets).square().mean().sqrt()
            self.last_relative_change.copy_(
                change
                / (self.targets.square().mean() + o["field_floor_gauss"] ** 2).sqrt()
            )
        self.targets.copy_(proposed)
        self._update_normalization()
        self.sample_targets.copy_(self.targets[:, self.sample_order])
        self.sample_normalization.copy_(self.normalization[:, self.sample_order])
        self.source_flux.copy_(torch.stack(fluxes))
        self.last_update = self.step
        self.update_count += 1
        self.frozen = self.step >= o["freeze_step"]

    def prepare(self, model):
        o = self.options
        if not self.training or self.frozen or self.step < o["start_step"]:
            return
        due = self.last_update < 0 or self.step >= o["freeze_step"]
        due |= (self.step - o["start_step"]) // o["update_every_n_steps"] > (
            self.last_update - o["start_step"]
        ) // o["update_every_n_steps"]
        if due:
            self._refresh(model)

    @torch.no_grad()
    def _update_normalization(self):
        """Cache detached mean target-field scales for the exterior faces.

        The top scale is averaged over the complete top surface.  Side scales
        are averaged over every side face at each discrete height, so a given
        height has one shared characteristic field strength rather than one
        independent scale per face.  Jittered side samples interpolate these
        height scales in ``_training_samples``.
        """

        time_count = len(self.times)
        for name in ("top", "photosphere"):
            if name not in self.surface_ranges:
                continue
            begin, end = self.surface_ranges[name]
            shape = self.surface_shapes[name]
            strength = self.targets[:, begin:end].norm(dim=-1).reshape(
                time_count, *shape
            )
            axes = (-2, -1)
            mean = strength.mean(dim=axes, keepdim=True).expand_as(strength)
            self.normalization[:, begin:end].copy_(mean.reshape(time_count, -1))

        side_names = [
            name
            for name in self.surface_ranges
            if name not in ("top", "photosphere")
        ]
        if not side_names:
            return
        side_strengths = []
        side_shapes = {}
        for name in side_names:
            begin, end = self.surface_ranges[name]
            shape = self.surface_shapes[name]
            side_shapes[name] = shape
            strength = self.targets[:, begin:end].norm(dim=-1).reshape(
                time_count, *shape
            )
            if side_strengths and strength.shape[1] != side_strengths[0].shape[1]:
                raise ValueError("All potential side faces must share height samples")
            side_strengths.append(strength)
        mean_by_height = torch.cat(side_strengths, dim=-1).mean(dim=-1)
        for name in side_names:
            begin, end = self.surface_ranges[name]
            height_count, horizontal_count = side_shapes[name]
            mean = mean_by_height[:, :height_count, None].expand(
                time_count, height_count, horizontal_count
            )
            self.normalization[:, begin:end].copy_(mean.reshape(time_count, -1))

    def _time_values(self, values, times):
        if len(self.times) == 1:
            return values[0]
        high = torch.searchsorted(self.times, times.contiguous()).clamp(
            1, len(self.times) - 1
        )
        low = high - 1
        fraction = (times - self.times[low]) / (self.times[high] - self.times[low])
        if values.ndim == 3:
            fraction = fraction[:, None]
        index = torch.arange(len(times), device=times.device)
        return torch.lerp(values[low, index], values[high, index], fraction)

    def _surface_interpolant(self, values, name, coordinates, *, clamp=False):
        """Interpolate cached [time,point,...] values on one surface chart.

        coordinates are in cell units, with cell centres at integers. The outer
        half cells use linear extrapolation for B. Normalization clamps to the
        nearest height endpoint, preserving positive mean-field scales.
        """
        begin, end = self.surface_ranges[name]
        n0, n1 = self.surface_shapes[name]
        grid = values[:, begin:end]

        def axis(value, size):
            if size == 1:
                index = torch.zeros_like(value, dtype=torch.long)
                return index, index, torch.zeros_like(value).to(values)
            if clamp:
                value = value.clamp(0, size - 1)
            low = value.floor().long().clamp(0, size - 2)
            return low, low + 1, (value - low).to(values)

        i0, i1, u = axis(coordinates[:, 0], n0)
        j0, j1, v = axis(coordinates[:, 1], n1)
        # Broadcast over time and optional vector components.
        shape = (1, len(coordinates)) + (1,) * (values.ndim - 2)
        u, v = u.reshape(shape), v.reshape(shape)
        low = torch.lerp(grid[:, i0 * n1 + j0], grid[:, i0 * n1 + j1], v)
        high = torch.lerp(grid[:, i1 * n1 + j0], grid[:, i1 * n1 + j1], v)
        return torch.lerp(low, high, u)

    @torch.no_grad()
    def _training_samples(self, name, start, stop, times):
        fraction = self.options["jitter_fraction"]
        if not fraction:
            return (
                self.sample_positions[start:stop],
                self._time_values(self.sample_targets[:, start:stop], times),
                self._time_values(self.sample_normalization[:, start:stop], times),
            )
        begin, _ = self.surface_ranges[name]
        n0, n1 = self.surface_shapes[name]
        cells = self.sample_order[start:stop] - begin
        centres = torch.stack((cells // n1, cells % n1), dim=-1).to(self.positions)
        noise = torch.rand(centres.shape, device=centres.device, dtype=centres.dtype)
        # Exclude exact cell edges (including side points at r=1) numerically.
        coordinates = centres + (noise.clamp(1e-6, 1 - 1e-6) - 0.5) * fraction
        uv = (coordinates + 0.5) / coordinates.new_tensor([n0, n1])
        domain = self.surface_domains[
            "photosphere" if name == "photosphere" else "outer"
        ]
        if name in ("top", "photosphere"):
            lon, lat = uv.unbind(-1)
            height = torch.full_like(
                lon, 0.0 if name == "photosphere" else domain.height_bounds_Mm[1]
            )
        else:
            height = uv[:, 0] * domain.height_bounds_Mm[1]
            edge = uv[:, 1]
            if name in ("west", "east"):
                lon, lat = torch.full_like(edge, float(name == "west")), edge
            else:
                lon, lat = edge, torch.full_like(edge, float(name == "north"))
        positions = domain._positions(
            height, lon, lat, torch.zeros_like(height), device=height.device
        )["position_m"]
        targets = self._time_values(
            self._surface_interpolant(self.targets, name, coordinates), times
        )
        if name == "photosphere":
            # Interpolate the tangent field, but retain the exact piecewise-
            # constant Neumann Br of the containing cell at the moved point.
            radial = self._time_values(self.source_reference[:, cells], times)
            direction = (positions / self.radius).to(targets)
            targets = (
                targets + (radial - (targets * direction).sum(-1))[:, None] * direction
            )
        normalization = self._time_values(
            self._surface_interpolant(
                self.normalization, name, coordinates, clamp=True
            ),
            times,
        )
        return positions, targets, normalization

    def _residual(self, model, positions, times, targets, normalization):
        """Field error normalized by a detached local characteristic strength."""
        normalized_evaluator = getattr(model, "evaluate_position_rsun_normalized", None)
        if normalized_evaluator is None:
            prediction = model.evaluate_position_rsun(positions / self.radius, times)[
                "magnetic_field"
            ]
            target = targets.to(prediction)
            field_scale = normalization.to(prediction)
            floor = prediction.new_tensor(self.options["field_floor_gauss"])
        else:
            prediction = normalized_evaluator(
                positions / self.radius,
                times,
            )["magnetic_field"]
            magnetic_scale = prediction.new_tensor(
                float(model.magnetic_scale_gauss)
            )
            target = targets.to(prediction) / magnetic_scale
            field_scale = normalization.to(prediction) / magnetic_scale
            floor = prediction.new_tensor(
                self.options["field_floor_gauss"]
            ) / magnetic_scale
        denominator = torch.sqrt(field_scale.square() + floor.square())
        denominator = denominator.clamp_min(torch.finfo(prediction.dtype).eps)
        return ((prediction - target) / denominator.reshape(-1, 1)).square()

    def _photosphere_residual(self, model, positions, times, targets, normalization):
        """Penalize >45-degree disagreement with a strong reference tangent field."""
        normalized_evaluator = getattr(model, "evaluate_position_rsun_normalized", None)
        if normalized_evaluator is None:
            prediction = model.evaluate_position_rsun(positions / self.radius, times)[
                "magnetic_field"
            ]
            field_scale = 1.0
        else:
            prediction = normalized_evaluator(
                positions / self.radius,
                times,
            )["magnetic_field"]
            field_scale = float(model.magnetic_scale_gauss)
            targets = targets / field_scale
        radial = torch.nn.functional.normalize(positions, dim=-1).to(prediction)
        horizontal = prediction - (prediction * radial).sum(-1, keepdim=True) * radial
        reference = targets - (targets * radial).sum(-1, keepdim=True) * radial
        strength = reference.norm(dim=-1)
        floor = self.options["field_floor_gauss"] / field_scale
        direction = reference / strength.clamp_min(floor)[:, None]
        dot = (horizontal * direction).sum(-1)
        cross = (torch.linalg.cross(direction, horizontal, dim=-1) * radial).sum(-1)
        # atan2 supplies a corrective angular gradient even at an exact 180-degree
        # reversal. At a numerically zero model field the direction is undefined;
        # use a bounded directional seed with a finite gradient at zero instead.
        epsilon = floor * 1e-6
        resolved = horizontal.norm(dim=-1) >= epsilon
        angle = torch.atan2(
            torch.where(resolved, cross, 0.0), torch.where(resolved, dot, 1.0)
        )
        # Zero at the 45-degree threshold, rising to 1 at a full 180-degree
        # reversal (kept fixed as the natural upper end of the scale).
        threshold = torch.pi / 4
        angular = (
            (angle.abs() - threshold) / (torch.pi - threshold)
        ).clamp_min(0).square()
        seed = 0.5 * (1 - dot / epsilon)
        return torch.where(
            strength
            >= self.options["photosphere"]["minimum_horizontal_field_gauss"]
            / field_scale,
            torch.where(resolved, angular, seed),
            0.0,
        )[:, None]

    def evaluate(self, atmosphere_model):
        self.prepare(atmosphere_model)
        zero = next(atmosphere_model.parameters()).new_zeros(())
        weight = self._weight()
        photo_weight = self._photosphere_weight()
        metrics = {
            "weight": zero + weight,
            "photosphere_weight": zero + photo_weight,
            "updates": zero + self.update_count,
            "frozen": zero + int(self.frozen),
            "time_anchors": zero + len(self.times),
            "last_update": zero + self.last_update,
            "target_relative_change": self.last_relative_change.detach(),
        }
        losses = {}
        if self.last_update < 0:
            return SharedTermResult(zero, losses, metrics, active=False)
        for name, (begin, end) in self.surface_ranges.items():
            is_photo = name == "photosphere"
            face_weight = photo_weight if is_photo else weight
            if not face_weight:
                continue
            batch = (
                self.options["photosphere"]["batch_size"]
                if is_photo
                else self.options["batch_size"]
            )
            # Keep the total side budget approximately equal to the top budget.
            if name not in ("top", "photosphere"):
                batch = max(1, batch // 4)
                face_weight /= 4
            count = end - begin
            residual_function = (
                self._photosphere_residual if is_photo else self._residual
            )
            components = 1 if is_photo else 3
            raw = zero
            if self.training:
                chunks = (count + batch - 1) // batch
                start = begin + int(torch.randint(chunks, ()).item()) * batch
                stop = min(start + batch, end)
                times = self.times[0] + torch.rand(
                    stop - start, device=self.times.device
                ) * (self.times[-1] - self.times[0])
                positions, target, normalization = self._training_samples(
                    name, start, stop, times
                )
                residual = residual_function(
                    atmosphere_model, positions, times, target, normalization
                )
                if is_photo:
                    # Average only over points that are not excluded (too weak
                    # a reference field) and not already within 45 degrees (a
                    # zero residual there): otherwise a chunk dominated by
                    # already-satisfied or excluded points dilutes the signal
                    # from the few points that still disagree. Exactly zero
                    # when nothing in the chunk is considered.
                    considered = (residual.reshape(-1) > 0).sum()
                    raw = residual.sum() / considered.clamp_min(1)
                else:
                    raw = residual.mean()
                    raw = raw * ((stop - start) * chunks / count)
            else:
                if is_photo:
                    residual_total = zero
                    considered_total = zero
                    for anchor, time in enumerate(self.times):
                        for start in range(begin, end, batch):
                            stop = min(start + batch, end)
                            residual = residual_function(
                                atmosphere_model,
                                self.positions[start:stop],
                                time.expand(stop - start),
                                self.targets[anchor, start:stop],
                                self.normalization[anchor, start:stop],
                            )
                            residual_total = residual_total + residual.sum()
                            considered_total = considered_total + (
                                residual.reshape(-1) > 0
                            ).sum()
                    raw = residual_total / considered_total.clamp_min(1)
                else:
                    for anchor, time in enumerate(self.times):
                        for start in range(begin, end, batch):
                            stop = min(start + batch, end)
                            residual = residual_function(
                                atmosphere_model,
                                self.positions[start:stop],
                                time.expand(stop - start),
                                self.targets[anchor, start:stop],
                                self.normalization[anchor, start:stop],
                            )
                            raw = raw + residual.sum() / (
                                components * count * len(self.times)
                            )
            key = name if name in ("top", "photosphere") else "side"
            losses[key] = losses.get(key, zero) + face_weight * raw
        return SharedTermResult(
            sum(losses.values(), zero), losses, metrics, active=bool(losses)
        )
