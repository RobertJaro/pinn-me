"""Common query interface for PROM3THEUS P3S save states."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
from .view import StreamEvaluationView
from .validation import _parse_observation_spec
from prom3theus.config import parse_config
from typing import Any

import numpy as np
import torch

from prom3theus.core import spherical_to_cartesian
from prom3theus.diagnostics.evaluation import AtmosphereEvaluator, AtmosphereSampling
from prom3theus.observations import (
    ObservationRaster,
    ObservationStore,
    StoredObservationDataModule,
)

from .checkpoint import ValidatedSaveState, load_validated_save_state
from .contracts import StoredRasterSelection
from .errors import ArtifactExportError
from .evaluation import (
    depth_grid,
    evaluate_atmosphere,
    resolve_storage_dtype,
    select_export_device,
)
from .full_shell import evaluate_full_shell_atmosphere, full_shell_height_grid
from .stokes import evaluate_stokes
from .validation import (
    scene_basis,
    validate_raster_scientific_contract,
    validate_raster_velocity_contract,
)


class P3SLoader:
    """Validate one P3S bundle and expose all common scientific queries."""

    def __init__(
        self, path: str | Path, *, device: str = "auto", stream_id: str | None = None
    ):
        self.state: ValidatedSaveState = load_validated_save_state(path)
        self.device_name = str(device)
        self._data_module = None
        configuration = deepcopy(self.state.context["configuration"])
        potential = configuration.get("physics", {}).get("potential_boundary", {})
        if "interior" in potential:
            # Old potential sampling is training-only and has no role in P3S
            # queries. Keep its original metadata in state.context; do not
            # reinterpret a historical volume prior as a photospheric prior.
            configuration["physics"].pop("potential_boundary")
        self._config = parse_config(configuration, base_directory=self.state.path.parent)
        self.stream_id = stream_id or self._config.scene.reference_stream
        if self.stream_id not in self.state.module.terms:
            raise KeyError(f"Unknown stream: {self.stream_id}")
        self._view = StreamEvaluationView(self.state.module, self.stream_id).eval()

    @property
    def path(self) -> Path:
        return self.state.path

    @property
    def module(self):
        return self._view

    @property
    def config(self):
        return self._config

    @property
    def observation(self):
        record = self.state.context["streams"][self.stream_id]
        metadata = record["store_metadata"]
        return SimpleNamespace(
            source_signature=record["source_signature"],
            spec=_parse_observation_spec(record["specification"]),
            raster_names=tuple(metadata.get("raster_names", ())),
            validation_raster_index=int(metadata.get("validation_raster_index", 0)),
            # Snapshots serialize sequences as lists and may retain FITS scale
            # casing. Match _time_records without relaxing timestamp equality.
            times=tuple(
                {**record, "scale": str(record["scale"]).lower()}
                for record in metadata.get("times", [])
            ),
            bounds=metadata.get("bounds", {}),
        )

    @property
    def resources(self) -> dict[str, Any]:
        return self.state.context["resources"]

    @property
    def epoch(self) -> int:
        return self.state.epoch

    @property
    def global_step(self) -> int:
        return self.state.global_step

    @property
    def raster_names(self) -> tuple[str, ...]:
        return self.observation.raster_names

    @property
    def raster_count(self) -> int:
        return len(self.raster_names)

    @property
    def store_metadata(self) -> dict[str, Any]:
        return {
            "source_signature": self.observation.source_signature,
            "validation_raster_index": self.observation.validation_raster_index,
            "times": [dict(item) for item in self.observation.times],
            "bounds": dict(self.observation.bounds),
        }

    @property
    def observation_cache_path(self) -> Path:
        """Exact signature-addressed converted observation store for this state."""

        return Path(self.state.context["streams"][self.stream_id]["store_path"])

    @property
    def stream_ids(self):
        return tuple(self.state.module.terms)

    @torch.no_grad()
    def predict(self, stream_id, batch):
        """Predict any configured observation from a canonical batch."""
        from prom3theus.components.forward import predict_term

        self._evaluation_module(
            include_stokes=self.state.module.terms[stream_id].observation_kind
            == "stokes"
        )
        model = self.state.module
        return predict_term(model.terms[stream_id], batch)

    @staticmethod
    def _time_records(
        rasters: list[ObservationRaster],
    ) -> tuple[dict[str, Any], ...]:
        records = []
        for raster in rasters:
            times = raster.metadata.get("times")
            coordinates = raster.metadata.get("coordinates")
            if not isinstance(times, list) or not isinstance(coordinates, Mapping):
                raise ArtifactExportError("Reloaded observation times are invalid.")
            scale = str(coordinates.get("time_scale", "utc")).lower()
            records.append({"values": list(times), "scale": scale})
        return tuple(records)

    def _observations(self):
        if self._data_module is None:
            observation_config = next(
                stream.observation.to_dict()
                for stream in self.config.streams
                if stream.id == self.stream_id
            )
            observation_type = str(observation_config["type"])
            loader_config = observation_config["loader"]
            store_path = self.observation_cache_path
            try:
                rasters, names, metadata = ObservationStore.load_sequence(
                    store_path,
                    mmap=True,
                    verify=True,
                    expected_signature=self.observation.source_signature,
                )
                if metadata.get("adapter") != observation_type:
                    raise ValueError("Observation-cache adapter type differs.")
                from prom3theus.observations.scene import rebase_stokes_raster

                rasters = [
                    rebase_stokes_raster(raster, self.state.scene) for raster in rasters
                ]
                data = StoredObservationDataModule(
                    rasters,
                    names,
                    validation_raster_index=int(metadata["validation_raster_index"]),
                    store_metadata=metadata,
                    batch_size=int(loader_config["batch_size"]),
                    validation_batch_size=int(loader_config["validation_batch_size"]),
                    validation_stride=int(loader_config["validation_stride"]),
                    num_workers=max(1, int(loader_config.get("workers", 2))),
                    pin_memory=False,
                )
                data.observation_store_path = store_path
                data.setup("fit")
            except (
                FileNotFoundError,
                IndexError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                raise ArtifactExportError(
                    f"Could not load the P3S observation cache {store_path}: {error}"
                ) from error
            reference = self.observation
            if metadata.get("observation") != reference.spec.metadata():
                raise ArtifactExportError("Reloaded observation specification differs.")
            if tuple(data.raster_names) != reference.raster_names:
                raise ArtifactExportError("Reloaded observation raster names differ.")
            if data.validation_raster_index != reference.validation_raster_index:
                raise ArtifactExportError(
                    "Reloaded validation raster selection differs."
                )
            if self._time_records(data.rasters) != reference.times:
                raise ArtifactExportError("Reloaded observation times differ.")
            if data.observation_sampling_bounds != reference.bounds:
                raise ArtifactExportError("Reloaded observation bounds differ.")
            self._data_module = data
        return self._data_module

    def _validated_selection(self, index: int) -> StoredRasterSelection:
        rasters = self._observations().rasters
        if type(index) is not int or index < 0 or index >= len(rasters):
            raise IndexError(f"raster index {index!r} is outside the stored sequence.")
        raster = rasters[index]
        validate_raster_scientific_contract(raster, self.observation.spec)
        validate_raster_velocity_contract(
            raster, self.observation.spec.velocity_synthesis_mode.value
        )
        scene_basis(raster)
        return StoredRasterSelection(
            raster=raster,
            name=self.observation.raster_names[index],
            index=index,
            raster_count=len(rasters),
        )

    def select_raster(
        self, *, index: int | None = None, name: str | None = None
    ) -> StoredRasterSelection:
        """Select one validated raster by index, name, or stored validation role."""

        if index is not None and name is not None:
            raise ValueError("Select a raster by index or name, not both.")
        if name is not None:
            if not isinstance(name, str) or not name:
                raise ValueError("raster name must be a non-empty string.")
            try:
                index = self.raster_names.index(name)
            except ValueError as error:
                raise KeyError(f"Unknown raster name: {name!r}.") from error
        if index is None:
            index = self.observation.validation_raster_index
        return self._validated_selection(index)

    @staticmethod
    def _raster_time(record: Mapping[str, Any]):
        try:
            from astropy.time import Time
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Time-based raster selection requires astropy."
            ) from error
        times = record.get("values")
        if (
            not isinstance(times, list)
            or len(times) != 1
            or not isinstance(times[0], str)
        ):
            raise ValueError(
                "Stored raster must contain one explicit observation time."
            )
        scale = str(record.get("scale", "")).lower()
        if scale not in {"tai", "utc"}:
            raise ValueError("Stored raster time_scale must be TAI or UTC.")
        return Time(times[0], scale=scale)

    def select_nearest_time(
        self, target: Any, *, tolerance_seconds: float
    ) -> tuple[StoredRasterSelection, float]:
        """Select the stored raster nearest an Astropy-compatible time."""

        index, separation = self.match_time(target, tolerance_seconds=tolerance_seconds)
        return self._validated_selection(index), separation

    def match_time(self, target: Any, *, tolerance_seconds: float) -> tuple[int, float]:
        """Match a physical time without loading the observation cache."""

        if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
            raise ValueError("tolerance_seconds must be finite and non-negative.")
        try:
            from astropy.time import Time
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Time-based raster selection requires astropy."
            ) from error
        if not isinstance(target, Time):
            target = Time(target)
        separations = np.asarray(
            [
                abs(float((self._raster_time(record) - target).to_value("s")))
                for record in self.observation.times
            ]
        )
        index = int(np.argmin(separations))
        separation = float(separations[index])
        if separation > tolerance_seconds:
            raise ValueError(
                "No stored raster matches the requested time within "
                f"{tolerance_seconds:g} s; nearest is {separation:.3f} s away."
            )
        return index, separation

    def _evaluation_module(self, *, include_stokes: bool = False):
        parameter = next(self.module.parameters())
        device = select_export_device(
            self.device_name,
            parameter.dtype,
            include_stokes=include_stokes,
        )
        return self.module.to(device).eval()

    def cube(
        self,
        *,
        depth_samples: int | None = 101,
        batch_size: int = 4096,
        storage_dtype: str = "float32",
        raster_index: int | None = None,
        raster_name: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate the atmosphere cube along one stored observation raster."""

        resolve_storage_dtype(storage_dtype)
        module = self._evaluation_module()
        selection = self.select_raster(index=raster_index, name=raster_name)
        return evaluate_atmosphere(
            module,
            selection.raster,
            depth_grid=depth_grid(module, depth_samples),
            batch_size=batch_size,
            storage_dtype=storage_dtype,
        )

    def full_cube(
        self,
        *,
        height_samples: int = 101,
        batch_size: int = 4096,
        storage_dtype: str = "float32",
        raster_index: int | None = None,
        raster_name: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate radial columns over the complete configured shell."""

        resolve_storage_dtype(storage_dtype)
        module = self._evaluation_module()
        selection = self.select_raster(index=raster_index, name=raster_name)
        heights = full_shell_height_grid(module, height_samples)
        return evaluate_full_shell_atmosphere(
            module,
            selection.raster,
            height_grid_m=heights,
            batch_size=batch_size,
            storage_dtype=storage_dtype,
        )

    def stokes(
        self,
        *,
        batch_size: int = 16,
        storage_dtype: str = "float32",
        raster_index: int | None = None,
        raster_name: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Forward-render Stokes profiles for one stored observation raster."""

        resolve_storage_dtype(storage_dtype)
        module = self._evaluation_module(include_stokes=True)
        selection = self.select_raster(index=raster_index, name=raster_name)
        return evaluate_stokes(
            module,
            selection.raster,
            self.observation.spec,
            batch_size=batch_size,
            storage_dtype=storage_dtype,
        )

    def _slice_evaluator(
        self,
        *,
        longitude_points: int,
        latitude_points: int,
        radial_points: int,
        batch_size: int,
        meridional_longitude_deg: float | None = None,
    ) -> AtmosphereEvaluator:
        return AtmosphereEvaluator(
            AtmosphereSampling.from_options(
                slice_sampling={
                    "longitude_points": longitude_points,
                    "latitude_points": latitude_points,
                    "radial_points": radial_points,
                    "layer_count": 3,
                    "batch_size": batch_size,
                },
                meridional_slice={
                    "enabled": meridional_longitude_deg is not None,
                    "longitude_deg": meridional_longitude_deg,
                },
            )
        )

    def shell_slices(
        self,
        heights_m: np.ndarray | list[float] | tuple[float, ...],
        *,
        longitude_points: int = 256,
        latitude_points: int = 256,
        batch_size: int = 8192,
        raster_index: int | None = None,
        raster_name: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate regular longitude-latitude layers at explicit heights."""

        module = self._evaluation_module()
        selection = self.select_raster(index=raster_index, name=raster_name)
        evaluator = self._slice_evaluator(
            longitude_points=longitude_points,
            latitude_points=latitude_points,
            radial_points=2,
            batch_size=batch_size,
        )
        return evaluator.evaluate_shell_layers(
            module, selection.raster, np.asarray(heights_m)
        )

    def meridional_slice(
        self,
        longitude_deg: float,
        *,
        latitude_points: int = 256,
        radial_points: int = 192,
        batch_size: int = 8192,
        raster_index: int | None = None,
        raster_name: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a constant-Carrington-longitude radial slice."""

        module = self._evaluation_module()
        selection = self.select_raster(index=raster_index, name=raster_name)
        evaluator = self._slice_evaluator(
            longitude_points=2,
            latitude_points=latitude_points,
            radial_points=radial_points,
            batch_size=batch_size,
            meridional_longitude_deg=longitude_deg,
        )
        return evaluator.evaluate_meridional_slice(module, selection.raster)

    def spherical_time_series(
        self,
        *,
        height_m: float = 0.0,
        longitude_points: int = 96,
        latitude_points: int = 96,
        current_height_samples: int = 48,
        batch_size: int = 4096,
    ) -> dict[str, Any]:
        """Evaluate every stored time on one fixed Carrington grid.

        Surface magnetic and velocity vectors are returned in the spherical
        basis. The current-density diagnostic is the radial column integral
        of ``|J|`` over the complete model shell. This query uses only the
        compact P3S time/bounds contract and never loads observational arrays.
        """

        if not math.isfinite(height_m):
            raise ValueError("height_m must be finite.")
        if min(longitude_points, latitude_points) < 2:
            raise ValueError("Angular time-series grids require at least two points.")
        if current_height_samples < 3:
            raise ValueError("Current integration requires at least three heights.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        module = self._evaluation_module()
        model = module.atmosphere_model
        bounds = self.observation.bounds
        radius_m = float(bounds["solar_radius_m"])
        longitude_center = float(bounds["surface_longitude_center_rad"])
        longitude_offset = bounds["surface_longitude_offset_rad"]
        latitude_bounds = bounds["surface_latitude_rad"]
        longitude = torch.linspace(
            longitude_center + float(longitude_offset[0]),
            longitude_center + float(longitude_offset[1]),
            longitude_points,
            dtype=torch.float32,
        )
        latitude = torch.linspace(
            float(latitude_bounds[0]),
            float(latitude_bounds[1]),
            latitude_points,
            dtype=torch.float32,
        )
        latitude_grid, longitude_grid = torch.meshgrid(
            latitude, longitude, indexing="ij"
        )

        outer_height_m, inner_height_m = (
            float(value) * 1.0e6 for value in model.shell_height_bounds_Mm
        )
        offset_m = 1.0e5
        above_reference = (
            np.geomspace(
                offset_m,
                outer_height_m + offset_m,
                current_height_samples - 1,
            )
            - offset_m
        )
        integration_heights_m = np.concatenate(
            ([inner_height_m], above_reference)
        ).astype(np.float32)
        if not inner_height_m <= height_m <= outer_height_m:
            raise ValueError(
                "height_m must lie inside the configured atmosphere shell."
            )

        def positions(heights: torch.Tensor) -> torch.Tensor:
            spherical = torch.stack(
                (
                    (radius_m + heights[:, None, None]).expand(
                        -1, latitude_points, longitude_points
                    ),
                    (0.5 * math.pi - latitude_grid)[None].expand(
                        heights.numel(), -1, -1
                    ),
                    longitude_grid[None].expand(heights.numel(), -1, -1),
                ),
                dim=-1,
            )
            return spherical_to_cartesian(spherical, torch).permute(1, 2, 0, 3)

        surface_position = positions(torch.tensor([height_m], dtype=torch.float32))[
            ..., 0, :
        ]
        volume_position = positions(torch.from_numpy(integration_heights_m))
        evaluator = self._slice_evaluator(
            longitude_points=longitude_points,
            latitude_points=latitude_points,
            radial_points=current_height_samples,
            batch_size=batch_size,
        )

        first_time = self._raster_time(self.observation.times[0])
        physical_times = [
            self._raster_time(record) for record in self.observation.times
        ]
        elapsed_hours = np.asarray(
            [float((time - first_time).to_value("hour")) for time in physical_times],
            dtype=np.float32,
        )
        allowed_time = np.asarray(bounds["time_hours"], dtype=float)
        if (
            elapsed_hours.min() < allowed_time[0] - 1.0e-4
            or elapsed_hours.max() > allowed_time[1] + 1.0e-4
        ):
            raise ArtifactExportError(
                "Stored observation times fall outside the P3S time bounds."
            )

        names = (
            "b_r",
            "b_theta",
            "b_phi",
            "v_r",
            "v_theta",
            "v_phi",
        )
        maps = {name: [] for name in names}
        maps.update(
            {
                "field_strength": [],
                "integrated_current_density": [],
                "radial_poynting_flux": [],
            }
        )
        mu0 = 4.0 * math.pi * 1.0e-7
        trapezoid = getattr(np, "trapezoid", None)
        if trapezoid is None:  # NumPy < 2.0
            trapezoid = np.trapz
        for time_hours in elapsed_hours:
            surface = evaluator._evaluate_physical_positions(
                module, surface_position, time_hours=float(time_hours)
            )
            volume = evaluator._evaluate_physical_positions(
                module, volume_position, time_hours=float(time_hours)
            )
            for name in names:
                maps[name].append(surface[name])
            magnetic = np.stack(
                (surface["b_r"], surface["b_theta"], surface["b_phi"]), axis=-1
            )
            velocity_km_s = np.stack(
                (surface["v_r"], surface["v_theta"], surface["v_phi"]), axis=-1
            )
            magnetic_tesla = magnetic * 1.0e-4
            velocity_m_s = velocity_km_s * 1.0e3
            magnetic_squared = np.sum(magnetic_tesla**2, axis=-1)
            velocity_dot_magnetic = np.sum(velocity_m_s * magnetic_tesla, axis=-1)
            radial_poynting = (
                magnetic_squared * velocity_m_s[..., 0]
                - velocity_dot_magnetic * magnetic_tesla[..., 0]
            ) / mu0
            maps["field_strength"].append(np.linalg.norm(magnetic, axis=-1))
            maps["integrated_current_density"].append(
                trapezoid(
                    volume["current_density"],
                    x=integration_heights_m,
                    axis=-1,
                )
            )
            maps["radial_poynting_flux"].append(radial_poynting)

        return {
            "times": tuple(time.isot for time in physical_times),
            "time_scale": str(first_time.scale),
            "elapsed_hours": elapsed_hours,
            "height_m": float(height_m),
            "integration_height_m": integration_heights_m,
            "longitude_deg": np.rad2deg(longitude_grid.numpy()),
            "latitude_deg": np.rad2deg(latitude_grid.numpy()),
            "map_fields": {
                name: np.stack(values).astype(np.float32, copy=False)
                for name, values in maps.items()
            },
            "component_convention": {
                "r": "outward",
                "theta": "southward",
                "phi": "westward/increasing Carrington longitude",
            },
        }

    @torch.inference_mode()
    def raster_fields_at_height(
        self,
        height_m: float,
        *,
        batch_size: int = 4096,
        raster_index: int | None = None,
        raster_name: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate common atmosphere fields at one height on stored pixels."""

        if not math.isfinite(height_m):
            raise ValueError("height_m must be finite.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        module = self._evaluation_module()
        selection = self.select_raster(index=raster_index, name=raster_name)
        raster = selection.raster
        return self.fields_at_coordinates(
            raster.coordinates,
            raster.valid_mask,
            height_m,
            batch_size=batch_size,
            module=module,
        )

    @torch.inference_mode()
    def fields_at_coordinates(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        height_m: float,
        *,
        batch_size: int = 4096,
        module=None,
    ) -> dict[str, np.ndarray]:
        """Evaluate common fields on an explicit P3S-coordinate grid."""

        from prom3theus.observations.arrays import materialize_array
        coordinates = torch.as_tensor(materialize_array(coordinates))
        valid_mask = torch.as_tensor(materialize_array(valid_mask))
        if coordinates.shape != (*valid_mask.shape, 3):
            raise ValueError("coordinates must have shape [*valid_mask.shape, 3].")
        if valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must be boolean.")
        if not torch.isfinite(coordinates[valid_mask]).all():
            raise ValueError("Valid grid coordinates must be finite.")
        if not math.isfinite(height_m):
            raise ValueError("height_m must be finite.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if module is None:
            module = self._evaluation_module()
        model = module.atmosphere_model
        parameter = next(model.parameters())
        valid = valid_mask.reshape(-1)
        indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            raise ArtifactExportError("The selected raster contains no valid pixels.")
        coordinates = coordinates.reshape(-1, 3).index_select(0, indices)
        shape = valid_mask.shape
        scalar_names = ("temperature", "microturbulence", "gas_pressure")
        scalars = {
            name: np.full(valid.numel(), np.nan, dtype=np.float32)
            for name in scalar_names
        }
        vectors = {
            name: np.full((valid.numel(), 3), np.nan, dtype=np.float32)
            for name in ("magnetic_field", "velocity_field")
        }
        was_training = model.training
        model.eval()
        try:
            for start in range(0, indices.numel(), batch_size):
                stop = min(start + batch_size, indices.numel())
                batch_coordinates = coordinates[start:stop].to(parameter)
                heights = torch.full(
                    (stop - start,),
                    height_m,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                fields = model.evaluate_chart_height_points(batch_coordinates, heights)
                selected = indices[start:stop].cpu().numpy()
                for name in scalar_names:
                    scalars[name][selected] = (
                        fields[name].detach().float().cpu().numpy()
                    )
                for name in vectors:
                    vectors[name][selected] = (
                        fields[name].detach().float().cpu().numpy()
                    )
        finally:
            model.train(was_training)
        result = {
            "temperature_k": scalars["temperature"].reshape(shape),
            "microturbulence_m_per_s": scalars["microturbulence"].reshape(shape),
            "gas_pressure_pa": scalars["gas_pressure"].reshape(shape),
            "magnetic_field_gauss": vectors["magnetic_field"].reshape(*shape, 3),
            "velocity_field_m_per_s": vectors["velocity_field"].reshape(*shape, 3),
            "valid_mask": valid_mask.detach().cpu().numpy(),
        }
        return result


__all__ = ["P3SLoader"]
