"""Forward-render synthetic HMI and AIA observables from a MURaM snapshot.

This is the validation entry point: known MURaM thermodynamics and magnetic
field are pushed through an independent numerical LTE reference solver
(:mod:`prom3theus.validation.reference_synthesis`) and the instrument response
code (:mod:`prom3theus.instruments`). Production inversions use parallel formal LTE integration. The source atmosphere is known, but inversion recovery remains
limited by its geometry, EOS, finite sampling, and observational degeneracies.
"""

from __future__ import annotations

import logging

from dataclasses import asdict, dataclass, field, replace
from numbers import Integral

import numpy as np
import torch

from prom3theus.core import carrington_rotation_velocity_cartesian
from prom3theus.instruments.aia_euv.operator import AIAEmissionOperator
from prom3theus.instruments.registry import build_instrument
from prom3theus.observations.contracts import ObservationRaster
from prom3theus.rt import HybridSolarEOS
from prom3theus.validation.reference_synthesis import LTESynthesizer
from prom3theus.validation.reference_transfer import GeometricHeightPath, RayDistancePath

from .atmosphere_builder import (
    DEFAULT_STOKES_HEIGHT_MEGAMETERS, NATIVE_ATMOSPHERE_FIELDS, UniformShellGrid, build_native_atmosphere, stokes_depth_count,
)
from .columns import column_batches, snapshot_columns, strided_columns
from .diagnostic_maps import DiagnosticAccumulator, DiagnosticMaps
from .geometry import SOLAR_RADIUS_M, DiskCenterGeometry, build_disk_center_geometry
from .hmi_filter import build_idealized_hmi_filter
from .provenance import stokes_forward_provenance
from .reader import MURaMSnapshot
from .reporting import timed_stage

logger = logging.getLogger(__name__)


DEFAULT_AIA_CHANNELS = (171, 193, 211)
AIA_RESPONSE_RESOURCE = "aia_euv_v1:aia_temperature_response"

# Purely an internal normalization constant to keep the LTE forward-model's
# numeric magnitude near unity (see LTESynthesizer's radiance_scale
# argument); it is not a physical calibration. It must match exactly between
# the Stokes values written here and prom3theus.muram.observation's
# ObservationSpec.radiance_scale_w_m3_sr, which is why both import it from
# here rather than repeating the literal.
MURAM_HMI_RADIANCE_SCALE_W_M3_SR = 1.0e13


@dataclass(frozen=True)
class SyntheticHMIResult:
    stokes: torch.Tensor  # [ny, nx, 4, 6], HMI-filter-integrated at 6 tuning positions
    observed_wavelength_angstrom: torch.Tensor
    raw_stokes: torch.Tensor | None  # Optional [ny, nx, 4, wavelength] pre-instrument spectra
    raw_wavelength_angstrom: torch.Tensor
    geometry: DiskCenterGeometry
    dx_m: float
    dy_m: float
    instrument_config: dict = field(default_factory=dict)
    line_ids: tuple[str, ...] = ("FeI_6173.3352",)
    response: dict = field(default_factory=dict)
    native_depth_samples: int = 0
    native_vertical_spacing_m: float = 0.0
    stokes_depth_samples: int = 0
    stokes_height_megameters: float | None = DEFAULT_STOKES_HEIGHT_MEGAMETERS
    native_component_order: tuple[int, int, int] = (0, 1, 2)
    spatial_stride: int = 1
    spatial_binning: int = 1
    forward_model_provenance: dict = field(default_factory=dict)
    native_reference: dict = field(default_factory=dict)
    diagnostics: DiagnosticMaps | None = None
    uniform_shell_grid: UniformShellGrid | None = None

    @property
    def mu(self) -> torch.Tensor:
        return self.geometry.mu

    def observation_metadata(self) -> dict:
        """Describe the spectrum without constructing spatial response arrays."""
        ny, nx = self.geometry.mu.shape
        return {
            "ray_geometry": {
                "solar_radius_m": SOLAR_RADIUS_M,
                "scene_basis_rows": self.geometry.stokes_basis[0, 0].tolist(),
            },
            "coordinates": {
                "time_origin": "2000-01-01T00:00:00",
                "time_scale": "tai",
                "network_affine": {
                    "center_megameter": [0.0, 0.0],
                    "scale_megameter": [
                        max(nx * self.dx_m / 2e6, 1e-6),
                        max(ny * self.dy_m / 2e6, 1e-6),
                    ],
                },
            },
            "stokes_order": ["I", "Q", "U", "V"],
            "normalization": {
                "indices": [0, self.stokes.shape[-1] - 1],
                "radiometric_calibration": {
                    "atlas_disk_center_continuum_radiance_w_m3_sr": MURAM_HMI_RADIANCE_SCALE_W_M3_SR
                },
            },
            "observer_velocity_correction": {
                "removed_solar_los_velocity_m_per_s": [0.0] * nx
            },
            "muram_synthesis": {
                "forward_model_provenance": self.forward_model_provenance,
                **({"native_reference": self.native_reference} if self.native_reference else {}),
                "instrument": self.instrument_config,
                "line_ids": list(self.line_ids),
                "radiance_scale": MURAM_HMI_RADIANCE_SCALE_W_M3_SR,
                "line_support_angstrom": [
                    float(self.raw_wavelength_angstrom[0]),
                    float(self.raw_wavelength_angstrom[-1]),
                ],
                "geometry": "parallel vertical columns; spherical surface embedding",
                "transfer_path": "ray_distance_m" if self.uniform_shell_grid else "geometric_height_m",
                "velocity_frame": "local co-rotating flows plus Carrington rotation; stationary observer",
                "depth_sampling": ("uniform_shell_ray_distance" if self.uniform_shell_grid else
                                   "native cells within Stokes height range; no vertical resampling"),
                **({"uniform_shell_grid": asdict(self.uniform_shell_grid)} if self.uniform_shell_grid else {}),
                "stokes_height_megameters": self.stokes_height_megameters,
                "stokes_depth_samples": self.stokes_depth_samples,
                "stokes_height_range_m": (self.uniform_shell_grid.native_height_range_m if self.uniform_shell_grid else
                                          [0.0, (self.stokes_depth_samples - 1) * self.native_vertical_spacing_m]),
                "native_depth_samples": self.native_depth_samples,
                "native_vertical_spacing_m": self.native_vertical_spacing_m,
                "spatial_sampling": "native columns [::stride, ::stride]; block mean after synthesis",
                "spatial_stride": self.spatial_stride,
                "spatial_binning": self.spatial_binning,
                "dx_m": self.dx_m,
                "dy_m": self.dy_m,
                "native_component_order_by_header_axis": list(self.native_component_order),
            },
        }

    def to_observation_raster(self) -> ObservationRaster:
        """Package the HMI-filter-integrated Stokes cube as the canonical
        :class:`ObservationRaster` the inversion pipeline consumes."""

        ny, nx = self.geometry.mu.shape
        return ObservationRaster(
            stokes=self.stokes,
            wavelength_angstrom=self.observed_wavelength_angstrom,
            coordinates=self.geometry.coordinates,
            ray_direction=self.geometry.ray_direction,
            surface_position_m=self.geometry.surface_position_m,
            stokes_basis=self.geometry.stokes_basis,
            valid_mask=torch.ones((ny, nx), dtype=torch.bool),
            metadata=self.observation_metadata(),
            auxiliary={
                "removed_solar_los_velocity_m_per_s": torch.zeros(
                    (ny, nx), dtype=self.stokes.dtype
                ),
                **{
                    "instrument_response:" + name: value.clone().expand(
                        ny, nx, *value.shape
                    )
                    for name, value in self.response.items()
                },
            },
        )


@dataclass(frozen=True)
class SyntheticAIAResult:
    intensity: torch.Tensor  # [ny, nx, channels]
    channels_angstrom: tuple[int, ...]
    dx_m: float
    dy_m: float
    native_reference: dict = field(default_factory=dict)
    response_resource: str = AIA_RESPONSE_RESOURCE
    diagnostics: DiagnosticMaps | None = None


def _validate_batch_size(batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    return int(batch_size)


def _validate_binning(binning, shape):
    factor = 1 if binning is None else binning
    if isinstance(factor, bool) or not isinstance(factor, Integral) or factor < 1:
        raise ValueError("binning must be a positive integer.")
    if any(size % factor for size in shape[:2]):
        raise ValueError("Horizontal dimensions after stride must be divisible by binning.")
    return int(factor)


def _bin_image(values, factor):
    """Average non-overlapping spatial blocks, preserving trailing axes."""
    if factor == 1:
        return values
    ny, nx = values.shape[:2]
    return values.reshape(ny // factor, factor, nx // factor, factor,
                          *values.shape[2:]).mean(dim=(1, 3))


def resolve_synthesis_device(device: torch.device | str = "auto") -> torch.device:
    """Use a visible CUDA GPU when available, matching the training selector."""
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("MURaM synthesis supports cpu, cuda, or cuda:N devices.")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA synthesis requested, but no CUDA GPU is available."
            )
        index = (
            torch.cuda.current_device() if resolved.index is None else resolved.index
        )
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {index}.")
        resolved = torch.device("cuda", index)
    return resolved


class _SynthesisGrid:
    """Shared column selection, output placement, and source identity."""

    def __init__(self, snapshot, *, stride, binning, batch_size, device, dtype):
        self.batch_size = _validate_batch_size(batch_size)
        self.device, self.dtype = resolve_synthesis_device(device), dtype
        source = snapshot_columns(snapshot)
        self.native_shape, self.ds_m = source.shape, source.ds_m
        self.columns, self.stride = strided_columns(source, stride)
        self.nx, self.ny, self.nz = self.columns.shape
        self.binning = _validate_binning(binning, self.columns.shape)
        self.time_s = snapshot.time_s
        self.component_order = tuple(getattr(snapshot, "component_order", (0, 1, 2)))
        descriptor = getattr(snapshot, "native_reference", None)
        self.native_reference = (
            descriptor(stride=self.stride, binning=self.binning) if descriptor is not None else {}
        )

    def geometry(self, *, binned=False):
        return build_disk_center_geometry(
            *self.native_shape[:2], *self.ds_m[:2], stride=self.stride,
            binning=self.binning if binned else 1,
            reference_time_hours=self.time_s / 3600, dtype=self.dtype,
        )

    def image(self, values):
        image = values.reshape(self.nx, self.ny, *values.shape[1:]).transpose(0, 1)
        return _bin_image(image, self.binning)


class _StokesRenderer:
    """Prepare the LTE/instrument models once and render one column batch at a time."""

    fields = NATIVE_ATMOSPHERE_FIELDS

    def __init__(self, grid, *, instrument, observed_wavelength_angstrom, line_ids,
                 response=None, keep_raw_stokes=False, stokes_height_megameters=DEFAULT_STOKES_HEIGHT_MEGAMETERS,
                 uniform_shell_grid=None):
        if not isinstance(keep_raw_stokes, bool):
            raise ValueError("keep_raw_stokes must be a boolean.")
        self.grid, self.instrument, self.line_ids = grid, dict(instrument), tuple(line_ids)
        self.stokes_height_megameters = stokes_height_megameters
        self.uniform_shell_grid = uniform_shell_grid
        if uniform_shell_grid is None:
            self.depth_count = stokes_depth_count(grid.nz, grid.ds_m[2], stokes_height_megameters)
            logger.info("Stokes: %d/%d native depths, height 0–%.3f megameters above bottom cell center",
                        self.depth_count, grid.nz, (self.depth_count - 1) * grid.ds_m[2] / 1e6)
        else:
            self.depth_count = uniform_shell_grid.depth_count
            logger.info("Stokes: %d equally spaced ray distances, native heights %.3f–%.3f Mm",
                        self.depth_count, *(h / 1e6 for h in uniform_shell_grid.native_height_range_m))
        # Absolute wavelengths remain float64 during spectral preparation.
        self.operator = build_instrument(instrument).double()
        observed = torch.as_tensor(observed_wavelength_angstrom, dtype=torch.float64, device="cpu")
        wavelength = self.operator.synthesis_grid(observed)
        if callable(getattr(self.operator, "prepare", None)):
            self.operator.prepare(wavelength, observed)
        self.response = {
            key: torch.as_tensor(value, dtype=grid.dtype, device=grid.device)
            for key, value in (response or {}).items()
        }
        self.synthesizer = LTESynthesizer(line_ids=self.line_ids).to(device=grid.device, dtype=grid.dtype)
        self.synthesizer.prepare_wavelength_grid(wavelength)
        self.synthesizer.to(device=grid.device, dtype=grid.dtype)
        self.operator.to(device=grid.device, dtype=grid.dtype)
        self.wavelength = wavelength.to(device=grid.device)
        self.observed = observed.to(device=grid.device)
        self.geometry = grid.geometry()
        # The native flows are co-rotating. Rotation's LOS contribution is
        # constant along a parallel vertical ray, so its surface value suffices.
        rotation = carrington_rotation_velocity_cartesian(self.geometry.surface_position_m, torch)
        self.rotation_los = (rotation * self.geometry.stokes_basis[..., 2, :]).sum(-1).T.reshape(-1)
        self.raw = (
            torch.empty((grid.nx * grid.ny, 4, wavelength.numel()), dtype=grid.dtype)
            if keep_raw_stokes else None
        )
        self.integrated = torch.empty((grid.nx * grid.ny, 4, observed.numel()), dtype=grid.dtype)

    def __call__(self, ix, iy, columns):
        indices = torch.from_numpy(ix * self.grid.ny + iy)
        tensor_options = dict(dtype=self.grid.dtype, device=self.grid.device)
        if self.uniform_shell_grid is None:
            lower_columns = {name: columns[name][..., :self.depth_count] for name in self.fields}
            lower_columns["ds_m"] = columns["ds_m"]
            atmosphere = build_native_atmosphere(lower_columns, **tensor_options)
            path = GeometricHeightPath(atmosphere.geometric_height_m, mu=1.0)
        else:
            distance, height = self.uniform_shell_grid.sample(self.geometry, ix, iy)
            atmosphere = build_native_atmosphere(columns, sample_height_m=height, **tensor_options)
            path = RayDistancePath(distance.to(**tensor_options))
        velocity = atmosphere.velocity_field.clone()
        velocity[..., 2] += self.rotation_los[indices].to(velocity)[:, None]
        atmosphere = replace(
            atmosphere, velocity_field=velocity,
            magnetic_field=self.operator.polarization_convention.to_synthesis_frame(atmosphere.magnetic_field),
        )
        profiles = self.synthesizer(
            atmosphere, None,
            path=path,
            radiance_scale=MURAM_HMI_RADIANCE_SCALE_W_M3_SR,
        )
        if self.raw is not None:
            self.raw[indices] = profiles.cpu()
        self.integrated[indices] = self.operator(
            profiles, self.wavelength, self.observed,
            **{key: value.expand(len(ix), *value.shape) for key, value in self.response.items()},
        ).cpu()

    def result(self):
        grid = self.grid
        return SyntheticHMIResult(
            stokes=grid.image(self.integrated),
            observed_wavelength_angstrom=self.observed.cpu(),
            raw_stokes=grid.image(self.raw) if self.raw is not None else None,
            raw_wavelength_angstrom=self.wavelength.cpu(),
            geometry=self.geometry if grid.binning == 1 else grid.geometry(binned=True),
            dx_m=grid.ds_m[0] * grid.stride * grid.binning,
            dy_m=grid.ds_m[1] * grid.stride * grid.binning,
            instrument_config=self.instrument, line_ids=self.line_ids,
            response={key: value.cpu() for key, value in self.response.items()},
            native_depth_samples=grid.nz, native_vertical_spacing_m=grid.ds_m[2],
            stokes_depth_samples=self.depth_count, stokes_height_megameters=self.stokes_height_megameters,
            uniform_shell_grid=self.uniform_shell_grid,
            spatial_stride=grid.stride, spatial_binning=grid.binning,
            native_component_order=grid.component_order,
            forward_model_provenance=stokes_forward_provenance(self.instrument["type"]),
            native_reference=grid.native_reference,
        )


class AIAAccumulator:
    """Integrate AIA from loaded columns, retaining only the output image."""

    fields = ("temperature", "pressure", "tau")

    def __init__(self, quantities, *, stride=1, binning=None,
                 channels_angstrom=DEFAULT_AIA_CHANNELS,
                 response_resource=AIA_RESPONSE_RESOURCE, device="cpu", dtype=torch.float32,
                 native_reference=None):
        self.nx, self.ny, _ = quantities.shape
        dx, dy, self.dz = quantities.ds_m
        self.binning = _validate_binning(binning, quantities.shape)
        self.dx, self.dy = dx * stride * self.binning, dy * stride * self.binning
        self.device, self.dtype = device, dtype
        self.native_reference = {} if native_reference is None else native_reference
        self.channels = tuple(channels_angstrom)
        self.response_resource = response_resource
        self.operator = AIAEmissionOperator(self.channels, response_resource=response_resource).to(
            device=device, dtype=dtype,
        )
        self.eos = HybridSolarEOS().to(device=device, dtype=dtype)
        self.intensity = torch.empty((self.nx * self.ny, len(self.channels)), dtype=dtype)

    @torch.no_grad()
    def __call__(self, ix, iy, columns):
        tau = columns["tau"][:, 0, :]
        if not np.isfinite(tau).all() or np.any(tau < 0):
            raise ValueError("AIA visibility requires finite, non-negative native tau values.")
        temperature = torch.as_tensor(columns["temperature"][:, 0, :], dtype=self.dtype, device=self.device)
        pressure = torch.as_tensor(columns["pressure"][:, 0, :], dtype=self.dtype, device=self.device)
        ne = self.eos.electron_density(temperature, pressure)
        ne = ne * torch.as_tensor(tau <= 1.0, device=self.device)
        # local_emission_measure_integrand uses SI densities and excludes 1e-10.
        local = self.operator.local_emission_measure_integrand(temperature, ne)
        self.intensity[torch.from_numpy(ix * self.ny + iy)] = (
            local.sum(dim=-2) * self.dz * 1e-10
        ).cpu()

    def result(self):
        image = self.intensity.reshape(self.nx, self.ny, len(self.channels)).transpose(0, 1)
        return SyntheticAIAResult(
            _bin_image(image, self.binning), self.channels, self.dx, self.dy,
            native_reference=self.native_reference, response_resource=self.response_resource,
        )


@torch.no_grad()
def _run_synthesis(snapshot, *, stokes=None, aia=None, stride=None, binning=None,
                   batch_size=256, progress=True, device="auto", dtype=torch.float32, diagnostics=True):
    """Load each selected batch once and execute its requested instruments sequentially."""
    iteration = getattr(snapshot, "iteration", "unknown")
    with timed_stage(logger, "Iteration %s: preparing native columns", iteration):
        grid = _SynthesisGrid(snapshot, stride=stride, binning=binning, batch_size=batch_size,
                              device=device, dtype=dtype)
    if stokes is not None and aia is None:
        shell = stokes.get("uniform_shell_grid")
        depth = (shell.retained_depth(grid.nz, grid.ds_m[2]) if shell else
                 stokes_depth_count(grid.nz, grid.ds_m[2], stokes.get("stokes_height_megameters", DEFAULT_STOKES_HEIGHT_MEGAMETERS)))
        grid.columns = grid.columns.cropped(depth)
        if grid.native_reference and depth < grid.nz:
            grid.native_reference = {**grid.native_reference, "column_selection": {
                "depth_samples": depth, "height_range_m": [0., (depth - 1) * grid.ds_m[2]],
            }}
        logger.info("Loading only selected XY columns and lower %d/%d native depths (0–%.3f megameters)",
                    depth, grid.nz, (depth - 1) * grid.ds_m[2] / 1e6)
    logger.info("Iteration %s: %d x %d columns, %d native depths; device=%s, dtype=%s, batch_size=%d, stride=%d, binning=%d",
                iteration, grid.nx, grid.ny, grid.nz, grid.device, grid.dtype,
                grid.batch_size, grid.stride, grid.binning)
    if not isinstance(diagnostics, bool):
        raise ValueError("diagnostics must be a boolean.")
    diagnostic = DiagnosticAccumulator(
        grid.columns.shape, grid.ds_m, native_shape=grid.native_shape, stride=grid.stride,
        synthesis_binning=grid.binning, native_reference=grid.native_reference, time_s=grid.time_s,
    ) if diagnostics else None
    with timed_stage(logger, "Iteration %s: preparing instrument models", iteration):
        renderers = []
        if stokes is not None:
            renderers.append(_StokesRenderer(grid, **stokes))
        if aia is not None:
            renderers.append(AIAAccumulator(
                grid.columns, stride=grid.stride, binning=grid.binning,
                device=grid.device, dtype=grid.dtype, native_reference=grid.native_reference, **aia,
            ))
    consumers = ([diagnostic] if diagnostic is not None else []) + renderers
    fields = tuple(dict.fromkeys(name for consumer in consumers for name in consumer.fields))
    for ix, iy, columns in column_batches(
        grid.columns, grid.batch_size, fields=fields, progress=progress,
        description=f"Iteration {iteration}: synthesis",
    ):
        for consumer in consumers:
            consumer(ix, iy, columns)
    maps = diagnostic.result() if diagnostic is not None else None
    return tuple(replace(renderer.result(), diagnostics=maps) for renderer in renderers)


def _hmi_options(dtype, keep_raw_stokes, stokes_height_megameters):
    operator, observed, weights, continuum = build_idealized_hmi_filter(dtype=dtype)
    return {
        "instrument": {
            "type": "hmi_filter_profiles",
            "quadrature_wavelength_angstrom": operator.quadrature_wavelength_angstrom.tolist(),
            "inner_half_width_angstrom": operator.inner_half_width_angstrom,
            "magnetic_azimuth_offset_deg": 90.0,
        },
        "observed_wavelength_angstrom": observed,
        "line_ids": ("FeI_6173.3352",),
        "response": {"spectral_weights": weights, "continuum_weights": continuum},
        "keep_raw_stokes": keep_raw_stokes,
        "stokes_height_megameters": stokes_height_megameters,
    }


def synthesize_stokes(
    snapshot: MURaMSnapshot, *, instrument: dict, observed_wavelength_angstrom, line_ids,
    response: dict | None = None, stride: int | None = None, binning: int | None = None,
    batch_size: int = 256, progress: bool = True, device: torch.device | str = "auto",
    dtype: torch.dtype = torch.float32, keep_raw_stokes: bool = False, diagnostics: bool = True,
    stokes_height_megameters: float | None = DEFAULT_STOKES_HEIGHT_MEGAMETERS,
    uniform_shell_grid: dict | None = None,
) -> SyntheticHMIResult:
    """Synthesize the lower native column through a registered LTE instrument.

    Stokes uses cell centers within stokes_height_megameters above the bottom center;
    None retains the full column. A uniform shell grid instead interpolates native
    fields at the registered shell heights and integrates by ray distance.
    Stride selects columns before transfer; binning averages final spectra.
    No optical-depth check, extra spatial PSF, or noise is applied. Retain pre-instrument spectra only when explicitly
    requested with keep_raw_stokes=True; wavelengths always remain available.
    """
    shell = UniformShellGrid(**uniform_shell_grid) if uniform_shell_grid is not None else None
    return _run_synthesis(
        snapshot, stokes=dict(instrument=instrument, observed_wavelength_angstrom=observed_wavelength_angstrom,
                             line_ids=line_ids, response=response, keep_raw_stokes=keep_raw_stokes,
                             uniform_shell_grid=shell, stokes_height_megameters=(
                                 shell.native_height_range_m[1] / 1e6 if shell else stokes_height_megameters)),
        stride=stride, binning=binning, batch_size=batch_size, progress=progress, device=device, dtype=dtype, diagnostics=diagnostics,
    )[0]


def synthesize_hmi(
    snapshot: MURaMSnapshot, *, stride: int | None = None, binning: int | None = None,
    dtype: torch.dtype = torch.float32, batch_size: int = 256, progress: bool = True,
    device: torch.device | str = "auto", keep_raw_stokes: bool = False, diagnostics: bool = True,
    stokes_height_megameters: float | None = DEFAULT_STOKES_HEIGHT_MEGAMETERS,
) -> SyntheticHMIResult:
    """Synthesize Stokes profiles with explicitly idealized Gaussian HMI filters."""
    return _run_synthesis(
        snapshot, stokes=_hmi_options(dtype, keep_raw_stokes, stokes_height_megameters), stride=stride, binning=binning,
        batch_size=batch_size, progress=progress, device=device, dtype=dtype, diagnostics=diagnostics,
    )[0]


def synthesize_aia(
    snapshot: MURaMSnapshot, *, stride: int | None = None, binning: int | None = None,
    channels_angstrom=DEFAULT_AIA_CHANNELS, response_resource: str = AIA_RESPONSE_RESOURCE,
    dtype: torch.dtype = torch.float32, batch_size: int = 256, progress: bool = True,
    device: torch.device | str = "auto", diagnostics: bool = True,
) -> SyntheticAIAResult:
    """Integrate optically thin AIA emissivity over native cell widths.

    The native tau<=1 visibility mask is a photospheric occultation proxy, not
    a model of EUV absorption or optically thick chromospheric emission.
    """
    return _run_synthesis(
        snapshot, aia=dict(channels_angstrom=channels_angstrom, response_resource=response_resource),
        stride=stride, binning=binning, batch_size=batch_size, progress=progress, device=device, dtype=dtype, diagnostics=diagnostics,
    )[0]


def synthesize_sdo(
    snapshot: MURaMSnapshot, *, stride: int | None = None, binning: int | None = None,
    dtype: torch.dtype = torch.float32, batch_size: int = 256, progress: bool = True,
    device: torch.device | str = "auto", keep_raw_stokes: bool = False, diagnostics: bool = True,
    stokes_height_megameters: float | None = DEFAULT_STOKES_HEIGHT_MEGAMETERS,
) -> tuple[SyntheticHMIResult, SyntheticAIAResult]:
    """Produce HMI and AIA sequentially from one shared pass over the columns."""
    return _run_synthesis(
        snapshot, stokes=_hmi_options(dtype, keep_raw_stokes, stokes_height_megameters), aia={},
        stride=stride, binning=binning, batch_size=batch_size, progress=progress, device=device, dtype=dtype, diagnostics=diagnostics,
    )


# Instrument-neutral name; retain the old result name for existing callers.
SyntheticStokesResult = SyntheticHMIResult
