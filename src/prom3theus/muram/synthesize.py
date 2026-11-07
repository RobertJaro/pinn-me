"""Forward-render synthetic HMI and AIA observables from a MURaM snapshot.

This is the validation entry point: known MURaM thermodynamics and magnetic
field are pushed through an independent numerical LTE reference solver
(:mod:`prom3theus.validation.reference_synthesis`) and the instrument response
code (:mod:`prom3theus.instruments`). Production inversions use parallel formal LTE integration. The source atmosphere is known, but inversion recovery remains
limited by its geometry, EOS, finite sampling, and observational degeneracies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral

import numpy as np
import torch
from prom3theus.instruments.registry import build_instrument

from prom3theus.rt import HybridSolarEOS
from prom3theus.validation.reference_synthesis import LTESynthesizer
from prom3theus.validation.reference_transfer import GeometricHeightPath
from prom3theus.instruments.aia_euv.operator import AIAEmissionOperator
from prom3theus.observations.contracts import ObservationRaster

from .atmosphere_builder import NATIVE_ATMOSPHERE_FIELDS, build_native_atmosphere
from .columns import column_batches, snapshot_columns, strided_columns
from .geometry import SOLAR_RADIUS_M, DiskCenterGeometry, build_disk_center_geometry
from .hmi_filter import build_idealized_hmi_filter
from .reader import MURaMSnapshot
from .provenance import stokes_forward_provenance


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
    native_component_order: tuple[int, int, int] = (0, 1, 2)
    spatial_stride: int = 1
    spatial_binning: int = 1
    forward_model_provenance: dict = field(default_factory=dict)

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
                "instrument": self.instrument_config,
                "line_ids": list(self.line_ids),
                "radiance_scale": MURAM_HMI_RADIANCE_SCALE_W_M3_SR,
                "line_support_angstrom": [
                    float(self.raw_wavelength_angstrom[0]),
                    float(self.raw_wavelength_angstrom[-1]),
                ],
                "geometry": "parallel vertical columns; spherical surface embedding",
                "transfer_path": "geometric_height_m",
                "velocity_frame": "local co-rotating flows plus Carrington rotation; stationary observer",
                "depth_sampling": "all native cells; no vertical resampling",
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
                    "instrument_response:" + name: value.expand(
                        ny, nx, *value.shape
                    ).clone()
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


@torch.no_grad()
def synthesize_stokes(
    snapshot: MURaMSnapshot,
    *,
    instrument: dict,
    observed_wavelength_angstrom,
    line_ids,
    response: dict | None = None,
    stride: int | None = None,
    binning: int | None = None,
    batch_size: int = 256,
    progress: bool = True,
    device: torch.device | str = "auto",
    dtype: torch.dtype = torch.float32,
    column_callback=None,
    keep_raw_stokes: bool = False,
) -> SyntheticHMIResult:
    """Synthesize any registered LTE instrument in bounded column batches.

    Response arrays are spatially uniform; HMI requires spectral_weights and
    continuum_weights. Hinode uses its configured spectral PSF. Stride selects
    native columns as [::stride, ::stride] before synthesis, without averaging.
    Binning averages synthesized profiles over non-overlapping spatial blocks.
    Horizontal dimensions after stride must be divisible by binning.
    No additional spatial PSF or noise is added. The default reads each selected
    column once and uses all native vertical cells with their physical heights.
    No log-tau interpolation, continuum optical-depth check, or top truncation
    is applied. Transfer uses geometric distances and the backend's opacity.
    Set keep_raw_stokes=True only when the pre-instrument spectra are needed;
    normal synthesis retains only the instrument-sampled output map.
    """
    device = resolve_synthesis_device(device)
    quantities = snapshot_columns(snapshot)
    dx_m, dy_m, _ = quantities.ds_m
    native_nx, native_ny, _ = quantities.shape
    quantities, stride = strided_columns(quantities, stride)
    nx, ny, nz = quantities.shape
    binning = _validate_binning(binning, (nx, ny))
    batch_size = _validate_batch_size(batch_size)
    if not isinstance(keep_raw_stokes, bool):
        raise ValueError("keep_raw_stokes must be a boolean.")
    line_ids = tuple(line_ids)
    # Absolute optical wavelengths are calibration metadata. Reduce their
    # differences in double precision once, then run transfer in the requested
    # dtype using cached Doppler offsets and response weights.
    operator = build_instrument(instrument).double()
    observed = torch.as_tensor(
        observed_wavelength_angstrom, dtype=torch.float64, device="cpu"
    )
    wavelength = operator.synthesis_grid(observed)
    if callable(getattr(operator, "prepare", None)):
        operator.prepare(wavelength, observed)
    response = {
        key: torch.as_tensor(value, dtype=dtype, device=device)
        for key, value in (response or {}).items()
    }
    synthesizer = LTESynthesizer(line_ids=line_ids).to(
        device=device, dtype=dtype
    )
    synthesizer.prepare_wavelength_grid(wavelength)
    synthesizer.to(device=device, dtype=dtype)
    operator.to(device=device, dtype=dtype)
    wavelength = wavelength.to(device=device)
    observed = observed.to(device=device)
    geometry = build_disk_center_geometry(
        native_nx,
        native_ny,
        dx_m,
        dy_m,
        stride=stride,
        reference_time_hours=snapshot.time_s / 3600,
        dtype=dtype,
    )
    # MURaM velocities are local co-rotating flows. The HMI training forward
    # model adds Carrington rotation before observer projection; reproduce
    # that same inertial wavelength gauge. For parallel rays, rotation's LOS
    # projection is constant along each ray, so the surface anchor is exact.
    from prom3theus.core import carrington_rotation_velocity_cartesian
    rotation = carrington_rotation_velocity_cartesian(geometry.surface_position_m, torch)
    rotation_los = (rotation * geometry.stokes_basis[..., 2, :]).sum(-1).T.reshape(-1)
    raw = (torch.empty((nx * ny, 4, wavelength.numel()), dtype=dtype)
           if keep_raw_stokes else None)
    integrated = torch.empty((nx * ny, 4, observed.numel()), dtype=dtype)
    for ix, iy, columns in column_batches(
        quantities,
        batch_size,
        fields=NATIVE_ATMOSPHERE_FIELDS if column_callback is None else None,
        progress=progress,
        description=f"Stokes synthesis ({device}, {dtype}, batch={batch_size})",
    ):
        if column_callback is not None:
            column_callback(ix, iy, columns)
        # Flattened pixel indices preserve image placement, including striding.
        indices = torch.from_numpy(ix * ny + iy)
        atmosphere = build_native_atmosphere(columns, dtype=dtype, device=device)
        velocity = atmosphere.velocity_field.clone()
        velocity[..., 2] += rotation_los[indices].to(velocity)[:, None]
        atmosphere = replace(
            atmosphere,
            velocity_field=velocity,
            magnetic_field=operator.polarization_convention.to_synthesis_frame(
                atmosphere.magnetic_field
            ),
        )
        profiles = synthesizer(
            atmosphere,
            None,
            path=GeometricHeightPath(atmosphere.geometric_height_m, mu=1.0),
            radiance_scale=MURAM_HMI_RADIANCE_SCALE_W_M3_SR,
        )
        if raw is not None:
            raw[indices] = profiles.cpu()
        integrated[indices] = operator(
            profiles,
            wavelength,
            observed,
            **{
                key: value.expand(len(ix), *value.shape)
                for key, value in response.items()
            },
        ).cpu()

    def image(values):
        return _bin_image(values.reshape(nx, ny, *values.shape[1:]).transpose(0, 1), binning)

    return SyntheticHMIResult(
        stokes=image(integrated),
        observed_wavelength_angstrom=observed.cpu(),
        raw_stokes=image(raw) if raw is not None else None,
        raw_wavelength_angstrom=wavelength.cpu(),
        geometry=(geometry if binning == 1 else build_disk_center_geometry(
            native_nx, native_ny, dx_m, dy_m, stride=stride, binning=binning,
            reference_time_hours=snapshot.time_s / 3600, dtype=dtype,
        )),
        dx_m=dx_m * stride * binning,
        dy_m=dy_m * stride * binning,
        instrument_config=dict(instrument),
        line_ids=line_ids,
        response={key: value.cpu() for key, value in response.items()},
        native_depth_samples=nz,
        native_vertical_spacing_m=quantities.ds_m[2],
        spatial_stride=stride,
        spatial_binning=binning,
        native_component_order=tuple(getattr(snapshot, "component_order", (0, 1, 2))),
        forward_model_provenance=stokes_forward_provenance(instrument["type"]),
    )


def synthesize_hmi(
    snapshot: MURaMSnapshot,
    *,
    stride: int | None = None,
    binning: int | None = None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 256,
    progress: bool = True,
    device: torch.device | str = "auto",
    column_callback=None,
    keep_raw_stokes: bool = False,
):
    """Convenience wrapper for explicitly idealized Gaussian HMI filters."""
    operator, observed, weights, continuum = build_idealized_hmi_filter(dtype=dtype)
    return synthesize_stokes(
        snapshot,
        column_callback=column_callback,
        keep_raw_stokes=keep_raw_stokes,
        stride=stride,
        binning=binning,
        batch_size=batch_size,
        progress=progress,
        device=device,
        dtype=dtype,
        instrument={
            "type": "hmi_filter_profiles",
            "quadrature_wavelength_angstrom": operator.quadrature_wavelength_angstrom.tolist(),
            "inner_half_width_angstrom": operator.inner_half_width_angstrom,
            "magnetic_azimuth_offset_deg": 90.0,
        },
        observed_wavelength_angstrom=observed,
        line_ids=("FeI_6173.3352",),
        response={"spectral_weights": weights, "continuum_weights": continuum},
    )


# Instrument-neutral name; retain the old result name for existing callers.
SyntheticStokesResult = SyntheticHMIResult


@torch.no_grad()
def synthesize_aia(
    snapshot: MURaMSnapshot,
    *,
    stride: int | None = None,
    binning: int | None = None,
    channels_angstrom=DEFAULT_AIA_CHANNELS,
    response_resource: str = AIA_RESPONSE_RESOURCE,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 256,
    progress: bool = True,
    device: torch.device | str = "auto",
) -> SyntheticAIAResult:
    """Approximate optically thin coronal count rates, in reference AIA pixels.

    Integrate cell-centered emissivity over full cell widths for selected
    native columns, then optionally average spatial blocks with binning.
    The tau500<=1 occultation mask is only a proxy: this
    does not model EUV absorption or optically thick chromospheric emission.
    """
    device = resolve_synthesis_device(device)
    quantities = snapshot_columns(snapshot)
    quantities, stride = strided_columns(quantities, stride)
    binning = _validate_binning(binning, quantities.shape)
    batch_size = _validate_batch_size(batch_size)
    accumulator = AIAAccumulator(
        quantities, stride=stride, binning=binning, channels_angstrom=channels_angstrom,
        response_resource=response_resource, device=device, dtype=dtype,
    )
    for ix, iy, columns in column_batches(
        quantities,
        batch_size,
        fields=("temperature", "pressure", "tau"),
        progress=progress,
        description=f"AIA synthesis ({device}, {dtype}, batch={batch_size})",
    ):
        accumulator(ix, iy, columns)
    return accumulator.result()


class AIAAccumulator:
    """Integrate AIA from loaded columns, retaining only the output image."""

    def __init__(self, quantities, *, stride=1, binning=None,
                 channels_angstrom=DEFAULT_AIA_CHANNELS,
                 response_resource=AIA_RESPONSE_RESOURCE, device="cpu", dtype=torch.float32):
        self.nx, self.ny, _ = quantities.shape
        dx, dy, self.dz = quantities.ds_m
        self.binning = _validate_binning(binning, quantities.shape)
        self.dx, self.dy = dx * stride * self.binning, dy * stride * self.binning
        self.device, self.dtype = device, dtype
        self.channels = tuple(channels_angstrom)
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
        temperature = torch.as_tensor(
            columns["temperature"][:, 0, :], dtype=self.dtype, device=self.device,
        )
        pressure = torch.as_tensor(
            columns["pressure"][:, 0, :], dtype=self.dtype, device=self.device,
        )
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
        )
