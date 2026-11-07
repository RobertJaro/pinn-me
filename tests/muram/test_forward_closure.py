"""Close synthetic-data and independent numerical reference paths at known columns."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from prom3theus.core.solar_velocity import CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S
from prom3theus.instruments.hmi.raster import load_raster
from prom3theus.inversion.depth_sampling import DepthRefinement
from prom3theus.validation.reference_forward import (
    ForwardRuntime,
    LTEForwardComposition,
    LTESynthesisBackend,
)
from prom3theus.muram.atmosphere_builder import build_native_atmosphere
from prom3theus.muram.fits import write_hmi_fits
from prom3theus.muram.hmi_filter import build_idealized_hmi_filter
from prom3theus.muram.synthesize import synthesize_hmi
from prom3theus.validation.reference_synthesis import LTESynthesizer
from prom3theus.rt import RadialReferenceAtmosphere, RayTraceResult


class FrozenNativeColumns:
    """Preserve native columns while exercising production projection/transfer."""

    def __init__(self, atmosphere, geometry):
        self.solar_radius_m = torch.tensor(6.96e8, dtype=atmosphere.temperature.dtype)
        self.reference_atmosphere = RadialReferenceAtmosphere("falc_82").double()
        self.line_formation_height_bounds_megameter = (1.5, -1.0)
        basis = geometry.stokes_basis.transpose(0, 1).reshape(-1, 3, 3)
        self.atmosphere = replace(
            atmosphere,
            magnetic_field=torch.einsum(
                "bji,bdj->bdi", basis, atmosphere.magnetic_field
            ),
            velocity_field=torch.einsum(
                "bji,bdj->bdi", basis, atmosphere.velocity_field
            ),
        )
        height = atmosphere.geometric_height_m
        surface = geometry.surface_position_m.transpose(0, 1).reshape(-1, 3)
        position = surface[:, None, :] + height[..., None] * basis[:, None, 2, :]
        self.trace = RayTraceResult(
            position_m=position,
            distance_m=height[:, :1] - height,
            chart_xy_megameter=geometry.coordinates.transpose(0, 1)
            .reshape(-1, 3)[:, None, :2]
            .expand(-1, height.shape[-1], -1),
            geometric_height_m=height,
        )

    def trace_rays(self, coordinates, ray_direction, depth_grid):
        return self.atmosphere, self.trace


def test_synthetic_fits_targets_match_inversion_at_native_atmosphere(tmp_path):
    """Catch wavelength reversal, radiance factors, basis and LOS gauge drift."""
    shape = (3, 2, 41)
    depth = np.geomspace(100, 1e-7, shape[-1])
    q = {name: np.zeros(shape) for name in ("bx", "by", "bz", "vx", "vy", "vz")}
    q.update(
        temperature=np.broadcast_to(np.linspace(7200, 4200, shape[-1]), shape).copy(),
        pressure=np.broadcast_to(np.geomspace(3e6, 1, shape[-1]), shape).copy(),
        tau=np.broadcast_to(depth, shape).copy(),
        ds_m=(2e7, 1e7, 2e4),
    )
    q["bx"][:] = np.array([300, -1500, 2000])[:, None, None]
    q["by"][:] = np.array([1100, -600])[None, :, None]
    q["bz"][:] = np.array([-2300, 1700])[None, :, None]
    q["vz"][:] = np.array([-2100, 900, 3400])[:, None, None]
    q["temperature"] += np.array([0, 650])[None, :, None]
    snapshot = SimpleNamespace(time_s=7200, synthesis_ready_quantities=lambda: q)
    result = synthesize_hmi(
        snapshot, dtype=torch.float64, progress=False
    )
    paths = write_hmi_fits(tmp_path, result, time_s=7200, iteration=1)
    raster = load_raster(paths, tmp_path)
    atmosphere = build_native_atmosphere(q)
    # The inversion adapter requires bounded reference labels; these do not
    # alter the native heights or cell values used by geometric transfer.
    grid = torch.linspace(-5, 1, atmosphere.temperature.shape[-1], dtype=torch.float64)
    atmosphere = replace(atmosphere, depth_coordinate=grid)
    sampler = FrozenNativeColumns(atmosphere, result.geometry)
    operator, observed, weights, continuum = build_idealized_hmi_filter()
    synthesizer = LTESynthesizer(log_tau500=grid, line_ids=("FeI_6173.3352",)).double()
    forward = LTEForwardComposition(
        atmosphere_model=sampler,
        backend=LTESynthesisBackend(synthesizer),
        instrument=operator,
        velocity_synthesis_mode="carrington_observer_relative",
        depth_refinement=DepthRefinement(False, 1, 0.0),
    )
    wavelength = forward.prepare_wavelength_grid(observed)
    scale = raster.metadata["normalization"]["radiometric_calibration"][
        "atlas_disk_center_continuum_radiance_w_m3_sr"
    ]
    runtime = ForwardRuntime(
        torch.as_tensor(grid),
        observed,
        wavelength,
        scale,
        CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
        0.0,
    )
    batch = shape[0] * shape[1]

    def flat(value):
        return value.transpose(0, 1).reshape(batch, *value.shape[2:]).double()

    prediction = forward.synthesize(
        flat(raster.coordinates),
        runtime=runtime,
        ray_direction=flat(raster.ray_direction),
        stokes_basis=flat(raster.stokes_basis),
        observer_los_velocity_m_per_s=torch.zeros(batch, dtype=torch.float64),
        instrument_response={
            "spectral_weights": weights.expand(batch, -1, -1),
            "continuum_weights": continuum.expand(batch, -1),
        },
    )["stokes"]
    # Targets are serialized to float32 SI radiance by FITS, then atlas-scaled.
    torch.testing.assert_close(prediction, flat(raster.stokes), rtol=2e-7, atol=1e-8)
    assert prediction[:, 1:3].abs().max() > 0.01
    assert prediction[:, 3].abs().max() > 0.01
