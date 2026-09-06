"""LTE radiative-transfer primitives and differentiable synthesis."""

from .atmosphere import (
    RadialReferenceAtmosphere,
    StratifiedAtmosphere,
    StratifiedAtmosphereModel,
)
from .atomic import AtomicDatabase, ElementData, SpectralLine
from .eos import HybridSolarEOS
from .geometry import (
    RayTraceResult,
    chart_height_to_position_m,
    chart_to_direction,
    direction_to_chart_mm,
    intersect_sphere_near_side,
    intersect_sphere_near_side_from_local_point,
    position_to_chart_height,
    project_vectors_to_stokes,
    validate_scene_basis,
)
from .opacity import (
    ContinuumOpacity,
    STICLookupState,
    STICSpectralState,
    damping_rate,
    doppler_velocity,
    doppler_width_frequency,
    integrated_line_opacity,
    planck_lambda,
)
from .plasma import SolarPlasmaState, SolarPlasmaTable, THOMSON_CROSS_SECTION_M2
from .polarization import PolarizedLineOpacity, PropagationDiagnostics
from .profiles import VoigtFaraday, voigt_faraday
from .radiometry import (
    HINODE_SOLAR_REFERENCE_RESOURCE,
    disk_center_continuum_radiance,
    disk_center_intensity_radiance,
    load_solar_reference,
    neckel_continuum_limb_darkening,
    reference_summary,
)
from .synthesis import LTESynthesizer, SynthesisDiagnostics
from .transfer import (
    GeometricHeightPath,
    OpticalDepthPath,
    PolarizedFormalSolver,
    RayDistancePath,
    TransferPath,
    scalar_formal_solution,
)
from .wavelength import air_to_vacuum_angstrom
from .zeeman import (
    ZeemanComponent,
    ZeemanPattern,
    wigner_3j,
    zeeman_component_strength,
    zeeman_components,
)

__all__ = [
    "AtomicDatabase",
    "ContinuumOpacity",
    "ElementData",
    "GeometricHeightPath",
    "HybridSolarEOS",
    "LTESynthesizer",
    "OpticalDepthPath",
    "PolarizedFormalSolver",
    "PolarizedLineOpacity",
    "PropagationDiagnostics",
    "RadialReferenceAtmosphere",
    "RayDistancePath",
    "RayTraceResult",
    "HINODE_SOLAR_REFERENCE_RESOURCE",
    "STICLookupState",
    "STICSpectralState",
    "SolarPlasmaState",
    "SolarPlasmaTable",
    "SpectralLine",
    "StratifiedAtmosphere",
    "StratifiedAtmosphereModel",
    "SynthesisDiagnostics",
    "TransferPath",
    "THOMSON_CROSS_SECTION_M2",
    "VoigtFaraday",
    "ZeemanComponent",
    "ZeemanPattern",
    "air_to_vacuum_angstrom",
    "chart_height_to_position_m",
    "chart_to_direction",
    "damping_rate",
    "direction_to_chart_mm",
    "disk_center_continuum_radiance",
    "disk_center_intensity_radiance",
    "doppler_velocity",
    "doppler_width_frequency",
    "integrated_line_opacity",
    "intersect_sphere_near_side",
    "intersect_sphere_near_side_from_local_point",
    "load_solar_reference",
    "neckel_continuum_limb_darkening",
    "planck_lambda",
    "position_to_chart_height",
    "project_vectors_to_stokes",
    "reference_summary",
    "scalar_formal_solution",
    "validate_scene_basis",
    "voigt_faraday",
    "wigner_3j",
    "zeeman_component_strength",
    "zeeman_components",
]
