"""Integrity and scientific-contract checks for packaged LTE resources."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from importlib.resources import files
import json
import math
from pathlib import PurePosixPath


BUNDLE_TYPE = "prom3theus-lte-runtime-resources"
SUPPORTED_INSTRUMENTS = ("hinode_sp", "hmi_stokes")
TRAINING_RESOURCE_CONTRACT = "offline-read-only; no downloads or table conversion"
REQUIRED_COMMON_PRODUCTION_FILES = frozenset(
    {
        "common/abundances.json",
        "common/chianti_thermodynamic_table.json",
        "common/falc_reference_atmosphere.json",
        "common/lines.json",
        "common/stic_continuum_table.json",
    }
)
HINODE_PRODUCTION_FILES = frozenset(
    {
        "hinode_sp/blend_inventory.json",
        "hinode_sp/instrument_hinode_sp.json",
        "hinode_sp/solar_reference_630nm.json",
    }
)
HMI_PRODUCTION_FILES = frozenset(
    {
        "hmi_stokes/instrument_hmi.json",
        "hmi_stokes/solar_reference_617nm.json",
    }
)
REQUIRED_HINODE_PRODUCTION_FILES = (
    REQUIRED_COMMON_PRODUCTION_FILES | HINODE_PRODUCTION_FILES
)
REQUIRED_HMI_PRODUCTION_FILES = REQUIRED_COMMON_PRODUCTION_FILES | HMI_PRODUCTION_FILES
HMI_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM = (6160.0, 6170.0, 6180.0, 6190.0)
HINODE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM = (6290.0, 6300.0, 6310.0, 6320.0)
REFERENCE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM = (5000.0,)
HMI_STIC_REQUIRED_DOMAIN_ANGSTROM = (6160.0, 6190.0)
HINODE_STIC_REQUIRED_DOMAIN_ANGSTROM = (6290.0, 6320.0)


def resource_path(resource_name: str | None = None):
    """Return the bundle root or one safe bundle-relative resource path."""

    root = files("prom3theus.resources").joinpath("data")
    if resource_name is None:
        return root
    if not isinstance(resource_name, str) or not resource_name or "\\" in resource_name:
        raise ValueError(
            "A resource name must be a safe non-empty relative POSIX path."
        )
    relative = PurePosixPath(resource_name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(
            "A resource name must be a safe non-empty relative POSIX path."
        )
    return root.joinpath(*relative.parts)


def _open_binary(resource):
    return resource.open("rb") if hasattr(resource, "open") else open(resource, "rb")


def _open_text(resource):
    return (
        resource.open("r", encoding="utf-8")
        if hasattr(resource, "open")
        else open(resource, encoding="utf-8")
    )


def file_sha256(resource) -> str:
    """Calculate a resource SHA256 digest without loading it all into memory."""

    digest = hashlib.sha256()
    with _open_binary(resource) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_resource(
    resource, manifest: Mapping, *, resource_name: str, kind: str
) -> str:
    """Verify one scientific resource against a pinned source manifest."""

    resource_path(resource_name)
    matches = [
        manifest.get(section, {}).get(resource_name)
        for section in ("vendored_files", "generated_files", "reviewed_files")
        if isinstance(manifest.get(section), Mapping)
        and manifest[section].get(resource_name) is not None
    ]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise RuntimeError(
            f"{kind} resource {resource_name!r} is not recorded exactly once in the "
            "LTE source manifest."
        )
    expected = matches[0].get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise RuntimeError(
            f"{kind} resource {resource_name!r} has no valid SHA256 digest."
        )
    actual = file_sha256(resource)
    if actual != expected:
        raise RuntimeError(
            f"{kind} resource {resource_name!r} failed SHA256 verification: "
            f"expected {expected}, got {actual}."
        )
    return actual


def _load_json(resource) -> dict:
    with _open_text(resource) as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise RuntimeError(f"Expected a JSON object in {resource}.")
    return document


def _packaged_file_names(root, prefix: PurePosixPath = PurePosixPath()) -> set[str]:
    names: set[str] = set()
    for child in root.iterdir():
        relative = prefix / child.name
        if child.is_file():
            names.add(relative.as_posix())
        elif child.is_dir():
            names.update(_packaged_file_names(child, relative))
    return names


def load_verified_source_manifest() -> dict:
    """Load the source manifest after checking its bundle-pinned digest."""

    root = resource_path()
    bundle_resource = root.joinpath("bundle.json")
    manifest_resource = root.joinpath("sources.json")
    if not bundle_resource.is_file() or not manifest_resource.is_file():
        raise FileNotFoundError(f"LTE resource bundle is incomplete in {root}.")
    bundle = _load_json(bundle_resource)
    if bundle.get("schema_version") != 2 or bundle.get("bundle_type") != BUNDLE_TYPE:
        raise RuntimeError(f"Unsupported LTE resource bundle in {bundle_resource}.")
    expected = bundle.get("source_manifest_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise RuntimeError("bundle.json has no valid source-manifest SHA256 digest.")
    if file_sha256(manifest_resource) != expected:
        raise RuntimeError("The LTE source manifest does not match bundle.json.")
    manifest = _load_json(manifest_resource)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported source manifest in {manifest_resource}.")
    return manifest


def _validate_stic_coverage(table: Mapping) -> dict:
    axes = table.get("axes", {})
    try:
        wavelengths = [float(value) for value in axes["wavelength_vacuum_angstrom"]]
        domains = [
            tuple(float(value) for value in interval)
            for interval in table["validated_wavelength_domains_angstrom"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid STiC wavelength-coverage metadata.") from error
    if (
        len(wavelengths) < 2
        or not all(math.isfinite(value) for value in wavelengths)
        or any(right <= left for left, right in zip(wavelengths, wavelengths[1:]))
        or not domains
        or any(
            len(interval) != 2
            or not all(math.isfinite(value) for value in interval)
            or interval[1] < interval[0]
            for interval in domains
        )
    ):
        raise RuntimeError("Invalid STiC wavelength coverage.")
    required_nodes = (
        *REFERENCE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM,
        *HMI_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM,
        *HINODE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM,
    )
    missing_nodes = [
        node
        for node in required_nodes
        if not any(
            math.isclose(value, node, rel_tol=0.0, abs_tol=1.0e-8)
            for value in wavelengths
        )
    ]
    required_domains = {
        "tau500_reference": (5000.0, 5000.0),
        "hmi_stokes": HMI_STIC_REQUIRED_DOMAIN_ANGSTROM,
        "hinode_sp": HINODE_STIC_REQUIRED_DOMAIN_ANGSTROM,
    }
    missing_domains = [
        instrument
        for instrument, (lower, upper) in required_domains.items()
        if not any(a <= lower and b >= upper for a, b in domains)
    ]
    if missing_nodes or missing_domains:
        raise RuntimeError(
            "The packaged STiC table does not cover the exact Hinode SP and HMI "
            f"production contract (missing nodes={missing_nodes}, "
            f"missing domains={missing_domains})."
        )
    return {
        "wavelength_vacuum_angstrom": wavelengths,
        "validated_wavelength_domains_angstrom": [
            list(interval) for interval in domains
        ],
    }


def _validate_falc_reference_atmosphere(
    table: Mapping,
    manifest: Mapping,
    top_boundary: Mapping,
) -> dict:
    """Validate the complete native FALC profile and legacy LTE resampling."""

    expected_fields = {
        "schema_version",
        "model",
        "native_depth_count",
        "source",
        "coordinates",
        "temperature_k",
        "gas_pressure_pa",
        "microturbulence_m_per_s",
        "line_formation_reference",
        "construction",
    }
    if table.get("schema_version") != 1 or set(table) != expected_fields:
        raise RuntimeError("LTE bundle has no valid native FALC reference.")
    try:
        native_count = int(table["native_depth_count"])
        coordinates = table["coordinates"]
        log_tau500 = [float(value) for value in coordinates["log10_tau500"]]
        height_m = [float(value) for value in coordinates["height_m"]]
        log_column_mass = [
            float(value) for value in coordinates["log10_column_mass_g_cm2"]
        ]
        temperature_k = [float(value) for value in table["temperature_k"]]
        pressure_pa = [float(value) for value in table["gas_pressure_pa"]]
        microturbulence = [float(value) for value in table["microturbulence_m_per_s"]]
        source = table["source"]
        construction = table["construction"]
        line = table["line_formation_reference"]
        line_log_tau500 = [float(value) for value in line["log10_tau500"]]
        line_height_m = [float(value) for value in line["height_m"]]
        line_temperature_k = [float(value) for value in line["temperature_k"]]
        line_pressure_pa = [float(value) for value in line["gas_pressure_pa"]]
        line_microturbulence = [
            float(value) for value in line["microturbulence_m_per_s"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid native FALC reference metadata.") from error

    native_arrays = (
        log_tau500,
        height_m,
        log_column_mass,
        temperature_k,
        pressure_pa,
        microturbulence,
    )
    line_arrays = (
        line_log_tau500,
        line_height_m,
        line_temperature_k,
        line_pressure_pa,
        line_microturbulence,
    )
    if (
        table.get("model") != "FALC_82"
        or native_count != 82
        or set(coordinates) != {"log10_tau500", "height_m", "log10_column_mass_g_cm2"}
        or any(len(values) != native_count for values in native_arrays)
        or any(len(values) != 25 for values in line_arrays)
        or not all(
            math.isfinite(value)
            for values in (*native_arrays, *line_arrays)
            for value in values
        )
        or any(right <= left for left, right in zip(log_tau500, log_tau500[1:]))
        or any(right >= left for left, right in zip(height_m, height_m[1:]))
        or any(
            right <= left for left, right in zip(log_column_mass, log_column_mass[1:])
        )
        or any(right <= left for left, right in zip(pressure_pa, pressure_pa[1:]))
        or min(temperature_k) <= 0.0
        or min(pressure_pa) <= 0.0
        or min(microturbulence) <= 0.0
        or any(
            not math.isclose(value, -5.0 + 0.25 * index, abs_tol=1.0e-12)
            for index, value in enumerate(line_log_tau500)
        )
    ):
        raise RuntimeError("Invalid native FALC reference arrays.")

    stic_source_ids = (
        "stic_witt_eos",
        "stic_kurucz_partition_functions",
        "stic_abundances",
        "stic_falc_82",
    )
    source_records = manifest.get("sources", {})
    if (
        not isinstance(source, Mapping)
        or not isinstance(source_records, Mapping)
        or not isinstance(source.get("source_sha256"), Mapping)
        or source.get("project") != "STiC"
        or source.get("repository") != "https://github.com/jaimedelacruz/stic"
        or source.get("commit") != "18cda77d038a97f007a783dcb61ea9a9a1244bf7"
        or source.get("file") != "input/Atmos/FALC_82.atmos"
        or set(source["source_sha256"]) != set(stic_source_ids)
        or any(
            source["source_sha256"].get(source_id)
            != source_records.get(source_id, {}).get("sha256")
            for source_id in stic_source_ids
        )
        or construction
        != {
            "pressure": "Pgas = g * column_mass in cgs, converted to pascal",
            "density": (
                "pinned STiC witt.py EOS evaluated at each native (T, Pgas) row"
            ),
            "height": (
                "dm = -rho dz with trapezoidal reciprocal density; additive "
                "origin fixed by linear interpolation to z(log10(tau500)=0)=0"
            ),
            "tau500": (
                "STiC ceos::hydrostatic_cmass convention: tau[0]=m[0]*kappa[0]; "
                "subsequent increments use trapezoidal total kappa500 over column mass"
            ),
            "tau500_extinction": (
                "true absorption plus scattering at vacuum 5000 Angstrom"
            ),
            "gravity_cm_s2": 27542.28703338169,
            "native_order": "top to bottom",
        }
        or set(line)
        != {
            "log10_tau500",
            "height_m",
            "temperature_k",
            "gas_pressure_pa",
            "microturbulence_m_per_s",
            "resampling",
        }
        or line.get("resampling")
        != {
            "interval": "log10(tau500)=-5..1 at 0.25 dex spacing",
            "linear_fields": [
                "height_m",
                "temperature_k",
                "microturbulence_m_per_s",
            ],
            "log_linear_fields": ["gas_pressure_pa"],
            "purpose": (
                "preserve the established LTE ray/depth mapping while the previously "
                "unused native FALC upper nodes remain available"
            ),
        }
    ):
        raise RuntimeError("Invalid native FALC provenance or construction contract.")

    def interpolate(query: float, axis: list[float], values: list[float]) -> float:
        upper = next(index for index, value in enumerate(axis) if value >= query)
        if axis[upper] == query or upper == 0:
            return values[upper]
        lower = upper - 1
        fraction = (query - axis[lower]) / (axis[upper] - axis[lower])
        return values[lower] + fraction * (values[upper] - values[lower])

    expected_pressure = [
        0.1 * construction["gravity_cm_s2"] * 10.0**value for value in log_column_mass
    ]
    resampled_native = (
        [interpolate(q, log_tau500, height_m) for q in line_log_tau500],
        [interpolate(q, log_tau500, temperature_k) for q in line_log_tau500],
        [
            10.0
            ** interpolate(q, log_tau500, [math.log10(value) for value in pressure_pa])
            for q in line_log_tau500
        ],
        [interpolate(q, log_tau500, microturbulence) for q in line_log_tau500],
    )
    if (
        any(
            not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
            for actual, expected in zip(pressure_pa, expected_pressure, strict=True)
        )
        or any(
            not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-9)
            for actual_values, expected_values in zip(
                (
                    line_height_m,
                    line_temperature_k,
                    line_pressure_pa,
                    line_microturbulence,
                ),
                resampled_native,
                strict=True,
            )
            for actual, expected in zip(actual_values, expected_values, strict=True)
        )
        or not math.isclose(
            interpolate(0.0, log_tau500, height_m),
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or not math.isclose(log_tau500[0], -5.405870771873386, abs_tol=1.0e-12)
        or not math.isclose(height_m[0], 2073502.4593743724, rel_tol=1.0e-12)
        or not math.isclose(temperature_k[0], 1.0e5, rel_tol=0.0)
        or not math.isclose(pressure_pa[0], 0.031934421263776124, rel_tol=1.0e-12)
        or not math.isclose(microturbulence[0], 10680.96, rel_tol=0.0, abs_tol=1.0e-9)
        or not math.isclose(
            line_pressure_pa[0],
            float(top_boundary.get("gas_pressure_pa", math.nan)),
            rel_tol=1.0e-12,
        )
        or not math.isclose(
            line_temperature_k[0],
            float(top_boundary.get("temperature_k", math.nan)),
            rel_tol=1.0e-12,
        )
        or [log_tau500[0], log_tau500[-1]]
        != list(top_boundary.get("falc_log10_tau500_range", ()))
    ):
        raise RuntimeError("Native FALC reference violates its physical contract.")

    return {
        "schema_version": 1,
        "model": "FALC_82",
        "native_depth_count": native_count,
        "log10_tau500_range": [log_tau500[0], log_tau500[-1]],
        "height_range_m": [height_m[-1], height_m[0]],
        "native_top": {
            "height_m": height_m[0],
            "temperature_k": temperature_k[0],
            "gas_pressure_pa": pressure_pa[0],
            "microturbulence_m_per_s": microturbulence[0],
        },
        "line_formation_log10_tau500_range": [
            line_log_tau500[0],
            line_log_tau500[-1],
        ],
    }


def _pchip_slopes(axis: list[float], values: list[float]) -> list[float]:
    """Reconstruct the shape-preserving slopes stored by the EoS builder."""

    spacing = [right - left for left, right in zip(axis, axis[1:])]
    secant = [
        (right - left) / width
        for left, right, width in zip(values[:-1], values[1:], spacing, strict=True)
    ]
    slopes = [0.0] * len(values)
    for index in range(1, len(values) - 1):
        previous = secant[index - 1]
        following = secant[index]
        if previous * following <= 0.0:
            continue
        weight_previous = 2.0 * spacing[index] + spacing[index - 1]
        weight_following = spacing[index] + 2.0 * spacing[index - 1]
        slopes[index] = (weight_previous + weight_following) / (
            weight_previous / previous + weight_following / following
        )
    first = ((2.0 * spacing[0] + spacing[1]) * secant[0] - spacing[0] * secant[1]) / (
        spacing[0] + spacing[1]
    )
    if first * secant[0] <= 0.0:
        first = 0.0
    elif secant[0] * secant[1] < 0.0 and abs(first) > 3.0 * abs(secant[0]):
        first = 3.0 * secant[0]
    last = (
        (2.0 * spacing[-1] + spacing[-2]) * secant[-1] - spacing[-1] * secant[-2]
    ) / (spacing[-1] + spacing[-2])
    if last * secant[-1] <= 0.0:
        last = 0.0
    elif secant[-1] * secant[-2] < 0.0 and abs(last) > 3.0 * abs(secant[-1]):
        last = 3.0 * secant[-1]
    slopes[0], slopes[-1] = first, last
    return slopes


def _validate_chianti_eos_table(table: Mapping, manifest: Mapping) -> dict:
    """Validate the reduced CHIANTI thermodynamic closure and its provenance."""

    if table.get("schema_version") != 1 or set(table) != {
        "schema_version",
        "axes",
        "log10_free_electrons_per_h_nucleus",
        "pchip_d_log10_free_electrons_per_h_d_log10_temperature",
        "interpolation",
        "fully_ionized_limit",
        "composition",
        "reference_ionization_equilibrium",
        "thermodynamic_mapping",
        "runtime_contract",
    }:
        raise RuntimeError("LTE bundle has no valid CHIANTI thermodynamic table.")
    try:
        log_temperature = [
            float(value) for value in table["axes"]["log10_temperature_k"]
        ]
        log_electrons = [
            float(value) for value in table["log10_free_electrons_per_h_nucleus"]
        ]
        electron_slopes = [
            float(value)
            for value in table["pchip_d_log10_free_electrons_per_h_d_log10_temperature"]
        ]
        interpolation = table["interpolation"]
        composition = table["composition"]
        mass_per_h = float(composition["mass_u_per_h_nucleus"])
        nuclei_per_h = float(composition["nuclei_per_h_nucleus"])
        ideal_electrons_per_h = float(
            composition["fully_ionized_electrons_per_h_nucleus"]
        )
        high_z_nuclei_fraction = float(composition["z_greater_than_30_nuclei_fraction"])
        high_z_mass_fraction = float(composition["z_greater_than_30_mass_fraction"])
        high_z_electron_fraction = float(
            composition["z_greater_than_30_fully_ionized_electron_fraction"]
        )
        ideal = table["fully_ionized_limit"]
        ideal_mu = float(ideal["mean_molecular_weight"])
        ideal_mu_e = float(ideal["mean_molecular_weight_per_electron"])
        source = table["reference_ionization_equilibrium"]
        contract = table["runtime_contract"]
        minimum_temperature = float(contract["minimum_temperature_k"])
        stic_max = float(contract["stic_exact_max_temperature_k"])
        lower_join = [
            float(value)
            for value in contract["stic_to_chianti_log_blend_temperature_k"]
        ]
        upper_join = [
            float(value)
            for value in contract["chianti_to_fully_ionized_log_blend_temperature_k"]
        ]
        recorded_ideal_error = float(
            contract["maximum_relative_error_at_ideal_blend_start"]
        )
        lower_transition = contract["lower_transition"]
        upper_transition = contract["upper_transition"]
        thermodynamic_mapping = table["thermodynamic_mapping"]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid CHIANTI thermodynamic metadata.") from error
    if (
        set(table["axes"]) != {"log10_temperature_k"}
        or len(log_temperature) != 101
        or len(log_electrons) != len(log_temperature)
        or len(electron_slopes) != len(log_temperature)
        or not all(
            math.isfinite(value)
            for value in (*log_temperature, *log_electrons, *electron_slopes)
        )
        or any(value < 0.0 for value in electron_slopes)
        or any(b <= a for a, b in zip(log_temperature, log_temperature[1:]))
        or any(b <= a for a, b in zip(log_electrons, log_electrons[1:]))
        or any(
            not math.isclose(value, 4.0 + 0.05 * index, rel_tol=0.0, abs_tol=1.0e-12)
            for index, value in enumerate(log_temperature)
        )
    ):
        raise RuntimeError("Invalid CHIANTI thermodynamic lookup arrays.")
    reconstructed_slopes = _pchip_slopes(log_temperature, log_electrons)
    if any(
        not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-14)
        for actual, expected in zip(electron_slopes, reconstructed_slopes, strict=True)
    ) or interpolation != {
        "method": "shape-preserving piecewise cubic Hermite (PCHIP)",
        "segment_coordinate": "t = (x - x_i) / (x_(i+1) - x_i)",
        "segment_polynomial": (
            "y = (2t^3-3t^2+1)y_i + (t^3-2t^2+t)h_i m_i + "
            "(-2t^3+3t^2)y_(i+1) + (t^3-t^2)h_i m_(i+1)"
        ),
        "node_slopes": (
            "stored explicitly; generated with weighted harmonic interior secants "
            "and one-sided shape-preserving endpoint limiters"
        ),
        "outside_native_grid": "clamp to the nearest endpoint value",
    }:
        raise RuntimeError("Invalid CHIANTI PCHIP interpolation contract.")
    if (
        min(mass_per_h, nuclei_per_h, ideal_electrons_per_h) <= 0.0
        or any(
            not math.isclose(value, expected, rel_tol=1.0e-12, abs_tol=0.0)
            for value, expected in zip(
                (mass_per_h, nuclei_per_h, ideal_electrons_per_h),
                (1.4178596896340696, 1.09878109693557, 1.2046951743544212),
                strict=True,
            )
        )
        or not all(
            math.isfinite(value)
            for value in (
                mass_per_h,
                nuclei_per_h,
                ideal_electrons_per_h,
                ideal_mu,
                ideal_mu_e,
            )
        )
        or not math.isclose(
            ideal_mu,
            mass_per_h / (nuclei_per_h + ideal_electrons_per_h),
            rel_tol=1.0e-12,
        )
        or not math.isclose(
            ideal_mu_e,
            mass_per_h / ideal_electrons_per_h,
            rel_tol=1.0e-12,
        )
        or composition.get("chianti_atomic_numbers") != [1, 30]
        or composition.get("stic_mixture_atomic_numbers") != [1, 99]
        or composition.get("abundance_source")
        != "STiC input/Atoms/abundance.input at the pinned STiC commit"
        or composition.get("atomic_mass_source")
        != "STiC witt.py AMASS at the pinned STiC commit"
        or composition.get("higher_atomic_number_treatment")
        != (
            "Z > 30 nuclei and mass remain in the STiC mixture; their negligible "
            "free-electron contribution is ignored in the CHIANTI regime and "
            "included only in the fully ionized analytic limit"
        )
        or any(
            not math.isclose(value, expected, rel_tol=1.0e-12, abs_tol=0.0)
            for value, expected in zip(
                (
                    high_z_nuclei_fraction,
                    high_z_mass_fraction,
                    high_z_electron_fraction,
                ),
                (
                    1.009858530132135e-08,
                    6.757485072617087e-07,
                    3.425923402744502e-07,
                ),
                strict=True,
            )
        )
        or composition.get("source_sha256", {}).get("stic_abundances")
        != manifest.get("sources", {}).get("stic_abundances", {}).get("sha256")
        or composition.get("source_sha256", {}).get("stic_witt_eos")
        != manifest.get("sources", {}).get("stic_witt_eos", {}).get("sha256")
    ):
        raise RuntimeError("Invalid CHIANTI/STiC composition closure.")
    source_records = manifest.get("sources", {})
    if (
        source.get("database_release") != "11.0.2"
        or source.get("file") != "ioneq/chianti.ioneq"
        or source.get("file_embedded_revision_note") != "prepared for CHIANTI 10.1"
        or source.get("source_sha256")
        != source_records.get("chianti_ioneq_v11_0_2", {}).get("sha256")
        or source.get("version_source_sha256")
        != source_records.get("chianti_database_version_11_0_2", {}).get("sha256")
        or source.get("model")
        != (
            "standard zero-density coronal ionization equilibrium; Maxwellian "
            "electron distribution"
        )
        or source.get("fraction_normalization")
        != (
            "each element and temperature renormalized to unit sum after parsing "
            "the source's e10.3 fractions"
        )
        or source.get("references")
        != [
            "https://doi.org/10.5281/zenodo.14263073",
            "https://doi.org/10.3847/1538-4365/acec79",
            "https://doi.org/10.3847/1538-4357/ad6765",
        ]
        or source.get("advanced_model_exclusion")
        != (
            "CHIANTI 11 density-dependent and charge-transfer advanced models "
            "require additional atmosphere-specific inputs and are not represented"
        )
    ):
        raise RuntimeError("Invalid CHIANTI thermodynamic provenance.")
    if thermodynamic_mapping != {
        "mean_charge": "qbar_Z(T) = sum_q q f_Zq(T)",
        "free_electrons_per_h_nucleus": "e(T) = sum_{Z<=30} A_Z qbar_Z(T)",
        "chianti_particles_per_h_nucleus": "g(T) = sum_Z A_Z + e(T)",
        "hybrid_particles_per_h_nucleus": (
            "g(T,P) = h(T,P) + e(T,P), where h transitions from the STiC "
            "non-electron particle count to sum_Z A_Z"
        ),
        "mean_molecular_weight": "mu(T,P) = M / g(T,P)",
        "mean_molecular_weight_per_electron": "mu_e(T,P) = M / e(T,P)",
        "hydrogen_nucleus_density": "n_H(T,P) = P / [k_B T g(T,P)]",
        "mass_density": "rho(T,P) = M m_u n_H(T,P)",
        "electron_density": "n_e(T,P) = e(T,P) n_H(T,P)",
    }:
        raise RuntimeError("Invalid CHIANTI thermodynamic mapping contract.")
    if (
        len(lower_join) != 2
        or len(upper_join) != 2
        or not math.isclose(minimum_temperature, 10.0**3.4, rel_tol=1.0e-12)
        or not math.isclose(stic_max, lower_join[0], rel_tol=1.0e-12)
        or any(
            not math.isclose(actual, expected, rel_tol=1.0e-12)
            for actual, expected in zip(lower_join, (1.0e4, 10.0**4.5), strict=True)
        )
        or any(
            not math.isclose(actual, expected, rel_tol=1.0e-12)
            for actual, expected in zip(upper_join, (1.0e7, 2.0e7), strict=True)
        )
        or lower_transition
        != {
            "electron_coordinate": (
                "logit(free_electrons_per_H / fully_ionized_electrons_per_H)"
            ),
            "non_electron_coordinate": "log10(non_electron_particles_per_H)",
            "interior_weight": "quintic smootherstep in log10(T)",
            "endpoint_tangent_match_width_log10_temperature": 0.05,
            "endpoint_tangent_ramp_power": 5,
            "lower_endpoint_ramp": "1 - (1 - u)^p",
            "upper_endpoint_ramp": "u^p",
            "integrated_tangent_change": "slope * width / p",
            "closure": (
                "particles_per_H = non_electron_particles_per_H + electrons_per_H"
            ),
        }
        or upper_transition
        != {
            "electron_coordinate": "log10(free_electrons_per_H)",
            "interior_weight": "quintic smootherstep in log10(T)",
            "closure": "particles_per_H = nuclei_per_H + electrons_per_H",
        }
        or contract.get("stic_interpolation")
        != {
            "coordinates": "tensor product in log10(T/K) and log10(Pgas/Pa)",
            "uniform_axis_coordinate": ("(x - x_0) * (N - 1) / (x_(N-1) - x_0)"),
            "cells": "Catmull-Rom cubic with endpoint-index clamping",
            "boundary_derivative": (
                "one half of the outer one-sided secant, matching the established "
                "STiC runtime interpolant"
            ),
        }
        or contract.get("pressure_continuation")
        != {
            "domain": "outside the packaged STiC log10(Pgas/Pa) axis",
            "coordinates": (
                "logit(electrons_per_H / fully_ionized_electrons_per_H) and "
                "log10(non_electron_particles_per_H)"
            ),
            "edge_derivative": (
                "derived from the clamped-Catmull boundary derivatives of "
                "log10(rho) and log10(n_e)"
            ),
            "tangent_decay_width_log10_pressure": 0.05,
            "tangent_decay_power": 5,
            "derivative_ramp": "(1 - u)^p",
            "integrated_coordinate_change": (
                "signed slope * width * [1 - (1 - u)^(p + 1)] / (p + 1)"
            ),
            "asymptotic_closure": (
                "constant particles_per_H and electrons_per_H; rho and n_e are "
                "linear in P"
            ),
        }
        or contract.get("smoothness")
        != (
            "C1 in log10(T) at all regime joins and in log10(P) at both STiC "
            "pressure edges; second derivatives are not constrained"
        )
        or contract.get("pressure_dependence")
        != (
            "STiC retains its tabulated (T,P) dependence on the native pressure "
            "axis; bounded physical-coordinate shoulders continue it outside; "
            "CHIANTI and the fully ionized limit depend on T only, with rho and "
            "n_e linear in P"
        )
        or contract.get("lower_transition_rationale")
        != (
            "the bounded electron logit prevents charge states above the fully "
            "ionized composition; the shared non-electron count preserves exact "
            "STiC at 10 kK and reaches the CHIANTI ideal-mixture closure at "
            "log10(T/K)=4.5; endpoint tangent ramps span one native CHIANTI node"
        )
        or contract.get("ideal_transition_criterion")
        != (
            "CHIANTI remains exact through the native 10 MK node; a C1 "
            "log-temperature blend then reaches the fully ionized limit at "
            "exactly 20 MK, and starts with both CHIANTI-derived mu and mu_e "
            "within 1e-3 relative error of that limit"
        )
        or not (0.0 <= recorded_ideal_error < 1.0e-3)
    ):
        raise RuntimeError("Invalid hybrid thermodynamic transition contract.")
    ideal_errors = []
    for log_t, log_e in zip(log_temperature, log_electrons):
        electrons_per_h = 10.0**log_e
        if not electrons_per_h < ideal_electrons_per_h:
            raise RuntimeError("CHIANTI charge mapping exceeds its composition limit.")
        mu = mass_per_h / (nuclei_per_h + electrons_per_h)
        mu_e = mass_per_h / electrons_per_h
        ideal_errors.append(max(abs(mu / ideal_mu - 1.0), abs(mu_e / ideal_mu_e - 1.0)))
    persistent_errors = [
        max(ideal_errors[index:]) for index in range(len(ideal_errors))
    ]
    transition_start = [
        index
        for index, value in enumerate(log_temperature)
        if math.isclose(
            value,
            math.log10(upper_join[0]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    if (
        len(transition_start) != 1
        or persistent_errors[transition_start[0]] >= 1.0e-3
        or not math.isclose(
            persistent_errors[transition_start[0]],
            recorded_ideal_error,
            rel_tol=1.0e-12,
        )
    ):
        raise RuntimeError("CHIANTI table violates its 10 MK transition policy.")
    return {
        "lookup_log10_temperature_k": [log_temperature[0], log_temperature[-1]],
        "runtime_contract": dict(contract),
        "fully_ionized_limit": {
            "mean_molecular_weight": ideal_mu,
            "mean_molecular_weight_per_electron": ideal_mu_e,
        },
        "composition": {
            "mass_u_per_h_nucleus": mass_per_h,
            "nuclei_per_h_nucleus": nuclei_per_h,
            "fully_ionized_electrons_per_h_nucleus": ideal_electrons_per_h,
        },
        "reference_ionization_equilibrium": dict(source),
    }


def validate_resource_bundle() -> dict:
    """Validate the immutable packaged LTE bundle and return its metadata."""

    root = resource_path()
    bundle_resource = root.joinpath("bundle.json")
    manifest_resource = root.joinpath("sources.json")
    bundle = _load_json(bundle_resource)
    manifest_digest = file_sha256(manifest_resource)
    manifest = load_verified_source_manifest()

    file_records: dict[str, Mapping] = {}
    for group in ("vendored_files", "generated_files", "reviewed_files"):
        records = manifest.get(group)
        if not isinstance(records, dict):
            raise RuntimeError(f"Source manifest has no valid {group!r} inventory.")
        overlap = set(file_records).intersection(records)
        if overlap:
            raise RuntimeError(f"Source manifest repeats files: {sorted(overlap)}")
        file_records.update(records)

    supported_instruments = bundle.get("supported_instruments")
    if supported_instruments != list(SUPPORTED_INSTRUMENTS):
        raise RuntimeError(
            "LTE bundle does not declare the supported instruments exactly."
        )
    common_files = bundle.get("common_production_files")
    instrument_files = bundle.get("instrument_production_files")
    if common_files != sorted(REQUIRED_COMMON_PRODUCTION_FILES):
        raise RuntimeError("LTE bundle does not declare its exact common contract.")
    if not isinstance(instrument_files, dict) or set(instrument_files) != set(
        SUPPORTED_INSTRUMENTS
    ):
        raise RuntimeError("LTE bundle has no exact per-instrument contract.")
    expected_instrument_files = {
        "hinode_sp": HINODE_PRODUCTION_FILES,
        "hmi_stokes": HMI_PRODUCTION_FILES,
    }
    if any(
        instrument_files[name] != sorted(expected_instrument_files[name])
        for name in SUPPORTED_INSTRUMENTS
    ):
        raise RuntimeError("LTE bundle has an invalid per-instrument contract.")
    if bundle.get("training_contract") != TRAINING_RESOURCE_CONTRACT:
        raise RuntimeError("LTE bundle has an invalid offline-training contract.")
    declared = set(common_files)
    for names in instrument_files.values():
        declared.update(map(str, names))
    if declared - set(file_records):
        raise RuntimeError("Production resources are absent from the source manifest.")
    if declared != set(file_records):
        raise RuntimeError(
            "Source manifest contains resources outside the production contract."
        )

    runtime_files = bundle.get("runtime_files")
    expected_runtime_files = sorted([*file_records, "sources.json"])
    if runtime_files != expected_runtime_files:
        raise RuntimeError("LTE runtime inventory does not match the source manifest.")
    packaged_files = _packaged_file_names(root)
    expected_packaged_files = {*expected_runtime_files, "bundle.json"}
    if packaged_files != expected_packaged_files:
        missing_packaged_files = sorted(expected_packaged_files - packaged_files)
        unexpected_packaged_files = sorted(packaged_files - expected_packaged_files)
        raise RuntimeError(
            "LTE bundle directory does not match its exact production inventory; "
            f"missing={missing_packaged_files}, "
            f"unexpected={unexpected_packaged_files}. Remove stale resource files "
            "or deploy the complete packaged bundle."
        )
    missing_files = [
        filename for filename in runtime_files if not resource_path(filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"LTE resource bundle is incomplete: {missing_files}")
    digests = {
        filename: verify_manifest_resource(
            resource_path(filename),
            manifest,
            resource_name=filename,
            kind="LTE",
        )
        for filename in file_records
    }

    table = _load_json(resource_path("common/stic_continuum_table.json"))
    raw_contract = table.get("continuum_contract")
    top_boundary = table.get("falc_top_boundary")
    reference_solver = table.get("reference_solver", {})
    axes = table.get("axes", {})
    if table.get("schema_version") != 2 or not isinstance(raw_contract, dict):
        raise RuntimeError("LTE bundle has no valid STiC continuum contract.")
    contract_fields = (
        "tau500_quantity",
        "scattering_is_stored_separately",
        "physical_height_qualified",
        "runtime_fallback",
        "source_function_limitation",
        "thermodynamic_population_contract",
    )
    try:
        contract = {field: raw_contract[field] for field in contract_fields}
    except KeyError as error:
        raise RuntimeError(
            "STiC table has an incomplete continuum contract."
        ) from error
    required_populations = (
        "log10_mass_density_kg_m3",
        "log10_electron_density_m3",
        "log10_neutral_hydrogen_density_m3",
        "log10_fe_i_population_over_partition_m3",
    )
    if any(not isinstance(table.get(field), list) for field in required_populations):
        raise RuntimeError("STiC table lacks coherent thermodynamic populations.")
    if not isinstance(contract.get("thermodynamic_population_contract"), str):
        raise RuntimeError("STiC table lacks its population contract.")
    if (
        contract.get("physical_height_qualified") is not True
        or contract.get("runtime_fallback")
        != "none; missing or invalid STiC table is a hard error"
    ):
        raise RuntimeError("STiC table is not qualified for physical height synthesis.")
    if (
        reference_solver.get("commit") != "18cda77d038a97f007a783dcb61ea9a9a1244bf7"
        or reference_solver.get("runtime_iterations") != 0
    ):
        raise RuntimeError("LTE bundle is not the pinned offline STiC reference.")
    try:
        temperature_axis = [float(value) for value in axes["log10_temperature_k"]]
        pressure_axis = [float(value) for value in axes["log10_gas_pressure_pa"]]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid STiC thermodynamic axes.") from error
    if (
        len(temperature_axis) < 2
        or len(pressure_axis) < 2
        or not all(
            math.isfinite(value) for value in (*temperature_axis, *pressure_axis)
        )
        or any(b <= a for a, b in zip(temperature_axis, temperature_axis[1:]))
        or any(b <= a for a, b in zip(pressure_axis, pressure_axis[1:]))
    ):
        raise RuntimeError("Invalid STiC thermodynamic axes.")
    if not isinstance(top_boundary, dict):
        raise RuntimeError("LTE bundle lacks its FALC top boundary.")
    try:
        top_pressure = float(top_boundary["gas_pressure_pa"])
        gravity = float(top_boundary["gravity_cm_s2"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid FALC top boundary.") from error
    if (
        top_boundary.get("target_log10_tau500") != -5.0
        or not math.isfinite(top_pressure)
        or top_pressure <= 0
        or not math.isfinite(gravity)
        or gravity <= 0
    ):
        raise RuntimeError("Invalid FALC top boundary.")
    if (
        bundle.get("physical_depth_contract") != contract
        or bundle.get("falc_top_boundary") != top_boundary
    ):
        raise RuntimeError("Bundle metadata disagrees with the STiC table.")

    coverage = _validate_stic_coverage(table)
    falc_reference = _validate_falc_reference_atmosphere(
        _load_json(resource_path("common/falc_reference_atmosphere.json")),
        manifest,
        top_boundary,
    )
    chianti_table = _load_json(resource_path("common/chianti_thermodynamic_table.json"))
    thermodynamic_eos = _validate_chianti_eos_table(chianti_table, manifest)
    if not math.isclose(
        thermodynamic_eos["runtime_contract"]["stic_exact_max_temperature_k"],
        10.0 ** temperature_axis[-1],
        rel_tol=1.0e-12,
    ):
        raise RuntimeError("STiC and CHIANTI thermodynamic domains do not meet.")
    return {
        "bundle_schema_version": 2,
        "bundle_type": BUNDLE_TYPE,
        "supported_instruments": list(SUPPORTED_INSTRUMENTS),
        "source_manifest_sha256": manifest_digest,
        "resource_sha256": digests,
        "training_contract": TRAINING_RESOURCE_CONTRACT,
        "common_production_files": sorted(REQUIRED_COMMON_PRODUCTION_FILES),
        "instrument_production_files": {
            name: sorted(expected_instrument_files[name])
            for name in SUPPORTED_INSTRUMENTS
        },
        "physical_depth_contract": contract,
        "falc_top_boundary": top_boundary,
        "falc_reference_atmosphere": falc_reference,
        "stic_lookup_bounds": {
            "temperature_log10_k": [temperature_axis[0], temperature_axis[-1]],
            "gas_pressure_log10_pa": [pressure_axis[0], pressure_axis[-1]],
        },
        "stic_wavelength_coverage": coverage,
        "thermodynamic_eos": thermodynamic_eos,
    }


def validate_instrument_resource_bundle(instrument: str) -> dict:
    """Validate and return the complete packaged contract for one instrument."""

    if instrument not in SUPPORTED_INSTRUMENTS:
        raise ValueError(
            f"Unsupported LTE resource instrument {instrument!r}; expected one of "
            f"{list(SUPPORTED_INSTRUMENTS)}."
        )
    bundle = validate_resource_bundle()
    production_files = sorted(
        REQUIRED_COMMON_PRODUCTION_FILES
        | (
            HINODE_PRODUCTION_FILES
            if instrument == "hinode_sp"
            else HMI_PRODUCTION_FILES
        )
    )
    return {
        "instrument": instrument,
        "bundle_schema_version": bundle["bundle_schema_version"],
        "bundle_type": bundle["bundle_type"],
        "training_contract": bundle["training_contract"],
        "production_files": production_files,
        "resource_sha256": {
            name: bundle["resource_sha256"][name] for name in production_files
        },
        "physical_depth_contract": bundle["physical_depth_contract"],
        "falc_top_boundary": bundle["falc_top_boundary"],
        "falc_reference_atmosphere": bundle["falc_reference_atmosphere"],
        "stic_lookup_bounds": bundle["stic_lookup_bounds"],
        "stic_wavelength_coverage": bundle["stic_wavelength_coverage"],
        "thermodynamic_eos": bundle["thermodynamic_eos"],
    }


__all__ = [
    "BUNDLE_TYPE",
    "HINODE_PRODUCTION_FILES",
    "HINODE_STIC_REQUIRED_DOMAIN_ANGSTROM",
    "HINODE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM",
    "HMI_PRODUCTION_FILES",
    "HMI_STIC_REQUIRED_DOMAIN_ANGSTROM",
    "HMI_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM",
    "REQUIRED_COMMON_PRODUCTION_FILES",
    "REQUIRED_HINODE_PRODUCTION_FILES",
    "REQUIRED_HMI_PRODUCTION_FILES",
    "REFERENCE_STIC_REQUIRED_WAVELENGTH_NODES_ANGSTROM",
    "SUPPORTED_INSTRUMENTS",
    "TRAINING_RESOURCE_CONTRACT",
    "file_sha256",
    "load_verified_source_manifest",
    "resource_path",
    "validate_resource_bundle",
    "validate_instrument_resource_bundle",
    "verify_manifest_resource",
]
