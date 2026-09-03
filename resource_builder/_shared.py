"""Shared implementation for the offline LTE resource builder."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen

import numpy as np


LOGGER = logging.getLogger("lte-resource-reproduction")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_ROOT = PROJECT_ROOT / "src" / "prom3theus" / "resources" / "data"
STIC_COMMIT = "18cda77d038a97f007a783dcb61ea9a9a1244bf7"
STIC_SOURCE_IDS = (
    "stic_witt_eos",
    "stic_kurucz_partition_functions",
    "stic_abundances",
    "stic_falc_82",
)
FTS_SOURCE_ID = "stic_fts_disk_center"
CONTINUUM_WAVELENGTHS_ANGSTROM = (
    5000.0,
    6160.0,
    6170.0,
    6180.0,
    6190.0,
    6290.0,
    6300.0,
    6310.0,
    6320.0,
)
VALIDATED_WAVELENGTH_DOMAINS_ANGSTROM = (
    (5000.0, 5000.0),
    (6160.0, 6190.0),
    (6290.0, 6320.0),
)
USER_AGENT = "prom3theus-resource-reproduction/1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}.")
    return value


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, document: dict, *, compact: bool = False) -> None:
    options = {"separators": (",", ":")} if compact else {"indent": 2}
    _write_bytes(path, (json.dumps(document, **options) + "\n").encode("utf-8"))


def _default_cache_directory() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "prom3theus" / "resource-sources-v1"


def _fetch_one(source_id: str, source: dict, cache_directory: Path) -> bytes:
    expected = source["sha256"]
    cached = cache_directory / source["filename"]
    if cached.is_file():
        payload = cached.read_bytes()
        if _sha256(payload) == expected:
            LOGGER.info("Verified cache hit: %s", source_id)
            return payload
        raise RuntimeError(f"Cached source has the wrong SHA256 digest: {cached}")
    request = Request(source["url"], headers={"User-Agent": USER_AGENT})
    LOGGER.info("Downloading pinned source: %s", source_id)
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    actual = _sha256(payload)
    if actual != expected:
        raise RuntimeError(
            f"Source {source_id!r} failed SHA256 verification: "
            f"expected {expected}, got {actual}."
        )
    cache_directory.mkdir(parents=True, exist_ok=True)
    _write_bytes(cached, payload)
    return payload


def _fetch_inputs(
    source_manifest: dict, cache_directory: Path, workers: int
) -> dict[str, bytes]:
    sources = source_manifest["sources"]
    source_ids = tuple(source_manifest["source_roles"]["bundle_input_ids"])
    if set(source_ids) != {*STIC_SOURCE_IDS, FTS_SOURCE_ID}:
        raise RuntimeError(
            "The source manifest has an unexpected generation-input set."
        )
    payloads: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(source_ids))) as executor:
        futures = {
            executor.submit(
                _fetch_one, source_id, sources[source_id], cache_directory
            ): source_id
            for source_id in source_ids
        }
        for future in as_completed(futures):
            source_id = futures[future]
            payloads[source_id] = future.result()
    return payloads


@contextmanager
def _loaded_stic_witt(payloads: dict[str, bytes]):
    """Import the checksum-verified STiC generator from a temporary directory."""

    with tempfile.TemporaryDirectory(prefix="prom3theus-stic-") as temporary:
        source_root = Path(temporary)
        _write_bytes(source_root / "witt.py", payloads["stic_witt_eos"])
        _write_bytes(
            source_root / "pf_Kurucz.input",
            payloads["stic_kurucz_partition_functions"],
        )
        previous = sys.modules.pop("witt", None)
        sys.path.insert(0, str(source_root))
        try:
            yield importlib.import_module("witt")
        finally:
            sys.modules.pop("witt", None)
            sys.path.remove(str(source_root))
            if previous is not None:
                sys.modules["witt"] = previous


def _stic_abundance_ratios(payload: bytes) -> np.ndarray:
    values = []
    for raw_line in payload.decode("ascii").splitlines():
        fields = raw_line.split("#", 1)[0].split()
        if len(fields) < 2:
            continue
        try:
            values.append(10.0 ** (float(fields[1]) - 12.0))
        except ValueError:
            continue
    if len(values) != 99 or not np.isclose(values[0], 1.0, atol=1.0e-15):
        raise RuntimeError("The pinned STiC abundance input has an unexpected format.")
    return np.asarray(values, dtype=np.float64)


def _parse_falc_mass_scale(payload: bytes) -> tuple[float, np.ndarray, np.ndarray]:
    lines = payload.decode("ascii").splitlines()
    try:
        depth_marker = next(i for i, line in enumerate(lines) if "Ndep" in line)
        ndep = int(
            next(
                line.strip()
                for line in lines[depth_marker + 1 :]
                if line.strip() and not line.lstrip().startswith("*")
            )
        )
        gravity_marker = next(i for i, line in enumerate(lines) if "log g" in line)
        log_gravity = float(
            next(
                line.strip()
                for line in lines[gravity_marker + 1 :]
                if line.strip() and not line.lstrip().startswith("*")
            )
        )
        table_marker = next(
            i for i, line in enumerate(lines) if "lg column Mass" in line
        )
    except (StopIteration, ValueError) as error:
        raise RuntimeError("Could not parse the pinned FALC_82 header.") from error
    rows = []
    for line in lines[table_marker + 1 :]:
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            rows.append(tuple(float(value) for value in fields))
        except ValueError:
            continue
        if len(rows) == ndep:
            break
    if len(rows) != ndep:
        raise RuntimeError(f"FALC_82 declares {ndep} depths but yielded {len(rows)}.")
    array = np.asarray(rows, dtype=np.float64)
    if np.any(np.diff(array[:, 0]) <= 0.0):
        raise RuntimeError("FALC_82 column mass is not strictly increasing.")
    return log_gravity, array[:, 0], array[:, 1]


def _falc_top_boundary(solver, payload: bytes) -> dict:
    target_log_tau500 = -5.0
    log_gravity, log_column_mass, temperature = _parse_falc_mass_scale(payload)
    gravity_cgs = 10.0**log_gravity
    column_mass = 10.0**log_column_mass
    gas_pressure = gravity_cgs * column_mass
    density = np.empty_like(temperature)
    extinction = np.empty_like(temperature)
    for index, (temp, pressure) in enumerate(zip(temperature, gas_pressure)):
        electron_pressure, electron_fraction = solver.pe_from_pg(
            temp, pressure, get_fe=True
        )
        density[index] = (
            electron_pressure * solver.rho_from_H / (electron_fraction * temp)
        )
        total, _ = solver.contOpacity(
            temp, pressure, electron_pressure, np.asarray([5000.0]), True
        )
        extinction[index] = total[0]
    mass_extinction = extinction / density
    tau500 = np.empty_like(temperature)
    tau500[0] = column_mass[0] * mass_extinction[0]
    tau500[1:] = tau500[0] + np.cumsum(
        np.diff(column_mass) * 0.5 * (mass_extinction[1:] + mass_extinction[:-1])
    )
    log_tau500 = np.log10(tau500)
    bracket_upper = int(np.searchsorted(log_tau500, target_log_tau500))
    bracket_lower = bracket_upper - 1
    pressure_pa = float(
        10.0
        ** np.interp(
            target_log_tau500,
            log_tau500,
            np.log10(gas_pressure * 0.1),
        )
    )
    boundary_temperature = float(np.interp(target_log_tau500, log_tau500, temperature))
    electron_pressure, electron_fraction = solver.pe_from_pg(
        boundary_temperature, pressure_pa * 10.0, get_fe=True
    )
    density_kg_m3 = (
        electron_pressure
        * solver.rho_from_H
        / (electron_fraction * boundary_temperature)
        * 1000.0
    )
    total_cm1, scattering_cm1 = solver.contOpacity(
        boundary_temperature,
        pressure_pa * 10.0,
        electron_pressure,
        np.asarray([5000.0]),
        True,
    )
    total_m1 = float(total_cm1[0] * 100.0)
    scattering_m1 = float(scattering_cm1[0] * 100.0)
    metric = math.log(10.0) * 10.0**target_log_tau500 / total_m1
    pressure_derivative = density_kg_m3 * (gravity_cgs / 100.0) * metric / pressure_pa
    return {
        "model": "FALC_82",
        "target_log10_tau500": target_log_tau500,
        "gas_pressure_pa": pressure_pa,
        "temperature_k": boundary_temperature,
        "log10_column_mass_g_cm2": float(
            np.interp(target_log_tau500, log_tau500, log_column_mass)
        ),
        "gravity_cm_s2": float(gravity_cgs),
        "falc_log10_tau500_range": [float(log_tau500[0]), float(log_tau500[-1])],
        "bracketing_depth_indices": [bracket_lower, bracket_upper],
        "integration": (
            "STiC ceos::hydrostatic_cmass convention: tau[0]=m[0]*kappa[0]; "
            "subsequent tau increments use trapezoidal kappa500 over column mass"
        ),
        "tau500_extinction": "true absorption plus scattering at vacuum 5000 Angstrom",
        "pressure_interpolation": "linear in log10(Pgas) versus log10(tau500)",
        "stic_state_at_interpolated_boundary": {
            "total_extinction_m1": total_m1,
            "true_absorption_m1": float(total_m1 - scattering_m1),
            "scattering_extinction_m1": scattering_m1,
            "mass_density_kg_m3": float(density_kg_m3),
            "mass_extinction_m2_kg": float(total_m1 / density_kg_m3),
            "required_metric_m_per_log10_tau": float(metric),
            "dlog_pressure_dlog10_tau": float(pressure_derivative),
            "pressure_efold_width_log10_tau": float(1.0 / pressure_derivative),
        },
    }


def _generate_stic_table(payloads: dict[str, bytes]) -> dict:
    log_temperature = np.linspace(3.4, 4.0, 129, dtype=np.float64)
    log_pressure = np.linspace(-1.5, 6.0, 173, dtype=np.float64)
    wavelengths = np.asarray(CONTINUUM_WAVELENGTHS_ANGSTROM, dtype=np.float64)
    shape = (log_temperature.size, log_pressure.size)
    log_absorption = np.empty((*shape, wavelengths.size), dtype=np.float64)
    log_scattering = np.empty_like(log_absorption)
    log_density = np.empty(shape, dtype=np.float64)
    log_electron_density = np.empty(shape, dtype=np.float64)
    log_hydrogen_neutral = np.empty(shape, dtype=np.float64)
    log_fe_i_over_partition = np.empty(shape, dtype=np.float64)
    abundances = _stic_abundance_ratios(payloads["stic_abundances"])
    with _loaded_stic_witt(payloads) as stic_witt:
        solver = stic_witt.witt(abund_init=abundances.copy())
        for temperature_index, log_t in enumerate(log_temperature):
            temperature = 10.0**log_t
            for pressure_index, log_p in enumerate(log_pressure):
                pressure_pa = 10.0**log_p
                pressure_cgs = pressure_pa * 10.0
                electron_pressure, electron_fraction = solver.pe_from_pg(
                    temperature, pressure_cgs, get_fe=True
                )
                _, hydrogen_partials = solver.gasc(temperature, electron_pressure)
                hydrogen_neutral_cm3 = (
                    hydrogen_partials[solver.ncontr]
                    * hydrogen_partials[solver.ncontr + 4]
                    / (solver.BK * temperature)
                )
                fe_populations, fe_partitions = solver.getXparts(
                    25,
                    temperature,
                    pressure_cgs,
                    electron_pressure,
                    divide_by_u=False,
                    return_u=True,
                )
                fe_i_over_partition_cm3 = fe_populations[0] / fe_partitions[0]
                density_g_cm3 = (
                    electron_pressure
                    * solver.rho_from_H
                    / (electron_fraction * temperature)
                )
                total_cm1, scattering_cm1 = solver.contOpacity(
                    temperature,
                    pressure_cgs,
                    electron_pressure,
                    wavelengths,
                    True,
                )
                absorption_cm1 = total_cm1 - scattering_cm1
                values = (
                    absorption_cm1,
                    scattering_cm1,
                    density_g_cm3,
                    electron_pressure,
                    hydrogen_neutral_cm3,
                    fe_i_over_partition_cm3,
                )
                if any(not np.all(np.isfinite(value)) for value in values) or any(
                    np.any(value <= 0.0) for value in values
                ):
                    raise RuntimeError(
                        "STiC returned a non-positive or non-finite state at "
                        f"T={temperature:g} K, Pgas={pressure_pa:g} Pa."
                    )
                log_absorption[temperature_index, pressure_index] = np.log10(
                    absorption_cm1 * 100.0
                )
                log_scattering[temperature_index, pressure_index] = np.log10(
                    scattering_cm1 * 100.0
                )
                log_density[temperature_index, pressure_index] = math.log10(
                    density_g_cm3 * 1000.0
                )
                log_electron_density[temperature_index, pressure_index] = math.log10(
                    electron_pressure / (solver.BK * temperature) * 1.0e6
                )
                log_hydrogen_neutral[temperature_index, pressure_index] = math.log10(
                    hydrogen_neutral_cm3 * 1.0e6
                )
                log_fe_i_over_partition[temperature_index, pressure_index] = math.log10(
                    fe_i_over_partition_cm3 * 1.0e6
                )
            LOGGER.info(
                "STiC table row %d/%d", temperature_index + 1, log_temperature.size
            )
        top_boundary = _falc_top_boundary(solver, payloads["stic_falc_82"])
    source_digests = {
        source_id: _sha256(payloads[source_id]) for source_id in STIC_SOURCE_IDS
    }
    return {
        "schema_version": 2,
        "axes": {
            "log10_temperature_k": log_temperature.tolist(),
            "log10_gas_pressure_pa": log_pressure.tolist(),
            "wavelength_vacuum_angstrom": wavelengths.tolist(),
        },
        "log10_true_absorption_m1": log_absorption.tolist(),
        "log10_scattering_extinction_m1": log_scattering.tolist(),
        "log10_mass_density_kg_m3": log_density.tolist(),
        "log10_electron_density_m3": log_electron_density.tolist(),
        "log10_neutral_hydrogen_density_m3": log_hydrogen_neutral.tolist(),
        "log10_fe_i_population_over_partition_m3": log_fe_i_over_partition.tolist(),
        "interpolation": (
            "tensor-product cubic in log10(T) and log10(Pgas), then linear in "
            "vacuum wavelength; logarithmic coefficients are interpolated before "
            "exponentiation"
        ),
        "validated_wavelength_domains_angstrom": [
            list(domain) for domain in VALIDATED_WAVELENGTH_DOMAINS_ANGSTROM
        ],
        "reference_solver": {
            "project": "STiC",
            "repository": "https://github.com/jaimedelacruz/stic",
            "commit": STIC_COMMIT,
            "implementation": "pythontools/py2/witt.py (Wittmann EOS and cop continuum)",
            "runtime_iterations": 0,
            "source_sha256": source_digests,
            "abundance_input_contract": {
                "file": "input/Atoms/abundance.input at the pinned STiC commit",
                "mixture": (
                    "Grevesse & Anders (1991)/Kurucz baseline with the file's "
                    "selective historical Asplund overrides"
                ),
                "iron_log10_n_over_n_h_plus_12": 7.44,
                "iron_annotation": "Asplund et al. (2000), as recorded by STiC",
                "not_used": (
                    "The separate Asplund, Amarsi & Grevesse (2021) reviewed "
                    "abundance resource is not used by this production STiC lookup."
                ),
            },
            "population_formulas": {
                "neutral_hydrogen_density": (
                    "witt.gasc: n(H I) = [p(H I)/p(H')] p(H') / (k_B T); "
                    "stored after cm^-3 to m^-3 conversion"
                ),
                "fe_i_population_over_partition": (
                    "witt.getXparts(atomic_number_minus_one=25, divide_by_u=False, "
                    "return_u=True): n(Fe I)/U(Fe I); stored after cm^-3 to m^-3 "
                    "conversion"
                ),
                "line_lower_population": (
                    "n_l = [n(Fe I)/U(Fe I)] (2 J_l + 1) "
                    "exp[-E_l/(k_B T)], with reviewed line E_l and J_l"
                ),
            },
        },
        "continuum_contract": {
            "tau500_quantity": "total extinction = true absorption + scattering",
            "scattering_is_stored_separately": True,
            "physical_height_qualified": True,
            "runtime_fallback": "none; missing or invalid STiC table is a hard error",
            "source_function_limitation": (
                "The current LTE formal solver applies B_lambda to total continuum "
                "extinction and does not iterate a coherent-scattering source. The "
                "tau500/geometric metric uses STiC total extinction, but upper-layer "
                "continuum intensities retain this explicit thermal-source approximation."
            ),
            "thermodynamic_population_contract": (
                "Continuum coefficients, mass density, electron density, neutral atomic-H "
                "density, and n(Fe I)/U(Fe I) all come from the same converged pinned "
                "STiC/Wittmann (T, Pgas) state. Fe-I lower-level populations apply the "
                "LTE Boltzmann factor using reviewed line E_l and J_l at runtime."
            ),
        },
        "falc_top_boundary": top_boundary,
    }


def _solar_reference(payload: bytes, lower: float, upper: float) -> dict:
    from scipy.io import readsav

    with tempfile.NamedTemporaryFile(suffix=".idlsave") as handle:
        handle.write(payload)
        handle.flush()
        atlas = readsav(handle.name)
    wavelength = np.asarray(atlas["ftswav"], dtype=np.float64)
    intensity = np.asarray(atlas["ftsint"], dtype=np.float64)
    continuum = np.asarray(atlas["ftscnt"], dtype=np.float64)
    selected = (wavelength >= lower) & (wavelength <= upper)
    if np.count_nonzero(selected) < 2:
        raise RuntimeError(f"FTS atlas has no samples in [{lower}, {upper}] A.")
    conversion = 1.0e14
    return {
        "schema_version": 1,
        "source": {
            "description": (
                "Absolute Kitt Peak FTS disk-center solar intensity and fitted "
                "continuum distributed with STiC"
            ),
            "repository": "https://github.com/jaimedelacruz/stic",
            "commit": STIC_COMMIT,
            "file": "pythontools/py2/fts_disk_center.idlsave",
            "sha256": _sha256(payload),
            "absolute_scale_reference": (
                "Neckel and Labs (1984), Solar Physics 90, 205-258, "
                "doi:10.1007/BF00173953"
            ),
        },
        "units": {
            "wavelength": "standard-air angstrom",
            "intensity_radiance": "W m^-3 sr^-1",
            "continuum_radiance": "W m^-3 sr^-1",
            "source_intensity_radiance": "W cm^-2 sr^-1 angstrom^-1",
            "source_to_runtime_factor": conversion,
        },
        "wavelength_range_air_angstrom": [lower, upper],
        "wavelength_air_angstrom": wavelength[selected].tolist(),
        "intensity_radiance_w_m3_sr": (conversion * intensity[selected]).tolist(),
        "continuum_radiance_w_m3_sr": (conversion * continuum[selected]).tolist(),
        "limb_darkening": {
            "reference": (
                "Neckel (2005), Solar Physics 229, 13-33, doi:10.1007/s11207-005-4081-z"
            ),
            "valid_wavelength_nm": [422.57, 1100.0],
            "formula": "I_c(lambda,mu)/I_c(lambda,1)=sum(A_i(lambda)*mu^i,i=0..5)",
            "wavelength_unit_in_coefficient_formula": "micrometre",
            "coefficients_422_57_to_1100_nm": {
                "a00": 0.75267,
                "a01": -0.265577,
                "a10": 0.93874,
                "a11": 0.265577,
                "a15": -0.004095,
                "a20": -1.89287,
                "a25": 0.012582,
                "a30": 2.42234,
                "a35": -0.017117,
                "a40": -1.71150,
                "a45": 0.011977,
                "a50": 0.49062,
                "a55": -0.003347,
            },
        },
    }


def _seal_bundle(output: Path, source_manifest: dict, stic_table: dict) -> None:
    reviewed = source_manifest["reviewed_files"]
    for relative, record in reviewed.items():
        resource = output / relative
        if not resource.is_file() or _sha256(resource.read_bytes()) != record["sha256"]:
            raise RuntimeError(
                f"Instrument/common builder did not reproduce reviewed resource: {relative}"
            )

    generated_paths = tuple(source_manifest["generated_files"])
    for relative in generated_paths:
        actual = _sha256((output / relative).read_bytes())
        expected = source_manifest["generated_files"][relative]["sha256"]
        if actual != expected:
            raise RuntimeError(
                f"Reproduced {relative} differs from the production digest: "
                f"expected {expected}, got {actual}."
            )
    _write_json(output / "sources.json", source_manifest)
    packaged_bundle_path = PACKAGED_ROOT / "bundle.json"
    packaged_bundle = _read_json(packaged_bundle_path)
    bundle = deepcopy(packaged_bundle)
    bundle["source_manifest_sha256"] = _sha256((output / "sources.json").read_bytes())
    bundle["runtime_files"] = sorted(
        [*reviewed, *source_manifest["generated_files"], "sources.json"]
    )
    bundle["physical_depth_contract"] = stic_table["continuum_contract"]
    bundle["falc_top_boundary"] = stic_table["falc_top_boundary"]
    if bundle != packaged_bundle:
        raise RuntimeError(
            "Reconstructed bundle metadata differs from the committed production "
            "contract. Review the scientific metadata before updating resources."
        )
    # Preserve the reviewed human-readable formatting after proving that every
    # reconstructed field has the same value as the committed contract.
    _write_bytes(output / "bundle.json", packaged_bundle_path.read_bytes())


def _validate_reproduction(output: Path, source_manifest: dict) -> None:
    expected = {
        "bundle.json",
        "sources.json",
        *source_manifest["reviewed_files"],
        *source_manifest["generated_files"],
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(
            "Reproduced inventory mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    for relative in sorted(expected):
        reproduced = _sha256((output / relative).read_bytes())
        packaged = _sha256((PACKAGED_ROOT / relative).read_bytes())
        if reproduced != packaged:
            raise RuntimeError(
                f"Final reproduction mismatch for {relative}: "
                f"packaged={packaged}, reproduced={reproduced}."
            )


def _copy_reviewed_resources(
    output: Path, source_manifest: dict, relative_paths: tuple[str, ...]
) -> None:
    reviewed = source_manifest["reviewed_files"]
    for relative in relative_paths:
        if relative not in reviewed:
            raise RuntimeError(f"Reviewed resource is absent from manifest: {relative}")
        payload = (PACKAGED_ROOT / relative).read_bytes()
        expected = reviewed[relative]["sha256"]
        if _sha256(payload) != expected:
            raise RuntimeError(f"Reviewed resource changed without review: {relative}")
        _write_bytes(output / relative, payload)
