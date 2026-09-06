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
CHIANTI_IONEQ_SOURCE_ID = "chianti_ioneq_v11_0_2"
CHIANTI_VERSION_SOURCE_ID = "chianti_database_version_11_0_2"
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
    if set(source_ids) != {
        *STIC_SOURCE_IDS,
        FTS_SOURCE_ID,
        CHIANTI_IONEQ_SOURCE_ID,
        CHIANTI_VERSION_SOURCE_ID,
    }:
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


def _parse_chianti_ionization_equilibrium(
    payload: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse the fixed-width standard CHIANTI ion-equilibrium distribution."""

    try:
        lines = payload.decode("ascii").splitlines()
        temperature_count, element_count = (int(value) for value in lines[0].split())
    except (UnicodeDecodeError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Could not parse the pinned CHIANTI ioneq header."
        ) from error
    if (temperature_count, element_count) != (101, 30):
        raise RuntimeError(
            "The pinned CHIANTI ioneq grid is not the expected H-through-Zn table."
        )
    if len(lines) < 2:
        raise RuntimeError("Could not parse the pinned CHIANTI ioneq header.")
    axis_line = lines[1]
    try:
        log_temperature = np.asarray(
            [
                float(axis_line[index : index + 6])
                for index in range(0, 6 * temperature_count, 6)
            ],
            dtype=np.float64,
        )
    except ValueError as error:
        raise RuntimeError(
            "Could not parse the pinned CHIANTI temperature grid."
        ) from error
    if (
        log_temperature.size != temperature_count
        or not np.all(np.isfinite(log_temperature))
        or not np.all(np.diff(log_temperature) > 0.0)
        or not np.isclose(log_temperature[0], 4.0, rtol=0.0, atol=1.0e-12)
        or not np.isclose(log_temperature[-1], 9.0, rtol=0.0, atol=1.0e-12)
        or not np.allclose(np.diff(log_temperature), 0.05, rtol=0.0, atol=1.0e-12)
    ):
        raise RuntimeError("The pinned CHIANTI temperature grid is invalid.")

    row_count = element_count * (element_count + 3) // 2
    rows = lines[2 : 2 + row_count]
    if (
        len(rows) != row_count
        or len(lines) <= 2 + row_count
        or lines[2 + row_count].strip() != "-1"
    ):
        raise RuntimeError("The pinned CHIANTI ioneq table is incomplete.")
    footer = "\n".join(lines[3 + row_count :])
    if (
        "%filename:  chianti.ioneq" not in footer
        or "Prepared for the release of CHIANTI 10.1." not in footer
        or lines[-1].strip() != "-1"
    ):
        raise RuntimeError("The pinned CHIANTI ioneq provenance footer is invalid.")
    fractions = np.zeros(
        (element_count, element_count + 1, temperature_count), dtype=np.float64
    )
    stages: set[tuple[int, int]] = set()
    for line in rows:
        try:
            atomic_number = int(line[:3])
            stage_number = int(line[3:6])
            values = np.asarray(
                [
                    float(line[6 + 10 * index : 16 + 10 * index])
                    for index in range(temperature_count)
                ],
                dtype=np.float64,
            )
        except ValueError as error:
            raise RuntimeError("Could not parse a pinned CHIANTI ioneq row.") from error
        stage = (atomic_number, stage_number)
        if (
            atomic_number < 1
            or atomic_number > element_count
            or stage_number < 1
            or stage_number > atomic_number + 1
            or stage in stages
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise RuntimeError("The pinned CHIANTI ioneq table has an invalid row.")
        stages.add(stage)
        fractions[atomic_number - 1, stage_number - 1] = values
    expected_stages = {
        (atomic_number, stage_number)
        for atomic_number in range(1, element_count + 1)
        for stage_number in range(1, atomic_number + 2)
    }
    if stages != expected_stages:
        raise RuntimeError("The pinned CHIANTI ioneq stage inventory is incomplete.")
    totals = fractions.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0) or not np.allclose(totals, 1.0, rtol=0.0, atol=5.0e-4):
        raise RuntimeError("The pinned CHIANTI ion fractions are not normalized.")
    # The source stores three digits after the decimal in e10.3 fields, so
    # explicitly restore a unit sum before taking charge moments.
    return log_temperature, fractions / totals


def _pchip_slopes(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the stored shape-preserving Hermite slopes for one mapping."""

    if (
        axis.ndim != 1
        or values.shape != axis.shape
        or axis.size < 3
        or not np.all(np.diff(axis) > 0.0)
        or not np.all(np.diff(values) > 0.0)
    ):
        raise RuntimeError("Cannot construct CHIANTI PCHIP interpolation slopes.")
    spacing = np.diff(axis)
    secant = np.diff(values) / spacing
    slopes = np.zeros_like(values)
    previous, following = secant[:-1], secant[1:]
    weight_previous = 2.0 * spacing[1:] + spacing[:-1]
    weight_following = spacing[1:] + 2.0 * spacing[:-1]
    slopes[1:-1] = (weight_previous + weight_following) / (
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


def _generate_chianti_eos_table(payloads: dict[str, bytes]) -> dict:
    """Reduce CHIANTI ion fractions to the one thermodynamic mapping we need."""

    source_id = CHIANTI_IONEQ_SOURCE_ID
    if payloads[CHIANTI_VERSION_SOURCE_ID] != b"11.0.2\n":
        raise RuntimeError("The pinned CHIANTI database version is not 11.0.2.")
    log_temperature, ion_fractions = _parse_chianti_ionization_equilibrium(
        payloads[source_id]
    )
    element_count = ion_fractions.shape[0]
    abundance_ratio = _stic_abundance_ratios(payloads["stic_abundances"])
    with _loaded_stic_witt(payloads) as stic_witt:
        atomic_mass_u = np.asarray(stic_witt.witt.AMASS, dtype=np.float64).copy()
    if (
        abundance_ratio.shape != (99,)
        or atomic_mass_u.shape != (99,)
        or np.any(abundance_ratio <= 0.0)
        or np.any(atomic_mass_u <= 0.0)
        or not np.all(np.isfinite(abundance_ratio))
        or not np.all(np.isfinite(atomic_mass_u))
    ):
        raise RuntimeError("The pinned STiC mixture is invalid for CHIANTI reduction.")

    charge = np.arange(element_count + 1, dtype=np.float64)
    mean_charge = np.sum(ion_fractions * charge[None, :, None], axis=1)
    nuclei_per_h = float(np.sum(abundance_ratio))
    mass_u_per_h = float(np.sum(abundance_ratio * atomic_mass_u))
    chianti_nuclei_per_h = float(np.sum(abundance_ratio[:element_count]))
    chianti_mass_u_per_h = float(
        np.sum(abundance_ratio[:element_count] * atomic_mass_u[:element_count])
    )
    electrons_per_h = np.sum(
        abundance_ratio[:element_count, None] * mean_charge, axis=0
    )
    fully_ionized_electrons_per_h = float(
        np.sum(abundance_ratio * np.arange(1, abundance_ratio.size + 1))
    )
    chianti_fully_ionized_electrons_per_h = float(
        np.sum(abundance_ratio[:element_count] * np.arange(1, element_count + 1))
    )
    fully_ionized_mean_molecular_weight = mass_u_per_h / (
        nuclei_per_h + fully_ionized_electrons_per_h
    )
    fully_ionized_mean_molecular_weight_per_electron = (
        mass_u_per_h / fully_ionized_electrons_per_h
    )
    if (
        np.any(electrons_per_h <= 0.0)
        or not np.all(np.isfinite(electrons_per_h))
        or not np.all(np.diff(electrons_per_h) > 0.0)
    ):
        raise RuntimeError("CHIANTI reduction produced an invalid EoS mapping.")

    log_electrons_per_h = np.log10(electrons_per_h)
    electron_slopes = _pchip_slopes(log_temperature, log_electrons_per_h)
    mean_molecular_weight = mass_u_per_h / (nuclei_per_h + electrons_per_h)
    mean_molecular_weight_per_electron = mass_u_per_h / electrons_per_h
    ideal_relative_error = np.maximum(
        np.abs(mean_molecular_weight / fully_ionized_mean_molecular_weight - 1.0),
        np.abs(
            mean_molecular_weight_per_electron
            / fully_ionized_mean_molecular_weight_per_electron
            - 1.0
        ),
    )
    persistent_error = np.maximum.accumulate(ideal_relative_error[::-1])[::-1]
    ideal_transition_temperature_k = (1.0e7, 2.0e7)
    ideal_transition_log_temperature = tuple(
        math.log10(value) for value in ideal_transition_temperature_k
    )
    ideal_transition_start = np.flatnonzero(
        np.isclose(
            log_temperature,
            ideal_transition_log_temperature[0],
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    if (
        ideal_transition_start.size != 1
        or ideal_transition_log_temperature[1] > log_temperature[-1]
        or persistent_error[ideal_transition_start[0]] >= 1.0e-3
    ):
        raise RuntimeError(
            "The pinned CHIANTI table no longer supports the reviewed 10 MK "
            "ideal-gas transition policy."
        )

    return {
        "schema_version": 1,
        "axes": {"log10_temperature_k": log_temperature.tolist()},
        "log10_free_electrons_per_h_nucleus": log_electrons_per_h.tolist(),
        "pchip_d_log10_free_electrons_per_h_d_log10_temperature": (
            electron_slopes.tolist()
        ),
        "interpolation": {
            "method": "shape-preserving piecewise cubic Hermite (PCHIP)",
            "segment_coordinate": "t = (x - x_i) / (x_(i+1) - x_i)",
            "segment_polynomial": (
                "y = (2t^3-3t^2+1)y_i + (t^3-2t^2+t)h_i m_i + "
                "(-2t^3+3t^2)y_(i+1) + (t^3-t^2)h_i m_(i+1)"
            ),
            "node_slopes": (
                "stored explicitly; generated with weighted harmonic interior "
                "secants and one-sided shape-preserving endpoint limiters"
            ),
            "outside_native_grid": "clamp to the nearest endpoint value",
        },
        "fully_ionized_limit": {
            "mean_molecular_weight": fully_ionized_mean_molecular_weight,
            "mean_molecular_weight_per_electron": (
                fully_ionized_mean_molecular_weight_per_electron
            ),
        },
        "composition": {
            "abundance_source": (
                "STiC input/Atoms/abundance.input at the pinned STiC commit"
            ),
            "atomic_mass_source": "STiC witt.py AMASS at the pinned STiC commit",
            "chianti_atomic_numbers": [1, element_count],
            "stic_mixture_atomic_numbers": [1, int(abundance_ratio.size)],
            "higher_atomic_number_treatment": (
                "Z > 30 nuclei and mass remain in the STiC mixture; their negligible "
                "free-electron contribution is ignored in the CHIANTI regime and "
                "included only in the fully ionized analytic limit"
            ),
            "z_greater_than_30_nuclei_fraction": (
                1.0 - chianti_nuclei_per_h / nuclei_per_h
            ),
            "z_greater_than_30_mass_fraction": (
                1.0 - chianti_mass_u_per_h / mass_u_per_h
            ),
            "z_greater_than_30_fully_ionized_electron_fraction": (
                1.0
                - chianti_fully_ionized_electrons_per_h / fully_ionized_electrons_per_h
            ),
            "source_sha256": {
                "stic_abundances": _sha256(payloads["stic_abundances"]),
                "stic_witt_eos": _sha256(payloads["stic_witt_eos"]),
            },
            "nuclei_per_h_nucleus": nuclei_per_h,
            "mass_u_per_h_nucleus": mass_u_per_h,
            "fully_ionized_electrons_per_h_nucleus": (fully_ionized_electrons_per_h),
        },
        "reference_ionization_equilibrium": {
            "project": "CHIANTI atomic database",
            "database_release": "11.0.2",
            "distribution": "SolarSoft CHIANTI dbase mirror",
            "file": "ioneq/chianti.ioneq",
            "file_embedded_revision_note": "prepared for CHIANTI 10.1",
            "source_sha256": _sha256(payloads[source_id]),
            "version_source_sha256": _sha256(payloads[CHIANTI_VERSION_SOURCE_ID]),
            "model": (
                "standard zero-density coronal ionization equilibrium; Maxwellian "
                "electron distribution"
            ),
            "fraction_normalization": (
                "each element and temperature renormalized to unit sum after parsing "
                "the source's e10.3 fractions"
            ),
            "references": [
                "https://doi.org/10.5281/zenodo.14263073",
                "https://doi.org/10.3847/1538-4365/acec79",
                "https://doi.org/10.3847/1538-4357/ad6765",
            ],
            "advanced_model_exclusion": (
                "CHIANTI 11 density-dependent and charge-transfer advanced models "
                "require additional atmosphere-specific inputs and are not represented"
            ),
        },
        "thermodynamic_mapping": {
            "mean_charge": "qbar_Z(T) = sum_q q f_Zq(T)",
            "free_electrons_per_h_nucleus": ("e(T) = sum_{Z<=30} A_Z qbar_Z(T)"),
            "chianti_particles_per_h_nucleus": "g(T) = sum_Z A_Z + e(T)",
            "hybrid_particles_per_h_nucleus": (
                "g(T,P) = h(T,P) + e(T,P), where h transitions from the "
                "STiC non-electron particle count to sum_Z A_Z"
            ),
            "mean_molecular_weight": "mu(T,P) = M / g(T,P)",
            "mean_molecular_weight_per_electron": ("mu_e(T,P) = M / e(T,P)"),
            "hydrogen_nucleus_density": "n_H(T,P) = P / [k_B T g(T,P)]",
            "mass_density": "rho(T,P) = M m_u n_H(T,P)",
            "electron_density": "n_e(T,P) = e(T,P) n_H(T,P)",
        },
        "runtime_contract": {
            "minimum_temperature_k": 10.0**3.4,
            "stic_exact_max_temperature_k": 1.0e4,
            "stic_to_chianti_log_blend_temperature_k": [1.0e4, 10.0**4.5],
            "chianti_to_fully_ionized_log_blend_temperature_k": list(
                ideal_transition_temperature_k
            ),
            "lower_transition": {
                "electron_coordinate": (
                    "logit(free_electrons_per_H / fully_ionized_electrons_per_H)"
                ),
                "non_electron_coordinate": ("log10(non_electron_particles_per_H)"),
                "interior_weight": "quintic smootherstep in log10(T)",
                "endpoint_tangent_match_width_log10_temperature": 0.05,
                "endpoint_tangent_ramp_power": 5,
                "lower_endpoint_ramp": "1 - (1 - u)^p",
                "upper_endpoint_ramp": "u^p",
                "integrated_tangent_change": "slope * width / p",
                "closure": "particles_per_H = non_electron_particles_per_H + electrons_per_H",
            },
            "upper_transition": {
                "electron_coordinate": "log10(free_electrons_per_H)",
                "interior_weight": "quintic smootherstep in log10(T)",
                "closure": "particles_per_H = nuclei_per_H + electrons_per_H",
            },
            "stic_interpolation": {
                "coordinates": "tensor product in log10(T/K) and log10(Pgas/Pa)",
                "uniform_axis_coordinate": "(x - x_0) * (N - 1) / (x_(N-1) - x_0)",
                "cells": "Catmull-Rom cubic with endpoint-index clamping",
                "boundary_derivative": (
                    "one half of the outer one-sided secant, matching the "
                    "established STiC runtime interpolant"
                ),
            },
            "pressure_continuation": {
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
                    "constant particles_per_H and electrons_per_H; rho and n_e "
                    "are linear in P"
                ),
            },
            "smoothness": (
                "C1 in log10(T) at all regime joins and in log10(P) at both STiC "
                "pressure edges; second derivatives are not constrained"
            ),
            "pressure_dependence": (
                "STiC retains its tabulated (T,P) dependence on the native pressure "
                "axis; bounded physical-coordinate shoulders continue it outside; "
                "CHIANTI and the fully ionized limit depend on T only, with rho and "
                "n_e linear in P"
            ),
            "lower_transition_rationale": (
                "the bounded electron logit prevents charge states above the fully "
                "ionized composition; the shared non-electron count preserves exact "
                "STiC at 10 kK and reaches the CHIANTI ideal-mixture closure at "
                "log10(T/K)=4.5; endpoint tangent ramps span one native CHIANTI node"
            ),
            "ideal_transition_criterion": (
                "CHIANTI remains exact through the native 10 MK node; a C1 "
                "log-temperature blend then reaches the fully ionized limit at "
                "exactly 20 MK, and starts with both CHIANTI-derived mu and mu_e "
                "within 1e-3 relative error of that limit"
            ),
            "maximum_relative_error_at_ideal_blend_start": float(
                persistent_error[ideal_transition_start[0]]
            ),
        },
    }


def _parse_falc_atmosphere(payload: bytes) -> tuple[float, np.ndarray]:
    """Parse every native thermodynamic row from the pinned FALC atmosphere."""

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
    if (
        array.shape != (82, 5)
        or not np.all(np.isfinite(array))
        or np.any(np.diff(array[:, 0]) <= 0.0)
        or np.any(array[:, 1] <= 0.0)
        or np.any(array[:, 4] <= 0.0)
    ):
        raise RuntimeError("FALC_82 contains invalid native atmosphere rows.")
    return log_gravity, array


def _parse_falc_mass_scale(payload: bytes) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the legacy FALC mass/temperature subset used by older builders."""

    log_gravity, atmosphere = _parse_falc_atmosphere(payload)
    return log_gravity, atmosphere[:, 0], atmosphere[:, 1]


def _falc_reference_state(solver, payload: bytes) -> dict[str, np.ndarray | float]:
    """Construct the native FALC optical-depth and geometric-height coordinates."""

    log_gravity, atmosphere = _parse_falc_atmosphere(payload)
    log_column_mass = atmosphere[:, 0]
    temperature = atmosphere[:, 1]
    microturbulence_km_s = atmosphere[:, 4]
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
    if (
        np.any(density <= 0.0)
        or np.any(extinction <= 0.0)
        or not np.all(np.isfinite(density))
        or not np.all(np.isfinite(extinction))
    ):
        raise RuntimeError("STiC returned an invalid native FALC state.")

    mass_extinction = extinction / density
    tau500 = np.empty_like(temperature)
    tau500[0] = column_mass[0] * mass_extinction[0]
    tau500[1:] = tau500[0] + np.cumsum(
        np.diff(column_mass) * 0.5 * (mass_extinction[1:] + mass_extinction[:-1])
    )
    log_tau500 = np.log10(tau500)

    # dm = -rho dz.  Integrating the reciprocal density preserves the original
    # native mass cells; the additive height origin is fixed at tau500 = 1.
    height_m = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            -np.cumsum(
                np.diff(column_mass) * 0.5 * (1.0 / density[1:] + 1.0 / density[:-1])
            )
            * 1.0e-2,
        )
    )
    if not log_tau500[0] < 0.0 < log_tau500[-1]:
        raise RuntimeError("The native FALC atmosphere does not bracket tau500=1.")
    height_m -= np.interp(0.0, log_tau500, height_m)
    if (
        np.any(np.diff(log_tau500) <= 0.0)
        or np.any(np.diff(height_m) >= 0.0)
        or not np.all(np.isfinite(height_m))
    ):
        raise RuntimeError("Could not construct monotone native FALC coordinates.")
    return {
        "log_gravity": float(log_gravity),
        "log_column_mass": log_column_mass,
        "temperature": temperature,
        "microturbulence_km_s": microturbulence_km_s,
        "gravity_cgs": float(gravity_cgs),
        "column_mass": column_mass,
        "gas_pressure": gas_pressure,
        "density": density,
        "extinction": extinction,
        "mass_extinction": mass_extinction,
        "log_tau500": log_tau500,
        "height_m": height_m,
    }


def _generate_falc_reference_atmosphere(payloads: dict[str, bytes]) -> dict:
    """Package the complete native FALC atmosphere without runtime truncation."""

    abundances = _stic_abundance_ratios(payloads["stic_abundances"])
    with _loaded_stic_witt(payloads) as stic_witt:
        solver = stic_witt.witt(abund_init=abundances.copy())
        state = _falc_reference_state(solver, payloads["stic_falc_82"])
    line_log_tau500 = np.linspace(-5.0, 1.0, 25, dtype=np.float64)
    native_log_tau500 = state["log_tau500"]
    native_pressure_pa = state["gas_pressure"] * 0.1
    line_height_m = np.interp(line_log_tau500, native_log_tau500, state["height_m"])
    line_height_m[line_log_tau500 == 0.0] = 0.0
    line_temperature_k = np.interp(
        line_log_tau500, native_log_tau500, state["temperature"]
    )
    line_pressure_pa = 10.0 ** np.interp(
        line_log_tau500, native_log_tau500, np.log10(native_pressure_pa)
    )
    line_microturbulence_m_per_s = np.interp(
        line_log_tau500,
        native_log_tau500,
        state["microturbulence_km_s"] * 1.0e3,
    )
    return {
        "schema_version": 1,
        "model": "FALC_82",
        "native_depth_count": 82,
        "source": {
            "project": "STiC",
            "repository": "https://github.com/jaimedelacruz/stic",
            "commit": STIC_COMMIT,
            "file": "input/Atmos/FALC_82.atmos",
            "source_sha256": {
                source_id: _sha256(payloads[source_id]) for source_id in STIC_SOURCE_IDS
            },
        },
        "coordinates": {
            "log10_tau500": state["log_tau500"].tolist(),
            "height_m": state["height_m"].tolist(),
            "log10_column_mass_g_cm2": state["log_column_mass"].tolist(),
        },
        "temperature_k": state["temperature"].tolist(),
        "gas_pressure_pa": (state["gas_pressure"] * 0.1).tolist(),
        "microturbulence_m_per_s": (state["microturbulence_km_s"] * 1.0e3).tolist(),
        "line_formation_reference": {
            "log10_tau500": line_log_tau500.tolist(),
            "height_m": line_height_m.tolist(),
            "temperature_k": line_temperature_k.tolist(),
            "gas_pressure_pa": line_pressure_pa.tolist(),
            "microturbulence_m_per_s": line_microturbulence_m_per_s.tolist(),
            "resampling": {
                "interval": "log10(tau500)=-5..1 at 0.25 dex spacing",
                "linear_fields": [
                    "height_m",
                    "temperature_k",
                    "microturbulence_m_per_s",
                ],
                "log_linear_fields": ["gas_pressure_pa"],
                "purpose": (
                    "preserve the established LTE ray/depth mapping while the "
                    "previously unused native FALC upper nodes remain available"
                ),
            },
        },
        "construction": {
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
            "gravity_cm_s2": state["gravity_cgs"],
            "native_order": "top to bottom",
        },
    }


def _falc_top_boundary(solver, payload: bytes) -> dict:
    target_log_tau500 = -5.0
    state = _falc_reference_state(solver, payload)
    log_column_mass = state["log_column_mass"]
    temperature = state["temperature"]
    gravity_cgs = state["gravity_cgs"]
    gas_pressure = state["gas_pressure"]
    log_tau500 = state["log_tau500"]
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
