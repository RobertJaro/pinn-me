"""Fetch and verify the external tables used by :mod:`pme.lte`.

The runtime package never needs network access. This preparation command pins
every remote resource to both an immutable URL (where available) and a SHA256
digest. It extracts diagnostic H-minus coefficients from Lightweaver and runs
the pinned STiC Wittmann EOS/background continuum once to generate a fixed
differentiable total-extinction lookup plus its FALC top boundary. Iterative
STiC code is never imported during training. Small reviewed abundance and
Fe-line metadata JSON files remain under version control.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import sys
import time
from urllib.request import Request, urlopen

import numpy as np

from pme.lte.atomic import AtomicDatabase
from pme.lte.eos import LTEEOS
from pme.lte.resources import validate_resource_bundle


LOGGER = logging.getLogger(__name__)


SOURCES = {
    "asplund_2021_abundances": {
        "url": "https://arxiv.org/pdf/2105.01661v2",
        "sha256": "b3aa4a0a2561992ce3f0aa75a6f9c14f9f2f4487de2bf3fe5eb8bf3782f7ef33",
        "filename": "asplund_amarsi_grevesse_2021_arxiv_2105.01661.pdf",
    },
    "ciaaw_standard_atomic_weights_2024": {
        "url": "https://ciaaw.org/atomic-weights.htm",
        "sha256": "b48282594b1fb01eee3cbc9d469ce5e3483b628157ea1b09ede33e3476895cf2",
        "filename": "ciaaw_standard_atomic_weights_2024.html",
    },
    "ciaaw_abridged_atomic_weights_2024": {
        "url": "https://ciaaw.org/abridged-atomic-weights.htm",
        "sha256": "f9e9554471749c55a624aec55151922470a7f4104c62811eb194fed9731b907d",
        "filename": "ciaaw_abridged_atomic_weights_2024.html",
    },
    "yang_2024_spin4d_table3": {
        "url": "https://arxiv.org/pdf/2407.20309v2",
        "sha256": "ef21353629a777f796e1d968eb5ad59fecde30934fe7ef7fca82cfeb5f23b392",
        "filename": "yang_et_al_2024_arxiv_2407.20309v2.pdf",
    },
    "barklem_collet_readme": {
        "url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/588/A96/ReadMe",
        "sha256": "e7a3b2c930971bed0945e50ba45e0d94dbb3030aca0391bc10968016c17c242a",
        "filename": "barklem_collet_2016_ReadMe.txt",
    },
    "barklem_collet_ionization": {
        "url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/588/A96/table4.dat",
        "sha256": "722c36a3294fc0d3d599145b2b19b502f9fa85362541a6acd789c098668632cf",
        "filename": "barklem_collet_2016_table4.dat",
    },
    "barklem_collet_partitions": {
        "url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/588/A96/table8.dat",
        "sha256": "c08800be067ccbbcc5486c61085569da153fbebcaccfd4bef3d3e203240846be",
        "filename": "barklem_collet_2016_table8.dat",
    },
    "lightweaver_background": {
        "url": (
            "https://raw.githubusercontent.com/Goobley/Lightweaver/"
            "d33058f8857acf28d9187e778b6f41e49d3aad95/Source/Background.cpp"
        ),
        "sha256": "d4d0f92e28ebf0dc5f82f4ffa7e6798acee9363023e2727689ca6e875031f048",
        "filename": "lightweaver_Background_d33058f.cpp",
    },
    "nist_fe1_lines": {
        "url": (
            "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?"
            "spectra=Fe%20I&low_w=6300.8&upp_w=6303.2&unit=0&de=0&format=2&"
            "line_out=0&en_unit=1&output=0&bibrefs=1&page_size=100&show_obs_wl=1&"
            "show_calc_wl=1&unc_out=1&order_out=0&show_av=2&A_out=0&f_out=1&"
            "S_out=1&loggf_out=1&intens_out=1&allowed_out=1&forbid_out=1&"
            "conf_out=1&term_out=1&enrg_out=1&J_out=1&g_out=1"
        ),
        "sha256": "b392a86c9f61c589cf68541fae983f12ea77cefe4c286d15b40eb15acfa54ccb",
        "filename": "nist_asd_5.12_fe1_6300.8_6303.2_lines.csv",
    },
    "nist_fe1_levels": {
        "url": (
            "https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0&spectrum=Fe%20I&"
            "units=0&format=2&output=0&page_size=5000&multiplet_ordered=1&"
            "conf_out=on&term_out=on&level_out=on&unc_out=1&j_out=on&g_out=on&"
            "lande_out=on&biblio=on"
        ),
        "sha256": "66736300d491520119991512e54ff05435e649b78055ddc146b7e90c5323d8a7",
        "filename": "nist_asd_5.12_fe1_levels.csv",
    },
    "hinode_sp_prep": {
        "url": (
            "https://sohoftp.nascom.nasa.gov/solarsoft/hinode/sot/idl/sp/util/"
            "sp_prep.pro"
        ),
        "sha256": "63bd10f742fae21cb32f62fb11abb29f00c82c2445eb8fabd65976f7d22f60a9",
        "filename": "hinode_sp_prep_2026-08-19.pro",
    },
    "sir2015_blends2": {
        "url": (
            "https://raw.githubusercontent.com/cdiazbas/SIRcode/"
            "def666eccddc92c0c2d0d3e72eaeb4b96939f8de/SIR2015/blends2.f"
        ),
        "sha256": "38c5e8344c6646db1349d41a34674565aca00faa4919d75f61248912bac2ad17",
        "filename": "sir2015_blends2_def666e.f",
    },
    # STiC is pinned to one immutable commit.  The Python Wittmann port is an
    # offline table generator only: none of this iterative code is imported
    # by training or synthesis.
    "stic_witt_eos": {
        "url": (
            "https://raw.githubusercontent.com/jaimedelacruz/stic/"
            "18cda77d038a97f007a783dcb61ea9a9a1244bf7/"
            "pythontools/py2/witt.py"
        ),
        "sha256": "49e62261302f45be29af287f1ea7c2eae33807d484ce6b54dca2bd3e3ee32394",
        "filename": "stic_18cda77_witt.py",
    },
    "stic_kurucz_partition_functions": {
        "url": (
            "https://raw.githubusercontent.com/jaimedelacruz/stic/"
            "18cda77d038a97f007a783dcb61ea9a9a1244bf7/"
            "pythontools/py2/pf_Kurucz.input"
        ),
        "sha256": "a7d15b25406b2736331253b6457d8ef4cfa3f3ef2275462120999121b8a734a8",
        "filename": "stic_18cda77_pf_Kurucz.input",
    },
    "stic_abundances": {
        "url": (
            "https://raw.githubusercontent.com/jaimedelacruz/stic/"
            "18cda77d038a97f007a783dcb61ea9a9a1244bf7/"
            "input/Atoms/abundance.input"
        ),
        "sha256": "6ed3664858107e423eac59a01d65735940c67ff7fd75afebdd2d0887a4108124",
        "filename": "stic_18cda77_abundance.input",
    },
    "stic_falc_82": {
        "url": (
            "https://raw.githubusercontent.com/jaimedelacruz/stic/"
            "18cda77d038a97f007a783dcb61ea9a9a1244bf7/"
            "input/Atmos/FALC_82.atmos"
        ),
        "sha256": "7725352edeec3c29efa551db8bf61e7ff7a43760b4ec6bb017392ab71b4e1697",
        "filename": "stic_18cda77_FALC_82.atmos",
    },
    "stic_fts_disk_center": {
        "url": (
            "https://raw.githubusercontent.com/jaimedelacruz/stic/"
            "18cda77d038a97f007a783dcb61ea9a9a1244bf7/"
            "pythontools/py2/fts_disk_center.idlsave"
        ),
        "sha256": "1332b2b9a10c390171dac6a7cc47d4d3e4590d89d9f122124a9b971125ad8550",
        "filename": "stic_18cda77_fts_disk_center.idlsave",
    },
    "neckel_2005_limb_darkening": {
        "url": "https://doi.org/10.1007/s11207-005-4081-z",
        "sha256": "citation-only:10.1007/s11207-005-4081-z",
        "filename": "neckel_2005_limb_darkening_citation",
    },
}


STIC_COMMIT = "18cda77d038a97f007a783dcb61ea9a9a1244bf7"
STIC_SOURCE_IDS = (
    "stic_witt_eos",
    "stic_kurucz_partition_functions",
    "stic_abundances",
    "stic_falc_82",
)
SOLAR_REFERENCE_SOURCE_IDS = ("stic_fts_disk_center",)
BUNDLE_INPUT_SOURCE_IDS = (
    "barklem_collet_readme",
    "barklem_collet_ionization",
    "barklem_collet_partitions",
    "lightweaver_background",
    *STIC_SOURCE_IDS,
    *SOLAR_REFERENCE_SOURCE_IDS,
)
CITATION_ONLY_SOURCE_IDS = tuple(
    source_id for source_id in SOURCES if source_id not in BUNDLE_INPUT_SOURCE_IDS
)
SOURCE_ROLES = {
    "bundle_input_ids": list(BUNDLE_INPUT_SOURCE_IDS),
    "citation_only_ids": list(CITATION_ONLY_SOURCE_IDS),
    "citation_only_contract": (
        "reviewed provenance only; these mutable endpoints are not downloaded or "
        "required to reproduce the generated runtime tables"
    ),
}
REVIEWED_RESOURCE_FILENAMES = (
    "lines.json",
    "abundances.json",
    "blend_inventory.json",
    "instrument_hinode_sp.json",
)
STIC_CONTINUUM_WAVELENGTHS_ANGSTROM = (5000.0, 6290.0, 6300.0, 6310.0, 6320.0)
SOLAR_REFERENCE_WINDOW_ANGSTROM = (6300.0, 6304.0)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


USER_AGENT = "pinn-me-lte-data-fetch/0.1 (https://github.com/RobertJaro/pinn-me)"


def _write_if_changed(destination: Path, payload: bytes) -> bool:
    """Write one artifact only when its verified content changed."""

    if destination.is_file() and destination.read_bytes() == payload:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def _default_cache_directory() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "pinn-me" / "lte" / "sources-v1"


def _fetch(
    name: str,
    source: dict[str, str],
    cache_dir: Path | None,
    *,
    refresh: bool = False,
) -> bytes:
    cached = None if cache_dir is None else cache_dir / source["filename"]
    if cached is not None and cached.exists() and not refresh:
        payload = cached.read_bytes()
        if _sha256(payload) == source["sha256"]:
            LOGGER.info(
                "Cache hit: %s (%s, %.1f KiB)",
                name,
                cached,
                len(payload) / 1024.0,
            )
            return payload
        LOGGER.warning(
            "Ignoring checksum-mismatched cache entry for %s: %s",
            name,
            cached,
        )
    LOGGER.info("Downloading %s from %s", name, source["url"])
    request = Request(source["url"], headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read()
    except Exception as error:
        raise RuntimeError(
            f"Failed to download pinned LTE source {name!r} from {source['url']}"
        ) from error
    digest = _sha256(payload)
    if digest != source["sha256"]:
        raise RuntimeError(
            f"SHA256 mismatch for {source['url']}: expected {source['sha256']}, got {digest}"
        )
    if cached is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_if_changed(cached, payload)
    LOGGER.info(
        "Downloaded and verified: %s (%s, %.1f KiB)",
        name,
        source["filename"],
        len(payload) / 1024.0,
    )
    return payload


def _fetch_all(cache_dir: Path, workers: int, refresh: bool) -> dict[str, bytes]:
    """Fetch only byte-for-byte inputs needed to generate a runtime bundle.

    Citation-only URLs remain pinned in :data:`SOURCES` and reviewed resource
    provenance, but mutable HTML/CGI endpoints are not live build dependencies.
    """

    payloads = {}
    source_count = len(BUNDLE_INPUT_SOURCE_IDS)
    LOGGER.info(
        "Preparing %d checksum-pinned generation inputs with %d worker(s)",
        source_count,
        min(workers, source_count),
    )
    with ThreadPoolExecutor(
        max_workers=min(workers, len(BUNDLE_INPUT_SOURCE_IDS))
    ) as executor:
        futures = {
            executor.submit(
                _fetch, name, source, cache_dir, refresh=refresh
            ): name
            for name, source in SOURCES.items()
            if name in BUNDLE_INPUT_SOURCE_IDS
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            payloads[name] = future.result()
            LOGGER.info("Source progress: %d/%d ready (%s)", completed, source_count, name)
    LOGGER.info("All %d generation inputs are checksum-verified", source_count)
    return payloads


def _verify_source_payloads(payloads: dict[str, bytes], source_ids) -> None:
    """Reject a generator input unless it is one of the pinned source bytes."""

    for source_id in source_ids:
        if source_id not in payloads:
            raise KeyError(f"Missing downloaded LTE source: {source_id!r}")
        actual = _sha256(payloads[source_id])
        expected = SOURCES[source_id]["sha256"]
        if actual != expected:
            raise RuntimeError(
                f"Source payload {source_id!r} failed SHA256 verification before "
                f"table generation: expected {expected}, got {actual}."
            )


@contextmanager
def _loaded_stic_witt(payloads: dict[str, bytes]):
    """Load the pinned STiC table generator in an isolated temporary module.

    STiC's Python port imports itself as ``witt`` and locates the Kurucz XDR
    file beside ``witt.py``.  Recreating that verified source layout in a
    temporary directory keeps the generated runtime bundle independent of a
    STiC installation while ensuring training never imports the iterative
    solver.
    """

    _verify_source_payloads(payloads, STIC_SOURCE_IDS)
    with tempfile.TemporaryDirectory(prefix="pinn-me-stic-") as temporary:
        source_root = Path(temporary)
        (source_root / "witt.py").write_bytes(payloads["stic_witt_eos"])
        (source_root / "pf_Kurucz.input").write_bytes(
            payloads["stic_kurucz_partition_functions"]
        )
        old_module = sys.modules.pop("witt", None)
        sys.path.insert(0, str(source_root))
        try:
            try:
                module = importlib.import_module("witt")
            except Exception as error:
                raise RuntimeError(
                    "The pinned STiC Wittmann EOS/continuum generator could not be loaded."
                ) from error
            yield module
        finally:
            sys.modules.pop("witt", None)
            try:
                sys.path.remove(str(source_root))
            except ValueError:
                pass
            if old_module is not None:
                sys.modules["witt"] = old_module


def _stic_abundance_ratios(payload: bytes) -> np.ndarray:
    """Parse STiC/RH ``abundance.input`` in atomic-number order."""

    values = []
    for raw_line in payload.decode("ascii").splitlines():
        fields = raw_line.split("#", 1)[0].split()
        if len(fields) < 2:
            continue
        try:
            log_epsilon = float(fields[1])
        except ValueError:
            continue
        values.append(10.0 ** (log_epsilon - 12.0))
    if len(values) != 99:
        raise RuntimeError(
            f"Expected 99 atomic abundances in STiC abundance.input, found {len(values)}."
        )
    if not np.isclose(values[0], 1.0, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("STiC abundance.input does not begin with log epsilon(H)=12.")
    return np.asarray(values, dtype=np.float64)


def _parse_falc_mass_scale(payload: bytes) -> tuple[float, np.ndarray, np.ndarray]:
    """Return FALC log(g), log column mass, and temperature from the text model."""

    lines = payload.decode("ascii").splitlines()
    try:
        depth_marker = next(i for i, line in enumerate(lines) if "Ndep" in line)
        ndep = int(next(
            line.strip()
            for line in lines[depth_marker + 1:]
            if line.strip() and not line.lstrip().startswith("*")
        ))
        gravity_marker = next(i for i, line in enumerate(lines) if "log g" in line)
        log_gravity = float(next(
            line.strip()
            for line in lines[gravity_marker + 1:]
            if line.strip() and not line.lstrip().startswith("*")
        ))
        table_marker = next(i for i, line in enumerate(lines) if "lg column Mass" in line)
    except (StopIteration, ValueError) as error:
        raise RuntimeError("Could not parse the pinned STiC FALC_82 atmosphere header.") from error
    rows = []
    for line in lines[table_marker + 1:]:
        fields = line.split()
        if len(fields) != 5:
            if rows and len(rows) == ndep:
                break
            continue
        try:
            rows.append(tuple(float(value) for value in fields))
        except ValueError:
            continue
        if len(rows) == ndep:
            break
    if len(rows) != ndep:
        raise RuntimeError(f"FALC_82 declares {ndep} depths but yielded {len(rows)} rows.")
    array = np.asarray(rows, dtype=np.float64)
    if np.any(np.diff(array[:, 0]) <= 0.0):
        raise RuntimeError("FALC_82 column mass is not strictly increasing.")
    return log_gravity, array[:, 0], array[:, 1]


def _falc_reference_boundary(
    solver,
    falc_payload: bytes,
    *,
    target_log_tau500: float = -5.0,
) -> dict:
    """Reproduce STiC's FALC column-mass to total-tau500 conversion."""

    log_gravity, log_column_mass, temperature = _parse_falc_mass_scale(falc_payload)
    gravity_cgs = 10.0**log_gravity
    column_mass_g_cm2 = 10.0**log_column_mass
    gas_pressure_dyn_cm2 = gravity_cgs * column_mass_g_cm2
    density_g_cm3 = np.empty_like(temperature)
    extinction_cm1 = np.empty_like(temperature)
    for index, (temp, pressure) in enumerate(zip(temperature, gas_pressure_dyn_cm2)):
        electron_pressure, electron_fraction = solver.pe_from_pg(
            temp, pressure, get_fe=True
        )
        density_g_cm3[index] = (
            electron_pressure
            * solver.rho_from_H
            / (electron_fraction * temp)
        )
        total, _ = solver.contOpacity(
            temp, pressure, electron_pressure, np.asarray([5000.0]), True
        )
        extinction_cm1[index] = total[0]
    mass_extinction_cm2_g = extinction_cm1 / density_g_cm3
    tau500 = np.empty_like(temperature)
    # This is the boundary and trapezoid convention used by
    # ceos::hydrostatic_cmass at the pinned commit.
    tau500[0] = column_mass_g_cm2[0] * mass_extinction_cm2_g[0]
    increments = np.diff(column_mass_g_cm2) * 0.5 * (
        mass_extinction_cm2_g[1:] + mass_extinction_cm2_g[:-1]
    )
    tau500[1:] = tau500[0] + np.cumsum(increments)
    log_tau500 = np.log10(tau500)
    if not np.all(np.diff(log_tau500) > 0.0):
        raise RuntimeError("The STiC FALC tau500 scale is not strictly increasing.")
    if not (log_tau500[0] <= target_log_tau500 <= log_tau500[-1]):
        raise RuntimeError("Requested top boundary lies outside the STiC FALC tau500 scale.")
    log_pressure_pa = np.log10(gas_pressure_dyn_cm2 * 0.1)
    bracket_upper = int(np.searchsorted(log_tau500, target_log_tau500))
    bracket_lower = max(0, bracket_upper - 1)
    boundary_pressure_pa = float(10.0 ** np.interp(
        target_log_tau500, log_tau500, log_pressure_pa
    ))
    boundary_temperature = float(np.interp(
        target_log_tau500, log_tau500, temperature
    ))
    boundary_pressure_cgs = boundary_pressure_pa * 10.0
    boundary_electron_pressure, boundary_electron_fraction = solver.pe_from_pg(
        boundary_temperature, boundary_pressure_cgs, get_fe=True
    )
    boundary_density_kg_m3 = (
        boundary_electron_pressure
        * solver.rho_from_H
        / (boundary_electron_fraction * boundary_temperature)
        * 1000.0
    )
    boundary_total_cm1, boundary_scattering_cm1 = solver.contOpacity(
        boundary_temperature,
        boundary_pressure_cgs,
        boundary_electron_pressure,
        np.asarray([5000.0]),
        True,
    )
    boundary_total_m1 = float(boundary_total_cm1[0] * 100.0)
    boundary_scattering_m1 = float(boundary_scattering_cm1[0] * 100.0)
    boundary_metric_m_per_dex = (
        math.log(10.0) * 10.0**target_log_tau500 / boundary_total_m1
    )
    boundary_dlog_pressure_dq = (
        boundary_density_kg_m3
        * (gravity_cgs / 100.0)
        * boundary_metric_m_per_dex
        / boundary_pressure_pa
    )
    return {
        "model": "FALC_82",
        "target_log10_tau500": float(target_log_tau500),
        "gas_pressure_pa": boundary_pressure_pa,
        "temperature_k": boundary_temperature,
        "log10_column_mass_g_cm2": float(np.interp(
            target_log_tau500, log_tau500, log_column_mass
        )),
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
            "total_extinction_m1": boundary_total_m1,
            "true_absorption_m1": float(
                boundary_total_m1 - boundary_scattering_m1
            ),
            "scattering_extinction_m1": boundary_scattering_m1,
            "mass_density_kg_m3": float(boundary_density_kg_m3),
            "mass_extinction_m2_kg": float(
                boundary_total_m1 / boundary_density_kg_m3
            ),
            "required_metric_m_per_log10_tau": float(
                boundary_metric_m_per_dex
            ),
            "dlog_pressure_dlog10_tau": float(boundary_dlog_pressure_dq),
            "pressure_efold_width_log10_tau": float(
                1.0 / boundary_dlog_pressure_dq
            ),
        },
    }


def _write_stic_continuum_table(
    payloads: dict[str, bytes],
    destination: Path,
    *,
    temperature_count: int = 129,
    pressure_count: int = 173,
) -> None:
    """Generate the differentiable-runtime STiC continuum reference table.

    The expensive/iterative Wittmann EOS is evaluated only here.  Stored
    quantities are logarithms on a regular ``(log10(T), log10(Pgas))`` grid,
    so runtime evaluation is a fixed differentiable interpolation.
    """

    if temperature_count < 4 or pressure_count < 4:
        raise ValueError("STiC continuum axes require at least four points for cubic lookup.")
    _verify_source_payloads(payloads, STIC_SOURCE_IDS)
    log_temperature = np.linspace(3.4, 4.0, temperature_count, dtype=np.float64)
    log_pressure = np.linspace(-1.5, 6.0, pressure_count, dtype=np.float64)
    wavelengths = np.asarray(STIC_CONTINUUM_WAVELENGTHS_ANGSTROM, dtype=np.float64)
    shape = (temperature_count, pressure_count)
    log_absorption = np.empty((*shape, wavelengths.size), dtype=np.float64)
    log_scattering = np.empty_like(log_absorption)
    log_density = np.empty(shape, dtype=np.float64)
    log_electron_density = np.empty(shape, dtype=np.float64)
    log_hydrogen_neutral_density = np.empty(shape, dtype=np.float64)
    log_fe_i_population_over_partition = np.empty(shape, dtype=np.float64)
    abundance_ratios = _stic_abundance_ratios(payloads["stic_abundances"])

    LOGGER.info(
        "Generating STiC lookup: %d temperature rows x %d pressure points x %d wavelengths",
        temperature_count,
        pressure_count,
        wavelengths.size,
    )
    started = time.monotonic()
    progress_interval = max(1, math.ceil(temperature_count / 10))
    with _loaded_stic_witt(payloads) as stic_witt:
        solver = stic_witt.witt(abund_init=abundance_ratios.copy())
        for temperature_index, log_t in enumerate(log_temperature):
            temperature = 10.0**log_t
            for pressure_index, log_p in enumerate(log_pressure):
                pressure_pa = 10.0**log_p
                pressure_dyn_cm2 = pressure_pa * 10.0
                electron_pressure, electron_fraction = solver.pe_from_pg(
                    temperature, pressure_dyn_cm2, get_fe=True
                )
                # Export the physical perturber and line-reservoir populations
                # from the exact same converged Wittmann state used below by
                # ``contOpacity``.  ``getBackgroundPartials`` defaults to
                # divide_by_u=True, for which its H-I entry is n(H I)/U(H I)
                # = n(H I)/2 rather than the neutral-atom density required by
                # ABO broadening.  The direct ``gasc`` expression is the
                # corresponding undivided physical population.
                _, hydrogen_partials = solver.gasc(temperature, electron_pressure)
                hydrogen_neutral_cm3 = (
                    hydrogen_partials[solver.ncontr]
                    * hydrogen_partials[solver.ncontr + 4]
                    / (solver.BK * temperature)
                )
                fe_ion_populations_cm3, fe_partition_functions = solver.getXparts(
                    25,
                    temperature,
                    pressure_dyn_cm2,
                    electron_pressure,
                    divide_by_u=False,
                    return_u=True,
                )
                fe_i_population_over_partition_cm3 = (
                    fe_ion_populations_cm3[0] / fe_partition_functions[0]
                )
                density_g_cm3 = (
                    electron_pressure
                    * solver.rho_from_H
                    / (electron_fraction * temperature)
                )
                total_cm1, scattering_cm1 = solver.contOpacity(
                    temperature,
                    pressure_dyn_cm2,
                    electron_pressure,
                    wavelengths,
                    True,
                )
                absorption_cm1 = total_cm1 - scattering_cm1
                if (
                    not np.all(np.isfinite(absorption_cm1))
                    or not np.all(np.isfinite(scattering_cm1))
                    or np.any(absorption_cm1 <= 0.0)
                    or np.any(scattering_cm1 <= 0.0)
                    or not math.isfinite(density_g_cm3)
                    or density_g_cm3 <= 0.0
                    or electron_pressure <= 0.0
                    or not math.isfinite(hydrogen_neutral_cm3)
                    or hydrogen_neutral_cm3 <= 0.0
                    or not math.isfinite(fe_i_population_over_partition_cm3)
                    or fe_i_population_over_partition_cm3 <= 0.0
                ):
                    raise RuntimeError(
                        "STiC returned a non-positive or non-finite continuum/EOS state "
                        f"at T={temperature:g} K, Pgas={pressure_pa:g} Pa."
                    )
                # cm^-1 -> m^-1, g cm^-3 -> kg m^-3, cm^-3 -> m^-3.
                log_absorption[temperature_index, pressure_index] = np.log10(
                    absorption_cm1 * 100.0
                )
                log_scattering[temperature_index, pressure_index] = np.log10(
                    scattering_cm1 * 100.0
                )
                log_density[temperature_index, pressure_index] = math.log10(
                    density_g_cm3 * 1000.0
                )
                electron_density_cm3 = electron_pressure / (
                    solver.BK * temperature
                )
                log_electron_density[temperature_index, pressure_index] = math.log10(
                    electron_density_cm3 * 1.0e6
                )
                log_hydrogen_neutral_density[
                    temperature_index, pressure_index
                ] = math.log10(hydrogen_neutral_cm3 * 1.0e6)
                log_fe_i_population_over_partition[
                    temperature_index, pressure_index
                ] = math.log10(fe_i_population_over_partition_cm3 * 1.0e6)
            completed = temperature_index + 1
            if (
                completed == 1
                or completed % progress_interval == 0
                or completed == temperature_count
            ):
                LOGGER.info(
                    "STiC lookup progress: %d/%d temperature rows (%.0f%%, %.1f min)",
                    completed,
                    temperature_count,
                    100.0 * completed / temperature_count,
                    (time.monotonic() - started) / 60.0,
                )
        top_boundary = _falc_reference_boundary(
            solver, payloads["stic_falc_82"], target_log_tau500=-5.0
        )

    document = {
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
        "log10_neutral_hydrogen_density_m3": (
            log_hydrogen_neutral_density.tolist()
        ),
        "log10_fe_i_population_over_partition_m3": (
            log_fe_i_population_over_partition.tolist()
        ),
        "interpolation": (
            "tensor-product cubic in log10(T) and log10(Pgas), then linear in "
            "vacuum wavelength; logarithmic coefficients are interpolated before "
            "exponentiation"
        ),
        "validated_wavelength_domains_angstrom": [[5000.0, 5000.0], [6290.0, 6320.0]],
        "reference_solver": {
            "project": "STiC",
            "repository": "https://github.com/jaimedelacruz/stic",
            "commit": STIC_COMMIT,
            "implementation": "pythontools/py2/witt.py (Wittmann EOS and cop continuum)",
            "runtime_iterations": 0,
            "source_sha256": {
                source_id: SOURCES[source_id]["sha256"] for source_id in STIC_SOURCE_IDS
            },
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
            "legacy_eos_scope": (
                "The separately packaged Barklem/Saha EOS remains available for atomic "
                "diagnostics and the inspectable H-minus-only continuum decomposition; "
                "production continuum, configured MHS, damping perturbers, and Fe-I line "
                "populations use this pinned STiC table."
            ),
        },
        "falc_top_boundary": top_boundary,
    }
    _write_if_changed(
        destination,
        (json.dumps(document, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    LOGGER.info(
        "STiC lookup written: %s (%.1f min)",
        destination,
        (time.monotonic() - started) / 60.0,
    )


def _cpp_array(source: str, name: str) -> list[float]:
    pattern = rf"static constexpr double\s+{re.escape(name)}\s*\[[^]]+\]\s*=\s*\{{(.*?)\}};"
    match = re.search(pattern, source, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"Could not locate C++ coefficient array {name!r}")
    body = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.DOTALL)
    return [float(token) for token in body.replace("\n", " ").split(",") if token.strip()]


def _write_hminus_table(background: bytes, destination: Path) -> None:
    source = background.decode("utf-8")
    wavelength_bf_nm = _cpp_array(source, "lambdaBF")
    cross_section_bf_1e21_m2 = _cpp_array(source, "alphaBF")
    wavelength_ff_nm = _cpp_array(source, "lambdaFF")
    theta_ff = _cpp_array(source, "thetaFF")
    flat_kappa = _cpp_array(source, "kappaFF")
    if len(flat_kappa) != len(wavelength_ff_nm) * len(theta_ff):
        raise RuntimeError("Unexpected Stilley--Callaway free-free table dimensions")
    kappa_ff = [
        flat_kappa[index:index + len(theta_ff)]
        for index in range(0, len(flat_kappa), len(theta_ff))
    ]
    document = {
        "schema_version": 1,
        "bound_free": {
            "wavelength_nm": wavelength_bf_nm,
            "cross_section_1e-21_m2": cross_section_bf_1e21_m2,
            "reference": "Geltman 1962, ApJ 136, 935; Mihalas 1978, p. 102",
            "interpolation": "linear in wavelength; zero outside the tabulated photodetachment interval",
        },
        "free_free": {
            "wavelength_nm": wavelength_ff_nm,
            "theta_5040_over_T": theta_ff,
            "kappa_1e-29_m5_per_J": kappa_ff,
            "reference": (
                "Stilley & Callaway 1970, ApJ 160, 245; Mihalas 1978, p. 102; "
                "Mathisen 1984, Master's thesis, p. 17"
            ),
            "interpolation": "bilinear in wavelength and theta=5040/T with endpoint theta clipping",
        },
        "extracted_from": {
            "project": "Lightweaver v0.16.2",
            "commit": "d33058f8857acf28d9187e778b6f41e49d3aad95",
            "path": "Source/Background.cpp",
            "url": SOURCES["lightweaver_background"]["url"],
            "sha256": SOURCES["lightweaver_background"]["sha256"],
            "license": "MIT; see https://github.com/Goobley/Lightweaver/blob/v0.16.2/LICENSE",
        },
    }
    _write_if_changed(
        destination,
        (json.dumps(document, indent=2) + "\n").encode("utf-8"),
    )


def _write_partition_table(raw_table: bytes, destination: Path) -> None:
    """Convert the CDS fixed-width text table into fast runtime JSON once."""

    raw_lines = raw_table.decode("ascii").splitlines()
    temperature_line = next(line for line in raw_lines if "T [K]" in line)
    temperatures = [float(token) for token in temperature_line.split()[3:]]
    values = {}
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        row = [float(value) for value in fields[1:]]
        if len(row) != len(temperatures):
            raise RuntimeError(
                f"Partition row {fields[0]!r} has {len(row)} values; "
                f"expected {len(temperatures)}."
            )
        values[fields[0]] = row
    document = {
        "schema_version": 1,
        "source": {
            "id": "barklem_collet_partitions",
            "url": SOURCES["barklem_collet_partitions"]["url"],
            "sha256": SOURCES["barklem_collet_partitions"]["sha256"],
        },
        "temperature_k": temperatures,
        "partition_functions": values,
        "interpolation": "log(U) linear in temperature; constant endpoint extrapolation",
    }
    _write_if_changed(
        destination,
        (json.dumps(document, indent=2) + "\n").encode("utf-8"),
    )


def _write_eos_table(data_directory: Path) -> str:
    """Generate the charge-neutrality table from already verified inputs."""

    database = AtomicDatabase(data_directory=data_directory)
    document = LTEEOS.generate_table_document(database)
    destination = data_directory / "eos_table.json"
    _write_if_changed(
        destination,
        (json.dumps(document, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    return _sha256(destination.read_bytes())


def _has_current_stic_population_schema(path: Path) -> bool:
    """Return whether a prepared STiC table satisfies the production schema."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return document.get("schema_version") == 2 and all(
        isinstance(document.get(field), list)
        for field in (
            "log10_neutral_hydrogen_density_m3",
            "log10_fe_i_population_over_partition_m3",
        )
    )


def prepare_eos_table(
    data_directory: Path,
    *,
    allow_incomplete_stic_bundle: bool = False,
) -> dict:
    """Generate and register an EOS table in an existing verified data set.

    ``allow_incomplete_stic_bundle`` is private preparation plumbing used only
    while a full bundle is being assembled.  The public ``--eos-only`` path
    refuses to bless a legacy H-minus-only resource directory as production
    physical-height input.
    """

    data_directory = Path(data_directory).expanduser().resolve()
    manifest_path = data_directory / "sources.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"No LTE source manifest exists in {data_directory}. EOS-only preparation "
            "cannot create the required pinned STiC continuum/FALC resources; run "
            f"`pinn-me-lte-fetch-data --output-dir {data_directory}` for a full bundle."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not allow_incomplete_stic_bundle:
        sources = manifest.get("sources", {})
        generated = manifest.get("generated_files", {})
        stic_path = data_directory / "stic_continuum_table.json"
        stic_ready = (
            all(source_id in sources for source_id in STIC_SOURCE_IDS)
            and "stic_continuum_table.json" in generated
            and _has_current_stic_population_schema(stic_path)
        )
        if not stic_ready:
            raise RuntimeError(
                "EOS-only preparation cannot upgrade this legacy resource directory: "
                "physical tau500 now requires the pinned STiC total-continuum table and "
                "FALC boundary. Run the full command with downloads enabled: "
                f"`pinn-me-lte-fetch-data --output-dir {data_directory}`."
            )
    LOGGER.info("Generating optional Barklem/Saha reference EOS table")
    manifest.setdefault("generated_files", {})["eos_table.json"] = {
        "sha256": _write_eos_table(data_directory)
    }
    _write_if_changed(
        manifest_path,
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
    )
    bundle_path = data_directory / "bundle.json"
    if bundle_path.is_file():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        bundle["source_manifest_sha256"] = _sha256(manifest_path.read_bytes())
        bundle["runtime_files"] = sorted(
            [
                *manifest.get("vendored_files", {}),
                *manifest.get("generated_files", {}),
                *manifest.get("reviewed_files", {}),
                "sources.json",
            ]
        )
        _write_if_changed(
            bundle_path,
            (json.dumps(bundle, indent=2) + "\n").encode("utf-8"),
        )
    LOGGER.info("Reference EOS table ready: %s", data_directory / "eos_table.json")
    return manifest


def prepare_stic_continuum_table(
    data_directory: Path,
    payloads: dict[str, bytes],
) -> dict:
    """Generate and register the pinned STiC total-continuum lookup.

    This preparation-only operation runs the verified iterative STiC/Wittmann
    solver.  Runtime code sees only the resulting differentiable table.
    """

    data_directory = Path(data_directory).expanduser().resolve()
    manifest_path = data_directory / "sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = data_directory / "stic_continuum_table.json"
    LOGGER.info("Starting pinned STiC/Wittmann table conversion")
    _write_stic_continuum_table(
        payloads,
        destination,
    )
    manifest["sources"] = SOURCES
    manifest["source_roles"] = SOURCE_ROLES
    manifest.setdefault("generated_files", {})[destination.name] = {
        "sha256": _sha256(destination.read_bytes())
    }
    _write_if_changed(
        manifest_path,
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
    )
    bundle_path = data_directory / "bundle.json"
    if bundle_path.is_file():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        table_document = json.loads(destination.read_text(encoding="utf-8"))
        bundle["source_manifest_sha256"] = _sha256(manifest_path.read_bytes())
        bundle["runtime_files"] = sorted(
            [
                *manifest.get("vendored_files", {}),
                *manifest.get("generated_files", {}),
                *manifest.get("reviewed_files", {}),
                "sources.json",
            ]
        )
        bundle["physical_depth_contract"] = table_document["continuum_contract"]
        bundle["falc_top_boundary"] = table_document["falc_top_boundary"]
        bundle["required_production_files"] = sorted(
            (
                "abundances.json",
                "blend_inventory.json",
                "instrument_hinode_sp.json",
                "lines.json",
                "stic_continuum_table.json",
            )
        )
        bundle["optional_reference_files"] = sorted(
            (
                "barklem_collet_2016_ReadMe.txt",
                "barklem_collet_2016_table4.dat",
                "barklem_collet_2016_table8.dat",
                "eos_table.json",
                "hminus_continuum.json",
                "partition_functions.json",
            )
        )
        _write_if_changed(
            bundle_path,
            (json.dumps(bundle, indent=2) + "\n").encode("utf-8"),
        )
    return manifest


def prepare_solar_reference(
    data_directory: Path,
    payloads: dict[str, bytes],
) -> dict:
    """Extract a compact absolute 630 nm reference from the pinned FTS atlas."""

    from scipy.io import readsav

    _verify_source_payloads(payloads, SOLAR_REFERENCE_SOURCE_IDS)
    data_directory = Path(data_directory).expanduser().resolve()
    manifest_path = data_directory / "sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_payload = payloads["stic_fts_disk_center"]
    with tempfile.NamedTemporaryFile(suffix=".idlsave") as handle:
        handle.write(source_payload)
        handle.flush()
        atlas = readsav(handle.name)
    wavelength = np.asarray(atlas["ftswav"], dtype=np.float64)
    intensity = np.asarray(atlas["ftsint"], dtype=np.float64)
    continuum = np.asarray(atlas["ftscnt"], dtype=np.float64)
    lower, upper = SOLAR_REFERENCE_WINDOW_ANGSTROM
    selected = (wavelength >= lower) & (wavelength <= upper)
    if (
        np.count_nonzero(selected) < 2
        or not np.isfinite(intensity[selected]).all()
        or not np.isfinite(continuum[selected]).all()
        or np.any(intensity[selected] <= 0)
        or np.any(continuum[selected] <= 0)
    ):
        raise RuntimeError("Pinned STiC FTS atlas has no valid 630 nm reference window.")
    # The STiC atlas stores W cm^-2 sr^-1 A^-1. Converting area and spectral
    # density gives 1e4 * 1e10 = 1e14 W m^-3 sr^-1.
    conversion = 1.0e14
    document = {
        "schema_version": 1,
        "source": {
            "description": (
                "Absolute Kitt Peak FTS disk-center solar intensity and fitted "
                "continuum distributed with STiC"
            ),
            "repository": "https://github.com/jaimedelacruz/stic",
            "commit": STIC_COMMIT,
            "file": "pythontools/py2/fts_disk_center.idlsave",
            "sha256": SOURCES["stic_fts_disk_center"]["sha256"],
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
                "Neckel (2005), Solar Physics 229, 13-33, "
                "doi:10.1007/s11207-005-4081-z"
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
    destination = data_directory / "solar_reference_630nm.json"
    _write_if_changed(
        destination, (json.dumps(document, indent=2) + "\n").encode("utf-8")
    )
    manifest["sources"] = SOURCES
    manifest["source_roles"] = SOURCE_ROLES
    manifest.setdefault("generated_files", {})[destination.name] = {
        "sha256": _sha256(destination.read_bytes())
    }
    _write_if_changed(
        manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )
    return manifest


def _current_sealed_bundle(
    output_directory: Path,
    *,
    reviewed_source: Path | None = None,
) -> dict | None:
    """Return a fully current sealed bundle without touching source downloads."""

    output_directory = Path(output_directory).expanduser().resolve()
    reviewed_source = (
        Path(__file__).resolve().parent / "data"
        if reviewed_source is None
        else Path(reviewed_source)
    )
    bundle_path = output_directory / "bundle.json"
    manifest_path = output_directory / "sources.json"
    if not bundle_path.is_file() or not manifest_path.is_file():
        return None
    try:
        from pme.lte.resources import validate_resource_bundle

        validate_resource_bundle(output_directory)
        sealed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reviewed_match = all(
            (reviewed_source / filename).is_file()
            and _sha256((reviewed_source / filename).read_bytes())
            == sealed_manifest.get("reviewed_files", {})
            .get(filename, {})
            .get("sha256")
            for filename in REVIEWED_RESOURCE_FILENAMES
        )
        if (
            sealed_manifest.get("sources") == SOURCES
            and sealed_manifest.get("source_roles") == SOURCE_ROLES
            and reviewed_match
        ):
            return json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError):
        return None
    return None


def prepare_bundle(
    output_directory: Path,
    payloads: dict[str, bytes],
    *,
    reviewed_source: Path | None = None,
    force_regenerate: bool = False,
) -> dict:
    """Convert verified source payloads into one reusable runtime bundle.

    A checksum-valid sealed bundle made from the same pinned inputs and reviewed
    files is returned unchanged.  Set ``force_regenerate`` only when an explicit
    deterministic rebuild is desired.
    """

    reviewed_source = (
        Path(__file__).resolve().parent / "data"
        if reviewed_source is None
        else Path(reviewed_source)
    )
    if not force_regenerate:
        sealed = _current_sealed_bundle(
            output_directory, reviewed_source=reviewed_source
        )
        if sealed is not None:
            LOGGER.info("Resource bundle is already current: %s", output_directory)
            return sealed
    LOGGER.info("Building LTE resource bundle in %s", output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    missing_sources = set(BUNDLE_INPUT_SOURCE_IDS) - set(payloads)
    if missing_sources:
        raise KeyError(f"Missing downloaded LTE sources: {sorted(missing_sources)}")
    for name in BUNDLE_INPUT_SOURCE_IDS:
        source = SOURCES[name]
        digest = _sha256(payloads[name])
        if digest != source["sha256"]:
            raise RuntimeError(
                f"Source payload {name!r} failed SHA256 verification before conversion."
            )
    LOGGER.info("Converting verified reference tables and coefficient arrays")
    for name in ("barklem_collet_readme", "barklem_collet_ionization", "barklem_collet_partitions"):
        source = SOURCES[name]
        _write_if_changed(output_directory / source["filename"], payloads[name])
    _write_hminus_table(
        payloads["lightweaver_background"],
        output_directory / "hminus_continuum.json",
    )
    _write_partition_table(
        payloads["barklem_collet_partitions"],
        output_directory / "partition_functions.json",
    )

    for filename in REVIEWED_RESOURCE_FILENAMES:
        source = reviewed_source / filename
        if not source.is_file():
            raise FileNotFoundError(f"Reviewed LTE resource is missing: {source}")
        _write_if_changed(output_directory / filename, source.read_bytes())
    LOGGER.info("Copied %d reviewed scientific metadata files", len(REVIEWED_RESOURCE_FILENAMES))

    manifest = {
        "schema_version": 1,
        "sources": SOURCES,
        "source_roles": SOURCE_ROLES,
        "vendored_files": {
            SOURCES[name]["filename"]: {"sha256": SOURCES[name]["sha256"]}
            for name in (
                "barklem_collet_readme",
                "barklem_collet_ionization",
                "barklem_collet_partitions",
            )
        },
        "generated_files": {
            "hminus_continuum.json": {
                "sha256": _sha256(
                    (output_directory / "hminus_continuum.json").read_bytes()
                )
            },
            "partition_functions.json": {
                "sha256": _sha256((output_directory / "partition_functions.json").read_bytes())
            },
        },
        "reviewed_files": {
            filename: {"sha256": _sha256((output_directory / filename).read_bytes())}
            for filename in REVIEWED_RESOURCE_FILENAMES
        },
    }
    # AtomicDatabase validates its inputs against this provisional manifest.
    # The EOS table is then generated from those verified inputs and added to
    # the final manifest before the bundle is sealed.
    manifest_payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    _write_if_changed(output_directory / "sources.json", manifest_payload)
    manifest = prepare_eos_table(
        output_directory, allow_incomplete_stic_bundle=True
    )
    manifest = prepare_stic_continuum_table(output_directory, payloads)
    manifest = prepare_solar_reference(output_directory, payloads)
    manifest_payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    stic_table = json.loads(
        (output_directory / "stic_continuum_table.json").read_text(encoding="utf-8")
    )
    bundle = {
        "schema_version": 1,
        "bundle_type": "pinn-me-lte-runtime-resources",
        "instrument": "Hinode/SOT-SP",
        "source_manifest_sha256": _sha256(manifest_payload),
        "runtime_files": sorted(
            [
                *manifest["vendored_files"],
                *manifest["generated_files"],
                *manifest["reviewed_files"],
                "sources.json",
            ]
        ),
        "training_contract": "offline-read-only; no downloads or table conversion",
        "required_production_files": sorted(
            (
                "abundances.json",
                "blend_inventory.json",
                "instrument_hinode_sp.json",
                "lines.json",
                "solar_reference_630nm.json",
                "stic_continuum_table.json",
            )
        ),
        "optional_reference_files": sorted(
            (
                "barklem_collet_2016_ReadMe.txt",
                "barklem_collet_2016_table4.dat",
                "barklem_collet_2016_table8.dat",
                "eos_table.json",
                "hminus_continuum.json",
                "partition_functions.json",
            )
        ),
        "physical_depth_contract": stic_table["continuum_contract"],
        "falc_top_boundary": stic_table["falc_top_boundary"],
    }
    _write_if_changed(
        output_directory / "bundle.json",
        (json.dumps(bundle, indent=2) + "\n").encode("utf-8"),
    )
    LOGGER.info("Sealed LTE resource bundle: %s", output_directory / "bundle.json")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path.cwd()
            / "data"
            / "lte_resources"
            / "hinode_sp_v1"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_default_cache_directory(),
        help="Persistent source cache; verified files are never downloaded twice.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "redownload every bundle-generation input while retaining checksum "
            "verification; citation-only endpoints are never build dependencies"
        ),
    )
    parser.add_argument(
        "--eos-only",
        action="store_true",
        help=(
            "regenerate eos_table.json without downloads in an already complete "
            "current bundle; legacy H-minus-only bundles must use the full command "
            "to obtain pinned STiC continuum/FALC resources"
        ),
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="rebuild deterministic EOS/STiC tables even when the sealed bundle is valid",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING"),
        default="INFO",
        help="console logging verbosity (default: INFO)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.workers < 1:
        parser.error("--workers must be positive")
    LOGGER.info("LTE resource output: %s", args.output_dir.expanduser().resolve())
    if args.eos_only:
        if args.refresh or args.force_regenerate:
            parser.error(
                "--eos-only is mutually exclusive with --refresh/--force-regenerate"
            )
        prepare_eos_table(args.output_dir)
        metadata = validate_resource_bundle(args.output_dir)
        LOGGER.info(
            "Bundle validation complete (manifest %s)",
            metadata["source_manifest_sha256"],
        )
        return
    if not args.refresh and not args.force_regenerate:
        if _current_sealed_bundle(args.output_dir) is not None:
            LOGGER.info(
                "Bundle is current; no downloads or conversion required: %s",
                args.output_dir.expanduser().resolve(),
            )
            return
    LOGGER.info("Source cache: %s", args.cache_dir.expanduser().resolve())
    payloads = _fetch_all(args.cache_dir, args.workers, args.refresh)
    prepare_bundle(
        args.output_dir,
        payloads,
        force_regenerate=args.force_regenerate,
    )
    metadata = validate_resource_bundle(args.output_dir)
    LOGGER.info(
        "Bundle ready and validated: %s (manifest %s)",
        metadata["directory"],
        metadata["source_manifest_sha256"],
    )


if __name__ == "__main__":
    main()
