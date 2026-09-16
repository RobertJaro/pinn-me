"""Build a bolometric CHIANTI cooling table with FIASCO, offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from functools import cached_property
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fiasco():
    # PlasmaPy probes GitHub at import time; no remote data are needed here.
    import requests

    if "fiasco" in sys.modules:
        existing = sys.modules["fiasco"]
        if existing.__version__ != "0.8.2":
            raise ValueError("Use fiasco 0.8.2, matching the existing response builder")
        return existing
    with (
        patch.object(
            requests,
            "get",
            side_effect=requests.ConnectionError("Offline CHIANTI build"),
        ),
        redirect_stdout(io.StringIO()),
    ):
        import fiasco
    if fiasco.__version__ != "0.8.2":
        raise ValueError("Use fiasco 0.8.2, matching the existing response builder")
    return fiasco


def _calculate(database, abundance, density, log_temperature, ion_names=None):
    import astropy.units as u
    from tqdm.auto import tqdm

    fiasco = _fiasco()
    from fiasco.util.exceptions import MissingDatasetException

    full_rate_sources = {}

    class CachedIon(fiasco.Ion):
        # Atomic metadata are immutable for an instance. Avoid repeated HDF5
        # reads and PlasmaPy particle construction inside large line models.
        atomic_number = cached_property(fiasco.Ion.atomic_number.fget)
        atomic_symbol = cached_property(fiasco.Ion.atomic_symbol.fget)
        n_levels = cached_property(fiasco.Ion.n_levels.fget)
        transitions = cached_property(fiasco.Ion.transitions.fget)

        def _collision_rates(self, name):
            full_temperature = 10**log_temperature * u.K
            if np.array_equal(self.temperature, full_temperature):
                return getattr(fiasco.Ion, name).func(self)
            # Only rate vectors span the full temperature grid, never the
            # quadratic population matrices. Reuse the same spline fits in
            # every chunk, including the adjacent ion in the two-ion model.
            key = self._base_rep
            if key not in full_rate_sources:
                full_rate_sources[key] = CachedIon(
                    key, full_temperature, **self._instance_kwargs
                )
            source = full_rate_sources[key]
            indices = np.searchsorted(
                full_temperature.value, self.temperature.to_value(u.K)
            )
            return getattr(source, name)[indices]

        @cached_property
        def electron_collision_excitation_rate(self):
            return self._collision_rates("electron_collision_excitation_rate")

        @cached_property
        def electron_collision_deexcitation_rate(self):
            return self._collision_rates("electron_collision_deexcitation_rate")

    options = dict(
        hdf5_dbase_root=database, abundance=abundance, ionization_fraction="chianti"
    )
    components = {
        name: np.zeros(log_temperature.size)
        for name in ("lines", "free_free", "free_bound", "two_photon")
    }
    missing = set()
    if ion_names is None:
        ion_names = fiasco.list_ions(database)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [
                pool.submit(
                    _calculate, database, abundance, density, log_temperature, [name]
                )
                for name in ion_names
            ]
            # Sum in atomic order so repeated builds remain reproducible.
            for job in tqdm(jobs, desc="CHIANTI cooling ions"):
                values, skipped, _ = job.result()
                for component in components:
                    components[component] += values[component]
                missing.update(tuple(item) for item in skipped)
        return components, sorted(missing), ion_names
    for name in ion_names:
        full_rate_sources.clear()
        probe = CachedIon(name, 10**log_temperature * u.K, **options)
        full_rate_sources[probe._base_rep] = probe
        try:
            levels = probe.n_levels
        except MissingDatasetException:
            levels = 1
        # Small ions can use the complete grid; large rate matrices are chunked.
        chunk_size = max(1, min(log_temperature.size, 2_000_000 // max(levels, 1) ** 2))
        # Bound memory by temperature chunk and ion, not the atomic inventory.
        for start in range(0, log_temperature.size, chunk_size):
            selection = slice(start, start + chunk_size)
            ion = (
                probe
                if chunk_size == log_temperature.size
                else CachedIon(name, 10 ** log_temperature[selection] * u.K, **options)
            )
            weight = (ion.abundance * ion.ionization_fraction).value
            if not np.any(weight > 0):
                continue
            population_options = dict(
                include_protons=False,
                use_two_ion_model=ion._has_dataset("auto") or ion._has_dataset("rrlvl"),
            )
            for component in components:
                try:
                    if component == "lines":
                        rate = ion.contribution_function(
                            density * u.cm**-3, **population_options
                        ).sum(axis=-1)[:, 0]
                    elif component == "free_free":
                        rate = ion.free_free_radiative_loss() * weight
                    elif component == "free_bound":
                        rate = ion.free_bound_radiative_loss() * weight
                    else:
                        if not (ion.hydrogenic or ion.helium_like):
                            continue
                        label = "2s 2S1/2" if ion.hydrogenic else "1s 2s 1S0"
                        edge = (
                            ion.levels.energy[ion.levels.label == label]
                            .to_value(u.angstrom, equivalencies=u.spectral())
                            .item()
                        )
                        wavelength = edge * np.geomspace(1.0, 1.0e5, 2049)
                        spectrum = ion.two_photon(
                            wavelength * u.angstrom,
                            density * u.cm**-3,
                            **population_options,
                        )
                        rate = (
                            np.trapezoid(
                                spectrum.to_value("erg cm3 s-1 Angstrom-1")[:, 0],
                                wavelength,
                                axis=-1,
                            )
                            * weight
                            * u.Unit("erg cm3 s-1")
                        )
                    values = rate.to_value(u.W * u.m**3)
                except MissingDatasetException as error:
                    missing.add((name, component, str(error)))
                    continue
                if not np.all(np.isfinite(values) & (values >= 0)):
                    raise ValueError(
                        f"Invalid {component} cooling for {name} at {log_temperature[selection]}"
                    )
                components[component][selection] += values
        full_rate_sources.clear()
    return components, sorted(missing), ion_names


def build(
    output: Path,
    database_root: Path,
    abundance: str = "sun_coronal_2021_chianti",
    electron_density_cm3: float = 1.0e9,
):
    root = Path(database_root)
    ascii_root = root / "ascii"
    database = root / "chianti_11.0.2.h5"
    if (ascii_root / "VERSION").read_text().strip() != "11.0.2":
        raise ValueError("Use CHIANTI 11.0.2 to match the current AIA/EOS resources")
    abundance_paths = list((ascii_root / "abundance").rglob(f"{abundance}.abund"))
    if len(abundance_paths) != 1 or not database.is_file():
        raise ValueError(
            "Prepared CHIANTI HDF5 database and abundance file are required"
        )
    if not np.isfinite(electron_density_cm3) or electron_density_cm3 <= 0:
        raise ValueError("electron_density_cm3 must be finite and positive")
    log_temperature = np.linspace(4.0, 9.0, 201)
    components, missing, ions = _calculate(
        database, abundance, electron_density_cm3, log_temperature
    )
    rate = sum(components.values())
    if not np.all(np.isfinite(rate) & (rate > 0)):
        raise ValueError("CHIANTI did not return a positive cooling curve")
    document = {
        "schema_version": 1,
        "unit": "W m^3",
        "density_convention": "n_e * n_H",
        "log10_temperature_k": log_temperature.tolist(),
        "log10_lambda_w_m3": np.log10(rate).tolist(),
        "components_w_m3": {name: value.tolist() for name, value in components.items()},
        "provenance": {
            "database": "CHIANTI",
            "database_version": "11.0.2",
            "fiasco_version": importlib.metadata.version("fiasco"),
            "database_sha256": _sha256(database),
            "abundance": abundance,
            "abundance_sha256": _sha256(abundance_paths[0]),
            "ioneq_sha256": _sha256(ascii_root / "ioneq/chianti.ioneq"),
            "reference_electron_density_cm3": electron_density_cm3,
            "ions": ions,
            "missing_datasets": missing,
            "line_population_options": {
                "include_protons": False,
                "use_two_ion_model": True,
                "include_level_resolved_rate_correction": True,
            },
            "two_photon_quadrature": {
                "wavelength_over_edge": [1, 100000],
                "points": 2049,
            },
            "free_bound_method": "FIASCO integrated Mewe approximation",
            "composition_note": "Coronal abundances match AIA, not the STiC photospheric EOS mixture; review this enrichment assumption before physical interpretation.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(
        f"Wrote {output}: {len(rate)} temperatures, Lambda(1 MK)={rate[80]:.6g} W m^3"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-root",
        type=Path,
        required=True,
        help="Directory containing ascii/ and chianti_11.0.2.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/prom3theus/resources/sets/plasma/coronal_cooling_v1.json"),
    )
    parser.add_argument("--abundance", default="sun_coronal_2021_chianti")
    parser.add_argument("--electron-density-cm3", type=float, default=1.0e9)
    args = parser.parse_args()
    build(args.output, args.database_root, args.abundance, args.electron_density_cm3)


if __name__ == "__main__":
    main()
