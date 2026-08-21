"""Atomic metadata and differentiable LTE partition functions.

The partition-function parser reads the *raw* CDS ``table8.dat`` distributed
with Barklem & Collet (2016).  Values are never silently synthesized for a
missing ion: an absent species is a data error and raises :class:`KeyError`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Mapping

import torch


def verify_manifest_resource(resource, manifest: Mapping, *, kind: str) -> str:
    """Verify one runtime data resource against the pinned source manifest.

    The manifest separates raw vendored tables, generated tables, and reviewed
    metadata. Runtime synthesis accepts all three classes, but never accepts an
    unrecorded or checksum-mismatched scientific input.
    """

    filename = getattr(resource, "name", None) or Path(resource).name
    record = None
    for section in ("vendored_files", "generated_files", "reviewed_files"):
        candidate = manifest.get(section, {}).get(filename)
        if candidate is not None:
            record = candidate
            break
    if not isinstance(record, Mapping) or "sha256" not in record:
        repair = ""
        if kind == "EOS-table":
            parent = Path(str(resource)).expanduser().resolve().parent
            repair = (
                " Upgrade the existing offline bundle without downloading by running "
                f"`pinn-me-lte-fetch-data --eos-only --output-dir {parent}`."
            )
        raise RuntimeError(
            f"{kind} resource {filename!r} is not recorded in the LTE source manifest."
            + repair
        )
    digest = hashlib.sha256()
    handle_context = (
        resource.open("rb")
        if hasattr(resource, "open")
        else open(resource, "rb")
    )
    with handle_context as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = str(record["sha256"])
    if actual != expected:
        raise RuntimeError(
            f"{kind} resource {filename!r} failed SHA256 verification: "
            f"expected {expected}, got {actual}."
        )
    return actual


@dataclass(frozen=True)
class ElementData:
    """Abundance and atomic constants for one element.

    ``abundance`` is on the astronomical ``log10(N_element/N_H) + 12``
    scale, ``atomic_mass_u`` is in unified atomic-mass units, and
    ``ionization_ev`` is the neutral-to-singly-ionized energy in eV.
    """

    symbol: str
    atomic_number: int
    abundance: float
    atomic_mass_u: float
    ionization_ev: float
    second_ionization_ev: float | None = None

    @property
    def abundance_ratio(self) -> float:
        """Return ``N_element / N_H``."""

        return 10.0 ** (self.abundance - 12.0)


@dataclass(frozen=True)
class SpectralLine:
    """Atomic inputs for an LTE bound-bound transition.

    Damping conventions are deliberately encoded in the field names.  The
    tabulated ``log(gf)`` quantity includes the lower statistical weight; the
    :attr:`oscillator_strength` property divides it out for formulas written
    in terms of the absorption oscillator strength ``f``.
    """

    id: str
    element: str
    ion_stage: int
    wavelength_air_angstrom: float
    lower_excitation_ev: float
    log_gf: float
    j_lower: float
    j_upper: float
    lande_lower: float
    lande_upper: float
    log_gamma_rad_s: float | None
    log_gamma_stark_s_cm3: float | None
    stark_temperature_exponent: float | None
    abo_sigma_a0_squared: float
    abo_alpha: float
    field_provenance: Mapping[str, tuple[str, ...]]

    @property
    def lower_statistical_weight(self) -> float:
        return 2.0 * self.j_lower + 1.0

    @property
    def oscillator_strength(self) -> float:
        return 10.0 ** self.log_gf / self.lower_statistical_weight


class PartitionFunctionTable:
    """Barklem--Collet atomic partition functions on their native grid."""

    def __init__(
        self,
        temperatures: tuple[float, ...],
        values: Mapping[str, tuple[float, ...]],
        *,
        source: str,
    ):
        if len(temperatures) < 2 or any(b <= a for a, b in zip(temperatures, temperatures[1:])):
            raise ValueError("Partition-function temperatures must be strictly increasing")
        for species, row in values.items():
            if len(row) != len(temperatures):
                raise ValueError(f"Partition row {species!r} has the wrong length")
            if any(value <= 0 for value in row):
                raise ValueError(f"Partition row {species!r} contains a non-positive value")
        self.temperatures = temperatures
        self.values = dict(values)
        self.source = source

    @classmethod
    def from_barklem_collet(cls, path) -> "PartitionFunctionTable":
        """Parse the raw CDS ``J/A+A/588/A96/table8.dat`` representation."""

        with open(path, encoding="ascii") as handle:
            raw_lines = handle.readlines()
        temperature_line = next((line for line in raw_lines if "T [K]" in line), None)
        if temperature_line is None:
            raise ValueError("Barklem--Collet table lacks its T [K] header")
        temperatures = tuple(float(token) for token in temperature_line.split()[3:])
        values: dict[str, tuple[float, ...]] = {}
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            species = fields[0]
            row = tuple(float(value) for value in fields[1:])
            if len(row) != len(temperatures):
                raise ValueError(
                    f"Barklem--Collet row {species!r} has {len(row)} values; "
                    f"expected {len(temperatures)}"
                )
            if species in values:
                raise ValueError(f"Duplicate partition-function species {species!r}")
            values[species] = row
        return cls(temperatures, values, source=str(path))

    @classmethod
    def from_json(cls, path) -> "PartitionFunctionTable":
        """Load the preconverted runtime representation from a resource bundle."""

        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported partition-function JSON schema.")
        return cls(
            tuple(float(value) for value in document["temperature_k"]),
            {
                species: tuple(float(value) for value in row)
                for species, row in document["partition_functions"].items()
            },
            source=str(path),
        )

    def __contains__(self, species: str) -> bool:
        return species in self.values

    def interpolate(self, species: str, temperature) -> torch.Tensor:
        """Return ``U(T)`` using differentiable semi-log interpolation.

        ``log(U)`` is linear in temperature between tabulated points.  Values
        outside the published grid use the nearest endpoint; this is explicit
        constant extrapolation rather than an invented high-temperature law.
        """

        if species not in self.values:
            raise KeyError(f"No Barklem--Collet partition function for {species!r}")
        temperature = torch.as_tensor(temperature)
        if not temperature.is_floating_point():
            temperature = temperature.to(torch.get_default_dtype())
        grid = temperature.new_tensor(self.temperatures)
        log_values = torch.log(temperature.new_tensor(self.values[species]))
        flat = temperature.reshape(-1)
        bounded = flat.clamp(min=grid[0], max=grid[-1])
        upper = torch.searchsorted(grid, bounded, right=False).clamp(1, grid.numel() - 1)
        lower = upper - 1
        weight = (bounded - grid[lower]) / (grid[upper] - grid[lower])
        interpolated = log_values[lower] + weight * (log_values[upper] - log_values[lower])
        return torch.exp(interpolated).reshape(temperature.shape)


class AtomicDatabase:
    """Versioned runtime view of line, abundance, and partition-function data."""

    def __init__(
        self,
        data_directory=None,
        line_file=None,
        partition_file=None,
        abundance_file=None,
        source_manifest_file=None,
    ):
        data_root = (
            files("pme.lte").joinpath("data")
            if data_directory is None
            else Path(data_directory).expanduser().resolve()
        )
        self.data_root = data_root
        line_file = data_root.joinpath("lines.json") if line_file is None else Path(line_file)
        partition_file = (
            (
                data_root.joinpath("partition_functions.json")
                if data_root.joinpath("partition_functions.json").is_file()
                else data_root.joinpath("barklem_collet_2016_table8.dat")
            )
            if partition_file is None
            else Path(partition_file)
        )
        abundance_file = data_root.joinpath("abundances.json") if abundance_file is None else Path(abundance_file)
        source_manifest_file = (
            data_root.joinpath("sources.json")
            if source_manifest_file is None
            else Path(source_manifest_file)
        )

        with open(source_manifest_file, encoding="utf-8") as handle:
            self.source_manifest = json.load(handle)
        self.verified_resource_sha256 = {
            "lines": verify_manifest_resource(
                line_file, self.source_manifest, kind="atomic-line"
            ),
            "abundances": verify_manifest_resource(
                abundance_file, self.source_manifest, kind="abundance"
            ),
        }
        # Barklem--Collet partition functions are an explicit reference-EOS
        # dependency, not a production synthesis dependency.  Retain the path
        # and verify/parse it only when ``partition_table`` or
        # ``partition_function`` is actually requested.
        self._partition_file = partition_file
        self._partition_table: PartitionFunctionTable | None = None
        with open(line_file, encoding="utf-8") as handle:
            line_document = json.load(handle)
        self.line_sources = line_document["sources"]
        self.line_conventions = line_document["field_conventions"]
        parsed_lines = []
        for entry in line_document["lines"]:
            entry = dict(entry)
            provenance = {
                field: tuple(source_ids)
                for field, source_ids in entry.pop("field_provenance").items()
            }
            parsed_lines.append(SpectralLine(**entry, field_provenance=provenance))
        self.lines = tuple(parsed_lines)
        self._lines_by_id = {line.id: line for line in self.lines}
        if len(self._lines_by_id) != len(self.lines):
            raise ValueError("Duplicate spectral-line identifier")

        with open(abundance_file, encoding="utf-8") as handle:
            abundance_document = json.load(handle)
        self.abundance_sources = abundance_document["sources"]
        self.elements = {
            symbol: ElementData(symbol=symbol, **metadata)
            for symbol, metadata in abundance_document["species"].items()
        }
        for line in self.lines:
            if line.element not in self.elements:
                raise ValueError(f"Line {line.id!r} references unknown element {line.element!r}")
            if line.ion_stage not in (1, 2, 3):
                raise ValueError(
                    f"Line {line.id!r} has unsupported ion stage {line.ion_stage}; "
                    "expected I, II, or III."
                )

    @property
    def partition_table(self) -> PartitionFunctionTable:
        """Load the verified Barklem--Collet reference table on first use."""

        if self._partition_table is None:
            digest = verify_manifest_resource(
                self._partition_file,
                self.source_manifest,
                kind="partition-function",
            )
            table = (
                PartitionFunctionTable.from_json(self._partition_file)
                if str(self._partition_file).endswith(".json")
                else PartitionFunctionTable.from_barklem_collet(
                    self._partition_file
                )
            )
            for line in self.lines:
                key = (
                    f"{line.element}_"
                    f"{('I', 'II', 'III')[line.ion_stage - 1]}"
                )
                if key not in table:
                    raise ValueError(
                        f"Line {line.id!r} lacks partition function {key!r}"
                    )
            self._partition_table = table
            self.verified_resource_sha256["partition_functions"] = digest
        return self._partition_table

    def get_line(self, line_id: str) -> SpectralLine:
        """Return one transition by its stable identifier."""

        try:
            return self._lines_by_id[line_id]
        except KeyError as error:
            raise KeyError(f"Unknown spectral line {line_id!r}") from error

    def select_lines(self, wavelength_min: float, wavelength_max: float) -> tuple[SpectralLine, ...]:
        """Select transitions in an inclusive air-wavelength interval."""

        if wavelength_max < wavelength_min:
            raise ValueError("wavelength_max must not be smaller than wavelength_min")
        return tuple(
            line
            for line in self.lines
            if wavelength_min <= line.wavelength_air_angstrom <= wavelength_max
        )

    def element(self, symbol: str) -> ElementData:
        try:
            return self.elements[symbol]
        except KeyError as error:
            raise KeyError(f"Unknown element {symbol!r}") from error

    def abundance_ratio(self, element: str) -> float:
        return self.element(element).abundance_ratio

    def partition_function(self, species: str, temperature) -> torch.Tensor:
        return self.partition_table.interpolate(species, temperature)


__all__ = [
    "AtomicDatabase",
    "ElementData",
    "PartitionFunctionTable",
    "SpectralLine",
    "verify_manifest_resource",
]
