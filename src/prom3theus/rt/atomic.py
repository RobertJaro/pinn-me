"""Atomic line metadata for LTE bound-bound synthesis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math

from prom3theus.resources import (
    load_verified_source_manifest,
    resource_path,
    verify_manifest_resource,
)


def _open_text(resource):
    return (
        resource.open("r", encoding="utf-8")
        if hasattr(resource, "open")
        else open(resource, encoding="utf-8")
    )


@dataclass(frozen=True)
class ElementData:
    """Reviewed abundance and atomic constants for one element."""

    symbol: str
    atomic_number: int
    abundance: float
    atomic_mass_u: float
    ionization_ev: float
    second_ionization_ev: float | None = None

    @property
    def abundance_ratio(self) -> float:
        """Return the reviewed ``N_element / N_H`` abundance ratio."""

        return 10.0 ** (self.abundance - 12.0)


@dataclass(frozen=True)
class SpectralLine:
    """Reviewed atomic inputs for one LTE bound-bound transition."""

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
        return 10.0**self.log_gf / self.lower_statistical_weight


class AtomicDatabase:
    """Checksum-verified, immutable runtime view of LTE line metadata."""

    def __init__(self):
        data_root = resource_path("common")
        self.data_root = data_root
        line_resource = data_root.joinpath("lines.json")
        abundance_resource = data_root.joinpath("abundances.json")
        self.source_manifest = load_verified_source_manifest()
        self.verified_resource_sha256 = {
            "lines": verify_manifest_resource(
                line_resource,
                self.source_manifest,
                resource_name="common/lines.json",
                kind="atomic-line",
            ),
            "abundances": verify_manifest_resource(
                abundance_resource,
                self.source_manifest,
                resource_name="common/abundances.json",
                kind="abundance",
            ),
        }

        with _open_text(line_resource) as handle:
            line_document = json.load(handle)
        if line_document.get("schema_version") != 3:
            raise ValueError("Unsupported atomic-line schema.")
        self.line_sources = line_document["sources"]
        self.line_conventions = line_document["field_conventions"]
        if not isinstance(self.line_sources, Mapping) or not isinstance(
            self.line_conventions, Mapping
        ):
            raise ValueError(
                "Atomic-line sources and field conventions must be mappings."
            )
        if "air wavelength" not in str(
            self.line_conventions.get("wavelength_air_angstrom", "")
        ):
            raise ValueError(
                "Atomic-line wavelengths must declare the standard-air contract."
            )
        parsed_lines = []
        for raw_entry in line_document["lines"]:
            entry = dict(raw_entry)
            provenance = {
                field: tuple(source_ids)
                for field, source_ids in entry.pop("field_provenance").items()
            }
            parsed_lines.append(SpectralLine(**entry, field_provenance=provenance))
        self.lines = tuple(parsed_lines)
        self._lines_by_id = {line.id: line for line in self.lines}
        if len(self._lines_by_id) != len(self.lines):
            raise ValueError("Duplicate spectral-line identifier.")

        with _open_text(abundance_resource) as handle:
            abundance_document = json.load(handle)
        if abundance_document.get("schema_version") != 2:
            raise ValueError("Unsupported abundance schema.")
        self.abundance_sources = abundance_document["sources"]
        self.elements = {
            symbol: ElementData(symbol=symbol, **metadata)
            for symbol, metadata in abundance_document["species"].items()
        }
        for element in self.elements.values():
            optional_second = element.second_ionization_ev
            if (
                not element.symbol
                or not isinstance(element.atomic_number, int)
                or element.atomic_number <= 0
                or not math.isfinite(element.abundance)
                or not math.isfinite(element.atomic_mass_u)
                or element.atomic_mass_u <= 0
                or not math.isfinite(element.ionization_ev)
                or element.ionization_ev <= 0
                or (
                    optional_second is not None
                    and (
                        not math.isfinite(optional_second)
                        or optional_second <= element.ionization_ev
                    )
                )
            ):
                raise ValueError(
                    f"Element {element.symbol!r} has invalid atomic metadata."
                )

        provenance_fields = {
            "wavelength_air_angstrom",
            "lower_excitation_ev",
            "log_gf",
            "j_lower",
            "j_upper",
            "lande_lower",
            "lande_upper",
            "log_gamma_rad_s",
            "log_gamma_stark_s_cm3",
            "stark_temperature_exponent",
            "abo_sigma_a0_squared",
            "abo_alpha",
        }
        for line in self.lines:
            if line.element not in self.elements:
                raise ValueError(
                    f"Line {line.id!r} references unknown element {line.element!r}."
                )
            if line.ion_stage not in (1, 2, 3):
                raise ValueError(
                    f"Line {line.id!r} has unsupported ion stage {line.ion_stage}; "
                    "expected I, II, or III."
                )
            twice_lower = round(2.0 * line.j_lower)
            twice_upper = round(2.0 * line.j_upper)
            if (
                not line.id
                or not math.isfinite(line.wavelength_air_angstrom)
                or not 2000.0 <= line.wavelength_air_angstrom <= 100000.0
                or not math.isfinite(line.lower_excitation_ev)
                or line.lower_excitation_ev < 0
                or not math.isfinite(line.log_gf)
                or min(twice_lower, twice_upper) < 0
                or abs(twice_lower - 2.0 * line.j_lower) > 1.0e-8
                or abs(twice_upper - 2.0 * line.j_upper) > 1.0e-8
                or abs(twice_upper - twice_lower) not in (0, 2)
                or (twice_lower == 0 and twice_upper == 0)
                or not math.isfinite(line.lande_lower)
                or not math.isfinite(line.lande_upper)
                or (
                    line.log_gamma_rad_s is not None
                    and not math.isfinite(line.log_gamma_rad_s)
                )
                or (line.log_gamma_stark_s_cm3 is None)
                != (line.stark_temperature_exponent is None)
                or (
                    line.log_gamma_stark_s_cm3 is not None
                    and (
                        not math.isfinite(line.log_gamma_stark_s_cm3)
                        or not math.isfinite(line.stark_temperature_exponent)
                    )
                )
                or not math.isfinite(line.abo_sigma_a0_squared)
                or line.abo_sigma_a0_squared <= 0
                or not math.isfinite(line.abo_alpha)
                or not 0 <= line.abo_alpha < 4
            ):
                raise ValueError(f"Line {line.id!r} has invalid physical metadata.")
            if set(line.field_provenance) != provenance_fields:
                raise ValueError(f"Line {line.id!r} has incomplete field provenance.")
            for field, source_ids in line.field_provenance.items():
                if not source_ids or any(
                    not isinstance(source_id, str) or source_id not in self.line_sources
                    for source_id in source_ids
                ):
                    raise ValueError(
                        f"Line {line.id!r} field {field!r} has invalid provenance."
                    )

    def get_line(self, line_id: str) -> SpectralLine:
        """Return one transition by its stable identifier."""

        try:
            return self._lines_by_id[line_id]
        except KeyError as error:
            raise KeyError(f"Unknown spectral line {line_id!r}.") from error

    def select_lines(
        self, wavelength_min: float, wavelength_max: float
    ) -> tuple[SpectralLine, ...]:
        """Select transitions in an inclusive air-wavelength interval."""

        if not math.isfinite(wavelength_min) or not math.isfinite(wavelength_max):
            raise ValueError("Wavelength selection bounds must be finite.")
        if wavelength_max < wavelength_min:
            raise ValueError("wavelength_max must not be smaller than wavelength_min.")
        return tuple(
            line
            for line in self.lines
            if wavelength_min <= line.wavelength_air_angstrom <= wavelength_max
        )

    def element(self, symbol: str) -> ElementData:
        try:
            return self.elements[symbol]
        except KeyError as error:
            raise KeyError(f"Unknown element {symbol!r}.") from error


__all__ = ["AtomicDatabase", "ElementData", "SpectralLine"]
