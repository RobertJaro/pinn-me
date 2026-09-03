"""General anomalous-Zeeman component patterns.

Component strengths follow the Wigner-3j convention for electric-dipole
transitions. Patterns are generated once at construction time and stored as
module buffers, avoiding Python work in the differentiable forward path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def _twice_quantum_number(value: float, *, name: str) -> int:
    numeric = float(value)
    twice = round(2.0 * numeric)
    if not math.isfinite(numeric) or abs(twice - 2.0 * numeric) > 1.0e-8:
        raise ValueError(f"{name} must be an integer or half-integer, got {value}.")
    return twice


def _factorial(value: float) -> int:
    rounded = round(value)
    if value < 0 or abs(value - rounded) > 1.0e-8:
        raise ValueError(
            f"Expected a non-negative integer factorial argument, got {value}."
        )
    return math.factorial(rounded)


def wigner_3j(
    j1: float,
    j2: float,
    j3: float,
    m1: float,
    m2: float,
    m3: float,
) -> float:
    """Evaluate a Wigner 3-j symbol for integer or half-integer arguments."""

    twice_j = tuple(
        _twice_quantum_number(value, name=name)
        for name, value in (("j1", j1), ("j2", j2), ("j3", j3))
    )
    if any(value < 0 for value in twice_j):
        raise ValueError("Angular momenta j1, j2, and j3 must be non-negative.")
    twice_m = tuple(
        _twice_quantum_number(value, name=name)
        for name, value in (("m1", m1), ("m2", m2), ("m3", m3))
    )
    if any(abs(m) > j + 1.0e-8 for j, m in ((j1, m1), (j2, m2), (j3, m3))):
        return 0.0
    if any((j - m) % 2 for j, m in zip(twice_j, twice_m)):
        return 0.0
    if sum(twice_j) % 2:
        return 0.0
    if (
        any(value < -1.0e-8 for value in (j1 + j2 - j3, j2 + j3 - j1, j3 + j1 - j2))
        or abs(m1 + m2 + m3) > 1.0e-8
    ):
        return 0.0

    lower = max(0, round(-(j3 - j1 - m2)), round(-(j3 - j2 + m1)))
    upper = min(round(j1 + j2 - j3), round(j1 - m1), round(j2 + m2))
    if lower > upper:
        return 0.0
    triangle = (
        _factorial(j1 + j2 - j3)
        * _factorial(j2 + j3 - j1)
        * _factorial(j3 + j1 - j2)
        / _factorial(j1 + j2 + j3 + 1)
    )
    magnetic = math.prod(
        _factorial(value)
        for value in (
            j1 + m1,
            j1 - m1,
            j2 + m2,
            j2 - m2,
            j3 + m3,
            j3 - m3,
        )
    )
    front = math.sqrt(triangle * magnetic)
    phase = -1.0 if round(j1 - j2 - m3) % 2 else 1.0
    total = 0.0
    for index in range(lower, upper + 1):
        denominator = math.prod(
            _factorial(value)
            for value in (
                index,
                j3 - j1 - m2 + index,
                j3 - j2 + m1 + index,
                j1 + j2 - j3 - index,
                j1 - m1 - index,
                j2 + m2 - index,
            )
        )
        total += (-1.0 if index % 2 else 1.0) * phase * front / denominator
    return total


def zeeman_component_strength(
    j_upper: float, j_lower: float, m_upper: float, m_lower: float
) -> float:
    """Return the relative strength of one allowed Zeeman component."""

    coefficient = wigner_3j(
        j_upper,
        j_lower,
        1.0,
        -m_upper,
        m_lower,
        m_upper - m_lower,
    )
    return 3.0 * coefficient**2


@dataclass(frozen=True)
class ZeemanComponent:
    """One electric-dipole Zeeman component.

    ``wavelength_shift_factor`` multiplies
    ``4.668645e-13 * lambda_0[Angstrom]**2 * B[gauss]``.  A positive value is a
    redward wavelength displacement.  With the adopted energy convention it is
    ``g_l M_l - g_u M_u``.
    """

    delta_m: int
    m_lower: float
    m_upper: float
    wavelength_shift_factor: float
    strength: float

    @property
    def group(self) -> str:
        if self.delta_m == 0:
            return "pi"
        return "blue" if self.delta_m == 1 else "red"


def _magnetic_sublevels(j_value: float) -> tuple[float, ...]:
    twice_j = _twice_quantum_number(j_value, name="J")
    if twice_j < 0:
        raise ValueError("J must be a non-negative integer or half-integer")
    return tuple(-0.5 * twice_j + index for index in range(twice_j + 1))


def zeeman_components(
    j_lower: float,
    j_upper: float,
    lande_lower: float,
    lande_upper: float,
) -> tuple[ZeemanComponent, ...]:
    """Generate normalized allowed components for an E1 transition."""

    twice_lower = _twice_quantum_number(j_lower, name="j_lower")
    twice_upper = _twice_quantum_number(j_upper, name="j_upper")
    delta_twice = abs(twice_upper - twice_lower)
    if (
        twice_lower < 0
        or twice_upper < 0
        or delta_twice not in (0, 2)
        or (twice_upper == 0 and twice_lower == 0)
    ):
        raise ValueError(
            "Electric-dipole transitions require delta J = 0,+/-1 and forbid 0 -> 0"
        )
    if not math.isfinite(float(lande_lower)) or not math.isfinite(float(lande_upper)):
        raise ValueError("Lande factors must be finite.")

    upper_values = _magnetic_sublevels(j_upper)
    raw: list[ZeemanComponent] = []
    for m_lower in _magnetic_sublevels(j_lower):
        for delta_m in (-1, 0, 1):
            m_upper = m_lower + delta_m
            if not any(abs(m_upper - value) < 1e-8 for value in upper_values):
                continue
            strength = float(
                zeeman_component_strength(j_upper, j_lower, m_upper, m_lower)
            )
            if strength <= 1e-14:
                continue
            raw.append(
                ZeemanComponent(
                    delta_m=delta_m,
                    m_lower=float(m_lower),
                    m_upper=float(m_upper),
                    wavelength_shift_factor=(
                        float(lande_lower) * m_lower - float(lande_upper) * m_upper
                    ),
                    strength=strength,
                )
            )

    components: list[ZeemanComponent] = []
    for delta_m in (-1, 0, 1):
        group = [component for component in raw if component.delta_m == delta_m]
        total = sum(component.strength for component in group)
        if total == 0:
            continue
        components.extend(
            ZeemanComponent(
                delta_m=component.delta_m,
                m_lower=component.m_lower,
                m_upper=component.m_upper,
                wavelength_shift_factor=component.wavelength_shift_factor,
                strength=component.strength / total,
            )
            for component in group
        )
    return tuple(components)


class ZeemanPattern(nn.Module):
    """A reusable anomalous-Zeeman pattern stored as PyTorch buffers."""

    groups = ("blue", "pi", "red")

    def __init__(
        self,
        j_lower: float,
        j_upper: float,
        lande_lower: float,
        lande_upper: float,
    ):
        super().__init__()
        self.j_lower = float(j_lower)
        self.j_upper = float(j_upper)
        self.lande_lower = float(lande_lower)
        self.lande_upper = float(lande_upper)
        self._components = zeeman_components(
            j_lower=self.j_lower,
            j_upper=self.j_upper,
            lande_lower=self.lande_lower,
            lande_upper=self.lande_upper,
        )

        for group in self.groups:
            values = [
                component for component in self._components if component.group == group
            ]
            self.register_buffer(
                f"{group}_shift_factors",
                torch.tensor(
                    [component.wavelength_shift_factor for component in values],
                    dtype=torch.float64,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"{group}_strengths",
                torch.tensor(
                    [component.strength for component in values],
                    dtype=torch.float64,
                ),
                persistent=False,
            )

        # Production synthesis evaluates every component in one Faddeeva call
        # per spectral line. Components remain contiguous by polarization
        # group so their independently normalized sums retain the defined
        # arithmetic and sign convention.
        counts = tuple(
            int(getattr(self, f"{group}_shift_factors").numel())
            for group in self.groups
        )
        starts = (0, counts[0], counts[0] + counts[1])
        self._profile_group_slices = tuple(
            slice(start, start + count) for start, count in zip(starts, counts)
        )
        self.register_buffer(
            "profile_shift_factors",
            torch.cat(
                tuple(getattr(self, f"{group}_shift_factors") for group in self.groups)
            ),
            persistent=False,
        )
        self.register_buffer(
            "profile_strengths",
            torch.cat(
                tuple(getattr(self, f"{group}_strengths") for group in self.groups)
            ),
            persistent=False,
        )

    @property
    def effective_lande(self) -> float:
        """Return the effective Landé factor of the transition."""

        lower = self.j_lower * (self.j_lower + 1.0)
        upper = self.j_upper * (self.j_upper + 1.0)
        return 0.5 * (self.lande_upper + self.lande_lower) + 0.25 * (
            self.lande_upper - self.lande_lower
        ) * (upper - lower)

    @property
    def components(self) -> tuple[ZeemanComponent, ...]:
        return self._components

    def group_tensors(
        self,
        group: str,
        *,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(shift_factors, strengths)`` on ``like``'s dtype/device."""

        if group not in self.groups:
            raise KeyError(
                f"Unknown Zeeman group {group!r}; expected one of {self.groups}"
            )
        shifts = getattr(self, f"{group}_shift_factors").to(like)
        strengths = getattr(self, f"{group}_strengths").to(like)
        return shifts, strengths

    def profile_tensors(
        self,
        *,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[slice, ...]]:
        """Return concatenated components and their contiguous group slices."""

        return (
            self.profile_shift_factors.to(like),
            self.profile_strengths.to(like),
            self._profile_group_slices,
        )
