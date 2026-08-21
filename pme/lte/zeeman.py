"""General anomalous-Zeeman component patterns.

Component strengths use the same Wigner-3j convention as the established
PINN-ME synthesis.  Patterns are generated once at construction time and then
stored as module buffers, avoiding Python work in the differentiable forward
path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pme.train.atomic_functions import compute_zeeman_strength


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
    twice_j = round(2.0 * float(j_value))
    if j_value < 0 or abs(twice_j - 2.0 * float(j_value)) > 1e-8:
        raise ValueError("J must be a non-negative integer or half-integer")
    return tuple(-0.5 * twice_j + index for index in range(twice_j + 1))


def zeeman_components(
    j_lower: float,
    j_upper: float,
    lande_lower: float,
    lande_upper: float,
) -> tuple[ZeemanComponent, ...]:
    """Generate normalized allowed components for an E1 transition."""

    if abs(j_upper - j_lower) > 1.0 + 1e-8 or (j_upper == 0 and j_lower == 0):
        raise ValueError("Electric-dipole transitions require delta J = 0,+/-1 and forbid 0 -> 0")

    upper_values = _magnetic_sublevels(j_upper)
    raw: list[ZeemanComponent] = []
    for m_lower in _magnetic_sublevels(j_lower):
        for delta_m in (-1, 0, 1):
            m_upper = m_lower + delta_m
            if not any(abs(m_upper - value) < 1e-8 for value in upper_values):
                continue
            strength = float(
                compute_zeeman_strength(j_upper, j_lower, m_upper, m_lower)
            )
            if strength <= 1e-14:
                continue
            raw.append(
                ZeemanComponent(
                    delta_m=delta_m,
                    m_lower=float(m_lower),
                    m_upper=float(m_upper),
                    wavelength_shift_factor=(
                        float(lande_lower) * m_lower
                        - float(lande_upper) * m_upper
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
            values = [component for component in self._components if component.group == group]
            self.register_buffer(
                f"{group}_shift_factors",
                torch.tensor(
                    [component.wavelength_shift_factor for component in values],
                    dtype=torch.float64,
                ),
            )
            self.register_buffer(
                f"{group}_strengths",
                torch.tensor(
                    [component.strength for component in values],
                    dtype=torch.float64,
                ),
            )

    @property
    def effective_lande(self) -> float:
        """Return the effective Landé factor of the transition."""

        lower = self.j_lower * (self.j_lower + 1.0)
        upper = self.j_upper * (self.j_upper + 1.0)
        return (
            0.5 * (self.lande_upper + self.lande_lower)
            + 0.25 * (self.lande_upper - self.lande_lower) * (upper - lower)
        )

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
            raise KeyError(f"Unknown Zeeman group {group!r}; expected one of {self.groups}")
        shifts = getattr(self, f"{group}_shift_factors").to(like)
        strengths = getattr(self, f"{group}_strengths").to(like)
        return shifts, strengths
