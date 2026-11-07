"""Read MURaM MHD snapshots and forward-render synthetic observables.

This subpackage is a validation tool: it takes a known MURaM atmosphere,
forward-renders synthetic instrument observables with the independent numerical
reference solver (:mod:`prom3theus.validation.reference_synthesis`), and packages them as
:class:`~prom3theus.observations.contracts.ObservationRaster` so they can be
fed back through the existing inversion CLI. Recovering the known MURaM
atmosphere from the synthetic observation is the validation criterion.
"""

from __future__ import annotations

from .reader import MURaMSnapshot

__all__ = ["MURaMSnapshot"]
