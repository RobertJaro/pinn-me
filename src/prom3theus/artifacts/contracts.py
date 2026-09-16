from dataclasses import dataclass
from prom3theus.observations import ObservationRaster


@dataclass(frozen=True, slots=True)
class StoredRasterSelection:
    """The exact validation raster selected from the embedded store."""

    raster: ObservationRaster
    name: str
    index: int
    raster_count: int
