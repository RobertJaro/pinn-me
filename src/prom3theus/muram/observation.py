"""Observation-registry adapter for the ``muram_stored`` type.

Bridges a directory written by :func:`prom3theus.muram.synthesize` +
``ObservationStore.save_sequence`` to the same
``load_or_prepare_observation``/``ObservationAdapter`` machinery real
HMI/Hinode-SP data goes through, so the synthetic rasters are cached,
validated, and consumed exactly like real observations from that point on.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from prom3theus.observations.contracts import ObservationSpec
from prom3theus.observations.contracts import CARRINGTON_REGISTERED_RELATIVE_VELOCITY
from prom3theus.observations.store import MANIFEST_FILENAME, ObservationStore
from .provenance import check_stokes_forward_provenance


class MuramStoredObservationData:
    """Adapter-facing wrapper around a pre-built ``ObservationStore`` directory."""

    def __init__(self, directory, validation_raster_index: int, binning: int = 1):
        from .binning import validate_binning

        self.binning = validate_binning(binning)
        self.directory = Path(directory)
        self._validation_raster_index = int(validation_raster_index)
        self.rasters = None
        self.raster_names = None

    def setup(self, stage=None) -> None:
        if self.rasters is not None:
            return
        rasters, names, _ = ObservationStore.load_sequence(self.directory)
        if not 0 <= self._validation_raster_index < len(rasters):
            raise ValueError(
                f"validation_raster_index={self._validation_raster_index} is out of "
                f"range for {len(rasters)} raster(s) in {self.directory}."
            )
        from .binning import bin_stokes_raster

        self.rasters = [bin_stokes_raster(raster, self.binning) for raster in rasters]
        self.raster_names = names

    @property
    def validation_raster_index(self) -> int:
        return self._validation_raster_index

    @property
    def fits_consistency_metadata(self) -> dict:
        return {}

    @property
    def raster(self):
        return self.rasters[self._validation_raster_index]

    @property
    def run_metadata(self) -> dict:
        return {}


def _build_data(config: Mapping) -> MuramStoredObservationData:
    raw = dict(config)
    binning = raw.pop("binning", 1)
    expected = {"directory", "loader", "validation_raster_index"}
    if set(raw) != expected:
        raise TypeError(
            "MURaM stored observation config must contain exactly directory, "
            f"loader, and validation_raster_index; missing={sorted(expected - set(raw))}, "
            f"unknown={sorted(set(raw) - expected)}."
        )
    data = MuramStoredObservationData(raw["directory"], raw["validation_raster_index"], binning=binning)
    data.setup()
    return data


def _describe_data(data: MuramStoredObservationData) -> ObservationSpec:
    contract = data.raster.metadata.get("muram_synthesis")
    if contract is None:
        raise ValueError(
            "Legacy MURaM store lacks the synthesis contract; regenerate it."
        )

    def shared_contract(value):
        # Adaptive bottom extension is verified per snapshot; its resolved grid
        # and achieved optical depth need not be identical at every time.
        result = {
            key: item for key, item in value.items()
            if key not in {"native_log_tau_grid", "bottom_boundary"}
        }
        if "bottom_boundary" in value:
            result["bottom_boundary"] = {
                key: value["bottom_boundary"][key]
                for key in ("required_optical_depth", "status")
            }
        return result

    for raster in data.rasters:
        other_contract = raster.metadata.get("muram_synthesis")
        if (
            other_contract is None
            or shared_contract(other_contract) != shared_contract(contract)
        ):
            raise ValueError("MURaM rasters must share one synthesis contract.")
        if not raster.wavelength_angstrom.equal(data.raster.wavelength_angstrom):
            raise ValueError("MURaM rasters must share one wavelength grid.")
    instrument = dict(contract["instrument"])
    instrument_type = instrument.pop("type")
    check_stokes_forward_provenance(
        contract.get("forward_model_provenance"), instrument_type
    )
    return ObservationSpec(
        observation_id="muram_synthetic_" + instrument_type,
        observation_type="muram_stored",
        instrument_type=instrument_type,
        wavelength_angstrom=data.raster.wavelength_angstrom,
        continuum_indices=(0, data.raster.stokes.shape[-1] - 1),
        radiance_scale_w_m3_sr=contract["radiance_scale"],
        velocity_synthesis_mode=CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
        required_line_ids=tuple(contract["line_ids"]),
        line_support_angstrom=tuple(contract["line_support_angstrom"]),
        instrument_options=instrument,
    )


def _source_files(config: Mapping) -> tuple[Path, ...]:
    directory = Path(config["directory"]).expanduser().resolve()
    manifest = ObservationStore.manifest(directory)
    # Loading creates derived bulk-catalog sidecars. They must not change the
    # source identity and invalidate every resolution's prepared cache.
    paths = {directory / MANIFEST_FILENAME}
    for raster in manifest["rasters"]:
        for record in raster["arrays"].values():
            path = (directory / record["file"]).resolve()
            if path.parent != directory or not path.is_file():
                raise ValueError("Unsafe or missing MURaM observation array path")
            paths.add(path)
    return tuple(sorted(paths))


__all__ = ["MuramStoredObservationData"]
