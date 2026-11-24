"""Only retain requested spectra and avoid constructing unused export arrays."""

import numpy as np
import pytest
import torch

from prom3theus.muram.fits import read_metadata, write_hmi_fits
from prom3theus.muram.synthesize import SyntheticHMIResult, synthesize_aia, synthesize_hmi
from tests.muram.test_sdo_fits import snapshot as _snapshot


def snapshot():
    return _snapshot((2, 2, 30))


def test_raw_spectra_are_opt_in_without_changing_observations(monkeypatch):
    snap = snapshot()
    retained = synthesize_hmi(snap, batch_size=3, progress=False, keep_raw_stokes=True)
    raw_shape = (4, 4, retained.raw_wavelength_angstrom.numel())
    original_empty = torch.empty

    def bounded_empty(*args, **kwargs):
        shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
        assert tuple(shape) != raw_shape, "Unrequested full raw spectra must not be allocated"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", bounded_empty)
    compact = synthesize_hmi(snap, batch_size=3, progress=False)
    assert compact.raw_stokes is None
    assert retained.raw_stokes.shape == (2, 2, 4, raw_shape[-1])
    torch.testing.assert_close(compact.stokes, retained.stokes)
    torch.testing.assert_close(compact.raw_wavelength_angstrom, retained.raw_wavelength_angstrom)
    assert compact.observation_metadata() == retained.observation_metadata()


def test_fits_metadata_does_not_construct_spatial_response_cubes(tmp_path, monkeypatch):
    from astropy.io import fits

    result = synthesize_hmi(snapshot(), progress=False)
    expected = dict(result.to_observation_raster().metadata)
    assert result.observation_metadata() == expected

    def unexpected_raster(*args, **kwargs):
        pytest.fail("FITS metadata must not construct an unused ObservationRaster")

    monkeypatch.setattr(SyntheticHMIResult, "to_observation_raster", unexpected_raster)
    paths = write_hmi_fits(tmp_path, result, time_s=0, iteration=1)
    assert len(paths) == 24
    with fits.open(paths[0]) as hdus:
        metadata = read_metadata(hdus)
    assert metadata["muram_synthesis"] == expected["muram_synthesis"]


@pytest.mark.parametrize("batch_size", [True, 1.0, 0, -1])
@pytest.mark.parametrize("synthesize", [synthesize_hmi, synthesize_aia])
def test_batch_size_requires_a_positive_integer(synthesize, batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        synthesize(snapshot(), batch_size=batch_size, progress=False)


@pytest.mark.parametrize("tau", [np.nan, np.inf, -1.0])
def test_aia_rejects_invalid_visibility_labels(tau):
    snap = snapshot()
    snap.synthesis_ready_quantities()["tau"][0, 0, 0] = tau
    with pytest.raises(ValueError, match="finite, non-negative native tau"):
        synthesize_aia(snap, batch_size=2, progress=False)
