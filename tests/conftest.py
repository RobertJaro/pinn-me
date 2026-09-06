import glob
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest
from astropy.io import fits


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

_TEST_RUNTIME = Path(tempfile.gettempdir()) / "prom3theus-tests"
_TEST_RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SUNPY_CONFIGDIR", str(_TEST_RUNTIME / "sunpy"))
os.environ.setdefault("MPLCONFIGDIR", str(_TEST_RUNTIME / "matplotlib"))


HINODE_N_WAVELENGTH = 112
HINODE_CRVAL_ANGSTROM = 6301.5091 + 0.021549 * (138.0 - 111.5)
HINODE_CRPIX = 56.5
HINODE_CDELT_ANGSTROM = 0.021549
HINODE_SPCCDIY0 = 56
HINODE_SPCCDIY1 = 167


def hinode_header(date_obs="2007-01-05T23:59:07.816"):
    header = fits.Header()
    header["CUNIT1"] = "Angstrom"
    header["CTYPE1"] = "WAVE"
    header["CRVAL1"] = HINODE_CRVAL_ANGSTROM
    header["CRPIX1"] = HINODE_CRPIX
    header["CDELT1"] = HINODE_CDELT_ANGSTROM
    header["CRPIX2"] = 1.5
    header["CDELT2"] = 0.1585
    header["CTYPE2"] = "Solar-Y"
    header["CUNIT2"] = "arcsec"
    header["DATE_OBS"] = date_obs
    header["XCEN"] = -22.0
    header["YCEN"] = -4.0
    header["CROTA2"] = 0.0
    # SolarSoft thermd_sbsp defines DOP_RCV as m/s, positive for redshift, and
    # sp_prep has already removed it from these Level-1 spectra.
    header["DOP_RCV"] = 2011.0
    header["SPWLSHFT"] = 4.72823
    header["SPWLSFT0"] = -0.887668
    header["SPCCDIY0"] = HINODE_SPCCDIY0
    header["SPCCDIY1"] = HINODE_SPCCDIY1
    header.add_history("sp_prep VERSION:  1.07")
    return header


@pytest.fixture
def synthetic_hinode_files(tmp_path):
    """Two tiny scan steps with uniquely identifiable spectral samples."""
    paths = []
    for scan_index in range(2):
        data = np.empty((4, 3, HINODE_N_WAVELENGTH), dtype=np.float32)
        wavelength_index = np.arange(HINODE_N_WAVELENGTH, dtype=np.float32)
        for stokes_index in range(4):
            for slit_index in range(3):
                if stokes_index == 0:
                    data[stokes_index, slit_index] = (
                        10000 + scan_index * 1000 + slit_index * 100 + wavelength_index
                    )
                else:
                    data[stokes_index, slit_index] = (
                        stokes_index
                        + scan_index * 0.1
                        + slit_index * 0.01
                        + wavelength_index * 1.0e-3
                    )
        path = tmp_path / f"SP3D20070105_23590{7 + scan_index}.fits"
        header = hinode_header(f"2007-01-05T23:59:0{7 + scan_index}.816")
        # Represent a real raster step in the helioprojective pointing rather
        # than creating two detector columns at one identical Solar-X.
        header["XCEN"] = -22.0 + scan_index * 0.14857
        header["SLITINDX"] = scan_index
        header["NSLITPOS"] = 2
        fits.writeto(path, data, header, overwrite=False)
        paths.append(path)
    return paths


@pytest.fixture
def expected_hinode_wavelength():
    pixel = np.arange(HINODE_N_WAVELENGTH, dtype=np.float64) + 1.0
    return HINODE_CRVAL_ANGSTROM + (pixel - HINODE_CRPIX) * HINODE_CDELT_ANGSTROM


@pytest.fixture
def real_hinode_files():
    pattern = os.environ.get("PROM3THEUS_HINODE_GLOB")
    if not pattern:
        pytest.skip("set PROM3THEUS_HINODE_GLOB to run the real-data integration test")
    paths = sorted(glob.glob(pattern))
    if not paths:
        pytest.fail(f"PROM3THEUS_HINODE_GLOB matched no files: {pattern!r}")
    return paths
