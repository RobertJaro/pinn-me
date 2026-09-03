"""Scientific constants and provenance identifiers for Hinode/SOT-SP."""

from __future__ import annotations


SP_PREP_SOURCE_URL = (
    "https://sohoftp.nascom.nasa.gov/solarsoft/hinode/sot/idl/sp/util/sp_prep.pro"
)
SP_PREP_SOURCE_SHA256 = (
    "63bd10f742fae21cb32f62fb11abb29f00c82c2445eb8fabd65976f7d22f60a9"
)
CALIB_SBSP_SOURCE_URL = (
    "https://sohoftp.nascom.nasa.gov/solarsoft/hinode/sot/idl/sp/util/calib_sbsp.pro"
)
THERMD_SBSP_SOURCE_URL = (
    "https://sohoftp.nascom.nasa.gov/solarsoft/hinode/sot/idl/sp/util/thermd_sbsp.pro"
)
THERMD_SBSP_SOURCE_SHA256 = (
    "e539150351e99a7ed99e5d97c8a9435022b2c3215fb75816630f3ba6a79e69fe"
)

SP_PREP_REFERENCE_LINE_ANGSTROM = 6301.5091
HINODE_6301_LAB_AIR_WAVELENGTH_ANGSTROM = 6301.5008
HINODE_COORDINATE_NORMALIZATION_PIXELS = 512.0
HINODE_SP_FE_LINE_IDS = ("FeI_6301.5008", "FeI_6302.4932")

HINODE_KEYWORD_REFERENCE_URL = (
    "https://hinode.nao.ac.jp/uploads/2016/04/22/SB_MW_Key13.pdf"
)
SOLAR_WCS_REFERENCE_URL = "https://fits.gsfc.nasa.gov/wcs/coordinates.pdf"


__all__ = [
    "CALIB_SBSP_SOURCE_URL",
    "HINODE_6301_LAB_AIR_WAVELENGTH_ANGSTROM",
    "HINODE_COORDINATE_NORMALIZATION_PIXELS",
    "HINODE_KEYWORD_REFERENCE_URL",
    "HINODE_SP_FE_LINE_IDS",
    "SOLAR_WCS_REFERENCE_URL",
    "SP_PREP_REFERENCE_LINE_ANGSTROM",
    "SP_PREP_SOURCE_SHA256",
    "SP_PREP_SOURCE_URL",
    "THERMD_SBSP_SOURCE_SHA256",
    "THERMD_SBSP_SOURCE_URL",
]
