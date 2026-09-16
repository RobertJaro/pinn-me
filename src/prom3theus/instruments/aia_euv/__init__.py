"""SDO/AIA optically thin EUV observation support."""

from typing import TYPE_CHECKING, Any

from .calibration import AIAChannelCalibration
from .operator import AIAEmissionOperator
from .response import AIAResponseTable, canonical_aia_channel

if TYPE_CHECKING:
    from .acquisition import SelectedAIARecord
    from .calibration_download import (
        download_aia_preprocessing_calibration,
        load_aia_calibration_manifest,
    )
    from .download import (
        AIA_SMALL_SAMPLE_MAX_OFFSET_SECONDS,
        AIA_SMALL_SAMPLE_TARGET_UTC,
        download_aia_euv,
        download_aia_euv_small_sample,
        load_aia_acquisition_manifest,
    )


def __getattr__(name: str) -> Any:
    """Load acquisition support only when its public API is requested."""

    if name == "SelectedAIARecord":
        from .acquisition import SelectedAIARecord

        return SelectedAIARecord
    if name in {
        "AIA_SMALL_SAMPLE_MAX_OFFSET_SECONDS",
        "AIA_SMALL_SAMPLE_TARGET_UTC",
        "download_aia_euv",
        "download_aia_euv_small_sample",
        "load_aia_acquisition_manifest",
    }:
        from . import download

        return getattr(download, name)
    if name in {
        "download_aia_preprocessing_calibration",
        "load_aia_calibration_manifest",
    }:
        from . import calibration_download

        return getattr(calibration_download, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AIAChannelCalibration",
    "AIAEmissionOperator",
    "AIAResponseTable",
    "AIA_SMALL_SAMPLE_MAX_OFFSET_SECONDS",
    "AIA_SMALL_SAMPLE_TARGET_UTC",
    "SelectedAIARecord",
    "canonical_aia_channel",
    "download_aia_euv",
    "download_aia_euv_small_sample",
    "download_aia_preprocessing_calibration",
    "load_aia_calibration_manifest",
    "load_aia_acquisition_manifest",
]
