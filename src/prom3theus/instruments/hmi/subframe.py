"""Crop filename-matched HMI Stokes files using SunPy maps."""

from pathlib import Path
from numbers import Real

import numpy as np
from astropy.io import fits
from prom3theus.core.parallel import parallel_map
from prom3theus.observations.pixel_footprint import CenteredPixelCutout, centered_submap
from .acquisition import (
    hmi_image_hdu, hmi_image_header, hmi_observation_wcs_header,
    resolve_acquisition_groups,
)
from .constants import HMI_CCD_SIZE


def _detector_origin(header, shape):
    """Validate native detector geometry before cropping full disks or cutouts."""
    keys = ("CCD_X0", "CCD_Y0", "CCD_NX", "CCD_NY")
    if shape == (HMI_CCD_SIZE, HMI_CCD_SIZE) and not any(key in header for key in keys):
        return 0, 0
    if not all(key in header for key in keys):
        raise ValueError("HMI cutouts require CCD_X0, CCD_Y0, CCD_NX and CCD_NY detector metadata.")
    values = tuple(header[key] for key in keys)
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not np.isfinite(value) or value != int(value) for value in values):
        raise ValueError("HMI detector metadata must contain integer pixel coordinates and dimensions.")
    x, y, nx, ny = map(int, values)
    if (nx, ny) != (HMI_CCD_SIZE, HMI_CCD_SIZE):
        raise ValueError("HMI cutout CCD_NX/CCD_NY must describe the full native detector.")
    height, width = shape
    if min(x, y) < 0 or min(height, width) < 1 or x + width > nx or y + height > ny:
        raise ValueError("HMI cutout lies outside its declared detector bounds.")
    return x, y


def _hmi_map(path):
    from sunpy.map import Map

    with fits.open(path, memmap=False) as hdul:
        header = hmi_image_header(hdul)
        data = hmi_image_hdu(hdul).data
    _detector_origin(header, data.shape)
    # Astropy has decoded FITS storage scaling; do not apply it again on save.
    for key in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM"):
        header.pop(key, None)
    image = Map(data, hmi_observation_wcs_header(header))
    image.meta["date-obs"] = image.reference_date.tai.isot
    image.meta["t_obs"] = header["T_OBS"]
    return image


def hmi_subframe_bounds(header, *, longitude_deg, latitude_deg, width_pixels, height_pixels):
    """Return full-detector crop bounds for reuse checks, including nested cutouts."""
    from sunpy.map import Map
    import astropy.units as u

    shape = (header["NAXIS2"], header["NAXIS1"])
    origin_x, origin_y = _detector_origin(header, shape)
    image = Map(np.broadcast_to(0., shape),
                hmi_observation_wcs_header(header))
    crop = centered_submap(image, longitude_deg, latitude_deg, width_pixels, height_pixels)
    x, y = np.rint(u.Quantity(image.reference_pixel) - u.Quantity(crop.reference_pixel)).value.astype(int)
    x, y = x + origin_x, y + origin_y
    return x, x + width_pixels, y, y + height_pixels


def prepare_hmi_subframes(
    *, inputs, output_directory, longitude_deg, latitude_deg,
    width_pixels, height_pixels, overwrite=False,
):
    """Each worker reads one FITS image, calls SunPy submap, and saves directly."""
    import astropy.units as u

    CenteredPixelCutout(longitude_deg, latitude_deg, width_pixels, height_pixels)
    output = Path(output_directory).expanduser().resolve()
    groups = resolve_acquisition_groups(files=inputs)
    sources = [path.resolve() for _, paths in groups for path in paths]
    destinations = [output / path.name for path in sources]
    if len({path.name for path in sources}) != len(sources):
        raise ValueError("HMI input filenames collide in the output directory.")
    if any(source == target.resolve() for source, target in zip(sources, destinations)):
        raise ValueError("HMI subframe output must not overwrite an input file.")
    if output.exists() and any(path.name not in {p.name for p in destinations} for path in output.iterdir()):
        raise FileExistsError("Refusing unrelated output files, even with overwrite=True.")
    if any(path.is_symlink() or (path.exists() and not path.is_file()) for path in destinations):
        raise FileExistsError("Refusing non-file HMI subframe output collisions.")
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("Refusing partial/colliding HMI outputs without overwrite=True.")
    output.mkdir(parents=True, exist_ok=True)

    def crop_file(paths):
        source, target = paths
        image = _hmi_map(source)
        origin_x, origin_y = _detector_origin(image.meta, image.data.shape)
        crop = centered_submap(image, longitude_deg, latitude_deg, width_pixels, height_pixels)
        # Detector offsets are needed by the HMI filter-response loader.
        x, y = np.rint(u.Quantity(image.reference_pixel) - u.Quantity(crop.reference_pixel)).value.astype(int)
        x, y = x + origin_x, y + origin_y
        crop.meta.update({"CCD_X0": int(x), "CCD_Y0": int(y),
                          "CCD_NX": HMI_CCD_SIZE, "CCD_NY": HMI_CCD_SIZE})
        crop.save(target, overwrite=overwrite)
        return target

    return parallel_map(crop_file, list(zip(sources, destinations)), description="HMI crops")


__all__ = ["HMI_CCD_SIZE", "hmi_subframe_bounds", "prepare_hmi_subframes"]
