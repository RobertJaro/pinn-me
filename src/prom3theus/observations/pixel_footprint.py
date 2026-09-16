"""Independent image crops sized in nominal HMI pixels."""

from dataclasses import asdict, dataclass

import numpy as np

HMI_PIXEL_SCALE_ARCSEC = 0.5


def centered_submap(image_map, longitude_deg, latitude_deg, width, height):
    """Use SunPy to extract an exact native-pixel rectangle at a Carrington center."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    center = SkyCoord(
        longitude_deg * u.deg, latitude_deg * u.deg, image_map.rsun_meters,
        frame=frames.HeliographicCarrington(
            observer=image_map.observer_coordinate, obstime=image_map.reference_date,
        ),
    ).transform_to(image_map.coordinate_frame)
    if not center.is_visible():
        raise ValueError(f"Crop center is on the far side of the Sun at {image_map.reference_date.isot}.")
    pixel = u.Quantity(image_map.world_to_pixel(center)).to_value(u.pix)
    bottom_left = np.floor(pixel - (np.array([width, height]) - 1) / 2 + 0.5) * u.pix
    crop = image_map.submap(bottom_left, width=(width - 1) * u.pix, height=(height - 1) * u.pix)
    if crop.data.shape != (height, width):
        raise ValueError("Requested pixel crop extends outside the image.")
    return crop


def pixel_dimensions(width, height, source_scale, target_scale):
    """Convert angular size between pixel scales, rounding to nearest pixels."""
    sizes = np.asarray([width, height]) * np.abs(source_scale) / np.abs(target_scale)
    return tuple(int(value) for value in np.maximum(1, np.floor(sizes + 0.5)))


@dataclass(frozen=True)
class CenteredPixelCutout:
    """A Carrington center and dimensions in nominal 0.5-arcsec HMI pixels."""

    longitude_deg: float
    latitude_deg: float
    width_pixels: int
    height_pixels: int

    def __post_init__(self):
        if not np.isfinite([self.longitude_deg, self.latitude_deg]).all():
            raise ValueError("Crop center must be finite.")
        if not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude_deg must lie in [-90, 90].")
        if any(type(v) is not int or not 1 <= v <= 4096
               for v in (self.width_pixels, self.height_pixels)):
            raise ValueError("Crop dimensions must be integers in [1, 4096] HMI pixels.")

    def metadata(self):
        return dict(asdict(self), kind="centered_nominal_hmi_pixel_rectangle",
                    reference_pixel_scale_arcsec=HMI_PIXEL_SCALE_ARCSEC,
                    grid_policy="native_aiapy_registered_submap_no_reprojection")
