import numpy as np
import torch
from astropy import units as u
from astropy import constants as const

from pme.data.util import cartesian_to_spherical, spherical_to_cartesian


def solar_differential_rotation_velocity(latitude):
    """
    Compute solar differential rotation velocity at given latitude(s).

    Parameters
    ----------
    latitude : `~astropy.units.Quantity`
        Latitude(s) on the Sun (positive northward), with angular units (e.g., deg, rad).

    Returns
    -------
    velocity : `~astropy.units.Quantity`
        Tangential linear rotation velocity (m/s) at the given latitude(s).
    """
    # Ensure latitude is in radians
    theta = latitude.to(u.rad)

    # Differential rotation law (Snodgrass, 1983) in deg/day
    A = 14.713 * u.deg / u.day
    B = -2.396 * u.deg / u.day
    C = -1.787 * u.deg / u.day

    omega = A + B * np.sin(theta) ** 2 + C * np.sin(theta) ** 4  # deg/day
    omega = omega.to(u.rad / u.s)  # Convert to rad/s

    # Tangential velocity: v = R * omega * cos(latitude)
    v = const.R_sun * omega * np.cos(theta)
    return v.to(u.m / u.s)


def solar_differential_rotation_velocity_torch(latitude_rad: torch.Tensor) -> torch.Tensor:
    """
    Compute solar differential rotation tangential velocity at given latitude(s) using PyTorch.

    Parameters
    ----------
    latitude_rad : torch.Tensor
        Latitude(s) on the Sun in radians (positive northward).

    Returns
    -------
    velocity : torch.Tensor
        Tangential linear rotation velocity (m/s) at the given latitude(s).
    """
    R_sun = 6.957e8  # Solar radius in meters
    deg_per_day_to_rad_per_sec = (torch.pi / 180) / 86400  # Conversion factor

    # Differential rotation coefficients in deg/day
    A = 14.713
    B = -2.396
    C = -1.787

    sin_lat = torch.sin(latitude_rad)
    omega_deg_per_day = A + B * sin_lat**2 + C * sin_lat**4
    omega_rad_per_sec = omega_deg_per_day * deg_per_day_to_rad_per_sec

    v = R_sun * omega_rad_per_sec * torch.cos(latitude_rad)
    return v  # in m/s

def carrington_rotation_velocity(latitude, radius, f=torch):
    """
        Compute the rigid Carrington co-rotation velocity at a given
        heliographic latitude and radius, using the Sun's sidereal rotation period.

        The Carrington sidereal rotation period is ~25.38 days, corresponding
        to an angular velocity of ~2.865e-6 rad/s. This function returns the
        tangential speed in the azimuthal (phi-hat) direction for a point fixed
        in Carrington coordinates.

        Parameters
        ----------
        latitude : array-like or tensor
            Heliographic latitude(s) in radians (positive northward).
        radius : float or array-like/tensor
            Radial distance from the Sun's center in meters.
        f : module, optional
            Math module to use for calculations (default: torch; can also use numpy).

        Returns
        -------
        v_phi : array-like or tensor
            Tangential linear velocity (m/s) in the Carrington co-rotating frame.
            Positive in the direction of increasing Carrington longitude.
    """
    rotation_rate = 2 * f.pi / (25.38 * 24 * 3600) # rad/s
    # rotation_rate = 2 * f.pi / (27.2753 * 24 * 3600)  # rad/s
    return radius * rotation_rate * f.cos(latitude)  # in m/s


def differential_rotation_rate(latitude, f=torch):
    """
    Compute the solar differential rotation angular velocity at a given latitude.

    Parameters
    ----------
    latitude : array-like or tensor
        Heliographic latitude(s) in radians (positive northward).
    f : module, optional
        Math module to use for calculations (default: torch; can also use numpy).

    Returns
    -------
    omega : array-like or tensor
        Angular velocity (rad/s) at the given latitude(s).
    """
    A = 14.713 * (f.pi / 180) / 86400  # rad/s
    B = -2.396 * (f.pi / 180) / 86400  # rad/s
    C = -1.787 * (f.pi / 180) / 86400  # rad/s

    sin_lat = f.sin(latitude)
    omega = A + B * sin_lat**2 + C * sin_lat**4
    return omega  # in rad/s

def carrington_rotation_rate(f=torch):
    """
    Get the Carrington rotation angular velocity.

    Parameters
    ----------
    f : module, optional
        Math module to use for calculations (default: torch; can also use numpy).

    Returns
    -------
    omega_carrington : float
        Carrington angular velocity (rad/s).
    """
    omega_carrington = 2 * f.pi / (25.38 * 24 * 3600)  # rad/s
    return omega_carrington


def to_differential_rotation_frame(coords, seconds_per_dt):
    time = coords[..., 0:1]  # reference time at 0

    spherical = cartesian_to_spherical(coords[..., 1:], torch)  # [r, theta, phi]
    lat = torch.pi / 2 - spherical[..., 1:2]
    lon = spherical[..., 2:3]

    # keep away from exact poles if your omega fit is nasty there
    lat_safe = lat.clamp(-torch.pi/2 + 1e-4, torch.pi/2 - 1e-4)

    omega = differential_rotation_rate(lat_safe, torch)
    omega_car = carrington_rotation_rate(torch)

    lon_shift = (omega - omega_car) * time * seconds_per_dt
    lon_new = lon + lon_shift

    # wrap to avoid huge angles
    lon_new = torch.remainder(lon_new + torch.pi, 2 * torch.pi) - torch.pi

    spherical_shift = torch.cat([spherical[..., 0:1], spherical[..., 1:2], lon_new], dim=-1)
    xyz_shift = spherical_to_cartesian(spherical_shift, torch)
    return torch.cat([time, xyz_shift], dim=-1)