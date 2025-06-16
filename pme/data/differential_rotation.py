import numpy as np
import torch
from astropy import units as u
from astropy import constants as const

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

def carrington_rotation_velocity():
    R_sun = 6.957e8  # Solar radius in meters
    rotation_rate = 2 * np.pi / (27.2753 * 24 * 3600) # rad/s
    v = R_sun * rotation_rate # m/s
    return v