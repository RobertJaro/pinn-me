import numpy as np
import torch
from astropy import units as u

from pme.coordinates import (
    cartesian_to_spherical as _cartesian_to_spherical,
    cartesian_to_spherical_matrix as _cartesian_to_spherical_matrix,
    project_cartesian_to_spherical,
    project_spherical_to_cartesian,
    spherical_to_cartesian as _spherical_to_cartesian,
    spherical_to_cartesian_matrix as _spherical_to_cartesian_matrix,
)

def spherical_to_cartesian_matrix(c, f=np):
    return _spherical_to_cartesian_matrix(c, f)


def cartesian_to_spherical_matrix(c, f=np):
    return _cartesian_to_spherical_matrix(c, f)

def vector_spherical_to_cartesian(v, c, f=np):
    return project_spherical_to_cartesian(v, c, f)


def vector_cartesian_to_spherical(v, c, f=np):
    return project_cartesian_to_spherical(v, c, f)



def spherical_to_cartesian(v, f=np):
    return _spherical_to_cartesian(v, f)


def cartesian_to_spherical(v, f=np, eps=1e-8):
    del eps
    return _cartesian_to_spherical(v, f)

def image_to_spherical_matrix(lon, lat, lonc, latc, pAng):
    sin = np.sin
    cos = np.cos

    k11 = cos(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - sin(lat) * cos(
        latc) * sin(pAng)
    k12 = -cos(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + sin(lat) * cos(
        latc) * cos(pAng)
    k13 = cos(lat) * cos(latc) * cos(lon - lonc) + sin(lat) * sin(latc)
    k21 = sin(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) + cos(lat) * cos(latc) * sin(pAng)
    k22 = -sin(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) - cos(lat) * cos(latc) * cos(pAng)
    k23 = sin(lat) * cos(latc) * cos(lon - lonc) - cos(lat) * sin(latc)
    k31 = -sin(latc) * sin(pAng) * sin(lon - lonc) + cos(pAng) * cos(lon - lonc)
    k32 = sin(latc) * cos(pAng) * sin(lon - lonc) + sin(pAng) * cos(lon - lonc)
    k33 = -cos(latc) * sin(lon - lonc)


    a_matrix = np.stack([k11, k12, k13, k21, k22, k23, k31, k32, k33], axis=-1)
    a_matrix = a_matrix.reshape((*a_matrix.shape[:-1], 3, 3))
    return a_matrix

def vector_spherical_to_image(v, lon, lat, lonc, latc, pAng, f=np):
    sin = f.sin
    cos = f.cos

    k11 = cos(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - sin(lat) * cos(
        latc) * sin(pAng)
    k12 = -cos(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + sin(lat) * cos(
        latc) * cos(pAng)
    k13 = cos(lat) * cos(latc) * cos(lon - lonc) + sin(lat) * sin(latc)
    k21 = sin(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) + cos(lat) * cos(latc) * sin(pAng)
    k22 = -sin(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) - cos(lat) * cos(latc) * cos(pAng)
    k23 = sin(lat) * cos(latc) * cos(lon - lonc) - cos(lat) * sin(latc)
    k31 = -sin(latc) * sin(pAng) * sin(lon - lonc) + cos(pAng) * cos(lon - lonc)
    k32 = sin(latc) * cos(pAng) * sin(lon - lonc) + sin(pAng) * cos(lon - lonc)
    k33 = -cos(latc) * sin(lon - lonc)

    br, bt, bp = v[..., 0], v[..., 1], v[..., 2]

    bx = br * k11 + bt * k21 + bp * k31
    by = br * k12 + bt * k22 + bp * k32
    bz = br * k13 + bt * k23 + bp * k33

    return f.stack([bx, by, bz], -1)



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
    v = (1 * u.Rsun).to_value(u.m) * omega * np.cos(theta) / u.rad
    return v.to(u.m / u.s)
