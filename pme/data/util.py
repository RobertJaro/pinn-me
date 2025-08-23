import numpy as np
import torch
from astropy import units as u

def spherical_to_cartesian_matrix(c):
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    matrix = [sin(t) * cos(p), cos(t) * cos(p), - sin(p),
              sin(t) * sin(p), cos(t) * sin(p), cos(p),
              cos(t), -sin(t), np.zeros_like(t)]
    matrix = np.stack(matrix, axis=-1).reshape((*c.shape[:-1], 3, 3))
    #
    return matrix


def cartesian_to_spherical_matrix(c):
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    matrix = [sin(t) * cos(p), sin(t) * sin(p), cos(t),
              cos(t) * cos(p), cos(t) * sin(p), -sin(t),
              -sin(p), cos(p), np.zeros_like(p)]
    matrix = np.stack(matrix, axis=-1).reshape((*c.shape[:-1], 3, 3))
    #
    return matrix

def vector_spherical_to_cartesian(v, c, f=np):
    vr, vt, vp = v[..., 0], v[..., 1], v[..., 2]
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = f.sin
    cos = f.cos
    #
    vx = vr * sin(t) * cos(p) + vt * cos(t) * cos(p) - vp * sin(p)
    vy = vr * sin(t) * sin(p) + vt * cos(t) * sin(p) + vp * cos(p)
    vz = vr * cos(t) - vt * sin(t)
    #
    return f.stack([vx, vy, vz], -1)


def vector_cartesian_to_spherical(v, c, f=np):
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = f.sin
    cos = f.cos
    #
    vr = vx * sin(t) * cos(p) + vy * sin(t) * sin(p) + vz * cos(t)
    vt = vx * cos(t) * cos(p) + vy * cos(t) * sin(p) - vz * sin(t)
    vp = - vx * sin(p) + vy * cos(p)
    #
    return f.stack([vr, vt, vp], -1)



def spherical_to_cartesian(v, f=np):
    sin = f.sin
    cos = f.cos
    r, t, p = v[..., 0], v[..., 1], v[..., 2]
    x = r * sin(t) * cos(p)
    y = r * sin(t) * sin(p)
    z = r * cos(t)
    return f.stack([x, y, z], -1)


def cartesian_to_spherical(v, f=np):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    nudge = (f.abs(r) < 1e-6) * 1e-6  # assure numerical stability
    t = f.arccos(z / (r + nudge))
    nudge = (f.abs(x) < 1e-6) * 1e-6  # assure numerical stability
    p = f.arctan2(y, x + nudge)

    return f.stack([r, t, p], -1)

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