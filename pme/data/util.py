import numpy as np
import torch
from astropy import units as u

def spherical_to_cartesian_matrix(c):
    t, p = c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    matrix = [sin(t) * cos(p), cos(t) * cos(p), - sin(p),
              sin(t) * sin(p), cos(t) * sin(p), cos(p),
              cos(t), -sin(t), np.zeros_like(t)]
    matrix = np.stack(matrix, axis=-1).reshape((*c.shape[:-1], 3, 3))
    #
    return matrix


def cartesian_to_spherical_matrix(c, f=np):
    t, p = c[..., 1], c[..., 2]
    sin = f.sin
    cos = f.cos
    #
    matrix = [sin(t) * cos(p), sin(t) * sin(p), cos(t),
              cos(t) * cos(p), cos(t) * sin(p), -sin(t),
              -sin(p), cos(p), f.zeros_like(p)]
    matrix = f.stack(matrix, -1).reshape((*c.shape[:-1], 3, 3))
    #
    return matrix

def vector_spherical_to_cartesian(v, c, f=np):
    vr, vt, vp = v[..., 0], v[..., 1], v[..., 2]
    t, p = c[..., 1], c[..., 2]
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
    t, p = c[..., 1], c[..., 2]
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


def cartesian_to_spherical(v, f=np, eps=1e-8):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    r = f.sqrt(x*x + y*y + z*z + eps)

    arg = z / r
    if f is torch:
        arg = arg.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        p = torch.atan2(y, x + eps)  # regularize backward near axis
    else:
        arg = f.clip(arg, -1.0 + 1e-6, 1.0 - 1e-6)
        p = f.arctan2(y, x)

    t = f.arccos(arg)
    return f.stack([r, t, p], -1)

def observer_ray_and_mu(lon, lat, lonc, latc, observer_distance, radius=1.0):
    """Return the surface-to-observer unit ray and its dot product with radial.

    Angles are radians; distance and radius must use the same units. Image
    axes are solar west, solar north, and Sun-center to observer, respectively.
    """
    distance = np.asarray(observer_distance)
    radius = np.asarray(radius)
    if np.any(~np.isfinite(distance)) or np.any(distance <= radius):
        raise ValueError('observer_distance must be finite and outside the surface radius.')
    # Avoid allocating a full 3x3 transform for every full-disk pixel.
    delta_lon = np.asarray(lon) - lonc
    radial = np.stack([
        np.cos(lat) * np.sin(delta_lon),
        np.sin(lat) * np.cos(latc) - np.cos(lat) * np.sin(latc) * np.cos(delta_lon),
        np.sin(lat) * np.sin(latc) + np.cos(lat) * np.cos(latc) * np.cos(delta_lon),
    ], axis=-1)
    ray = -radius[..., None] * radial
    ray[..., 2] += distance
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    mu = np.clip(np.sum(radial * ray, axis=-1), -1., 1.)
    return ray, mu


def image_to_spherical_matrix(lon, lat, lonc, latc, pAng, observer_distance=None, radius=1.0):
    """Map image vectors to local (radial, colatitude, longitude) components.

    With observer_distance, use the finite-distance LOS and helioprojective
    tangent axes, followed by the image roll. The radial/LOS element [0, 2]
    is mu. Omitting distance retains the legacy parallel-ray transformation.
    """
    if observer_distance is not None:
        ray, _ = observer_ray_and_mu(lon, lat, lonc, latc, observer_distance, radius)
        # Tangent along increasing helioprojective longitude, then latitude.
        image_x = np.stack([ray[..., 2], np.zeros_like(ray[..., 0]), -ray[..., 0]], axis=-1)
        image_x /= np.linalg.norm(image_x, axis=-1, keepdims=True)
        image_y = np.cross(ray, image_x)
        cos_p, sin_p = np.cos(pAng), np.sin(pAng)
        rolled_x = cos_p * image_x - sin_p * image_y
        rolled_y = sin_p * image_x + cos_p * image_y
        basis = np.stack([rolled_x, rolled_y, ray], axis=-1)
        return image_to_spherical_matrix(lon, lat, lonc, latc, 0) @ basis
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
