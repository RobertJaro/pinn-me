import numpy as np
from astropy import units as u
from astropy.constants import R_sun
# Some documentation to be included

def spherical_to_cartesian_matrix(c):
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    matrix = np.stack([
        np.stack([cos(t) * cos(p), sin(t) * cos(p), -sin(p)], -1),
        np.stack([cos(t) * sin(p), sin(t) * sin(p), cos(p)], -1),
        np.stack([sin(t), -cos(t), np.zeros_like(t)], -1)
    ], -2)
    #
    return matrix


def cartesian_to_spherical_matrix(c):
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    matrix = np.stack([
        np.stack([cos(t) * cos(p), cos(t) * sin(p), sin(t)], -1),
        np.stack([sin(t) * cos(p), sin(t) * sin(p), -cos(t)], -1),
        np.stack([-sin(p), cos(p), np.zeros_like(p)], -1)
    ], -2)
    #
    return matrix

def vector_spherical_to_cartesian(v, c, f=np):
    vr, vt, vp = v[..., 0], v[..., 1], v[..., 2]
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = f.sin
    cos = f.cos
    #
    vx = vr * cos(t) * cos(p) + vt * sin(t) * cos(p) - vp * sin(p)
    vy = vr * cos(t) * sin(p) + vt * sin(t) * sin(p) + vp * cos(p)
    vz = vr * sin(t) - vt * cos(t)
    #
    return f.stack([vx, vy, vz], -1)


def vector_cartesian_to_spherical(v, c, f=np):
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = f.sin
    cos = f.cos
    #
    vr = vx * cos(t) * cos(p) + vy * cos(t) * sin(p) + vz * sin(t)
    vt = vx * sin(t) * cos(p) + vy * sin(t) * sin(p) - vz * cos(t)
    vp = - vx * sin(p) + vy * cos(p)
    #
    return f.stack([vr, vt, vp], -1)


def spherical_to_cartesian(v, f=np):
    sin = f.sin
    cos = f.cos
    r, t, p = v[..., 0], v[..., 1], v[..., 2]
    x = r * cos(t) * cos(p)
    y = r * cos(t) * sin(p)
    z = r * sin(t)
    return f.stack([x, y, z], -1)


def cartesian_to_spherical(v, f=np):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    xy = x ** 2 + y ** 2

    r = f.sqrt(xy + z ** 2)
    xy_sqrt = f.sqrt(xy)
    nudge = (f.abs(xy_sqrt) < 1e-6) * 1e-6  # assure numerical stability
    t = f.arctan2(z, xy_sqrt + nudge)
    nudge = (f.abs(x) < 1e-6) * 1e-6  # assure numerical stability
    p = f.arctan2(y, x + nudge)

    return f.stack([r, t, p], -1)


def img_to_los_trv_azi(b, f=np):
    b_x, b_y, b_z = b[..., 0], b[..., 1], b[..., 2]

    B_los = b_z
    B_trv = (b_x ** 2 + b_y ** 2) ** 0.5
    azi = f.arctan2(b_x, b_y)

    b = f.stack([B_los, B_trv, azi], -1)
    return b


def los_trv_azi_to_img(b, ambiguous=False, f=np):
    B_los, B_trv, azi = b[..., 0], b[..., 1], b[..., 2]
    B_x = B_trv * f.sin(azi % f.pi) if ambiguous else B_trv * f.sin(azi)
    B_y = B_trv * f.cos(azi % f.pi) if ambiguous else B_trv * f.cos(azi)
    B_z = B_los
    b = f.stack([B_x, B_y, B_z], -1)
    return b

def vector_cartesian_to_polar(v, f=np):
    B_x, B_y, B_z = v[..., 0], v[..., 1], v[..., 2]
    B_total = np.sqrt(B_x**2 + B_y **2 + B_z**2)
    theta = np.arccos(B_z / B_total)
    phi = np.arccos(B_y / np.sqrt(B_x**2 + B_y**2))
    b = f.stack([B_total, theta, phi], -1)
    return b

def image_to_spherical_matrix(lon, lat, latc, lonc, pAng, sin=np.sin, cos=np.cos):
    ''' Transformation of the POS to the heliocentric frame for any vector
    '''

    try:
        a11 = -sin(latc) * sin(pAng) * sin(lon - lonc) + cos(pAng) * cos(lon - lonc)
        a12 = sin(latc) * cos(pAng) * sin(lon - lonc) + sin(pAng) * cos(lon - lonc)
        a13 = -cos(latc) * sin(lon - lonc)
        a21 = -sin(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - cos(lat) * cos(
            latc) * sin(pAng)
        a22 = sin(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + cos(lat) * cos(
            latc) * cos(pAng)
        a23 = -cos(latc) * sin(lat) * cos(lon - lonc) + sin(latc) * cos(lat)
        a31 = cos(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - sin(lat) * cos(
            latc) * sin(pAng)
        a32 = -cos(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + sin(lat) * cos(
            latc) * cos(pAng)
        a33 = cos(lat) * cos(latc) * cos(lon - lonc) + sin(lat) * sin(latc)

        a_matrix = np.stack([a31, a32, a33, a21, a22, a23, a11, a12, a13], axis=-1)
        a_matrix = a_matrix.reshape((*a_matrix.shape[:-1], 3, 3))

    except:
        a_matrix = np.ones((3, 3)) + np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
    return a_matrix

# hmi_b2ptr
# def image_to_spherical_matrix(phi, lam, b, pAng, sin=np.sin, cos=np.cos):
#     sinb = sin(b)
#     cosb = cos(b)
#     sinp = sin(pAng)
#     cosp = cos(pAng)
#     sinphi = sin(phi)
#     cosphi = cos(phi)
#     sinlam = sin(lam)
#     coslam = cos(lam)
#
#     k11 = coslam * (sinb * sinp * cosphi + cosp * sinphi) - sinlam * cosb * sinp
#     k12 = - coslam * (sinb * cosp * cosphi - sinp * sinphi) + sinlam * cosb * cosp
#     k13 = coslam * cosb * cosphi + sinlam * sinb
#     k21 = sinlam * (sinb * sinp * cosphi + cosp * sinphi) + coslam * cosb * sinp
#     k22 = - sinlam * (sinb * cosp * cosphi - sinp * sinphi) - coslam * cosb * cosp
#     k23 = sinlam * cosb * cosphi - coslam * sinb
#     k31 = - sinb * sinp * sinphi + cosp * cosphi
#     k32 = sinb * cosp * sinphi + sinp * cosphi
#     k33 = - cosb * sinphi
#
#     # TODO check paper
#     a_matrix = np.stack([k11, k12, k13, k21, k22, k23, k31, k32, k33], axis=-1)
#     a_matrix = a_matrix.reshape((*a_matrix.shape[:-1], 3, 3))
#     return a_matrix

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
    v = R_sun * omega * np.cos(theta) / u.rad
    return v.to(u.m / u.s)