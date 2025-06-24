import numpy as np
import os
# Constants
from scipy.special import jvp
from tqdm import tqdm

from pme.convert.vtk import save_vtk


def vector_spherical_to_cartesian(v, c):
    vr, vt, vp = v[..., 0], v[..., 1], v[..., 2]
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    vx = vr * sin(t) * cos(p) + vt * cos(t) * cos(p) - vp * sin(p)
    vy = vr * sin(t) * sin(p) + vt * cos(t) * sin(p) + vp * cos(p)
    vz = vr * cos(t) - vt * sin(t)
    #
    return np.stack([vx, vy, vz], -1)


def to_spherical(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.stack([r, theta, phi], -1)

if __name__ == '__main__':

    # Initial conditions


    # Time parameters
    t0 = 0
    t_start = 0  # Starting time
    t_end = 26 * 60 * 60  # Ending time

    spatial_res = 64
    temporal_res = 60
    spatial_dim = 1e8  # 100 Mm

    ds = (2 * spatial_dim) / spatial_res
    print('Spatial resolution:', ds)

    # Spatial parameters
    R_0 = 500e5
    coords = np.stack(np.meshgrid(np.arange(-spatial_dim, spatial_dim + ds, ds),
                                  np.arange(-spatial_dim, spatial_dim + ds, ds),
                                  np.arange(1.e5, R_0 + ds, ds),
                                  np.linspace(t_start, t_end, temporal_res), indexing='ij'), -1)

    x, y, z, T = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
    # Calculate the spheromak solution
    # mesh grid
    d = to_spherical(np.stack([x, y, z], -1))
    R, THETA, PHI = d[..., 0], d[..., 1], d[..., 2]

    print('Grid initialized')
    #
    B_0 = 500
    #
    n = 1
    m = 0
    gamma = 5e2  # 2.5e5 # Mm/s
    C_alpha = 4.4934
    alpha_0 = C_alpha / R_0
    #
    A_r = np.zeros_like(R)
    A_theta = 0
    A_phi = 0
    #
    alpha = C_alpha * (R_0 + gamma * (T - t0) ** n) ** -1
    dalpha_dt = - C_alpha * (R_0 + gamma * (T - t0) ** n) ** -2 * gamma * n * (T - t0) ** (n - 1)
    #
    # f = (alpha/alpha_0) ** (2 + m)
    # df_dt = (1/alpha_0) ** (2 + m) * alpha ** (2 + m - 1) * dalpha_dt
    #
    j1 = jvp(1, alpha * R, n=0)
    dj1 = jvp(1, alpha * R, n=1)

    # E_r = np.zeros_like(R)
    # E_theta = - (df_dt/f - dalpha_dt/alpha) * A_theta - dalpha_dt * R * A_phi
    # E_phi = - (df_dt/f - 3 * dalpha_dt/alpha) * A_phi - B_0 * f * (dalpha_dt * np.sin(alpha * R) / alpha ** 2) * np.sin(theta)

    B_r = 2 * B_0 * alpha * j1 / (alpha_0 ** 2 * R) * np.cos(THETA)
    B_theta = - B_0 * alpha * (j1 + alpha * R * dj1) / (alpha_0 ** 2 * R) * np.sin(THETA)
    B_phi = B_0 * j1 * (alpha / alpha_0) ** 2 * np.sin(THETA)

    E_r = np.zeros_like(R)
    E_theta = - B_phi * dalpha_dt / alpha * R
    E_phi = B_theta * dalpha_dt / alpha * R
    # B_r = 2 * A_theta * np.cos(THETA) / (r * np.sin(THETA))
    # B_theta = A_phi / R - B_0 * f * np.sin(alpha * R) / (alpha * R) * np.sin(THETA)
    # B_phi = alpha * A_phi

    print('Fields computed')

    c = np.stack([R, THETA, PHI], -1)
    B = vector_spherical_to_cartesian(np.stack([B_r, B_theta, B_phi], -1), c)
    E = vector_spherical_to_cartesian(np.stack([E_r, E_theta, E_phi], -1), c)

    V = np.divide(np.cross(E, B), (B ** 2).sum(-1, keepdims=True))

    print('VERIFICATION B', B.min(), B.max())
    print('VERIFICATION V', V.min(), V.max())
    print('VERIFICATION Vz', V[..., 0, :, 2].min(), V[..., 0, :, 2].max())

    print('Fields converted to spherical coordinates')

    print('SAVING...')
    os.makedirs('/glade/work/rjarolim/data/induction/spheromak', exist_ok=True)
    for i in tqdm(range(B.shape[3])):
        save_vtk(f'/glade/work/rjarolim/data/induction/spheromak/spheromak_{i:03d}.vtk', {'B': B[:, :, :, i], 'V': V[:, :, :, i]})

    np.save('/glade/work/rjarolim/data/induction/spheromak/spheromak_B.npy', B)
    np.save('/glade/work/rjarolim/data/induction/spheromak/spheromak_V.npy', V)
    np.save('/glade/work/rjarolim/data/induction/spheromak/spheromak_times.npy', np.linspace(t_start, t_end, temporal_res))
