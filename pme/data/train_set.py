from multiprocessing import Pool

import torch

from pme.train.me_equations import MEAtmosphere

"""
Test the ME module with PSF
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.signal import convolve2d
import matplotlib.pyplot as pl

from tqdm import tqdm


def plot_all_Stokes(atmos):
    fig, ax = pl.subplots(2, 2, dpi=150)
    ax[0, 0].plot(atmos.lambdaGrid / 1e-10, atmos.I)
    ax[0, 0].set_title("Stokes I")

    ax[0, 1].plot(atmos.lambdaGrid / 1e-10, atmos.Q / atmos.I)
    ax[0, 1].set_title("Stokes Q/I")

    ax[1, 0].plot(atmos.lambdaGrid / 1e-10, atmos.U / atmos.I)
    ax[1, 0].set_title("Stokes U/I")

    ax[1, 1].plot(atmos.lambdaGrid / 1e-10, atmos.V / atmos.I)
    ax[1, 1].set_title("Stokes V/I")

    for el in ax.flatten():
        el.grid(alpha=0.1)

    ax[0, 0].set_ylabel("Intensity")
    ax[1, 0].set_ylabel("Intensity")
    ax[1, 0].set_xlabel("$\\lambda$ [$\\AA$]")
    ax[1, 1].set_xlabel("$\\lambda$ [$\\AA$]")

    pl.tight_layout()
    pl.show()


def convert_xy_to_rt(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)

    return r, t


def convert_rt_to_xy(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)

    return int(x), int(y)


def create_tstep(tt):
    xx, yy = np.meshgrid(np.linspace(-0.5 * nx, 0.5 * nx, nx),
                         np.linspace(-0.5 * ny, 0.5 * ny, ny))
    r0 = 50 + tt / 2

    r, t = convert_xy_to_rt(xx, yy)
    r = r[..., None]
    t = t[..., None]

    # B --> (100, 100); lambda --> (50,); B * lambda --> (100, 100, 50)
    # B[..., None] --> (100, 100, 1); lambda[None, None, :] --> (1, 1, 50)

    BField = BField0 * (r0 / (r + r0)) ** 2
    # theta is defined between 0 and pi
    t_arr = ((r % r0) / r0 * np.pi)
    ch_arr = t + tt / 180 * np.pi # slow down the rotation

    B0_arr = B0 * (10 * r0 / (r + 10 * r0)) ** 2
    B1_arr = B1 * (10 * r0 / (r + 10 * r0)) ** 2

    lambdaEnd = (lambdaStart
                 + lambdaStep * (-1 + nLambda))

    lambdaGrid = np.linspace(-.5 * (lambdaEnd - lambdaStart),
                             .5 * (lambdaEnd - lambdaStart),
                             num=nLambda)
    lambdaGrid = lambdaGrid[None, None, :]

    atmos = MEAtmosphere(lambda0, jUp, jLow, gUp, gLow,
                         lambdaGrid)

    BField = torch.tensor(BField, dtype=torch.float32)
    t_arr = torch.tensor(t_arr, dtype=torch.float32)
    ch_arr = torch.tensor(ch_arr, dtype=torch.float32)
    B0_arr = torch.tensor(B0_arr, dtype=torch.float32)
    B1_arr = torch.tensor(B1_arr, dtype=torch.float32)

    vmac_arr = vmac * torch.ones_like(BField)
    damping_arr = damping * torch.ones_like(BField)
    mu_arr = mu * torch.ones_like(BField)
    vdop_arr = vdop * torch.ones_like(BField)
    kl_arr = kl * torch.ones_like(BField)

    I, Q, U, V = atmos.forward(BField, t_arr, ch_arr,
                               vmac_arr,
                               damping_arr,
                               B0_arr, B1_arr, mu_arr,
                               vdop_arr, kl_arr)
    Stokes_profiles = torch.cat([I, Q, U, V]).cpu().numpy()

    return {'Stokes_profiles': Stokes_profiles, 'BField': BField, 'theta': t_arr, 'chi': ch_arr,
                             'B0': B0_arr, 'B1': B1_arr, 'vmac': vmac_arr, 'damping': damping_arr, 'mu': mu_arr,
                             'vdop': vdop_arr, 'kl': kl_arr}


if __name__ == '__main__':

    lambda0 = 6301.5080
    jUp = 2.0
    jLow = 2.0
    gUp = 1.5
    gLow = 1.83
    lambdaStart = 6300.8 * 1e-10
    lambdaStep = 0.03 * 1e-10
    nLambda = 50
    BField0 = 2000.0
    theta = np.deg2rad(70.0)
    chi = np.deg2rad(0.0)
    vmac = 2.0
    damping = 0.2
    B0 = 0.8
    B1 = 0.2
    mu = 1.0
    vdop = 0.0
    kl = 5.0

    nStokes = 4
    nx = 400
    ny = 400

    nTime = 100
    rMax = np.sqrt((nx / 2) ** 2 + (ny / 2) ** 2)

    # Make the X, Y meshgrid instead of np.tile
    xs = np.linspace(-2 * np.pi, 2 * np.pi, nx)
    ys = np.linspace(-6 * np.pi, 6 * np.pi, ny)
    tau, phi = np.meshgrid(xs, ys)
    # Z evaluation
    amp = np.sin(tau + phi)

    with Pool(32) as p:
        profiles = list(tqdm(p.imap(create_tstep, range(nTime)), total=nTime))

    Stokes_profiles = np.stack([p['Stokes_profiles'] for p in profiles], axis=1)
    B0_arr = np.stack([p['B0'] for p in profiles], axis=0)
    B1_arr = np.stack([p['B1'] for p in profiles], axis=0)
    t_arr = np.stack([p['theta'] for p in profiles], axis=0)
    ch_arr = np.stack([p['chi'] for p in profiles], axis=0)
    Bf_arr = np.stack([p['BField'] for p in profiles], axis=0)

    keys = profiles[0].keys()
    parameters = {key: np.stack([p[key] for p in profiles], axis=2) for key in keys if key != 'Stokes_profiles'}


    # for tt in tqdm(range(nTime)):
    #     print(f"Convolving {tt} timestep out of {nTime}")
    #     for el in range(nStokes):
    #         for ll in range(nLambda):
    #             Stokes_profiles_PSF[tt, :, :,el,
    #                                 ll] = convolve2d(Stokes_profiles[tt, :, :, el, ll],
    #                                                       PSF, mode="same")

    # plot_all_Stokes(atmos)
    for el in range(nTime):
        im1 = pl.imshow(t_arr[el, :, :],
                        vmin=0,
                        vmax=np.pi,
                        cmap="RdBu_r")
        pl.colorbar(im1)
        pl.title(f"$\\theta$ Timestep {el}")
        pl.savefig(
            f"/glade/work/rjarolim/data/inversion/test/t_arr_{el:03d}.png")
        pl.close()

    for el in range(nTime):
        im1 = pl.imshow(np.log10(Bf_arr[el, :, :]),
                        # vmin=0, vmax=np.log10(2000),
                        cmap="cividis")
        pl.colorbar(im1)
        pl.title(f"Bfield [G] Timestep {el}")
        pl.savefig(
            f"/glade/work/rjarolim/data/inversion/test/Bf_arr_{el:03d}.png")
        pl.close()

    for el in range(nTime):
        im1 = pl.imshow(B0_arr[el, :, :],
                        cmap="cividis")
        pl.colorbar(im1)
        pl.title(f"B0 Timestep {el}")
        pl.savefig(
            f"/glade/work/rjarolim/data/inversion/test/B0_arr_{el:03d}.png")
        pl.close()
    for el in range(nTime):
        im1 = pl.imshow(B1_arr[el, :, :],
                        cmap="cividis")
        pl.colorbar(im1)
        pl.title(f"B1 Timestep {el}")
        pl.savefig(
            f"/glade/work/rjarolim/data/inversion/test/B1_arr_{el:03d}.png")
        pl.close()

    for el in range(nTime):
        im1 = pl.imshow(ch_arr[el, :, :] % np.pi,
                        vmin=0,
                        vmax=np.pi,
                        cmap="twilight_shifted")
        pl.colorbar(im1)
        pl.title(f"$\chi$ Timestep {el}")
        pl.savefig(
            f"/glade/work/rjarolim/data/inversion/test/ch_arr_{el:03d}.png")
        pl.close()

    nameTestCase = "MEset_v4_nt"
    np.savez(f"/glade/work/rjarolim/data/inversion/{nameTestCase}_{nTime}_spatial_{nx}_{ny}_withPSF_11x11_s_4.npz",
             Stokes_profiles=Stokes_profiles)

    np.savez(f"/glade/work/rjarolim/data/inversion/parameters_{nameTestCase}_{nTime}_spatial_{nx}_{ny}_withPSF_11x11_s_4.npz",
             **parameters)



