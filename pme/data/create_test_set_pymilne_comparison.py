from multiprocessing import Pool

import torch

from pme.train.me_atmosphere import MEAtmosphere

"""
Test the ME module with PSF
"""
import numpy as np
import matplotlib.pyplot as pl

from tqdm import tqdm

from astropy import units as u


def plot_all_Stokes(lambda_grid, atmos):
    """
    Plot all the stokes profiles from an atmosphere

    Input:
        -- atmos, ndarray [4, num_Intensity]
    """

    fig, ax = pl.subplots(2, 2, dpi=150)
    ax[0, 0].plot(lambda_grid, atmos[0, :])
    ax[0, 0].set_title("Stokes I")

    ax[0, 1].plot(lambda_grid, atmos[1, :])
    ax[0, 1].set_title("Stokes Q/I")

    ax[1, 0].plot(lambda_grid, atmos[2, :])
    ax[1, 0].set_title("Stokes U/I")

    ax[1, 1].plot(lambda_grid, atmos[3, :])
    ax[1, 1].set_title("Stokes V/I")

    for el in ax.flatten():
        el.grid(alpha=0.1)
        el.set_xlim(-.2, 0.2)

    ax[0, 0].set_ylabel("Intensity")
    ax[1, 0].set_ylabel("Intensity")
    ax[1, 0].set_xlabel("$\\lambda$ [$\\AA$]")
    ax[1, 1].set_xlabel("$\\lambda$ [$\\AA$]")

    pl.tight_layout()
    pl.savefig("/glade/u/home/mmolnar/Projects/PINNME/test_stokes.png")
    pl.show()


def convert_xy_to_rt(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)

    return r, t


def convert_rt_to_xy(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)

    return int(x), int(y)


def create_tstep(time_step, r0_inital=1, b_field_0=4000,
                 chi_rot_speed=1,
                 test_dataset_dir = "/glade/work/rjarolim/data/inversion/parameters_MEset_v4_nt_100_spatial_400_400_withPSF_11x11_s_4.npz"):
    """
    Create the test set for the PINNME code. The inputs
    time_step and r0_initial determine the shape of the
    resulting test set.

    Input:
        -- time_step, int
            Time step that increases the scale of the
            concentric circles in the test set.
        -- r0_initial, int
            Initial lengthscale for the concentric cirles in
            the test set.
        -- b_field_0, float, optional
            Central magnetic field strength, Gauss. Default = 1000
        -- chi_rot_speed, float, optional
            Rotation speed for the magnetic field azimuth.
            Default is 1.

    Output:
        Dictionary with the following keys:
        -- stokes_profiles:
            Stokes profiles from the ME calculation
        -- the rest of the keys are the 8 parameters defining
        the ME atmosphere used for the computation.

    """

    xx, yy = np.meshgrid(np.linspace(-0.5 * nx, 0.5 * nx, nx),
                         np.linspace(-0.5 * ny, 0.5 * ny, ny))

    r0 = r0_inital + time_step / 2

    r, t = convert_xy_to_rt(xx, yy)
    r = r[..., None]
    t = t[..., None]

    # Radial decline of the magnetic field with scale r0
    b_field = b_field_0 * (r0 / (r + r0)) ** 2

    # The field inclination goes through jumps every r0 pixels
    t_arr =  0.3 * r / r # ((r % r0) / r0 * np.pi)
    ch_arr =  0.5 * r / r #(t + time_step / 180 * np.pi / chi_rot_speed) # slow down the rotation

    b0_arr = b0 * (10 * r0 / (r + 10 * r0)) ** 2
    b1_arr = b1 * (10 * r0 / (r + 10 * r0)) ** 2

    lambda_end = (lambda_start + lambda_step * (-1 + n_lambda))

    lambda_grid = np.linspace(-.5 * (lambda_end - lambda_start),
                              .5 * (lambda_end - lambda_start),
                              num=n_lambda)

    atmos = MEAtmosphere(lambda0, jUp, jLow, gUp, gLow, lambda_grid)
    print(f"tarr shape is {t_arr.shape}")

    with np.load(test_dataset_dir) as blah:
        # print(f"Timestep {el} out of {ntimesteps}"
        B = blah["BField"][:, :, time_step, :]
        inc = blah["theta"][:, :, time_step, :]
        azi = blah["chi"][:, :, time_step, :]
        S0 = blah["B0"][:, :, time_step, :]
        S1 = blah["B1"][:, :, time_step, :]
        vDop = blah["vmac"][:, :, time_step, :]   # make it in AA
        damping = blah["damping"][:, :, time_step, :]
        eta_l = blah["kl"][:, :, time_step, :]
        vlos = blah["vdop"][:, :, time_step, :]  # make it in km/s

    b_field = torch.tensor(B, dtype=torch.float32)
    print(f"Shape of b_field is: {b_field.shape}")

    t_arr = torch.tensor(inc, dtype=torch.float32)
    ch_arr = torch.tensor(azi, dtype=torch.float32)
    b0_arr = torch.tensor(S0, dtype=torch.float32)
    b1_arr = torch.tensor(S1, dtype=torch.float32)
    vmac_arr = torch.tensor(vDop, dtype=torch.float32) * 1e3 * 1.35 # make it in meters!
    damping_arr = torch.tensor(damping, dtype=torch.float32) / 10
    mu_arr = mu * torch.ones_like(b_field)
    vdop_arr = torch.tensor(vlos, dtype=torch.float32)
    kl_arr = torch.tensor(eta_l, dtype=torch.float32)


    I, Q, U, V = atmos.forward(b_field, t_arr, ch_arr,
                               vmac_arr, damping_arr,
                               b0_arr, b1_arr, mu_arr,
                               vdop_arr, kl_arr)

    stokes_profiles = torch.cat([I, Q, U, V]).cpu().numpy()

    return {'stokes_profiles': stokes_profiles, 'b_field': b_field, 'theta': t_arr, 'chi': ch_arr,
            'b0': b0_arr, 'b1': b1_arr, 'vmac': vmac_arr, 'damping': damping_arr, 'mu': mu_arr,
            'vdop': vdop_arr, 'kl': kl_arr, 'lambda_grid':lambda_grid}


if __name__ == '__main__':

    lambda0 = 6302.4931 * u.AA
    jUp = 1
    jLow = 0
    gUp = 2.49
    gLow = 0
    lambda_start = 6301.87 * u.AA
    lambda_step = 0.022 * u.AA
    n_lambda = 56

    # Inputs for the inversion
    b_field_0 = 3000.0
    theta = np.deg2rad(10.0)
    chi = np.deg2rad(10.0)
    vmac = 0.5 * 1e3
    damping = 0.2
    b0 = 0.8
    b1 = 0.2
    mu = 1.0
    vdop = 2.0 * 1e3
    kl = 1.0

    nStokes = 4
    nx = 400
    ny = 400

    nTime = 1

    with Pool(1) as p:
        profiles = list(tqdm(p.imap(create_tstep, range(nTime)), total=nTime))


    stokes_profiles = np.stack([p['stokes_profiles'] for p in profiles],
                               axis=1)
    lambda_grid = profiles[0]['lambda_grid']

    print(stokes_profiles.shape)
    plot_all_Stokes(lambda_grid,
                    stokes_profiles[:, 0, 0, 0, :])

    pl.savefig('/glade/u/home/mmolnar/PINNME_results/test_stokes.png')

    b0_arr = np.stack([p['b0'] for p in profiles], axis=0)
    b1_arr = np.stack([p['b1'] for p in profiles], axis=0)
    t_arr = np.stack([p['theta'] for p in profiles], axis=0)
    ch_arr = np.stack([p['chi'] for p in profiles], axis=0)
    b_field_arr = np.stack([p['b_field'] for p in profiles], axis=0)
    vmac = np.stack([p['vmac'] for p in profiles], axis=0)
    # keys = profiles[0].keys()
    # parameters = {key: np.stack([p[key] for p in profiles], axis=2)
    #               for key in keys if key != 'stokes_profiles'}

    # plot_all_Stokes(atmos)
    for el in range(nTime):
        im1 = pl.imshow(t_arr[el, :, :],
                        vmin=0,
                        vmax=np.pi,
                        cmap="RdBu_r")
        pl.colorbar(im1)
        pl.title(f"$\\theta$ Timestep {el}")
        pl.savefig(
            f"/glade/work/mmolnar/data/inversion/test/t_arr_{el:03d}.png")
        pl.close()

    for el in range(nTime):
        im1 = pl.imshow(np.log10(b_field_arr[el, :, :]),
                        # vmin=0, vmax=np.log10(2000),
                        cmap="cividis")
        pl.colorbar(im1)
        pl.title(f"Bfield [G] Timestep {el}")
        pl.savefig(
            f"/glade/work/mmolnar/data/inversion/test/Bf_arr_{el:03d}.png")
        pl.close()

    for el in range(nTime):
        im1 = pl.imshow(b0_arr[el, :, :], cmap="cividis")
        pl.colorbar(im1)
        pl.title(f"B0 Timestep {el}")
        pl.savefig(
            f"/glade/work/mmolnar/data/inversion/test/B0_arr_{el:03d}.png")
        pl.close()
    for el in range(nTime):
        im1 = pl.imshow(b1_arr[el, :, :], cmap="cividis")
        pl.colorbar(im1)
        pl.title(f"B1 Timestep {el}")
        pl.savefig(
            f"/glade/work/mmolnar/data/inversion/test/B1_arr_{el:03d}.png")
        pl.close()

    for el in range(nTime):
        im1 = pl.imshow(ch_arr[el, :, :] % np.pi, vmin=0, vmax=np.pi, cmap="twilight_shifted")
        pl.colorbar(im1)
        pl.title(f"$\chi$ Timestep {el}")
        pl.savefig(f"/glade/work/mmolnar/data/inversion/test/ch_arr_{el:03d}.png")
        pl.close()



    nameTestCase = "dataset_PINNME_synthesis_PINNME_format"

    np.savez(f"/glade/work/mmolnar/PINN-ME/pymilne_comparison_v1/data/{nameTestCase}.npz",
             wavelength=lambda_grid,
             stokes_parameters=stokes_profiles)

    nameTestCase = "dataset_PINNME_synthesis_pymilne_format"

    np.savez(f"/glade/work/mmolnar/PINN-ME/pymilne_comparison_v1/data/{nameTestCase}.npz",
             wavelength=lambda_grid,
             stokes_parameters=np.swapaxes(np.swapaxes(stokes_profiles, 0, -2), 1, 2))

    # np.savez(f"/glade/work/mmolnar/data/inversion/{nameTestCase}_{nTime}_spatial_{nx}_{ny}.npz",
    #          stokes_profiles=stokes_profiles)
    #
    # np.savez(f"/glade/work/mmolnar/data/inversion/parameters_{nameTestCase}_{nTime}_spatial_{nx}_{ny}.npz",
    #          **parameters)