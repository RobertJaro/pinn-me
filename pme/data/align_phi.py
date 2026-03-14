import glob
import os.path
from multiprocessing.pool import Pool

import numpy as np
from astropy import units as u
from astropy.io import fits
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sunpy.map import Map, all_coordinates_from_map

from pme.data.phi_util import load_fix_phi_header


def _normalize_data(x, m):
    v = x[m]
    mu = np.nanmean(v)
    sig = np.nanstd(v)
    return (x - mu) / (sig + 1e-12)


def apply_roll_to_header(header: fits.Header, dtheta: u.Quantity) -> fits.Header:
    """
    Update FITS WCS keywords to apply a new roll angle WITHOUT rotating pixel data.

    - If PC matrix is present: PC' = R(dtheta) @ PC
    - Else if CD matrix is present: CD' = R(dtheta) @ CD
    - Else: update CROTA2
    """
    hdr = header.copy()

    dtheta = dtheta.to_value(u.rad)

    c, s = np.cos(dtheta), np.sin(dtheta)
    R = np.array([[c, -s],
                  [s, c]], dtype=float)

    # PC case
    if all(k in hdr for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
        PC = np.array([[float(hdr["PC1_1"]), float(hdr["PC1_2"])],
                       [float(hdr["PC2_1"]), float(hdr["PC2_2"])]], dtype=float)
        PCn = R @ PC
        hdr["PC1_1"], hdr["PC1_2"] = float(PCn[0, 0]), float(PCn[0, 1])
        hdr["PC2_1"], hdr["PC2_2"] = float(PCn[1, 0]), float(PCn[1, 1])

        return hdr

    # CD case
    elif all(k in hdr for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        CD = np.array([[float(hdr["CD1_1"]), float(hdr["CD1_2"])],
                       [float(hdr["CD2_1"]), float(hdr["CD2_2"])]], dtype=float)
        CDn = R @ CD
        hdr["CD1_1"], hdr["CD1_2"] = float(CDn[0, 0]), float(CDn[0, 1])
        hdr["CD2_1"], hdr["CD2_2"] = float(CDn[1, 0]), float(CDn[1, 1])

        return hdr

    else:
        raise ValueError("Header does not contain PC or CD matrix for WCS transformation.")


def crop_off_limb(s_map: Map, limit=0.9):
    coords = all_coordinates_from_map(s_map)
    radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs
    radius = radius.to_value(u.dimensionless_unscaled)
    # crop off-limb
    mask = radius > limit
    s_map.data[mask] = np.nan


def detrend_mu(s_map, deg=3):
    coords = all_coordinates_from_map(s_map)
    radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs
    radius = radius.to_value(u.dimensionless_unscaled)
    mu = np.sqrt(1 - radius ** 2)

    valid = (np.isnan(s_map.data) == False) & (mu >= 0)
    x = mu[valid].ravel()
    y = s_map.data[valid].ravel()

    # Fit polynomial y ≈ f(mu)
    p = np.polyfit(x, y, deg=deg)
    f = np.polyval(p, mu)

    s_map.data[:] = s_map.data - f  # additive detrend


class _MapAligner:
    def __init__(self, out_path, iterations=10, initial_angle_range=10):
        self.iterations = iterations
        self.initial_angle_range = initial_angle_range
        self.out_path = out_path

    def align_phi_to_hmi(self, file_pair):
        phi_file, hmi_file = file_pair
        base_hmi_name = hmi_file.replace('.I0.', '.I%d.')
        hmi_files = [base_hmi_name % i for i in range(0, 6)]
        # load PHI map
        phi_header = load_fix_phi_header(phi_file)
        phi_data = fits.getdata(phi_file)
        # --> (x, y, stokes, wl)
        phi_stokes = np.transpose(phi_data, (2, 3, 1, 0))

        phi_I = phi_stokes[:, :, 0, :].mean(-1)  # average over wavelength_grid
        phi_map = Map(phi_I, phi_header)

        # load HMI maps and average
        hmi_I = np.stack([Map(f).data for f in hmi_files], axis=-1).mean(-1)
        hmi_map = Map(hmi_I, fits.getheader(hmi_files[0], 1))

        # crop all off limb pixels
        crop_off_limb(hmi_map)
        crop_off_limb(phi_map)

        # detrend center-to-limb variation
        detrend_mu(hmi_map)
        detrend_mu(phi_map)

        # set common rsun_meters - is this necessary?
        rsun_ref = hmi_map.rsun_meters.to_value(u.m)
        phi_header['rsun_ref'] = rsun_ref

        # resample to lower resolution
        resampled_hmi_map = hmi_map.resample((256, 256) * u.pixel)

        # iteratively change PHI rotation angle
        # 1) sample rotation angles
        # 2) update PHI header with new rotation angle
        # 3) project PHI map to HMI map
        # 4) compute difference metric --> correlation coefficient
        # 5) find optimal rotation angle
        # 6) sample finer around optimal angle
        # 7) repeat until convergence

        initial_angle_range = self.initial_angle_range  # degrees
        angles = np.linspace(-initial_angle_range, initial_angle_range, 11)  # initial coarse sampling
        best_dangle = 0
        best_corr = -1

        for iteration in range(self.iterations):
            corrs = []
            for angle in angles:
                # update header
                rotated_phi_header = apply_roll_to_header(phi_header, angle * u.deg)
                rotated_phi_map = Map(phi_map.data, rotated_phi_header)
                # project to HMI
                projected_map = rotated_phi_map.reproject_to(resampled_hmi_map.wcs)
                # filter NaNs
                projected_data = projected_map.data
                hmi_data = resampled_hmi_map.data
                valid_mask = ~np.isnan(projected_data) & ~np.isnan(hmi_data)
                # normalize data
                projected_data = _normalize_data(projected_data, valid_mask)
                hmi_data = _normalize_data(hmi_data, valid_mask)
                # compute correlation
                if np.sum(valid_mask) == 0:
                    corr = -1
                else:
                    corr = np.corrcoef(projected_data[valid_mask].flatten(),
                                       hmi_data[valid_mask].flatten())[0, 1]
                corrs.append(corr)

            corrs = np.array(corrs)
            best_idx = np.argmax(corrs)
            best_dangle = angles[best_idx]
            best_corr = corrs[best_idx]

            # print(f"Iteration {iteration}: Best angle = {best_dangle:.4f}, Correlation = {best_corr:.4f}")

            # refine angles around best angle
            angle_range = initial_angle_range / (2 ** (iteration + 1))
            angles = np.linspace(best_dangle - angle_range, best_dangle + angle_range, 11)

            # increase map resolution to 512x512 after 5 iterations
            if iteration == 4:
                resampled_hmi_map = hmi_map.resample((512, 512) * u.pixel)
            # increase map resolution to 1024x1024 after 7 iterations
            if iteration == 6:
                resampled_hmi_map = hmi_map.resample((1024, 1024) * u.pixel)

        # plot comparison between original PHI, aligned PHI, and HMI
        original_phi_map = Map(phi_map.data, phi_header)
        original_phi = original_phi_map.reproject_to(resampled_hmi_map.wcs)

        rotated_phi_header = apply_roll_to_header(phi_header, best_dangle * u.deg)
        rotated_phi_map = Map(phi_map.data, rotated_phi_header)
        aligned_phi = rotated_phi_map.reproject_to(resampled_hmi_map.wcs)

        original_phi_data = original_phi.data
        aligned_phi_data = aligned_phi.data
        hmi_data = resampled_hmi_map.data
        # normalize
        valid_mask = ~np.isnan(original_phi_data) & ~np.isnan(hmi_data) & ~np.isnan(aligned_phi_data)
        original_phi_data[~valid_mask] = np.nan
        aligned_phi_data[~valid_mask] = np.nan
        hmi_data[~valid_mask] = np.nan
        original_phi_data = _normalize_data(original_phi_data, valid_mask)
        aligned_phi_data = _normalize_data(aligned_phi_data, valid_mask)
        hmi_data = _normalize_data(hmi_data, valid_mask)

        # compute differences
        diff_orig = original_phi_data - hmi_data
        diff_aligned = aligned_phi_data - hmi_data
        vmin = -max(np.nanmax(np.abs(diff_orig)), np.nanmax(np.abs(diff_aligned)))
        vmax = max(np.nanmax(np.abs(diff_orig)), np.nanmax(np.abs(diff_aligned)))

        norm = Normalize()

        fig, axes = plt.subplots(
            2, 3, figsize=(15, 9),
            subplot_kw={'projection': hmi_map}
        )

        # --- top row: images
        im00 = axes[0, 0].imshow(original_phi_data, origin='lower', cmap='gray', norm=norm)
        axes[0, 0].set_title('Original PHI (reproj)')
        plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im01 = axes[0, 1].imshow(aligned_phi_data, origin='lower', cmap='gray', norm=norm)
        axes[0, 1].set_title('Aligned PHI (reproj)')
        plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im02 = axes[0, 2].imshow(hmi_data, origin='lower', cmap='gray', norm=norm)
        axes[0, 2].set_title('HMI')
        plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # --- bottom row: differences (use diverging colormap)
        im10 = axes[1, 0].imshow(diff_orig, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Original PHI − HMI')
        plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im11 = axes[1, 1].imshow(diff_aligned, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Aligned PHI − HMI')
        plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # bottom-right: plot contour overlay of PHI and HMI
        im12 = axes[1, 2].imshow(hmi_data, origin='lower', cmap='gray', norm=norm)
        axes[1, 2].contour(aligned_phi_data, levels=[-10], colors='red', alpha=0.5)
        axes[1, 2].contour(original_phi_data, levels=[-10], colors='blue', alpha=0.5)
        axes[1, 2].set_title('Contour Overlay (red: aligned PHI, blue: original PHI)')
        plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # plot grid lines and solar limb for each axis
        for ax in axes.flatten():
            hmi_map.draw_grid(axes=ax, grid_spacing=20 * u.deg, color='white', alpha=0.5, linewidth=0.5)
            hmi_map.draw_limb(axes=ax, color='yellow', alpha=0.7, linewidth=1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_path, f"{os.path.basename(phi_file).replace('.fits.gz', '_alignment.jpg')}"),
                    dpi=300)
        plt.close(fig)

        # save final aligned map
        out_fits = os.path.join(self.out_path, os.path.basename(phi_file).replace(".fits.gz", "_rollcorr.fits.gz"))
        fits.writeto(out_fits,
                     data=phi_data, header=rotated_phi_header, overwrite=True, output_verify="ignore")

        return {'best_dangle': best_dangle, 'best_correlation': best_corr}


if __name__ == '__main__':
    hmi_path = "/glade/work/rjarolim/data/hmi_stokes/20240327_3h"
    phi_path = "/glade/work/rjarolim/data/phi_fdt/2024-03-27"
    out_path = "/glade/work/rjarolim/spinn_me/align_phi"
    os.makedirs(out_path, exist_ok=True)

    phi_files = sorted(glob.glob(os.path.join(phi_path, '*.fits.gz')))
    hmi_I_files = sorted(glob.glob(os.path.join(hmi_path, '*.I0.fits')))

    hmi_times = [parse(fits.getheader(f, 1)['DATE-OBS']) for f in hmi_I_files]
    phi_times = [parse(fits.getheader(f)['DATE-OBS']) for f in phi_files]

    # for each PHI file, find closest HMI file
    phi_hmi_pairs = []
    for phi_file, phi_time in zip(phi_files, phi_times):
        time_diffs = [abs((phi_time - hmi_time).total_seconds()) for hmi_time in hmi_times]
        best_hmi_idx = np.argmin(time_diffs)
        best_hmi_file = hmi_I_files[best_hmi_idx]
        phi_hmi_pairs.append((phi_file, best_hmi_file))

    aligner = _MapAligner(out_path, iterations=10)

    with Pool(4) as p:
        for aligned_out in p.imap(aligner.align_phi_to_hmi, phi_hmi_pairs):
            best_dangle, best_corr = aligned_out['best_dangle'], aligned_out['best_correlation']
            print(
                f"Optimal rotation for PHI: {best_dangle:.4f} degrees with correlation {best_corr:.4f}; file: {phi_file}")
