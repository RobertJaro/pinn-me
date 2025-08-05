import argparse
import glob
import os.path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, \
    image_to_spherical_matrix
from pme.evaluation.loader import PINNMEOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
    parser.add_argument('--input', type=str, help='the path to the input file')
    parser.add_argument('--ref_maps', type=str, help='the path to the reference map fld')
    parser.add_argument('--output', type=str, help='the path to the output file', default=None, required=False)
    args = parser.parse_args()

    in_path = args.input

    out_path = args.output
    out_path = out_path if out_path is not None else os.path.join(os.path.dirname(in_path), 'evaluation', 'ref_series')
    os.makedirs(out_path, exist_ok=True)

    # load
    pinnme = PINNMEOutput(in_path)

    # load reference maps
    ref_maps = sorted(glob.glob(args.ref_maps))

    for i, f in tqdm(enumerate(ref_maps), total=len(ref_maps)):
        img_path = os.path.join(out_path, f'step{i:03d}.jpg')
        # if os.path.exists(out_path):
        #     continue

        ref_map = Map(f)
        # bl = SkyCoord(-500 * u.arcsec, -500 * u.arcsec, frame=ref_map.coordinate_frame)
        # tr = SkyCoord(500 * u.arcsec, 500 * u.arcsec, frame=ref_map.coordinate_frame)
        # ref_map = ref_map.submap(bl, top_right=tr)
        ref_map = ref_map.resample((512, 512) * u.pix)  # resample to 128x128 pixels

        # load time
        target_time = ref_map.date.to_datetime()
        normalized_time = pinnme._normalize_time(target_time)

        coords = all_coordinates_from_map(ref_map).transform_to(frames.HeliographicCarrington)
        lat, lon = coords.lat.to_value(u.rad), coords.lon.to_value(u.rad)
        colat = np.pi / 2 - lat  # convert to colatitude
        r = np.ones_like(lat)  # coords.radius.to_value(u.solRad)

        spherical_coords = np.stack([r, colat, lon], axis=-1)
        #
        cartesian_coords = spherical_to_cartesian(spherical_coords) / pinnme.Rs_per_ds
        time_coords = np.ones((*cartesian_coords.shape[:-1], 1), dtype=np.float32) * normalized_time
        coords = np.concatenate([time_coords, cartesian_coords], axis=-1)
        cartesian_to_spherical_transform = cartesian_to_spherical_matrix(spherical_coords)

        pAng = -np.deg2rad(ref_map.meta.get('CROTA2', 0))
        latc, lonc = ref_map.carrington_latitude.to_value(u.rad), ref_map.carrington_longitude.to_value(u.rad)
        a_matrix = image_to_spherical_matrix(lon, lat, lonc, latc, pAng=pAng)
        rtp_to_img_transform = np.linalg.inv(a_matrix)

        parameter_cube = pinnme.load_parameters(coords=coords, progress=False)
        b_xyz = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']], axis=-1)
        b_xyz *= pinnme.gauss_per_dB
        b_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, b_xyz)

        v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']], axis=-1)
        v_xyz *= pinnme.meters_per_ds / pinnme.seconds_per_dt / 1000  # convert to km/s
        v_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, v_xyz)

        b_img = np.einsum('...ij,...j->...i', rtp_to_img_transform, b_rtp)
        v_img = np.einsum('...ij,...j->...i', rtp_to_img_transform, v_rtp)

        b_field = np.linalg.norm(b_img, axis=-1)
        inc = np.arccos(b_img[..., 2] / (b_field + 1e-8))
        azi = np.arctan2(-b_img[..., 0], b_img[..., 1])

        ########################################################################################################################
        # Plot subframe in B_r, B_theta, B_phi

        b_norm = Normalize(-500, 500)

        fig, axs = plt.subplots(4, 3, figsize=(10, 10), subplot_kw={'projection': ref_map})

        ax = axs[0, 0]
        im = ax.imshow(b_rtp[..., 0], cmap='gray', norm=b_norm, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
        ax.set_title(r'$B_\text{r}$ [G]')

        ax = axs[0, 1]
        im = ax.imshow(b_rtp[..., 1], cmap='gray', norm=b_norm, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
        ax.set_title(r'$B_\text{t}$ [G]')

        ax = axs[0, 2]
        im = ax.imshow(b_rtp[..., 2], cmap='gray', norm=b_norm, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
        ax.set_title(r'$B_\text{p}$ [G]')

        ax = axs[1, 0]
        im = ax.imshow(b_field, cmap='viridis', origin='lower', vmin=1, vmax=1e3, norm='log')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

        ax = axs[1, 1]
        im = ax.imshow(np.rad2deg(inc % np.pi), cmap='seismic_r', origin='lower', vmin=0, vmax=180)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\theta$ [deg]')

        ax = axs[1, 2]
        im = ax.imshow(np.rad2deg(azi % (2 * np.pi)), cmap='twilight', origin='lower', vmin=0, vmax=360)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [deg]')

        ax = axs[2, 0]
        im = ax.imshow(v_rtp[..., 0], cmap='seismic_r', origin='lower', vmin=-2, vmax=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\text{r  }$ [km/s]')
        ax.set_title(r'$v_\text{rad}$ [km/s]')

        ax = axs[2, 1]
        im = ax.imshow(v_rtp[..., 1], cmap='seismic_r', origin='lower', vmin=-2, vmax=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\theta$ [km/s]')
        ax.set_title(r'$v_\theta$ [km/s]')

        ax = axs[2, 2]
        im = ax.imshow(v_rtp[..., 2], cmap='seismic_r', origin='lower', vmin=-2, vmax=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\phi$ [km/s]')
        ax.set_title(r'$v_\phi$ [km/s]')

        ax = axs[3, 0]
        im = ax.imshow(v_img[..., 2], cmap='seismic_r', origin='lower', vmin=-2, vmax=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\text{dop}$ [km/s]')
        ax.set_title(r'$v_\text{LOS}$ [km/s]')

        ax = axs[3, 1]
        im = ax.imshow(np.rad2deg(spherical_coords[..., 1]), cmap='RdBu_r', vmin=0, vmax=180, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'Colatitude [deg]')
        ax.set_title(r'Colatitude [deg]')

        ax = axs[3, 2]
        im = ax.imshow(np.rad2deg(spherical_coords[..., 2]), cmap='twilight', vmin=0, vmax=360, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, orientation='vertical', label=r'Longitude [deg]')
        ax.set_title(r'Longitude [deg]')

        [ax.set_xlabel(' ') for ax in axs.flatten()]
        [ax.set_ylabel(' ') for ax in axs.flatten()]
        [ax.set_ylabel('Latitude [deg]') for ax in axs[:, 0]]
        [ax.set_xlabel('Longitude [deg]') for ax in axs[-1]]

        # add subtitle with date
        plt.suptitle(f'{target_time}', fontsize=16)

        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.close()
