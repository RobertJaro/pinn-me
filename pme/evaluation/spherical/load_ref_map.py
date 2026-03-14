import argparse
import os.path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from pme.data.differential_rotation import carrington_rotation_velocity
from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, \
    image_to_spherical_matrix, spherical_to_cartesian_matrix
from pme.evaluation.loader import PINNMEOutput, SPINNMEOutput
from pme.loader.spherical import load_v_observer_LOS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
    parser.add_argument('--input', type=str, help='the path to the input file')
    parser.add_argument('--ref_map_fld', type=str, help='the path to the reference map fld')
    parser.add_argument('--ref_map_inc', type=str, help='the path to the reference map inc')
    parser.add_argument('--ref_map_azi', type=str, help='the path to the reference map azi')
    parser.add_argument('--ref_map_disambig', type=str, help='the path to the reference map disambig')
    parser.add_argument('--ref_map_vlos_mag', type=str, help='the path to the reference map vlos_mag')
    parser.add_argument('--output', type=str, help='the path to the output file', default=None)
    parser.add_argument('--hpc_range', type=float, nargs=4, default=None, required=False)
    args = parser.parse_args()

    in_path = args.input

    out_path = args.output
    out_path = out_path if out_path is not None else os.path.join(os.path.dirname(in_path), 'evaluation')
    os.makedirs(out_path, exist_ok=True)

    # load
    pinnme = SPINNMEOutput(in_path)

    # load reference maps
    ref_map = Map(args.ref_map_fld)

    # convert HPC min/max to world pixel coordinates
    if args.hpc_range is not None:
        min_hpc_x, max_hpc_x, min_hpc_y, max_hpc_y = args.hpc_range
        bl_coord = SkyCoord(Tx=min_hpc_x * u.arcsec, Ty=min_hpc_y * u.arcsec, frame=ref_map.coordinate_frame)
        tr_coord = SkyCoord(Tx=max_hpc_x * u.arcsec, Ty=max_hpc_y * u.arcsec, frame=ref_map.coordinate_frame)
        ref_map = ref_map.submap(bottom_left=bl_coord, top_right=tr_coord)

    # load time
    target_time = ref_map.date.to_datetime()
    normalized_time = pinnme._normalize_time(target_time)

    coords = all_coordinates_from_map(ref_map).transform_to(frames.HeliographicCarrington)
    lat, lon = coords.lat.to_value(u.rad), coords.lon.to_value(u.rad)
    r = np.ones_like(lat)  # coords.radius.to_value(u.solRad)

    spherical_coords = np.stack([r, np.pi / 2 - lat, lon], axis=-1)
    #
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    time_coords = np.ones((*cartesian_coords.shape[:-1], 1), dtype=np.float32) * normalized_time
    coords = np.concatenate([time_coords, cartesian_coords], axis=-1)
    cartesian_to_spherical_transform = cartesian_to_spherical_matrix(spherical_coords)

    latc, lonc = np.deg2rad(ref_map.meta['CRLT_OBS']), np.deg2rad(ref_map.meta['CRLN_OBS'])
    pAng = -np.deg2rad(ref_map.meta['CROTA2'])
    a_matrix = image_to_spherical_matrix(lon, lat, lonc, latc, pAng=pAng)
    rtp_to_img_transform = np.linalg.inv(a_matrix)

    parameter_cube = pinnme.load_parameters(coords=coords)
    b_xyz= np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']], axis=-1) * pinnme.gauss_per_dB
    b_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, b_xyz)
    b_img = np.einsum('...ij,...j->...i', rtp_to_img_transform, b_rtp)

    fld = np.linalg.norm(b_img, axis=-1, keepdims=True)
    inc = np.arccos(b_img[..., 2:3] / (fld + 1e-8))
    azi = np.arctan2(-b_img[..., 0:1], b_img[..., 1:2])

    inc = np.rad2deg(inc)
    azi = np.rad2deg(azi)

    v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']], axis=-1) * pinnme.meters_per_ds / pinnme.seconds_per_dt
    v_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, v_xyz)
    v_img = np.einsum('...ij,...j->...i', rtp_to_img_transform, v_rtp)

    v_los = -v_img[..., 2]  # negative because doppler shift is defined in the observer frame

    ########################################################################################################################
    # load reference map
    fld_ref = Map(args.ref_map_fld).reproject_to(ref_map.wcs).data
    inc_ref = Map(args.ref_map_inc).reproject_to(ref_map.wcs).data
    azi_ref = Map(args.ref_map_azi).reproject_to(ref_map.wcs).data
    amb_ref = Map(args.ref_map_disambig).reproject_to(ref_map.wcs).data
    vlos_ref = Map(args.ref_map_vlos_mag).reproject_to(ref_map.wcs).data
    vlos_ref = vlos_ref / 100  # convert cm/s to m/s

    # correct for observer LOS
    v_obs_los = load_v_observer_LOS(ref_map)
    vlos_ref = vlos_ref - v_obs_los

    # correct for carrington rotation
    v_carr = carrington_rotation_velocity(latitude=lat, radius=r * pinnme.meters_per_ds, f=np)
    v_carr_rtp = np.stack([np.zeros_like(v_carr), np.zeros_like(v_carr), v_carr], axis=-1)
    v_carr_img = np.einsum('...ij,...j->...i', rtp_to_img_transform, v_carr_rtp)
    vlos_ref = vlos_ref + v_carr_img[..., 2]  # subtract LOS component of carrington rotation velocity

    # disambiguate
    amb_weak = 2
    condition = (amb_ref.astype(int) >> amb_weak).astype(bool)
    azi_ref[condition] += 180

    ########################################################################################################################
    # transform to B_r, B_theta, B_phi
    b_xi = - fld_ref * np.sin(np.deg2rad(inc_ref)) * np.sin(np.deg2rad(azi_ref))
    b_eta = fld_ref * np.sin(np.deg2rad(inc_ref)) * np.cos(np.deg2rad(azi_ref))
    b_zeta = fld_ref * np.cos(np.deg2rad(inc_ref))

    b_img_ref = np.stack([b_xi, b_eta, b_zeta], axis=-1)

    b_rtp_ref = np.einsum('...ij,...j->...i', a_matrix, b_img_ref)
    spherical_to_cartesian_matrix = spherical_to_cartesian_matrix(spherical_coords)
    b_xyz_ref = np.einsum('...ij,...j->...i', spherical_to_cartesian_matrix, b_rtp_ref)

    ########################################################################################################################
    # disambiguate PINN ME
    azi_disambig = azi % 180
    azi_disambig[condition] += 180

    b_xi_disambig = - fld * np.sin(np.deg2rad(inc)) * np.sin(np.deg2rad(azi_disambig))
    b_eta_disambig = fld * np.sin(np.deg2rad(inc)) * np.cos(np.deg2rad(azi_disambig))
    b_zeta_disambig = fld * np.cos(np.deg2rad(inc))

    b_img_disambig = np.concatenate([b_xi_disambig, b_eta_disambig, b_zeta_disambig], axis=-1)

    b_rtp_disambig = np.einsum('...ij,...j->...i', a_matrix, b_img_disambig)
    b_xyz_disambig = np.einsum('...ij,...j->...i', spherical_to_cartesian_matrix, b_rtp_disambig)

    ########################################################################################################################
    # Plot in B_r, B_theta, B_phi

    norm = Normalize(-500, 500)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(b_rtp[..., 0], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('PINN ME $B_r$')

    ax = axs[0, 1]
    im = ax.imshow(b_rtp[..., 1], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
    ax.set_title('PINN ME $B_t$')

    ax = axs[0, 2]
    im = ax.imshow(b_rtp[..., 2], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
    ax.set_title('PINN ME $B_p$')

    ax = axs[1, 0]
    im = ax.imshow(b_rtp_ref[..., 0], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('Reference $B_r$')

    ax = axs[1, 1]
    im = ax.imshow(b_rtp_ref[..., 1], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
    ax.set_title('Reference $B_t$')

    ax = axs[1, 2]
    im = ax.imshow(b_rtp_ref[..., 2], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
    ax.set_title('Reference $B_p$')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel('Latitude [deg]') for ax in axs[:, 0]]
    [ax.set_xlabel('Longitude [deg]') for ax in axs[-1]]

    # add subtitle with date
    plt.suptitle(f'Map at {target_time}', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'reference_comparison.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # Plot disambiguated B_r, B_theta, B_phi

    norm = Normalize(-500, 500)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(b_rtp_disambig[..., 0], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('PINN ME $B_r$')

    ax = axs[0, 1]
    im = ax.imshow(b_rtp_disambig[..., 1], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
    ax.set_title('PINN ME $B_t$')

    ax = axs[0, 2]
    im = ax.imshow(b_rtp_disambig[..., 2], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
    ax.set_title('PINN ME $B_p$')

    ax = axs[1, 0]
    im = ax.imshow(b_rtp_ref[..., 0], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('Reference $B_r$')

    ax = axs[1, 1]
    im = ax.imshow(b_rtp_ref[..., 1], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
    ax.set_title('Reference $B_t$')

    ax = axs[1, 2]
    im = ax.imshow(b_rtp_ref[..., 2], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
    ax.set_title('Reference $B_p$')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel('Latitude [deg]') for ax in axs[:, 0]]
    [ax.set_xlabel('Longitude [deg]') for ax in axs[-1]]

    # add subtitle with date
    plt.suptitle(f'Map at {target_time}', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'reference_comparison_disambiguated.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # plot field strength

    b_norm = np.linalg.norm(b_rtp, axis=-1)
    b_norm_ref = np.linalg.norm(b_rtp_ref, axis=-1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': ref_map})

    ax = axs[0]
    im = ax.imshow(b_norm, cmap='viridis', origin='lower', vmin=1, vmax=3000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')
    ax.set_title('PINN ME')

    ax = axs[1]
    im = ax.imshow(b_norm_ref, cmap='viridis', origin='lower', vmin=1, vmax=3000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')
    ax.set_title('Reference')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    fig.tight_layout()
    plt.savefig(os.path.join(out_path, 'field_strength_comparison.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # plot fld, inc, azi

    fig, axs = plt.subplots(2, 4, figsize=(15, 5), subplot_kw={'projection': ref_map})

    ax = axs[1, 0]
    im = ax.imshow(fld_ref, cmap='viridis', origin='lower', vmin=1, vmax=2000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

    ax = axs[1, 1]
    im = ax.imshow(inc_ref % 180, cmap='PiYG', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\theta$ [deg]')

    ax = axs[1, 2]
    im = ax.imshow(azi_ref % 180, cmap='twilight', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [deg]')

    ax = axs[1, 3]
    im = ax.imshow(azi_ref % 360, cmap='twilight', origin='lower', vmin=0, vmax=360)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [deg]')

    ax = axs[0, 0]
    im = ax.imshow(fld, cmap='viridis', origin='lower', vmin=1, vmax=2000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

    ax = axs[0, 1]
    im = ax.imshow(inc % 180, cmap='PiYG', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\theta$ [deg]')

    ax = axs[0, 2]
    im = ax.imshow(azi % 180, cmap='twilight', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [deg]')

    ax = axs[0, 3]
    im = ax.imshow(azi % 360, cmap='twilight', origin='lower', vmin=0, vmax=360)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [deg]')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    axs[0, 0].set_ylabel('PINN ME')
    axs[1, 0].set_ylabel('Reference')

    fig.tight_layout()
    plt.savefig(os.path.join(out_path, 'fld_inc_azi_comparison.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # plot coordinates
    fig, axs = plt.subplots(2, 3, figsize=(10, 5), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(spherical_coords[..., 0], cmap='viridis', origin='lower', vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Radius [R_s]')
    ax.set_title('Radius')

    ax = axs[0, 1]
    im = ax.imshow(np.rad2deg(spherical_coords[..., 1]), cmap='PiYG', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Latitude [deg]')
    ax.set_title('Co-Latitude')

    ax = axs[0, 2]
    im = ax.imshow(np.rad2deg(spherical_coords[..., 2]), cmap='twilight', origin='lower', vmin=0, vmax=360)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Longitude [deg]')
    ax.set_title('Longitude')

    ax = axs[1, 0]
    im = ax.imshow(cartesian_coords[..., 0], cmap='viridis', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='X [R_s]')
    ax.set_title('X')

    ax = axs[1, 1]
    im = ax.imshow(cartesian_coords[..., 1], cmap='viridis', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Y [R_s]')
    ax.set_title('Y')

    ax = axs[1, 2]
    im = ax.imshow(cartesian_coords[..., 2], cmap='viridis', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Z [R_s]')
    ax.set_title('Z')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    axs[0, 0].set_ylabel('Spherical Coordinates')
    axs[1, 0].set_ylabel('Cartesian Coordinates')

    fig.tight_layout()
    plt.savefig(os.path.join(out_path, 'coordinates.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # plot cartesian
    fig, axs = plt.subplots(3, 3, figsize=(10, 8), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(b_xyz_ref[..., 0], cmap='gray', origin='lower', vmin=-500, vmax=500)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='$B_x$ [G]')
    ax.set_title('Reference $B_x$')

    ax = axs[0, 1]
    im = ax.imshow(b_xyz_ref[..., 1], cmap='gray', origin='lower', vmin=-500, vmax=500)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='$B_y$ [G]')
    ax.set_title('Reference $B_y$')

    ax = axs[0, 2]
    im = ax.imshow(b_xyz_ref[..., 2], cmap='gray', origin='lower', vmin=-500, vmax=500)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='$B_z$ [G]')
    ax.set_title('Reference $B_z$')

    ax = axs[1, 0]
    im = ax.imshow(b_xyz[..., 0], cmap='gray', origin='lower', vmin=-500, vmax=500)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='$B_x$ [G]')
    ax.set_title('PINN ME $B_x$')

    ax = axs[1, 1]
    im = ax.imshow(b_xyz[..., 1], cmap='gray', origin='lower', vmin=-500, vmax=500)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='$B_y$ [G]')
    ax.set_title('PINN ME $B_y$')

    ax = axs[1, 2]
    im = ax.imshow(b_xyz[..., 2], cmap='gray', origin='lower', vmin=-500, vmax=500)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='$B_z$ [G]')
    ax.set_title('PINN ME $B_z$')

    ax = axs[2, 0]
    im = ax.imshow(cartesian_coords[..., 0], cmap='viridis', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='X [R_s]')
    ax.set_title('X')

    ax = axs[2, 1]
    im = ax.imshow(cartesian_coords[..., 1], cmap='viridis', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Y [R_s]')
    ax.set_title('Y')

    ax = axs[2, 2]
    im = ax.imshow(cartesian_coords[..., 2], cmap='viridis', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Z [R_s]')
    ax.set_title('Z')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    axs[0, 0].set_ylabel('Spherical Coordinates')
    axs[1, 0].set_ylabel('Cartesian Coordinates')

    fig.tight_layout()
    plt.savefig(os.path.join(out_path, 'cartesian.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # plot velocity
    v_norm = Normalize(vmin=-2000, vmax=2000)

    fig, axs = plt.subplots(2, 3, figsize=(12, 5), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(v_rtp[..., 0], cmap='RdBu_r', origin='lower', norm=v_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\text{r}$ [m/s]')
    ax.set_title('PINN ME $v_r$')

    ax = axs[0, 1]
    im = ax.imshow(v_rtp[..., 1], cmap='RdBu_r', origin='lower', norm=v_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\text{t}$ [m/s]')
    ax.set_title('PINN ME $v_t$')

    ax = axs[0, 2]
    im = ax.imshow(v_rtp[..., 2], cmap='RdBu_r', origin='lower', norm=v_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_\text{p}$ [m/s]')
    ax.set_title('PINN ME $v_p$')

    ax = axs[1, 0]
    im = ax.imshow(v_los, cmap='RdBu_r', origin='lower', norm=v_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_x$ [m/s]')
    ax.set_title(r'PINN ME $v_\text{LOS}$')

    ax = axs[1, 1]
    im = ax.imshow(vlos_ref, cmap='RdBu_r', origin='lower', norm=v_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$v_z$ [m/s]')
    ax.set_title(r'Reference $v_\text{LOS}$')

    axs[1, 2].axis('off')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel('Latitude [deg]') for ax in axs[:, 0]]
    [ax.set_xlabel('Longitude [deg]') for ax in axs[-1]]
    fig.tight_layout()
    plt.savefig(os.path.join(out_path, 'velocity.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # plot log B

    norm = SymLogNorm(1, vmin=-3000, vmax=3000)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(b_rtp[..., 0], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('PINN ME $B_r$')

    ax = axs[0, 1]
    im = ax.imshow(b_rtp[..., 1], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
    ax.set_title('PINN ME $B_t$')

    ax = axs[0, 2]
    im = ax.imshow(b_rtp[..., 2], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
    ax.set_title('PINN ME $B_p$')

    ax = axs[1, 0]
    im = ax.imshow(b_rtp_ref[..., 0], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('Reference $B_r$')

    ax = axs[1, 1]
    im = ax.imshow(b_rtp_ref[..., 1], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
    ax.set_title('Reference $B_t$')

    ax = axs[1, 2]
    im = ax.imshow(b_rtp_ref[..., 2], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
    ax.set_title('Reference $B_p$')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel('Latitude [deg]') for ax in axs[:, 0]]
    [ax.set_xlabel('Longitude [deg]') for ax in axs[-1]]

    # add subtitle with date
    plt.suptitle(f'Map at {target_time}', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'log_B.jpg'), dpi=300)
    plt.close()


    ########################################################################################################################
    # plot log B disambiguated

    norm = SymLogNorm(1, vmin=-3000, vmax=3000)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(b_rtp_disambig[..., 0], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('PINN ME $B_r$')

    ax = axs[0, 1]
    im = ax.imshow(b_rtp_disambig[..., 1], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
    ax.set_title('PINN ME $B_t$')

    ax = axs[0, 2]
    im = ax.imshow(b_rtp_disambig[..., 2], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
    ax.set_title('PINN ME $B_p$')

    ax = axs[1, 0]
    im = ax.imshow(b_rtp_ref[..., 0], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('Reference $B_r$')

    ax = axs[1, 1]
    im = ax.imshow(b_rtp_ref[..., 1], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')
    ax.set_title('Reference $B_t$')

    ax = axs[1, 2]
    im = ax.imshow(b_rtp_ref[..., 2], cmap='RdBu_r', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')
    ax.set_title('Reference $B_p$')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel('Latitude [deg]') for ax in axs[:, 0]]
    [ax.set_xlabel('Longitude [deg]') for ax in axs[-1]]

    # add subtitle with date
    plt.suptitle(f'Map at {target_time}', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'log_B_disambiguated.jpg'), dpi=300)
    plt.close()