import argparse
import os.path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from pme.data.differential_rotation import carrington_rotation_velocity
from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, \
    image_to_spherical_matrix, spherical_to_cartesian_matrix
from pme.evaluation.loader import SPINNMEOutput
from pme.loader.spherical import load_v_observer_LOS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
    parser.add_argument('--input', type=str, help='the path to the input file')
    parser.add_argument('--ref_map_fld', type=str, help='the path to the reference map fld')
    parser.add_argument('--ref_map_inc', type=str, help='the path to the reference map inc')
    parser.add_argument('--ref_map_azi', type=str, help='the path to the reference map azi')
    parser.add_argument('--ref_map_vlos', type=str, help='the path to the reference map vlos')
    parser.add_argument('--output', type=str, help='the path to the output file', default=None)
    args = parser.parse_args()

    in_path = args.input

    out_path = args.output
    out_path = out_path if out_path is not None else os.path.join(os.path.dirname(in_path), 'evaluation')
    os.makedirs(out_path, exist_ok=True)

    # load
    pinnme = SPINNMEOutput(in_path)

    # load reference maps
    ref_map = Map(args.ref_map_fld)

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
    if 'CROTA2' not in ref_map.meta:
        ref_map.meta['CROTA2'] = ref_map.meta['CROTA']
    pAng = -np.deg2rad(ref_map.meta['CROTA2'])
    a_matrix = image_to_spherical_matrix(lon, lat, lonc, latc, pAng=pAng)
    rtp_to_img_transform = np.linalg.inv(a_matrix)

    parameter_cube = pinnme.load_parameters(coords=coords)
    b_xyz = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']],
                           axis=-1) * pinnme.gauss_per_dB
    b_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, b_xyz)
    b_img = np.einsum('...ij,...j->...i', rtp_to_img_transform, b_rtp)

    fld = np.linalg.norm(b_img, axis=-1, keepdims=True)
    inc = np.arccos(b_img[..., 2:3] / (fld + 1e-8))
    azi = np.arctan2(-b_img[..., 0:1], b_img[..., 1:2])

    inc = np.rad2deg(inc)
    azi = np.rad2deg(azi)

    v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']],
                           axis=-1) * pinnme.meters_per_ds / pinnme.seconds_per_dt
    v_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, v_xyz)
    v_img = np.einsum('...ij,...j->...i', rtp_to_img_transform, v_rtp)

    v_los = -v_img[..., 2]  # negative because doppler shift is defined in the observer frame

    ########################################################################################################################
    # load reference map
    fld_ref = Map(args.ref_map_fld).reproject_to(ref_map.wcs).data
    inc_ref = Map(args.ref_map_inc).reproject_to(ref_map.wcs).data
    azi_ref = Map(args.ref_map_azi).reproject_to(ref_map.wcs).data
    vlos_ref = Map(args.ref_map_vlos).reproject_to(ref_map.wcs).data
    vlos_ref = vlos_ref / 100  # convert cm/s to m/s

    # correct for observer LOS
    v_obs_los = load_v_observer_LOS(ref_map)
    vlos_ref = vlos_ref - v_obs_los

    # correct for carrington rotation
    v_carr = carrington_rotation_velocity(latitude=lat, radius=r * pinnme.meters_per_ds, f=np)
    v_carr_rtp = np.stack([np.zeros_like(v_carr), np.zeros_like(v_carr), v_carr], axis=-1)
    v_carr_img = np.einsum('...ij,...j->...i', rtp_to_img_transform, v_carr_rtp)
    vlos_ref = vlos_ref + v_carr_img[..., 2]  # subtract LOS component of carrington rotation velocity

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
    # plot fld, inc, azi

    fig, axs = plt.subplots(2, 4, figsize=(15, 5), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(fld_ref, cmap='viridis', origin='lower', vmin=1, vmax=2000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

    ax = axs[0, 1]
    im = ax.imshow(inc_ref % 180, cmap='PiYG', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\theta$ [deg]')

    ax = axs[0, 2]
    im = ax.imshow(azi_ref % 180, cmap='twilight', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [deg]')

    ax = axs[0, 3]
    im = ax.imshow(azi_ref % 360, cmap='twilight', origin='lower', vmin=0, vmax=360)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [deg]')

    ax = axs[1, 0]
    im = ax.imshow(fld, cmap='viridis', origin='lower', vmin=1, vmax=2000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

    ax = axs[1, 1]
    im = ax.imshow(inc % 180, cmap='PiYG', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\theta$ [deg]')

    ax = axs[1, 2]
    im = ax.imshow(azi % 180, cmap='twilight', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [deg]')

    ax = axs[1, 3]
    im = ax.imshow(azi % 360, cmap='twilight', origin='lower', vmin=0, vmax=360)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [deg]')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]

    axs[0, 0].set_ylabel('Reference')
    axs[1, 0].set_ylabel('SPINN ME')

    fig.tight_layout()
    plt.savefig(os.path.join(out_path, 'fld_inc_azi_comparison.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # compare B_LOS
    b_los = b_img[..., 2]
    b_los_ref = b_img_ref[..., 2]

    b_mag = np.linalg.norm(b_img, axis=-1)
    b_mag_ref = np.linalg.norm(b_img_ref, axis=-1)

    b_trv = np.linalg.norm(b_img[..., :2], axis=-1)
    b_trv_ref = np.linalg.norm(b_img_ref[..., :2], axis=-1)

    fig, axs = plt.subplots(2, 3, figsize=(10, 5), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(b_los_ref, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{LOS}$ [G]')

    ax = axs[0, 1]
    im = ax.imshow(b_trv_ref, cmap='viridis', origin='lower', vmin=0, vmax=2000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{TRV}$ [G]')

    ax = axs[0, 2]
    im = ax.imshow(b_mag_ref, cmap='viridis', origin='lower', vmin=0, vmax=2000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

    ax = axs[1, 0]
    im = ax.imshow(b_los, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{LOS}$ [G]')

    ax = axs[1, 1]
    im = ax.imshow(b_trv, cmap='viridis', origin='lower', vmin=0, vmax=2000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{TRV}$ [G]')

    ax = axs[1, 2]
    im = ax.imshow(b_mag, cmap='viridis', origin='lower', vmin=10, vmax=2000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

    for ax in np.ravel(axs):
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')

    axs[0, 0].set_ylabel('Reference')
    axs[1, 0].set_ylabel('SPINN ME')

    fig.tight_layout()
    plt.savefig(os.path.join(out_path, 'b_los_comparison.jpg'), dpi=300)
    plt.close()


