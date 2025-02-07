import argparse
import os.path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from pme.data.util import spherical_to_cartesian, vector_cartesian_to_spherical, cartesian_to_spherical_matrix, \
    image_to_spherical_matrix
from pme.evaluation.loader import PINNMEOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
    parser.add_argument('--input', type=str, help='the path to the input file')
    parser.add_argument('--ref_map_r', type=str, help='the path to the reference map Br')
    parser.add_argument('--ref_map_t', type=str, help='the path to the reference map Bt')
    parser.add_argument('--ref_map_p', type=str, help='the path to the reference map Bp')
    parser.add_argument('--ref_map_fld', type=str, help='the path to the reference map fld')
    parser.add_argument('--ref_map_inc', type=str, help='the path to the reference map inc')
    parser.add_argument('--ref_map_azi', type=str, help='the path to the reference map azi')
    parser.add_argument('--ref_map_disambig', type=str, help='the path to the reference map disambig')
    parser.add_argument('--output', type=str, help='the path to the output file')
    args = parser.parse_args()

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    # load
    pinnme = PINNMEOutput(args.input)

    # load reference maps
    ref_map_r = Map(args.ref_map_r)
    ref_map_t = Map(args.ref_map_t)
    ref_map_p = Map(args.ref_map_p)

    # load time
    target_time = ref_map_r.date.to_datetime()
    normalized_time = pinnme._normalize_time(target_time)
    normalized_time = 0

    coords = all_coordinates_from_map(ref_map_r).transform_to(frames.HeliographicCarrington)
    lat, lon = coords.lat.to_value(u.rad), coords.lon.to_value(u.rad)
    r = coords.radius.to_value(u.solRad)

    spherical_coords = np.stack([r, lat, lon], axis=-1)
    #
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    time_coords = np.ones((*cartesian_coords.shape[:-1], 1), dtype=np.float32) * normalized_time
    coords = np.concatenate([time_coords, cartesian_coords], axis=-1)
    cartesian_to_spherical_transform = cartesian_to_spherical_matrix(spherical_coords)

    latc, lonc = np.deg2rad(ref_map_r.meta['CRLT_OBS']), np.deg2rad(ref_map_r.meta['CRLN_OBS'])
    pAng = -np.deg2rad(ref_map_r.meta['CROTA2'])
    a_matrix = image_to_spherical_matrix(lon, lat, latc, lonc, pAng=pAng)
    rtp_to_img_transform = np.linalg.inv(a_matrix)

    parameter_cube = pinnme.load_parameters(coords=coords)
    b_xyz = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']], axis=-1)
    b_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, b_xyz)
    b_rtp[..., 1] *= -1

    v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']], axis=-1)
    v_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, v_xyz)

    b_img = np.einsum('...ij,...j->...i', rtp_to_img_transform, b_rtp)

    fld = np.linalg.norm(b_img, axis=-1, keepdims=True)
    inc = np.arccos(b_img[..., 2:3] / (fld + 1e-8))
    azi = np.arctan2(-b_img[..., 0:1], b_img[..., 1:2])

    inc = np.rad2deg(inc)
    azi = np.rad2deg(azi)

    ########################################################################################################################
    # load reference map
    b_rtp_ref = np.stack([ref_map_r.data, ref_map_t.data, ref_map_p.data], axis=-1)

    fld_ref = Map(args.ref_map_fld).data
    inc_ref = Map(args.ref_map_inc).data
    azi_ref = Map(args.ref_map_azi).data
    amb_ref = Map(args.ref_map_disambig).data

    # disambiguate
    amb_weak = 2
    condition = (amb_ref.astype(int) >> amb_weak).astype(bool)
    azi_ref[condition] += 180

    ########################################################################################################################
    # Plot subframe in B_r, B_theta, B_phi

    norm = Normalize(-500, 500)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': ref_map_r})

    ax = axs[0, 0]
    im = ax.imshow(b_rtp[..., 0], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')

    ax = axs[0, 1]
    im = ax.imshow(b_rtp[..., 1], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')

    ax = axs[0, 2]
    im = ax.imshow(b_rtp[..., 2], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')

    ax = axs[1, 0]
    im = ax.imshow(b_rtp_ref[..., 0], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')

    ax = axs[1, 1]
    im = ax.imshow(b_rtp_ref[..., 1], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{t}$ [G]')

    ax = axs[1, 2]
    im = ax.imshow(b_rtp_ref[..., 2], cmap='gray', norm=norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{p}$ [G]')

    [ax.set_xlabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel(' ') for ax in axs.flatten()]
    [ax.set_ylabel('Latitude [deg]') for ax in axs[:, 0]]
    [ax.set_xlabel('Longitude [deg]') for ax in axs[1]]

    [ax.set_xlim(2048 - 512 - 256, 2048 + 256) for ax in axs.flatten()]
    [ax.set_ylim(2048, 2048 + 1024) for ax in axs.flatten()]

    # add subtitle with date
    plt.suptitle(f'Carrington map at {target_time}', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'reference_comparison.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # plot field strength

    b_norm = np.linalg.norm(b_rtp, axis=-1)
    b_norm_ref = np.linalg.norm(b_rtp_ref, axis=-1)

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), subplot_kw={'projection': ref_map_r})

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

    fig.tight_layout()
    plt.savefig(os.path.join(args.output, 'field_strength_comparison.jpg'), dpi=300)
    plt.close()

    ########################################################################################################################
    # plot fld, inc, azi

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': ref_map_r})

    ax = axs[0, 0]
    im = ax.imshow(fld_ref, cmap='viridis', origin='lower', vmin=1, vmax=2000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')
    ax.set_title('Reference')

    ax = axs[0, 1]
    im = ax.imshow(inc_ref % 180, cmap='PiYG', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\theta$ [rad]')
    ax.set_title('Reference')

    ax = axs[0, 2]
    im = ax.imshow(azi_ref % 360, cmap='twilight', origin='lower', vmin=0, vmax=360)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [rad]')
    ax.set_title('Reference')

    ax = axs[1, 0]
    im = ax.imshow(fld, cmap='viridis', origin='lower', vmin=1, vmax=2000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')
    ax.set_title('PINN ME')

    ax = axs[1, 1]
    im = ax.imshow(inc % 180, cmap='PiYG', origin='lower', vmin=0, vmax=180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\theta$ [rad]')
    ax.set_title('PINN ME')

    ax = axs[1, 2]
    im = ax.imshow(azi % 360, cmap='twilight', origin='lower', vmin=0, vmax=360)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\phi$ [rad]')
    ax.set_title('PINN ME')

    [ax.set_xlim(2048 - 512 - 256, 2048 + 256) for ax in axs.flatten()]
    [ax.set_ylim(2048, 2048 + 1024) for ax in axs.flatten()]

    fig.tight_layout()
    plt.savefig(os.path.join(args.output, 'fld_inc_azi_comparison.jpg'), dpi=300)
    plt.close()


