import argparse
import os.path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
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
    parser.add_argument('--ref_map_blos', type=str, help='the path to the reference map fld')
    parser.add_argument('--ref_map_bmag', type=str, help='the path to the reference map inc')
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
    ref_map = Map(args.ref_map_blos)

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
    cartesian_coords = spherical_to_cartesian(spherical_coords) / pinnme.Rs_per_ds
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
    b_los_ref = Map(args.ref_map_blos).reproject_to(ref_map.wcs).data
    b_mag_ref = Map(args.ref_map_bmag).reproject_to(ref_map.wcs).data

    ########################################################################################################################
    # compare B_LOS
    b_los = b_img[..., 2]
    b_mag = np.linalg.norm(b_img, axis=-1)

    fig, axs = plt.subplots(2, 2, figsize=(8, 5), subplot_kw={'projection': ref_map})

    ax = axs[0, 0]
    im = ax.imshow(b_los_ref, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{LOS}$ [G]')

    ax = axs[0, 1]
    im = ax.imshow(b_mag_ref, cmap='viridis', origin='lower', vmin=10, vmax=2000, norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

    ax = axs[1, 0]
    im = ax.imshow(b_los, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{LOS}$ [G]')

    ax = axs[1, 1]
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
    plt.savefig(os.path.join(out_path, 'b_mag_comparison.jpg'), dpi=300)
    plt.close()

