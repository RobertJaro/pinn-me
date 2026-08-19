import argparse
import os.path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, \
    image_to_spherical_matrix
from pme.evaluation.loader import SPINNMEOutput

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

    frame_out = SkyCoord(
        ref_map.carrington_longitude,
        0 * u.deg,
        frame=f"heliographic_carrington",
        obstime=ref_map.date,
        observer=ref_map.observer_coordinate,
        rsun=getattr(ref_map.observer_coordinate, "rsun", None),
    )
    shape = (256, 256)
    # scale = (40 / shape[0], 40 / shape[1]) * u.deg / u.pix
    scale = [40 / int(shape[1]), (40 / np.pi) / (int(shape[0]) / 2)] * u.deg / u.pix
    lat_offset = 70 / scale[0].to_value(u.deg / u.pix)
    reference_pixel = ((shape[1] - 1) / 2, (shape[0] - 1) / 2 + lat_offset) * u.pix
    carr_header = make_fitswcs_header(shape, frame_out, scale=scale, projection_code='CEA', reference_pixel=reference_pixel)
    carrington_ref_map = ref_map.reproject_to(carr_header)

    # shape = (720, 1440)
    # carr_header = make_heliographic_header(ref_map.date, ref_map.observer_coordinate, shape, frame='carrington')

    ########################################################################################################################
    # load time

    target_time = carrington_ref_map.date.to_datetime()
    normalized_time = pinnme._normalize_time(target_time)

    coords = all_coordinates_from_map(carrington_ref_map).transform_to(frames.HeliographicCarrington)
    lat, lon = coords.lat.to_value(u.rad), coords.lon.to_value(u.rad)
    r = np.ones_like(lat)  # coords.radius.to_value(u.solRad)

    spherical_coords = np.stack([r, np.pi / 2 - lat, lon], axis=-1)
    #
    cartesian_coords = spherical_to_cartesian(spherical_coords) / pinnme.Rs_per_ds
    time_coords = np.ones((*cartesian_coords.shape[:-1], 1), dtype=np.float32) * normalized_time
    coords = np.concatenate([time_coords, cartesian_coords], axis=-1)
    cartesian_to_spherical_transform = cartesian_to_spherical_matrix(spherical_coords)

    parameter_cube = pinnme.load_parameters(coords=coords)
    b_xyz = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']],
                           axis=-1) * pinnme.gauss_per_dB
    b_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, b_xyz)

    v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']],
                           axis=-1) * pinnme.meters_per_ds / pinnme.seconds_per_dt
    v_rtp = np.einsum('...ij,...j->...i', cartesian_to_spherical_transform, v_xyz)

    ########################################################################################################################
    # mask nans

    nan_maks = np.isnan(carrington_ref_map.data)
    b_rtp[nan_maks] = np.nan
    v_rtp[nan_maks] = np.nan

    ########################################################################################################################
    # load reference map
    fld_ref = Map(args.ref_map_fld).reproject_to(ref_map.wcs).data
    inc_ref = Map(args.ref_map_inc).reproject_to(ref_map.wcs).data
    azi_ref = Map(args.ref_map_azi).reproject_to(ref_map.wcs).data
    amb_ref = Map(args.ref_map_disambig).reproject_to(ref_map.wcs).data
    vlos_ref = Map(args.ref_map_vlos_mag).reproject_to(ref_map.wcs).data
    vlos_ref = vlos_ref / 100  # convert cm/s to m/s

    amb_weak = 2
    condition = (amb_ref.astype(int) >> amb_weak).astype(bool)
    azi_ref[condition] += 180

    ########################################################################################################################
    # transform to B_r, B_theta, B_phi
    coords = all_coordinates_from_map(ref_map).transform_to(frames.HeliographicCarrington)
    lat, lon = coords.lat.to_value(u.rad), coords.lon.to_value(u.rad)
    latc, lonc = np.deg2rad(ref_map.meta['CRLT_OBS']), np.deg2rad(ref_map.meta['CRLN_OBS'])
    pAng = -np.deg2rad(ref_map.meta['CROTA2'])
    a_matrix = image_to_spherical_matrix(lon, lat, lonc, latc, pAng=pAng)

    b_xi = - fld_ref * np.sin(np.deg2rad(inc_ref)) * np.sin(np.deg2rad(azi_ref))
    b_eta = fld_ref * np.sin(np.deg2rad(inc_ref)) * np.cos(np.deg2rad(azi_ref))
    b_zeta = fld_ref * np.cos(np.deg2rad(inc_ref))

    b_img_ref = np.stack([b_xi, b_eta, b_zeta], axis=-1)
    b_rtp_ref = np.einsum('...ij,...j->...i', a_matrix, b_img_ref)

    ########################################################################################################################
    # project to polar field
    b_r_ref = Map(b_rtp_ref[..., 0], ref_map.meta).reproject_to(carr_header).data
    b_t_ref = Map(b_rtp_ref[..., 1], ref_map.meta).reproject_to(carr_header).data
    b_p_ref = Map(b_rtp_ref[..., 2], ref_map.meta).reproject_to(carr_header).data

    b_r_pinnme = b_rtp[..., 0]
    b_t_pinnme = b_rtp[..., 1]
    b_p_pinnme = b_rtp[..., 2]

    #########################################################################################################################
    # plot comparison

    coords = all_coordinates_from_map(carrington_ref_map).transform_to(frames.HeliographicCarrington)
    lat, lon = coords.lat.to_value(u.deg), coords.lon.to_value(u.deg)
    r = coords.radius

    bl = carrington_ref_map.bottom_left_coord
    tr = carrington_ref_map.top_right_coord

    # extent = [0, shape[1] * scale[1].to_value(u.deg / u.pix), 0, shape[0] * scale[0].to_value(u.deg / u.pix)]
    if bl.lon > tr.lon:
        extent = [bl.lon.to_value(u.deg), tr.lon.to_value(u.deg) + 360, bl.lat.to_value(u.deg), tr.lat.to_value(u.deg)]
    else:
        extent = [bl.lon.to_value(u.deg), tr.lon.to_value(u.deg), bl.lat.to_value(u.deg), tr.lat.to_value(u.deg)]
    extent = None

    b_norm = SymLogNorm(linthresh=1, vmin=-3000, vmax=3000)
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    ax = axs[0, 0]
    im = ax.imshow(b_r_pinnme, cmap='seismic', norm=b_norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('PINN ME $B_r$')

    ax = axs[0, 1]
    im = ax.imshow(b_r_ref, cmap='seismic', norm=b_norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\text{r}$ [G]')
    ax.set_title('Reference $B_r$')

    ax = axs[0, 2]
    im = ax.imshow(r, cmap='viridis', origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$r$ [$R_\odot$]')
    ax.set_title('Radius $r$')

    ax = axs[1, 0]
    im = ax.imshow(b_t_pinnme, cmap='seismic', norm=b_norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\theta$ [G]')
    ax.set_title('PINN ME $B_\\theta$')

    ax = axs[1, 1]
    im = ax.imshow(b_t_ref, cmap='seismic', norm=b_norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\theta$ [G]')
    ax.set_title('Reference $B_\\theta$')

    ax = axs[1, 2]
    im = ax.imshow(lat, cmap='seismic', origin='lower', extent=extent, vmin=-90, vmax=90)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'Latitude [deg]')
    ax.set_title('Latitude $\\theta$')

    ax = axs[2, 0]
    im = ax.imshow(b_p_pinnme, cmap='seismic', norm=b_norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\phi$ [G]')
    ax.set_title('PINN ME $B_\\phi$')

    ax = axs[2, 1]
    im = ax.imshow(b_p_ref, cmap='seismic', norm=b_norm, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$B_\phi$ [G]')
    ax.set_title('Reference $B_\\phi$')

    ax = axs[2, 2]
    im = ax.imshow(lon, cmap='twilight', origin='lower', extent=extent, vmin=0, vmax=360)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'Longitude [deg]')
    ax.set_title('Longitude $\\phi$')

    [ax.set_xlabel('Longitude [deg]') for ax in axs[2, :]]
    [ax.set_ylabel('Latitude [deg]') for ax in axs[:, 0]]

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'polar_field_comparison.png'), dpi=300)
    plt.close()
