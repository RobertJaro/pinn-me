import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from pme.data.util import image_to_spherical_matrix, spherical_to_cartesian, cartesian_to_spherical_matrix

if __name__ == '__main__':
    ####################################################################################################################
    # load ME parameters
    B_field_map = Map('/glade/work/rjarolim/data/hmi_stokes/test/hmi.b_720s.20240501_000000_TAI.field.fits')
    B_az_map = Map('/glade/work/rjarolim/data/hmi_stokes/test/hmi.b_720s.20240501_000000_TAI.azimuth.fits')
    B_in_map = Map('/glade/work/rjarolim/data/hmi_stokes/test/hmi.b_720s.20240501_000000_TAI.inclination.fits')
    B_disamb_map = Map('/glade/work/rjarolim/data/hmi_stokes/test/hmi.b_720s.20240501_000000_TAI.disambig.fits')

    fld_ref = B_field_map.data
    inc_ref = np.deg2rad(B_in_map.data)
    azi_ref = np.deg2rad(B_az_map.data)
    amb = B_disamb_map.data
    # disambiguate
    amb_weak = 2
    condition = (amb.astype((int)) >> amb_weak).astype(bool)
    azi_ref[condition] += np.pi

    ####################################################################################################################
    # load Brtp
    ref_B_r = Map('/glade/work/rjarolim/data/hmi_stokes/test/hmi.B_720s.20240501_000000_TAI.Br.fits')
    ref_B_t = Map('/glade/work/rjarolim/data/hmi_stokes/test/hmi.B_720s.20240501_000000_TAI.Bt.fits')
    ref_B_p = Map('/glade/work/rjarolim/data/hmi_stokes/test/hmi.B_720s.20240501_000000_TAI.Bp.fits')

    ####################################################################################################################
    # prepare transformation matrix
    s_map = ref_B_r
    spherical_coords = all_coordinates_from_map(s_map)

    projective_coords = spherical_coords.transform_to(frames.Helioprojective)
    radial_distance = np.sqrt(projective_coords.Tx ** 2 + projective_coords.Ty ** 2) / s_map.rsun_obs
    mu = np.cos(radial_distance.to_value(u.dimensionless_unscaled) * np.pi / 2)
    mu = mu.astype(np.float32)

    carrington_coords = spherical_coords.transform_to(frames.HeliographicCarrington)
    lat, lon = carrington_coords.lat.to_value(u.rad), carrington_coords.lon.to_value(u.rad)
    r = carrington_coords.radius
    #
    r = r * u.solRad if r.unit == u.dimensionless_unscaled else r
    carrington_coords = np.stack([r.to_value(u.solRad), lat, lon], -1)
    cartesian_coords = spherical_to_cartesian(carrington_coords)

    # create rtp transform
    cartesian_to_spherical_transform = cartesian_to_spherical_matrix(carrington_coords)
    # create observer transform
    latc, lonc = np.deg2rad(s_map.meta['CRLT_OBS']), np.deg2rad(s_map.meta['CRLN_OBS'])
    pAng = -np.deg2rad(s_map.meta['CROTA2'])
    a_matrix = image_to_spherical_matrix(lon, lat, latc, lonc, pAng=pAng)
    rtp_to_img_transform = np.linalg.inv(a_matrix)

    ####################################################################################################################
    # apply transformation

    b_rtp = np.stack([ref_B_r.data, ref_B_t.data, ref_B_p.data], -1)
    b_rtp[..., 1] *= -1
    b_img = np.einsum("...ij,...j->...i", rtp_to_img_transform, b_rtp)

    b_field = np.linalg.norm(b_img, axis=-1, keepdims=True)
    theta = np.arccos(b_img[..., 2:3] / (b_field + 1e-8))
    chi = np.arctan2(-b_img[..., 0:1], b_img[..., 1:2])

    ####################################################################################################################
    # plot comparison

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    ax = axs[0, 0]
    im = ax.imshow(fld_ref, cmap='viridis', vmin=1, vmax=2000, norm='log')
    ax.set_title('B field - ref')
    plt.colorbar(im, ax=ax)

    ax = axs[0, 1]
    im = ax.imshow(np.rad2deg(inc_ref) % 180, cmap='PiYG', vmin=0, vmax=180)
    ax.set_title('Inclination - ref')
    plt.colorbar(im, ax=ax)

    ax = axs[0, 2]
    im = ax.imshow(np.rad2deg(azi_ref) % 360, cmap='twilight', vmin=0, vmax=360)
    ax.set_title('Azimuth - ref')
    plt.colorbar(im, ax=ax)

    ax = axs[1, 0]
    im = ax.imshow(b_field[..., 0], cmap='viridis', vmin=1, vmax=2000, norm='log')
    ax.set_title('B field - transformed')
    plt.colorbar(im, ax=ax)

    ax = axs[1, 1]
    im = ax.imshow(np.rad2deg(theta[..., 0]) % 180, cmap='PiYG', vmin=0, vmax=180)
    ax.set_title('Inclination - transformed')
    plt.colorbar(im, ax=ax)

    ax = axs[1, 2]
    im = ax.imshow(np.rad2deg(chi[..., 0]) % 360, cmap='twilight', vmin=0, vmax=360)
    ax.set_title('Azimuth - transformed')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig('/glade/work/rjarolim/data/hmi_stokes/test/comparison.jpg', dpi=300)
    plt.close(fig)
