import glob

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map, all_coordinates_from_map

from pme.data.phi_util import load_fix_phi_header

phi_file = '/glade/work/rjarolim/data/phi_fdt/2024-03-29/solo_L2_phi-fdt-stokes_20240329T000009_V202412051616_0443291511.fits.gz'
hmi_files = sorted(glob.glob('/glade/work/rjarolim/data/hmi_stokes/20240327_3h/hmi.s_720s.20240329_000000_TAI.3.I*.fits'))

phi_header = load_fix_phi_header(phi_file)
phi_data = fits.getdata(phi_file)
phi_data = phi_data[:, 0].sum(0)
phi_map = Map(phi_data, phi_header)

hmi_header = fits.getheader(hmi_files[0],1)
hmi_data = np.stack([fits.getdata(f) for f in hmi_files], -1).sum(-1)
hmi_map = Map(hmi_data, hmi_header)


hmi_map_projected = hmi_map.reproject_to(phi_map.wcs)

# filter off disc pixels
coords = all_coordinates_from_map(phi_map)
r = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / phi_map.rsun_obs
mask = r <= 1.0
phi_map.data[~mask] = np.nan
hmi_map_projected.data[~mask] = np.nan

# mask phi based on reprojected hmi data
mask = ~np.isnan(hmi_map_projected.data)
phi_map.data[~mask] = np.nan

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

levels = np.logspace(.2, 1, num=5)

ax = axs[0]
im = ax.imshow(phi_map.data, origin='lower', cmap='gray', norm='log')
contours = ax.contour(phi_map.data, levels=levels, colors='red', alpha=0.2)
ax.set_title('PHI Stokes I')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[1]
im = ax.imshow(hmi_map_projected.data, origin='lower', cmap='gray', norm='log')
# add contours from phi map - levels in log space
contours = ax.contour(phi_map.data, levels=levels, colors='red', alpha=0.2)
ax.set_title('HMI Stokes I projected to PHI')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')


plt.tight_layout()
plt.savefig('/glade/work/rjarolim/spinn_me/combined/phi_hmi_comparison.jpg', dpi=300)
plt.close()
