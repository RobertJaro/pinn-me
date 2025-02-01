import argparse
import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pme.data.util import spherical_to_cartesian, vector_cartesian_to_spherical
from pme.evaluation.loader import PINNMEOutput

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file')
args = parser.parse_args()

out_path = args.output
os.makedirs(out_path, exist_ok=True)

# load
pinnme = PINNMEOutput(args.input)

# lat = np.linspace(0, np.pi, 1800)
# lon = np.linspace(0, 2 * np.pi, 3600)

resolution = 0.005
lat = np.arange(40, 60, resolution)
lon = np.arange(210, 240, resolution)

# resolution = 0.1
# lat = np.arange(0, 180, resolution)
# lon = np.arange(0, 360, resolution)

lat, lon = np.deg2rad(lat), np.deg2rad(lon)

spherical_coords = np.stack(np.meshgrid(
    [1.], [1], lat, lon, indexing='ij'),
    axis=-1)
spherical_coords[..., 0] = pinnme._normalize_time(pinnme.times[5])
spherical_coords = spherical_coords[0, 0, :, :]

cartesian_coords = spherical_to_cartesian(spherical_coords[..., 1:])
cartesian_coords = np.concatenate([spherical_coords[..., :1], cartesian_coords], axis=-1)

parameter_cube = pinnme.load_parameters(coords=cartesian_coords)
b_xyz = np.concatenate([parameter_cube['b_x'], parameter_cube['b_y'], parameter_cube['b_z']], axis=-1)
b_rtp = vector_cartesian_to_spherical(b_xyz, spherical_coords)

v_xyz = np.concatenate([parameter_cube['v_x'], parameter_cube['v_y'], parameter_cube['v_z']], axis=-1)
v_rtp = vector_cartesian_to_spherical(v_xyz, spherical_coords)

########################################################################################################################
# Plot subframe in B_r, B_theta, B_phi
extent = [lon.min(), lon.max(), lat.max(), lat.min()]
extent = np.rad2deg(extent)

v_min_max = np.max(np.abs(b_rtp))
norm = SymLogNorm(linthresh=1, vmin=-v_min_max, vmax=v_min_max)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

ax = axs[0]
im = ax.imshow(b_rtp[..., 0], cmap='RdBu_r', norm=norm, origin='upper', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('B_r')

ax = axs[1]
im = ax.imshow(b_rtp[..., 1], cmap='RdBu_r', norm=norm, origin='upper', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('B_theta')

ax = axs[2]
im = ax.imshow(b_rtp[..., 2], cmap='RdBu_r', norm=norm, origin='upper', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('B_phi')

axs[0].set_ylabel('Latitude [deg]')
[ax.set_xlabel('Longitude [deg]') for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'carrington.jpg'), dpi=300)
plt.close()

########################################################################################################################
# Plot subframe in B, inclination, azimuth

b = np.linalg.norm(b_rtp, axis=-1)
inclination = np.arccos(b_rtp[..., 2] / b) % np.pi
azimuth = np.arctan2(b_rtp[..., 1], b_rtp[..., 0]) % (2 * np.pi)

extent = [lon.min(), lon.max(), lat.min(), lat.max()]
extent = np.rad2deg(extent)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

ax = axs[0]
im = ax.imshow(b, cmap='viridis', norm='log', origin='upper', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('|B| [G]', fontsize=16)

ax = axs[1]
im = ax.imshow(np.rad2deg(inclination), cmap='seismic_r', origin='upper', extent=extent, vmin=0, vmax=180)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([0, 45, 90, 135, 180])
cbar.set_label('$\Theta$ [deg]', fontsize=16)

ax = axs[2]
im = ax.imshow(np.rad2deg(azimuth), cmap='twilight', origin='upper', extent=extent, vmin=0, vmax=360)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar =  fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([0, 90, 180, 270, 360])
cbar.set_label('$\phi$ [deg]', fontsize=16)

axs[0].set_ylabel('Latitude [deg]', fontsize=16)
[ax.set_xlabel('Longitude [deg]', fontsize=16) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'carrington_los_inc_azi.jpg'), dpi=300)
plt.close()


########################################################################################################################
# Plot velocity

v_min_max = np.max(np.abs(v_rtp))
norm = Normalize(vmin=-v_min_max, vmax=v_min_max)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

ax = axs[0]
im = ax.imshow(v_rtp[..., 0], cmap='RdBu_r', norm=norm, origin='upper', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('V_r')

ax = axs[1]
im = ax.imshow(v_rtp[..., 1], cmap='RdBu_r', norm=norm, origin='upper', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('V_theta')

ax = axs[2]
im = ax.imshow(v_rtp[..., 2], cmap='RdBu_r', norm=norm, origin='upper', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('V_phi')

axs[0].set_ylabel('Latitude [deg]')
[ax.set_xlabel('Longitude [deg]') for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'carrington_velocity.jpg'), dpi=300)
plt.close()
