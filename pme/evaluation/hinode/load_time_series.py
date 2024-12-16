import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from sunpy.map import Map

from pme.evaluation.loader import to_cartesian, PINNMEOutput

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file')

args = parser.parse_args()

input_path = args.input
output_path = args.output

ref_files = sorted(glob.glob('/glade/work/rjarolim/data/inversion/hinode_2024_05_l2_1/*.fits'))


os.makedirs(output_path, exist_ok=True)

loader = PINNMEOutput(input_path)

times = loader.times
parameters = {}
for i, t in enumerate(times):
    res = loader.load_time(t)
    for k, v in res.items():
        if k not in parameters:
            parameters[k] = []
        parameters[k].append(v)

parameters = {k: np.concatenate(v) for k, v in parameters.items()}

b = parameters['b_field']
theta = parameters['theta']
chi = parameters['chi']
#
b_los = b * np.cos(theta)
b_trv = b * np.sin(theta)
azi = chi % np.pi
########################################################################################################################
# plot example images

Mm_per_pix = 0.07
b_max = 2.5e3
extent = [0, b_los.shape[2] * Mm_per_pix, 0, b_los.shape[1] * Mm_per_pix]

plot_kwargs = {'extent': extent, 'origin': 'lower'}

fig, axs = plt.subplots(2, b_los.shape[0], figsize=(10, 10))
axs = axs.reshape(2, -1)

for i in range(b_los.shape[0]):
    row = axs[:, i]
    im_los = row[0].imshow(b_los[i], cmap='RdBu_r', vmin=-b_max, vmax=b_max, **plot_kwargs)
    row[0].set_title(f'{times[i].isoformat(" ", timespec="minutes")}')
    #
    data = -fits.getdata(ref_files[i], 4)
    row[1].imshow(data, cmap='RdBu_r',  vmin=-b_max, vmax=b_max, **plot_kwargs)

# [ax.set_xticklabels([]) for ax in axs[0, :]]
# [ax.set_yticklabels([]) for ax in axs[:, 1:].ravel()]

fig.tight_layout()
fig.savefig(f'{output_path}/series.png', transparent=True, dpi=300)
plt.close(fig)
