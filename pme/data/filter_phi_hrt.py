import glob
import os.path
import shutil

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

files = sorted(glob.glob('/glade/work/rjarolim/data/phi_stokes/2024_03_flare/*.fits'))

os.makedirs('/glade/work/rjarolim/data/phi_stokes/imgs', exist_ok=True)

for file in files:
    stokes = fits.getdata(file)
    header = fits.getheader(file)
    integrated_I = stokes[:, 0, :, :].sum(axis=0)
    #
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(integrated_I, cmap='gray', origin='lower')
    ax.set_title(header['SOOPTYPE'])
    #
    fig.savefig(f'/glade/work/rjarolim/data/phi_stokes/imgs/{os.path.basename(file).replace(".fits", ".jpg")}', dpi=300)
    plt.close(fig)

sooptype = 'LB5'
os.makedirs(f'/glade/work/rjarolim/data/phi_stokes/{sooptype}/', exist_ok=True)
for file in files:
    header = fits.getheader(file)
    if header['SOOPTYPE'] != sooptype:
        continue
    shutil.copy(file, f'/glade/work/rjarolim/data/phi_stokes/{sooptype}/')


for f in sorted(glob.glob('/glade/work/rjarolim/data/phi_stokes/LB5/*.fits')):
    header =  fits.getheader(f)
    lambda_center = header['WAVELNTH']  # reference wavelength from header
    lambda_grid = np.array([header[f'WAVELN{i + 1:02d}'] for i in range(6)])
    print(f"File: {os.path.basename(f)}, Center Wavelength: {lambda_center}, Wavelength Grid: {lambda_grid}")
