import numpy as np
from astropy import units as u
from astropy.io import fits

from pme.train.data_loader import add_synthetic_noise

file = '/glade/work/rce/for_momo/SIRprofiles_plage.fits'
out_file = '/glade/work/rjarolim/data/pinnme/muram/SIRprofiles_plage.fits'

data = fits.getdata(file, 0)
header = fits.getheader(file, 0)

lambda_grid = fits.getdata(file, 1)

# clipping of the stokes vector
lambda_grid = lambda_grid[201::2] * u.mA
lambda_grid = lambda_grid.to_value(u.A) + 6301.502

data = data[201::2]
# (lambda, y, x, stokes) --> (x, y, stokes, lambda)
data = np.moveaxis(data, [0, 1, 2], [3, 1, 0])

data = add_synthetic_noise(data, 1e-3, "/glade/work/rjarolim/data/inversion/hinode_psf_0.16.fits", bin=4)

header['DATE_OBS'] = '2024-01-01T00:00:00'
header["CDELT1"] = np.gradient(lambda_grid)[0]
header["CRVAL1"] = (lambda_grid[-1] + lambda_grid[0]) / 2
header["CRPIX1"] = lambda_grid.shape[0] / 2

header['NAXIS'] = 4
header['NAXIS1'] = data.shape[0]
header['NAXIS2'] = data.shape[1]
header['NAXIS3'] = data.shape[2]
header['NAXIS4'] = data.shape[3]

fits.writeto(out_file, data.astype(np.float32), header, overwrite=True)
