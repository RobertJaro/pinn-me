import argparse

import numpy as np
from astropy.io import fits

from pme.train.data_loader import add_synthetic_noise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--wl_file', type=str, required=True)
    parser.add_argument('--psf_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=False, default=None)
    args = parser.parse_args()

    file = args.file
    wl_file = args.wl_file
    psf_file = args.psf_file
    out_file = args.out_file

    data = np.load(file)  # y, x, lambda, stokes
    lambda_grid = np.load(wl_file)

    lambda_grid = lambda_grid[201:]
    data = data[:, :, 201:, :]

    # (y, x, lambda, stokes) --> (y, x, stokes, lambda)
    data = np.moveaxis(data, [0, 1, 2, 3], [0, 1, 3, 2])

    data = add_synthetic_noise(data, 1e-3, psf_file, bin=4)

    header = fits.Header()
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
