import argparse

import numpy as np
from astropy.io import fits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--psf_file', type=str, required=True)
    parser.add_argument('--psf_size', type=int, required=False, nargs=2, default=(5, 5))
    parser.add_argument('--out_file', type=str, required=True)
    args = parser.parse_args()

    psf_size=args.psf_size
    psf_file=args.psf_file
    out_file=args.out_file

    if psf_file.endswith('.npy'):
        psf = np.load(psf_file)['PSF']
    elif psf_file.endswith('.fits'):
        psf = fits.getdata(psf_file).astype(np.float32)
    else:
        raise ValueError('PSF file must be either .npy or .fits.')
    print(f'Loaded PSF with shape {psf.shape}.')

    center = np.array(psf.shape) // 2
    psf = psf[center[0] - psf_size[0] // 2:center[0] + psf_size[0] // 2 + 1,
          center[1] - psf_size[1] // 2:center[1] + psf_size[1] // 2 + 1]

    # Normalize PSF
    psf /= np.sum(psf)

    print(f'Saving PSF with shape {psf.shape} to {out_file}.')
    np.save(out_file, psf)