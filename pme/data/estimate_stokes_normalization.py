import glob

import numpy as np
from astropy.io import fits

if __name__ == '__main__':
    files = sorted(glob.glob('/glade/work/rjarolim/data/phi_stokes/RS5/*.fits'))
    #
    mean_intensities = []
    for file in files:
        stokes = fits.getdata(file)
        mean_intensity = stokes[:, 0, :, :].mean()
        print('Mean intensity for {}: {:.5f}'.format(file, mean_intensity))
        mean_intensities.append(mean_intensity)
        header = fits.getheader(file)
        print(f'WAVELENGTHs:', header['WAVELENGTH1'], header['WAVELENGTH2'], header['WAVELENGTH3'], header['WAVELENGTH4'], header['WAVELENGTH5'])
    #
    print(f'Overall mean intensity: {np.mean(mean_intensities):.5f} ± {np.std(mean_intensities):.5f}')

    stokes_I_files = [sorted(glob.glob(f'/glade/work/rjarolim/data/phi_stokes/LB5_hmi/*I{i}.fits')) for i in range(1, 6)]
    stokes_I_files = zip(*stokes_I_files)
    mean_intensities = []
    for files in stokes_I_files:
        stokes_I = np.array([fits.getdata(file) for file in files])
        print(f'Mean intensities for Stokes I files: {np.mean(stokes_I)}')
        mean_intensities.append(stokes_I.mean())
    print(f'Mean intensities for Stokes I files: {np.mean(mean_intensities):.5f} ± {np.std(mean_intensities):.5f}')