import numpy as np
from astropy.io import fits


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\u001b[0m'


def printc(*args, color=bcolors.RESET, **kwargs):
    """My custom print() function.

    Parameters
    ----------
    *args:
        arguments to be printed
    color: string
        color of the text
    **kwargs:
        keyword arguments to be passed to print()

    Returns
    -------
    None

    From SPGPyLib PHITools
    """
    print(u"\u001b" + f"{color}", end='\r')
    print(*args, **kwargs)
    print(u"\u001b" + f"{bcolors.RESET}", end='\r')
    return


def fits_get_sampling(file, num_wl=6, TemperatureCorrection=True, TemperatureConstant=40.1225e-3, verbose=False):
    '''Open fits file, extract the wavelength axis and the continuum position, from Voltages in header

    Parameters
    ----------
    file: string
        location of the fits file
    num_wl: int
        number of wavelength
    TemperatureCorrection: bool
        if True, apply temperature correction to the wavelength axis
    TemperatureConstant: float
        Temperature constant to be used when TemperatureCorrection is True. Default: 40.1225e-3 Å/K. Old: 40.323e-3 Å/K. Suggested (old) value: 36.46e-3 Å/K
    verbose: bool
        if True, print the continuum position

    Returns
    -------
    wave_axis: numpy array
        wavelength axis
    voltagesData: numpy array
        voltages of the wavelength axis
    tunning_constant: float
        tunning constant of the etalon
    cpos: int
        continuum position

    Adapted from SPGPyLib

    Usage: wave_axis,voltagesData,tunning_constant,cpos = fits_get_sampling(file,num_wl = 6, TemperatureCorrection = False, verbose = False)
    No S/C velocity corrected!!!
    cpos = 0 if continuum is at first wavelength and = num_wl - 1 (usually 5) if continuum is at the end
    '''
    fg_head = 3
    with fits.open(file) as hdu_list:
        header = hdu_list[fg_head].data
        tunning_constant = float(header[0][4]) / 1e9
        ref_wavelength = float(header[0][5]) / 1e3
        Tfg = hdu_list[0].header['FGOV1PT1']  # ['FGH_TSP1'] #temperature of the FG

        try:
            voltagesData = np.zeros(num_wl)
            hi = np.histogram(header['PHI_FG_voltage'], bins=num_wl + 1)
            yi = hi[0]
            xi = hi[1]
            j = 0
            for i in range(num_wl + 1):
                if yi[i] != 0:
                    if i < num_wl:
                        idx = np.logical_and(header['PHI_FG_voltage'] >= xi[i], header['PHI_FG_voltage'] < xi[i + 1])
                    else:
                        idx = np.logical_and(header['PHI_FG_voltage'] >= xi[i], header['PHI_FG_voltage'] <= xi[i + 1])
                    voltagesData[j] = int(np.median(header['PHI_FG_voltage'][idx]))
                    j += 1
        except:
            printc('WARNING: Running fits_get_sampling_SPG', color=bcolors.WARNING)
            return fits_get_sampling_SPG(file, False)

    d1 = voltagesData[0] - voltagesData[1]
    d2 = voltagesData[num_wl - 2] - voltagesData[num_wl - 1]
    if np.abs(d1) > np.abs(d2):
        cpos = 0
    else:
        cpos = num_wl - 1
    if verbose:
        print('Continuum position at wave: ', cpos)
    wave_axis = voltagesData * tunning_constant + ref_wavelength  # 6173.341

    if TemperatureCorrection:
        if verbose:
            printc(
                '-->>>>>>> If FG temperature is not 61, the relation wl = wlref + V * tunning_constant is not valid anymore',
                color=bcolors.WARNING)
            printc('          Use instead: wl =  wlref + V * tunning_constant + temperature_constant_new*(Tfg-61)',
                   color=bcolors.WARNING)
        # temperature_constant_old = 40.323e-3 # old temperature constant, still used by Johann
        # temperature_constant_new = 37.625e-3 # new and more accurate temperature constant
        # temperature_constant_new = 36.46e-3 # value from HS
        wave_axis += TemperatureConstant * (Tfg - 61)  # 20221123 see cavity_maps.ipynb with example

    return wave_axis, voltagesData, tunning_constant, cpos, ref_wavelength


def fits_get_sampling_SPG(file, verbose=False):
    '''
    Obtains the wavelength and voltages from  fits header

    Parameters
    ----------
    file : str
        fits file path
    verbose : bool, optional
        More info printed. The default is False.

    Returns
    -------
    wave_axis : array
        wavelength axis
    voltagesData : array
        voltages
    tunning_constant : float
        tunning constant of etalon (FG)
    cpos : int
        continuum position

    From SPGPylibs PHITools
    '''
    fg_head = 3
    with fits.open(file) as hdu_list:
        header = hdu_list[fg_head].data
        j = 0
        dummy = 0
        voltagesData = np.zeros((6))
        tunning_constant = 0.0
        ref_wavelength = 0.0
        for v in header:
            # print(v)
            if (j < 6):
                if tunning_constant == 0:
                    tunning_constant = float(v[4]) / 1e9
                if ref_wavelength == 0:
                    ref_wavelength = float(v[5]) / 1e3
                if np.abs(np.abs(float(v[2])) - np.abs(
                        dummy)) > 5:  # check that the next voltage is more than 5 from the previous, as voltages change slightly
                    voltagesData[j] = float(v[2])
                    dummy = voltagesData[j]
                    j += 1

    d1 = voltagesData[0] - voltagesData[1]
    d2 = voltagesData[4] - voltagesData[5]
    if np.abs(d1) > np.abs(d2):
        cpos = 0
    else:
        cpos = 5
    if verbose:
        print('Continuum position at wave: ', cpos)
    wave_axis = voltagesData * tunning_constant + ref_wavelength  # 6173.3356

    return wave_axis, voltagesData, tunning_constant, cpos


def load_fix_phi_header(file):
    header = fits.getheader(file)

    # keep only spatial dimensions and transpose

    # fix CUNIT
    for i in range(1, 5):
        del header[f'CUNIT{i}']
    header['CUNIT1'] = 'arcsec'
    header['CUNIT2'] = 'arcsec'

    # fix CTYPE
    for i in range(1, 5):
        del header[f'CTYPE{i}']
    header['CTYPE1'] = 'HPLN-TAN'
    header['CTYPE2'] = 'HPLT-TAN'

    # fix CDELT
    cdelt1 = header['CDELT3']
    cdelt2 = header['CDELT4']
    for i in range(1, 5):
        del header[f'CDELT{i}']
    header['CDELT1'] = cdelt1
    header['CDELT2'] = cdelt2

    # fix CRPIX
    crpix1 = header['CRPIX3']
    crpix2 = header['CRPIX4']
    for i in range(1, 5):
        del header[f'CRPIX{i}']
    header['CRPIX1'] = crpix1
    header['CRPIX2'] = crpix2

    # fix CRVAL
    crval1 = header['CRVAL3']
    crval2 = header['CRVAL4']
    for i in range(1, 5):
        del header[f'CRVAL{i}']
    header['CRVAL1'] = crval1
    header['CRVAL2'] = crval2

    # get wavelength axis
    wave_axis, _, _, _, ref_wavelength = fits_get_sampling(file, verbose=False)
    for i in range(6):
        header[f'WAVELN{i + 1:02d}'] = wave_axis[i]
    header['WAVELNTH'] = ref_wavelength

    return header
