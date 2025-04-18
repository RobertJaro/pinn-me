"""
Create a test set for the spherical (SPINN ME) inversion

In this case, we take a 2D test case of ME parameters. Then we stretch them on a sphere,
and resample as if observed from a virtual spacecraft. Then we synthesize with ME and then
we create FITS files with appropriate headers and invert.
"""
from typing import Any

import argparse
import os

from tqdm import tqdm
import numpy as np
from astropy import units as u
from astropy.io import fits as fits
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header
from astropy.coordinates import SkyCoord
from astropy.time import Time

# from pme.evaluation.muram.compare_muram import muram_ref
from pme.train.me_atmosphere import MEAtmosphere


import sunpy.coordinates
from scipy.interpolate import interp2d, RectBivariateSpline

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, \
    image_to_spherical_matrix, vector_cartesian_to_polar
from pme.evaluation.loader import PINNMEOutput

from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.coordinates import frames

from datetime import datetime
import torch
from astropy.coordinates import SkyCoord, HeliocentricTrueEcliptic
from sunpy.coordinates import sun
from sunpy.map.header_helper import make_heliographic_header

def differential_rotation(theta, Rsol=699e3 * u.km / u.s):
    """
    define differential rotation based on the wikipedia description

    Input:
        -- theta, float
    output:
        -- velocity
    """

    A = 14.713
    B = -2.396
    C = -1.787

    omega = A + B * np.sin(theta)**2 + C * np.sin(theta)**4 # in deg/day
    omega_rad_per_second = omega / 180 * np.pi / 86400

    V = Rsol * np.cos(theta) * omega_rad_per_second

    return V

def make_map_from_dicts(dict1, ext, shape):

    array = np.zeros(shape)

    for xx in range(shape[0]):
        for yy in range(shape[1]):
            try:
                array[xx, yy] = dict1[f'{xx}_{yy}'][ext]
            except:
                zero_array = np.empty((shape[2:]))
                zero_array[...] = np.nan
                array[xx, yy] = zero_array
    return array

def compute_mu(x, y, r0):
    r = np.sqrt(x**2 + y**2)
    mu = np.sqrt(1 - r**2/r0**2)
    return mu

class Synthesizer():

    def __init__(self, lambda0=6173.3433 * u.AA,
                 j_up=1.0, j_low=0.0, g_up=2.50, g_low=0,
                 lambda_step=0.021743135134784097 * u.AA, n_lambda=102,):
        self.lambda0 = lambda0
        self.jUp = j_up
        self.jLow = j_low
        self.gUp = g_up
        self.gLow = g_low
        self.n_lambda = n_lambda

        lambda_range = (n_lambda - 1) * lambda_step
        self.lambda_grid = np.linspace(-0.5 * lambda_range, 0.5 * lambda_range, n_lambda)

    def synthesize(self, am):
        if am == None:
            return {'stokes_profiles': np.nan, 'b_field': np.nan, 'theta': np.nan, 'chi': np.nan,
                    'b0': np.nan, 'b1': np.nan, 'vmac': np.nan,
                    'damping': np.nan, 'mu': np.nan,
                    'vdop': np.nan, 'kl': np.nan}

        atmos = MEAtmosphere(self.lambda0, self.jUp, self.jLow, self.gUp, self.gLow, self.lambda_grid)

        B, theta, chi = vector_cartesian_to_polar(np.array([am['Bx'], am['By'], am['Bz']]))
        I, Q, U, V = atmos.forward(torch.tensor(B), torch.tensor(theta), torch.tensor(chi),
                                   torch.tensor(am['vmac']), torch.tensor(am['damping']),
                                   torch.tensor(am['b0']), torch.tensor(am['b1']), torch.tensor(am['mu']),
                                   torch.tensor(am['vdop']), torch.tensor(am['kl']))

        stokes_profiles = torch.stack([I, Q, U, V], -2).cpu().numpy()
        # (x, y, n_lambda, n_stokes)
        # stokes_profiles = stokes_profiles.reshape(*self.shape, 4, *self.lambda_grid.shape)

        return {'stokes_profiles': stokes_profiles, 'b_field': B, 'theta': theta, 'chi': chi,
                                                      'b0': am['b0'], 'b1': am['b1'], 'vmac': am['vmac'],
                                                      'damping': am['damping'], 'mu': am['mu'],
                                                      'vdop': am['vdop'], 'kl': am['kl']}

def observed_params(x_POS, y_POS, carrington_lon, carrington_lat, carrington_pAng,
                    parameters_interpolators, sun_radius=954*u.arcsec):
    """ Function to compute the observed parameters in the local LOS frame
        Input:
            -- x_POS, y_POS: float, astropy.units.arcsec
                x,y coordinates in the helioprojective reference frame
            -- carrington_lon, carrington_lat: float, astropy.units.deg
                carrington location of the observer
            -- parameters: dict
                Dictionary with all the ME parameters for the synthesis

        Output:
            -- params_list: list, 8 floats
                Reprojected ME parameters for the synthesis
    """

    lat_observed = np.arcsin(y_POS / sun_radius)
    lon_observed = np.arcsin(x_POS / sun_radius)
    a_matrix = image_to_spherical_matrix(lon_observed, lat_observed,
                                         carrington_lon, carrington_lat,
                                         pAng=carrington_pAng)

    a_matrix_inverse = np.linalg.inv(a_matrix)

    local_params = {}
    transformed_params = {}
    b_field_keys_polar_local = ['b_field', 'theta', 'chi'] # b field in the local polar frame (aka up is radial direction)
    non_transform_keys = ['b0', 'b1', 'vmac', 'damping', 'mu', 'vdop', 'kl']
    for el in parameters_interpolators.keys():
        local_params[el] = parameters_interpolators[el](lon_observed,
                                                        lat_observed)
    local_params['mu'] = compute_mu(x_POS, y_POS, sun_radius)

    for el in non_transform_keys:
        transformed_params[el] = local_params[el]

    Br = local_params['b_field'] * np.cos(local_params['theta'])
    Bt = local_params['b_field'] * np.sin(local_params['theta']) * np.sin(local_params['chi'])
    Bp = local_params['b_field'] * np.sin(local_params['theta']) * np.cos(local_params['chi'])

    B_transformed = np.matmul(a_matrix_inverse, np.array([Br, Bt, Bp]).squeeze().T)
    transformed_params['Bx'] = B_transformed[0]
    transformed_params['By'] = B_transformed[1]
    transformed_params['Bz'] = B_transformed[2]

    return transformed_params

class Carrington_map():

    def __init__(self, test_dataset, parameter_extension, map_name="blah"):
        """
        Make a carrington map from a given dataset

        Input:
            -- test_dataset: string
            Test dataset file name
             -- parameter_extension: int
            Parameter number in the npz file
        Output:
            -- carrington_map: Sunpy.Map.map object
            The resulting carringotn map
        """
        self.parameter_extension = parameter_extension
        self.map_name = map_name
        self.parameters_test_case = np.load(test_dataset)
        self.parameter_array = self.parameters_test_case[self.parameter_extension]

        self.nx_testset = self.parameter_array.shape[0]
        self.ny_testset = self.parameter_array.shape[1]

         # define the spherical coordinate system
        self.lat_array = np.linspace(-360 / 4, 360 / 4, num=self.nx_testset) * u.deg
        self.lon_array = np.linspace(-360 / 2, 360 / 2, num=self.ny_testset) * u.deg
        self.distance = np.ones(self.lon_array.shape) * u.AU
        self.obs_time = '2013-10-28'

        self.crln_obs_reference = 0.0  # Carrington longitude of observer
        self.crlt_obs_reference = 0.0  # Carrington latitude of observer
        # hgln_obs = 0.0  # Heliographic Stonyhurst longitude of observer
        # hglt_obs = 0.0  # Heliographic Stonyhurst latitude of observer
        from sunpy.coordinates import get_earth
        self.observer = get_earth(self.obs_time)

        coords = SkyCoord(
            lon=self.lon_array,
            lat=self.lat_array,
            radius=self.distance,
            frame=frames.HeliographicCarrington,
            obstime=self.obs_time
        )

        self.metadata = make_heliographic_header(self.obs_time, self.observer,
                                                 [400, 400],
                                                 frame='carrington')

        self.carrington_map = sunpy.map.Map(self.parameter_array,
                                            self.metadata)

        fig = plt.figure()
        ax = fig.add_subplot(projection=self.carrington_map)
        im = self.carrington_map.plot(axes=ax)
        plt.colorbar(im, ax=ax)
        overlay = ax.get_coords_overlay('heliographic_carrington')
        lon = overlay[0]
        lat = overlay[1]
        #
        lon.set_ticks(values = [0, 90, 360] * u.deg)
        lat.set_ticks(values = [-90+1, 0, 90-1] * u.deg)
        # Add additional plot elements if needed
        # self.carrington_map.draw_grid()
        # xlims_world = [-180, 180] * u.deg
        # ylims_world = [-90, 90] * u.deg
        #
        # world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world,
        #                         frame=self.carrington_map.coordinate_frame)
        # pixel_coords_x, pixel_coords_y = self.carrington_map.wcs.world_to_pixel(world_coords)
        # ax.set_xlim(pixel_coords_x)
        # ax.set_ylim(pixel_coords_y)

        ax.coords[0].set_major_formatter('d.ddd')  # Longitude format
        ax.coords[1].set_major_formatter('d.ddd')  # Latitude format

        # Add axis labelsc
        ax.set_xlabel('Carrington Longitude [deg]')
        ax.set_ylabel('Carrington Latitude [deg]')
        # Save the figure

        self.path_fig_dir = "/home/memolnar/Data/PINN-ME/datasets/spherical/test_case/"
        plot_name = f"CarrMap_{self.map_name}_{self.parameter_extension}_.png"

        file_path = os.path.join(self.path_fig_dir, plot_name)

        plt.savefig(file_path, dpi=300)


    def plot_reprojection(self, observer_longitude,
                          observer_latitude,
                          observer_radius = 1 * u.AU,
                          scale = [100, 100] * u.arcsec / u.pix,
                          shape_output_map = [4000, 4000]):


        # Define the observer's frame in Heliographic Stonyhurst coordinates
        observer_coord = SkyCoord(lon=observer_longitude,
                                       lat=observer_latitude,
                                       radius=observer_radius,
                                       frame=frames.HeliographicStonyhurst,
                                       obstime=self.obs_time)

        self.new_header = make_fitswcs_header(shape_output_map,
                                         observer_coord, scale=scale)

        # Reproject the Carrington map to the observer's perspective
        self.transformed_map = self.carrington_map.reproject_to(self.new_header)
        # Plot the resulting map
        fig = plt.figure()
        ax = plt.subplot(projection=self.transformed_map)
        im = self.transformed_map.plot(axes=ax)
        plt.colorbar(im, ax=ax)
        plot_name = f"CarrMap_{self.map_name}_{self.parameter_extension}_{observer_longitude}_{observer_latitude}.png"
        file_path = os.path.join(self.path_fig_dir, plot_name)
        plt.savefig(file_path, dpi=300)

        plt.clf()

if __name__ == '__main__':
    # resulting image size
    output_dir = "/home/memolnar/Data/PINN-ME/datasets/spherical/test_case/"
    nx_image = 100
    ny_image = 100
    time_string = "2010-01-01T00:00:00"

    # Define observer time and xyz
    observer_lon = 0 * u.deg
    observer_lat = 0 * u.deg
    observer_distance = 1 * u.AU
    obs_time = Time(time_string)
    observer_pAng = 0 * u.deg # Corresponding to CROTA angle for the observer

    sc = SkyCoord(observer_lon, observer_lat, observer_distance,
                  obstime=obs_time, observer="self", frame="heliographic_carrington")

    sun_radius = sun.angular_radius(obs_time)
    print(f"the sun radius in arcsed: {sun_radius}")

    # resulting image range on the sun in arcseconds
    # in notation [x0 x1 y0 y1] in terms of left bottom / right top x/y coordinates

    image_size = [2000 * u.arcsec, 2000 * u.arcsec]
    plate_scale = [image_size[0] / nx_image, image_size[1] / ny_image]

    x0 = -1000 * u.arcsec
    x1 = 1000 * u.arcsec
    y0 = -1000 * u.arcsec
    y1 = 1000 * u.arcsec

    shift_lon = -30 / 180 * 3.1415

    x_coords_image_plane = np.linspace(x0, x1, num=nx_image)
    y_coords_image_plane = np.linspace(y0, y1, num=ny_image)
    xy_meshgrid_image_plane = np.meshgrid(x_coords_image_plane,
                                          y_coords_image_plane)

    # Compute if a pixel is on the sun or not
    mask_image = np.sqrt((xy_meshgrid_image_plane[0])**2
                         + (xy_meshgrid_image_plane[1])**2) < sun_radius
    plt.imshow(mask_image)
    plt.savefig(os.path.join(output_dir, "carrington_mask.png"))

    test_data_file  = "/home/memolnar/Data/PINN-ME/datasets/spherical/test_case/parameters_000.npz"
    carrington_map_test_case = Carrington_map(test_dataset=test_data_file,
                                              parameter_extension="b_field",
                                              map_name="b_field")

    keys_test_data = [el for el in np.load(test_data_file).keys()]
    params_dict = {}
    parameters_interpolators = {}

    for el in keys_test_data:
        parameter_array = np.load(test_data_file)[el]

        nx_testset = parameter_array.shape[0]
        ny_testset = parameter_array.shape[1]

        # define the spherical coordinate system for the data
        lat_array = np.linspace(-np.pi/2 + shift_lon, np.pi/2 + shift_lon, num=nx_testset) * u.rad
        lon_array = np.linspace(-np.pi, np.pi, num=ny_testset) * u.rad
        params_dict[el] = parameter_array
        parameters_interpolators[el] = RectBivariateSpline(lon_array, lat_array, parameter_array)

    ## Compute the coordinates of the sampled points in the space of the dataset
    params_transformed = {}

    for xx in tqdm(range(mask_image.shape[0])):
        for yy in range(mask_image.shape[1]):
            if mask_image[xx, yy] == False:
                params_transformed[f'{xx}_{yy}'] = None
                continue

            params_transformed[f'{xx}_{yy}'] = observed_params(xy_meshgrid_image_plane[0][xx, yy],
                                                                 xy_meshgrid_image_plane[1][xx, yy],
                                                                 observer_lon, observer_lat, observer_pAng,
                                                                 parameters_interpolators, sun_radius=sun_radius)


    Bx = make_map_from_dicts(params_transformed, 'Bx', (nx_image, ny_image))
    By = make_map_from_dicts(params_transformed, 'By', (nx_image, ny_image))
    Bz = make_map_from_dicts(params_transformed, 'Bz', (nx_image, ny_image))
    b0 = make_map_from_dicts(params_transformed, 'b0', (nx_image, ny_image))
    mu = make_map_from_dicts(params_transformed, 'mu', (nx_image, ny_image))


    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    plt.clf()
    im1 = plt.imshow(Bz.T, cmap="seismic")
    plt.colorbar(im1)
    plt.savefig(os.path.join(output_dir, f"Bz_map_{shift_lon:0.3f}_deg_lon_shift.png"))
    plt.clf()
    im1 = plt.imshow(b0.T, cmap="plasma")
    plt.colorbar(im1)
    plt.savefig(os.path.join(output_dir, f"b0_{shift_lon:0.3f}_deg_lon_shift.png"))
    plt.clf()
    im1 = plt.imshow(b0.T, cmap="plasma")
    plt.colorbar(im1)
    plt.savefig(os.path.join(output_dir, "b0_{shift_lon:0.3f}_deg_lon_shift.png"))
    plt.clf()
    im1 = plt.imshow(Bx.T, cmap="seismic")
    plt.colorbar(im1)
    plt.savefig(os.path.join(output_dir, "Bx_map_{shift_lon:0.3f}_deg_lon_shift.png"))
    plt.clf()
    im1 = plt.imshow(By.T, cmap="seismic")
    plt.colorbar(im1)
    plt.savefig(os.path.join(output_dir, "By_map_{shift_lon:0.3f}_deg_lon_shift.png"))
    plt.clf()
    im1 = plt.imshow(Btot.T, cmap="seismic")
    plt.colorbar(im1)
    plt.savefig(os.path.join(output_dir, f"B_map_{shift_lon:0.3f}_deg_lon_shift.png"))

    synthesizer = Synthesizer()
    spectra = {}
    for xx in tqdm(range(mask_image.shape[0])):
        for yy in range(mask_image.shape[1]):
            if mask_image[xx, yy] == False or np.isnan(params_transformed[f'{xx}_{yy}']['b1']):
                spectra[f'{xx}_{yy}'] = synthesizer.synthesize(None)
                continue

            spectra[f'{xx}_{yy}'] = synthesizer.synthesize(params_transformed[f'{xx}_{yy}'])

    ## Create the fits files with headers compatible with the observations
    central_image_pixel_x = nx_image // 2
    central_image_pixel_y = ny_image // 2

    spectra_array = make_map_from_dicts(spectra, 'stokes_spectra',
                                       (nx_image, ny_image, 4, synthesizer.n_lambda))

    atmos_keys = spectra['0_0'].keys()
    atmos_keys = list(atmos_keys)[1:]

    num_atmos_parameters = 10
    atmos_params = np.zeros((nx_image, ny_image, num_atmos_parameters))

    for el in range(num_atmos_parameters):
        atmos_params[:, :, el] = make_map_from_dicts(spectra, atmos_keys[el],
                                                     (nx_image, ny_image))

    header = fits.Header()


    filename = "blah.fits"
    filename = os.path.join(output_dir, filename)
    # Create PrimaryHDU for the primary array (array1)
    hdu1 = fits.PrimaryHDU(data=atmos_params)

    # Create ImageHDU for the secondary array (array2)
    hdu2 = fits.ImageHDU(data=spectra_array)

    # Create header for the primary array
    hdu1.header['TITLE'] = 'Primary Array'
    hdu1.header['DIM'] = 3

    # Create header for the secondary array
    hdu2.header['TITLE'] = 'COMPRESSED IMAGE'
    hdu2.header['DIM'] = 4
    hdu2.header['CONTENT'] = 'Simulated Test case Data for PINNME'
    hdu2.header['TELESCOP'] = 'Momos computer'
    hdu2.header['DATE-OBS'] = time_string
    hdu2.header['WAVELNTH'] = 171
    hdu2.header['WAVEUNIT'] = 'angstrom'
    hdu2.header['CDELT1'] = plate_scale[0].value
    hdu2.header['CDELT2'] = plate_scale[1].value
    hdu2.header['CRPIX1'] = central_image_pixel_x
    hdu2.header['CRPIX2'] = central_image_pixel_y
    hdu2.header['CRVAL1'] = 0.0
    hdu2.header['CRVAL2'] = 0.0
    hdu2.header['CTYPE1'] = 'HPLN-TAN'
    hdu2.header['CTYPE2'] = 'HPLT-TAN'
    hdu2.header['CROTA2'] = 0
    hdu2.header['CUNIT1'] = 'arcsec'
    hdu2.header['CUNIT2'] = 'arcsec'
    hdu2.header['CRLT_OBS'] = 0
    hdu2.header['CRLN_OBS'] = 0

    # Combine HDUs into an HDUList
    hdulist = fits.HDUList([hdu1, hdu2])

    # Write to a FITS file
    hdulist.writeto(filename, overwrite=True)
    print(f"FITS file {filename} written successfully.")
