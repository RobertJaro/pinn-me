import argparse
import os.path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from pme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, \
    image_to_spherical_matrix
from pme.evaluation.loader import PINNMEOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare PINN-ME magnetic field output with reference maps and generate visualization plots')
    parser.add_argument('--input', type=str, required=True, help='Path to the PINN-ME output file')
    parser.add_argument('--ref_map_fld', type=str, required=True, help='Path to the reference field strength map')
    parser.add_argument('--ref_map_inc', type=str, required=True, help='Path to the reference inclination map')
    parser.add_argument('--ref_map_azi', type=str, required=True, help='Path to the reference azimuth map')
    parser.add_argument('--ref_map_disambig', type=str, required=True, help='Path to the reference disambiguation map')
    parser.add_argument('--output', type=str, required=True, help='Path to save output visualizations')
    args = parser.parse_args()

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    ########################################################################################################################
    # Load PINN-ME output and reference maps
    ########################################################################################################################
    pinnme = PINNMEOutput(args.input)

    pinnme.times