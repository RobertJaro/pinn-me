import argparse

import numpy as np

from pme.evaluation.loader import PINNMEOutput

parser = argparse.ArgumentParser(description='Create a video from a PINN ME file')
parser.add_argument('--input', type=str, help='the path to the input file')
parser.add_argument('--output', type=str, help='the path to the output file', default=None)
args = parser.parse_args()

output = args.output if args.output else args.input.replace('.pme', '.npz')

# load
pinnme = PINNMEOutput(args.input)
parameter_cube = pinnme.load_cube()

# save
np.savez(output, **parameter_cube)
