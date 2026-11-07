#!/bin/bash -l

#PBS -N prom3theus-muram
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

test "$#" -eq 0 || exit 2
cd /glade/u/home/rjarolim/projects/lte


# 1. scripts/muram/unpack_snapshots.sh   (unpacks the MURaM 3D snapshots)
# 2. scripts/muram/synthesize_snapshots.sh   (forward-renders the synthetic
#    HMI+AIA FITS and prepared inputs configs/muram/sdo/joint.yaml
#    points at)
# 3. this script, which starts the inversion against that synthetic data.
PYTHONPATH=src python -m prom3theus.muram.cli invert configs/muram/sdo/joint.yaml
# For the isolated Stokes control, run this command instead of the joint run:
#PYTHONPATH=src python -m prom3theus.muram.cli invert configs/muram/sdo/stokes_only.yaml

PYTHONPATH=src python -m prom3theus.muram.cli invert configs/muram/sdo/mhs.yaml
