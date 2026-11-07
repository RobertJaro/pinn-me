#!/bin/bash -l

#PBS -N prom3theus-hinode-stokes
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

set -euo pipefail
test "$#" -eq 0
cd /glade/u/home/rjarolim/projects/lte

# First run scripts/muram_hinode/synthesize.sh once.
# Automatic GPU selection launches Lightning DDP before data/model setup.
PYTHONPATH=src python -m prom3theus.muram.cli invert configs/muram/hinode/stokes_only.yaml
