#!/bin/bash -l

#PBS -N prom3theus-hmi
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

test "$#" -eq 0 || exit 2
cd /glade/u/home/rjarolim/projects/lte

# Dynamic HMI inversion with AIA EUV rendering.

PYTHONPATH=src python -m prom3theus.cli.main invert \
  configs/sdo/hmi_aia_dynamic.yaml
