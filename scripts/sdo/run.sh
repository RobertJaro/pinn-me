#!/bin/bash -l

#PBS -N prom3theus-hmi
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

cd /glade/u/home/rjarolim/projects/lte

#PYTHONPATH=src python -m prom3theus.cli.main invert \
#  configs/hmi_aia_dynamic.yaml
#
#
#PYTHONPATH=src python -m prom3theus.cli.main invert \
#  configs/hmi_aia_mhs.yaml

PYTHONPATH=src python -m prom3theus.cli.main invert \
  configs/hmi_aia_potential.yaml


PYTHONPATH=src python -m prom3theus.cli.main invert \
  configs/hmi_aia_vector_potential.yaml