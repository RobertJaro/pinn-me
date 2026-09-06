#!/bin/bash -l

#PBS -N prom3theus-hmi
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

PYTHONPATH=src python -m prom3theus.cli.main invert configs/hmi_lte_dynamic.yaml


PYTHONPATH=src python -m prom3theus.cli.main compare-hmi /glade/work/rjarolim/lte/hmi_subframe_dynamic_extrapolation_v07/state.p3s \
  /glade/work/rjarolim/data/hmi_stokes/test_2024_03_24 \
  --output /glade/work/rjarolim/lte/hmi_comparison
