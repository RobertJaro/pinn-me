#!/bin/bash -l

#PBS -N hmi-cadence
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

set -euo pipefail
cd /glade/u/home/rjarolim/projects/PINN-ME

python3 -m pme.inversion_spherical --config config/hmi/hmi_fd_20240323_2h_90s.yaml
