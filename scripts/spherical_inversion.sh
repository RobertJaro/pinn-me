#!/bin/bash -l

#PBS -N inversion
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

python3 -m pme.inversion_spherical --config config/hmi/hmi_spherical.yaml
