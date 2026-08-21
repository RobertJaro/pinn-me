#!/bin/bash -l

#PBS -N pinn-me
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00


cd /glade/u/home/rjarolim/projects/lte

#python -m pme.lte.fetch_data   --output-dir /glade/work/rjarolim/data/inversion/hinode_sp/resources

python -m pme.inversion_lte --config "config/hinode/lte_full_resolution.yaml"
