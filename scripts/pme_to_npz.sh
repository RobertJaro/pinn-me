#!/bin/bash -l

#PBS -N inversion
#PBS -A P22100000
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -l walltime=06:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME
python3 -m pme.convert.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_data/test_psf/inversion.pme
