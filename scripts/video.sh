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
python3 -i -m pme.evaluation.video --input /glade/work/rjarolim/pinn_me/train/test_v22/inversion.pme --output /glade/work/rjarolim/pinn_me/train/test_v22
