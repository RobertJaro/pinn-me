#!/bin/bash -l

#PBS -N inversion
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=16:ngpus=4:mem=256gb
#PBS -l walltime=04:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME
python3 -i -m pme.inversion --config config/simple_noise_1e-5.json
