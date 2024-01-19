#!/bin/bash -l

#PBS -N global
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=02:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME
python3 -i -m pme.inversion --config config/test.json
