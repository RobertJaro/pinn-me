#!/bin/bash -l

#PBS -N pinn-me
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

#################################################################################
# train

python3 -m pme.inversion --config config/hinode/2024_05_09.yaml
#python3 -m pme.inversion --config config/hinode/hinode.yaml
#python3 -m pme.inversion --config config/hinode/hinode_psf.yaml

#################################################################################
# convert to npz files

#python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/clear_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/clear_v01.npz

#################################################################################
# plot performance as a function of noise
