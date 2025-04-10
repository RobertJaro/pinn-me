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

#python3 -m pme.convert.pme_to_npz --input "/glade/work/rjarolim/pinn_me/hinode/20070105_psf_v01/inversion.pme" --output "/glade/campaign/hao/radmhd/rjarolim/PINN-ME/hinode/20070105_psf_v01.npz"
#python3 -m pme.convert.pme_to_npz --input "/glade/work/rjarolim/pinn_me/hinode/20070105_v01/inversion.pme" --output "/glade/campaign/hao/radmhd/rjarolim/PINN-ME/hinode/20070105_v01.npz"

#################################################################################
# plot performance as a function of noise

#python3 -i -m pme.evaluation.hinode.compare_hinode --input '/glade/campaign/hao/radmhd/rjarolim/PINN-ME/hinode' --output '/glade/work/rjarolim/pinn_me/hinode/evaluation'

#python3 -i -m pme.evaluation.hinode.load_time_series --input '/glade/work/rjarolim/pinn_me/hinode/20240509_v04/inversion.pme' --output '/glade/work/rjarolim/pinn_me/hinode/20240509_v04/evaluation'

