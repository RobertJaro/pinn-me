#!/bin/bash -l

#PBS -N pinn-me
#PBS -A P22100000
#PBS -q casper
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

#################################################################################
# train

#python3 -m pme.inversion --config config/muram/muram_sunspot.yaml
python3 -m pme.inversion --config config/muram/muram_sunspot_psf.yaml

#################################################################################
# convert to npz files

python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/muram/muram_sunspot_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram/muram_sunspot_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/muram/muram_sunspot_psf_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram/muram_sunspot_psf_v01.npz

#################################################################################
# evaluate uncertainty

#python3 -i -m pme.evaluation.compute_uncertainty --input "/glade/work/rjarolim/pinn_me/muram/muram_sunspot_v01/inversion.pme" --output "/glade/work/rjarolim/pinn_me/muram/muram_sunspot_v01" --ref_stokes "/glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram/data/SIRprofiles_sunspot.fits"
