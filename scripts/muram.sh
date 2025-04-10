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
# prepare data

#python3 -m pme.data.muram_sunspot_prep --file '/glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram/data/sunspot_jmb_sir_synth_profiles.npy' --wl_file '/glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram/data/jmb_wav.npy' --psf_file "/glade/campaign/hao/radmhd/rjarolim/PINN-ME/hinode_psf_0.16.fits" --out_file '/glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram/data/SIRprofiles_sunspot_v2.fits'

#################################################################################
# train

#python3 -m pme.inversion --config config/muram/muram_sunspot.yaml
python3 -m pme.inversion --config config/muram/muram_sunspot_psf.yaml

#################################################################################
# convert to npz files

python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/muram/muram_sunspot_v04/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram/muram_sunspot_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/muram/muram_sunspot_psf_v05/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram/muram_sunspot_psf_v02.npz

#################################################################################
# plot examples

python3 -m pme.evaluation.muram.compare_muram --input '/glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram' --output '/glade/work/rjarolim/pinn_me/muram/evaluation'


#################################################################################
# evaluate uncertainty

#python3 -i -m pme.evaluation.compute_uncertainty --input "/glade/work/rjarolim/pinn_me/muram/muram_sunspot_v04/inversion.pme" --output "/glade/work/rjarolim/pinn_me/muram/muram_sunspot_v04" --ref_stokes "/glade/campaign/hao/radmhd/rjarolim/PINN-ME/muram/data/SIRprofiles_sunspot.fits"
