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
#python3 -m pme.evaluation.video --input /glade/work/rjarolim/pinn_me/test_data/test_psf_noise_1e-3/inversion.pme --output /glade/work/rjarolim/pinn_me/test_data/test_psf_noise_1e-3 --reference /glade/work/rjarolim/data/inversion/parameters_MEset_v4_nt_100_spatial_400_400_withPSF_11x11_s_4.npz
#python3 -i -m pme.evaluation.parameters --input /glade/work/rjarolim/pinn_me/test_data/test_psf_noise_1e-3/inversion.pme --output /glade/work/rjarolim/pinn_me/test_data/test_psf_noise_1e-3/parameters.jpg --reference /glade/work/rjarolim/data/inversion/parameters_MEset_v4_nt_100_spatial_400_400_withPSF_11x11_s_4.npz
python3 -i -m pme.evaluation.noise --input /glade/work/rjarolim/pinn_me/test_data/test_psf_noise_ --output /glade/work/rjarolim/pinn_me/test_data/evaluation --reference /glade/work/rjarolim/data/inversion/parameters_MEset_v4_nt_100_spatial_400_400_withPSF_11x11_s_4.npz
