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
# create data set
#python3 -m pme.data.create_spherical_test_set --out_path "/glade/campaign/hao/radmhd/rjarolim/SPINN-ME/test_set/simple_obs_000" --n_time_steps 100 --obs_lon 0.0
#python3 -m pme.data.create_spherical_test_set --out_path "/glade/campaign/hao/radmhd/rjarolim/SPINN-ME/test_set/simple_obs_060" --n_time_steps 100 --obs_lon 60.0
#python3 -m pme.data.create_spherical_test_set --out_path "/glade/campaign/hao/radmhd/rjarolim/SPINN-ME/test_set/simple_obs_120" --n_time_steps 100 --obs_lon 120.0

python3 -m pme.data.create_spheromak_test_set --out_path "/glade/campaign/hao/radmhd/rjarolim/SPINN-ME/spheromak/simple_obs_120" --n_time_steps 100 --obs_lon 120.0


#################################################################################
# inversion
python3 -m pme.inversion_spherical --config config/spherical/2_obs.yaml

# test transformation
#python3 -m pme.data.test_B_transform
#python3 -i -m pme.evaluation.spherical.compare_test_set --input "/glade/work/rjarolim/spinn_me/3_obs_v01/inversion.pme" --output "/glade/work/rjarolim/spinn_me/3_obs_v01/evaluation" --ref_maps "/glade/campaign/hao/radmhd/rjarolim/SPINN-ME/test_set/data/*_I0.fits"
