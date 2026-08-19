#!/bin/bash -l

#PBS -N pinn-me
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null

module load cuda/12.3.2

cd /glade/u/home/rjarolim/projects/PINN-ME

##################################
# Prepare PHI data

python3 -m pme.data.align_phi


##################################
# Train PINN-ME model
python3 -m pme.inversion_spherical --config config/hmi/combined_fd_20240327.yaml


##################################
# Evaluate and plot results
python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/combined/combined_fd_20240327_v01/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.vlos_mag.fits"
