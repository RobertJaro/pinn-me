#!/bin/bash -l

#PBS -N pinn-me
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=16:ngpus=4:mem=400gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

python3 -m pme.inversion_spherical --config config/hmi/hmi_202405.yaml


########################################################################################
# load carrington map
#python3 -i -m pme.evaluation.spherical.load_carrington_map --input "/glade/work/rjarolim/pinn_me/hmi/202405_12h_v04/inversion.pme" --output "/glade/work/rjarolim/pinn_me/hmi/202405_12h_v04"
#python3 -i -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/pinn_me/hmi/202405_v09/inversion.pme" --output "/glade/work/rjarolim/pinn_me/hmi/202405_v09/evaluation" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits"
#python3 -i -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/pinn_me/hmi/201102_v09/inversion.pme" --output "/glade/work/rjarolim/pinn_me/hmi/201102_v09/evaluation" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.disambig.fits"
