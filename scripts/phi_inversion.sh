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
# PHI Inversions

# full disk inversions
python3 -m pme.inversion_spherical --config config/hmi/phi_fdt_202403.yaml

# high-res inversions
#python3 -m pme.inversion_spherical --config config/hmi/phi_202404.yaml

##################################
# Download data

#python3 -m pme.data.download_phi

##################################
# Evaluation

# HRT
python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/phi_202403_v07/inversion.pme" --ref_maps "/glade/work/rjarolim/data/phi_stokes/2024_04_flare/*.fits"
python3 -m pme.evaluation.phi.visualize_stokes
python3 -m pme.evaluation.spherical.load_ref_comparison --input "/glade/work/rjarolim/spinn_me/hmi/phi_202403_v07/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/phi_stokes/2024_03_23/solo_L2_phi-hrt-bmag_20240323T233008_V01_0443230203.fits" --ref_map_inc "/glade/work/rjarolim/data/phi_stokes/2024_03_23/solo_L2_phi-hrt-binc_20240323T233008_V01_0443230203.fits" --ref_map_azi "/glade/work/rjarolim/data/phi_stokes/2024_03_23/solo_L2_phi-hrt-bazi_20240323T233008_V01_0443230203.fits" --ref_map_vlos "/glade/work/rjarolim/data/phi_stokes/2024_03_23/solo_L2_phi-hrt-vlos_20240323T233008_V01_0443230203.fits"

# FDT
python3 -m pme.evaluation.spherical.load_ref_mag --input "/glade/work/rjarolim/spinn_me/hmi/phi_fdt_202403_v08/inversion.pme" --ref_map_blos "/glade/work/rjarolim/data/phi_stokes/fdt_2024_03_29/solo_L2_phi-fdt-blos_20240329T164009_V01.fits" --ref_map_bmag "/glade/work/rjarolim/data/phi_stokes/fdt_2024_03_29/solo_L2_phi-fdt-bmag_20240329T164009_V01.fits"
python3 -m pme.evaluation.spherical.load_ref_mag --input "/glade/work/rjarolim/spinn_me/hmi/phi_fdt_202403_v08/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/phi_fdt_202403_v08/evaluation/subframe" --ref_map_blos "/glade/work/rjarolim/data/phi_stokes/fdt_2024_03_29/solo_L2_phi-fdt-blos_20240329T164009_V01.fits" --ref_map_bmag "/glade/work/rjarolim/data/phi_stokes/fdt_2024_03_29/solo_L2_phi-fdt-bmag_20240329T164009_V01.fits" --hpc_range 1000 2000 -2000 1000
python3 -m pme.evaluation.spherical.load_ref_params --input "/glade/work/rjarolim/spinn_me/hmi/phi_fdt_202403_v08/inversion.pme" --ref_map "/glade/work/rjarolim/data/phi_stokes/fdt_2024_03_29/solo_L2_phi-fdt-blos_20240329T060009_V01.fits"
