#!/bin/bash -l

#PBS -N pinn-me
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

#python3 -m pme.inversion_spherical --config config/hmi/combined_fd_20240327.yaml

#python3 -m pme.inversion_spherical --config config/hmi/hmi_20240327.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_12h.yaml
python3 -m pme.inversion_spherical --config config/hmi/hmi_202405.yaml

#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_physics.yaml
#python3 -m pme.inversion_spherical --config config/hmi/phi_fdt_202403.yaml
#python3 -m pme.inversion_spherical --config config/hmi/phi_202404.yaml
python3 -m pme.inversion_spherical --config config/hmi/hmi_subframe.yaml
# finetune

#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_90s.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_90s_no_physics.yaml
#python3 -m pme.inversion_spherical --config config/hmi/solar_cycle.yaml

#python3 -m pme.inversion_spherical --config config/hmi/phi_fdt_202403.yaml
exit

# preprocess HMI subframe data
#python3 -m pme.data.subframe_hmi --input_files '/glade/work/rjarolim/data/hmi_stokes/202405_90s/*.fits' --out_path '/glade/work/rjarolim/data/phi_stokes/subframe' --longitude 347 --latitude -20 --size 1024 512 --overwrite
python3 -m pme.data.subframe_hmi --input_files '/glade/work/rjarolim/data/hmi_stokes/202405/*.fits' --out_path '/glade/work/rjarolim/data/hmi_stokes/202405_subframe' --longitude 347 --latitude -20 --size 1024 512 --overwrite



#python3 -m pme.inversion_spherical --config config/hmi/hmi_subframe_no_physics.yaml

#python3 -m pme.inversion_spherical --config config/hmi/hmi_202404.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_90s_no_physics.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_201102_3h_no_physics.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_90s.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_90s_VP.yaml


#python3 -m pme.inversion_spherical --config config/hmi/hmi_201102_3h.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_201102_3h_VP.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_201102_3h_no_physics.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_physics.yaml

#python3 -m pme.inversion_spherical --config config/hmi/hmi_201102_6h.yaml

#python3 -m pme.inversion_spherical --config config/hmi/combined_202404.yaml
#python3 -m pme.inversion_spherical --config config/hmi/phi_202404.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202404.yaml

#python3 -m pme.inversion_spherical --config config/hmi/phi_fdt_202403.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_fd_20240327.yaml
#python3 -m pme.inversion_spherical --config config/hmi/combined_fd_202403.yaml
#python3 -m pme.inversion_spherical --config config/hmi/combined_fd_20240327.yaml
#python3 -m pme.inversion_spherical --config config/hmi/physics_fd_20240327.yaml


########################################################################################
# load carrington map
#python3 -i -m pme.evaluation.spherical.load_carrington_map --input "/glade/work/rjarolim/pinn_me/hmi/202405_v17/inversion.pme"
#python3 -i -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_v17/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits"
#python3 -i -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_v17/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.disambig.fits"

# comparison to HMI
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v05/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.vlos_mag.fits"
# series
#python3 -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/inversion.pme" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/202405/*.I0.fits"

#python3 -i -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/evaluation" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits"
#python3 -i -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/evaluation/series" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/202405_12h/*.I0.fits"


# series
#python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/201102_no_physics_v03/inversion.pme" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/201102_3h/*.I0.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/201102_no_physics_v03/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.disambig.fits"
#python3 -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/201102_no_physics_v03/inversion.pme"

#
#python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/solar_cycle_v01/inversion.pme" --ref_maps "/glade/campaign/hao/radmhd/rjarolim/data_stash/solar_cycle/*.I0.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_v20/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.vlos_mag.fits" --hpc_range 0 400 -400 0
#python3 -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/201102_physics_v02/inversion.pme"


#python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_v20/inversion.pme" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/202405/*.I0.fits"
#python3 -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_v09/inversion.pme"

# 90s series
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_90s_v14/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.vlos_mag.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_90s_v14/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_90s_v14/evaluation/subframe/" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.vlos_mag.fits" --hpc_range 100 600 -500 0


#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_90s_no_physics_v17/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.vlos_mag.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_90s_no_physics_v17/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_90s_no_physics_v17/evaluation/subframe/" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.vlos_mag.fits" --hpc_range 100 600 -500 0
#python3 -m pme.evaluation.spherical.load_carrington_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_90s_no_physics_v17/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_90s_no_physics_v17/evaluation"
#python3 -m pme.evaluation.spherical.polar_field --input "/glade/work/rjarolim/spinn_me/hmi/202405_90s_no_physics_v17/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.vlos_mag.fits"


#python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v30/inversion.pme" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/202405_12h/*.I0.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v31/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.vlos_mag.fits" --hpc_range 0 500 -500 0
#python3 -m pme.evaluation.spherical.load_carrington_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v16/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v16/evaluation"
#python3 -m pme.evaluation.spherical.polar_field --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v16/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.vlos_mag.fits"

#python3 -m pme.evaluation.spherical.load_carrington_map --input "/glade/work/rjarolim/spinn_me/hmi/solar_cycle_v01/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/solar_cycle_v01/evaluation"
#python3 -m pme.evaluation.spherical.load_carrington_map --input "/glade/work/rjarolim/spinn_me/combined/combined_fd_20240327_v04/inversion.pme" --output "/glade/work/rjarolim/spinn_me/combined/combined_fd_20240327_v04/evaluation"


#python3 -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_90s_no_physics_v02/inversion.pme" --latitude_range -30 -10 --longitude_range 330 370 --resolution 0.1


# PHI full disk
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/phi_fdt_202403_v07/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.vlos_mag.fits" --hpc_range 400 800 -300 0
#python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/phi_fdt_202403_v07/inversion.pme" --ref_maps "/glade/work/rjarolim/data/phi_fdt/2024-03-29/*.fits.gz"

# combined
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/combined/combined_fd_20240327_v03/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.vlos_mag.fits" --hpc_range 400 800 -300 0
#python3 -m pme.evaluation.spherical.load_carrington_map --input "/glade/work/rjarolim/spinn_me/combined/combined_fd_20240327_v03/inversion.pme" --output "/glade/work/rjarolim/spinn_me/combined/combined_fd_20240327_v03/evaluation"
#python3 -m pme.evaluation.spherical.polar_field --input "/glade/work/rjarolim/spinn_me/combined/combined_fd_20240327_v03/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.vlos_mag.fits"

# HMI full disk
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/hmi_20240327_v21/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.vlos_mag.fits" --hpc_range 400 900 -300 100
python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/hmi_20240327_v21/inversion.pme" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/20240327_3h/*.I0.fits"


# cutouts
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/subframe_v13/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.vlos_mag.fits" --hpc_range 100 600 -500 0
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/subframe_no_physics_v01/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.vlos_mag.fits" --hpc_range -200 200 -200 100
#python3 -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/subframe_v08/inversion.pme" --longitude_range 90 140 --latitude_range -30 0 --resolution 0.1
#python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/subframe_v08/inversion.pme" --ref_maps "/glade/work/rjarolim/data/phi_stokes/subframe/*.I0.fits"


python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/subframe_v32/inversion.pme" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/202405_subframe/*.I0.fits"
python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/subframe_v32/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.vlos_mag.fits" --hpc_range -200 400 -500 0
python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/subframe_v32/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.vlos_mag.fits" --hpc_range 0 200 -350 -250


#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/combined_fd_202403_v01/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_15/hmi.b_720s.20240315_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_15/hmi.b_720s.20240315_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_15/hmi.b_720s.20240315_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_15/hmi.b_720s.20240315_060000_TAI.disambig.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/combined_fd_20240327_v01/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.vlos_mag.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/physics_fd_20240327_v03/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits"


python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/combined/combined_fd_20240327_v01/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.vlos_mag.fits"





# PHI
python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/phi_202403_v07/inversion.pme" --ref_maps "/glade/work/rjarolim/data/phi_stokes/2024_04_flare/*.fits"
python3 -m pme.evaluation.phi.visualize_stokes

python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/combined/combined_fd_20240327_v01/inversion.pme" --ref_maps "/glade/work/rjarolim/data/phi_stokes/2024_04_flare/*.fits"


python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_v28/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits" --ref_map_vlos_mag "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.vlos_mag.fits" --hpc_range 0 200 -300 -200
