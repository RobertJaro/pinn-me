#!/bin/bash -l

#PBS -N pinn-me
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

#python3 -m pme.inversion_spherical --config config/hmi/hmi_201102_3h.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_201102_3h_no_physics.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_physics.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_90s.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202405_12h.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_201102_6h.yaml

#python3 -m pme.inversion_spherical --config config/hmi/combined_202404.yaml
#python3 -m pme.inversion_spherical --config config/hmi/phi_202404.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_202404.yaml

#python3 -m pme.inversion_spherical --config config/hmi/phi_fdt_202403.yaml
#python3 -m pme.inversion_spherical --config config/hmi/hmi_fd_20240327.yaml
#python3 -m pme.inversion_spherical --config config/hmi/combined_fd_202403.yaml
python3 -m pme.inversion_spherical --config config/hmi/combined_fd_20240327.yaml
#python3 -m pme.inversion_spherical --config config/hmi/physics_fd_20240327.yaml


########################################################################################
# load carrington map
#python3 -i -m pme.evaluation.spherical.load_carrington_map --input "/glade/work/rjarolim/pinn_me/hmi/202405_12h_v04/inversion.pme" --output "/glade/work/rjarolim/pinn_me/hmi/202405_12h_v04"
#python3 -i -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_v08/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_v08/evaluation" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits"
#python3 -i -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_v07/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_v07/evaluation" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.disambig.fits"

# comparison to HMI
#python3 -i -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/evaluation" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits"
# series
#python3 -i -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/evaluation/series" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/202405/*.I0.fits"

#python3 -i -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/evaluation" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits"
#python3 -i -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/inversion.pme" --output "/glade/work/rjarolim/spinn_me/hmi/202405_12h_v01/evaluation/series" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/202405_12h/*.I0.fits"


# series
#python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/201102_physics_v43/inversion.pme" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/201102_3h/*.I0.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/201102_physics_v43/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/201102_test/hmi.b_720s.20110215_000000_TAI.disambig.fits"
#python3 -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/201102_physics_v43/inversion.pme"


#python3 -m pme.evaluation.spherical.load_ref_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_v03/inversion.pme" --ref_maps "/glade/work/rjarolim/data/hmi_stokes/202405/*.I0.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_v03/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2/hmi.b_720s.20240508_000000_TAI.disambig.fits"
#python3 -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/202405_v01/inversion.pme"

#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/202405_90s_v05/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_05_09/hmi.b_720s.20240509_010000_TAI.disambig.fits"

#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/hmi_202403_v02/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.disambig.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/phi_202403_v01/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24/hmi.b_720s.20240324_010000_TAI.disambig.fits"


#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/combined_fd_202403_v01/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_15/hmi.b_720s.20240315_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_15/hmi.b_720s.20240315_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_15/hmi.b_720s.20240315_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_15/hmi.b_720s.20240315_060000_TAI.disambig.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/combined_fd_20240327_v05/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits"
#python3 -m pme.evaluation.spherical.load_ref_map --input "/glade/work/rjarolim/spinn_me/hmi/physics_fd_20240327_v03/inversion.pme" --ref_map_fld "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.field.fits" --ref_map_inc "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.inclination.fits" --ref_map_azi "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.azimuth.fits" --ref_map_disambig "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_27/hmi.b_720s.20240327_060000_TAI.disambig.fits"

#python3 -m pme.evaluation.spherical.load_series --input "/glade/work/rjarolim/spinn_me/hmi/physics_fd_20240327_v03/inversion.pme"
