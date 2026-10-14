#!/bin/bash -l

#PBS -N hmi-comparison
#PBS -A P22100000
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -l walltime=06:00:00

set -euo pipefail
cd /glade/u/home/rjarolim/projects/PINN-ME
export MPLBACKEND=Agg

# One central-time 720s reference; HPC_RANGE selects the field of view.
RUN_PREFIX="/glade/work/rjarolim/spinn_me/hmi/hmi_fd_20240323_2h"
RUN_VERSION_90S="v09"
RUN_VERSION_720S="v01"
REFERENCE_PREFIX="/glade/work/rjarolim/data/hmi_stokes/20240323_0100_0300_reference_720s/hmi.b_720s.20240323_020000_TAI"
EVALUATION_SUBDIR="evaluation/reference_720s/020000"
HPC_RANGE=(-200 -100 -200 -100)

python3 -m pme.evaluation.spherical.load_ref_map \
  --input "${RUN_PREFIX}_90s_${RUN_VERSION_90S}/inversion.pme" \
  --ref_map_fld "${REFERENCE_PREFIX}.field.fits" \
  --ref_map_inc "${REFERENCE_PREFIX}.inclination.fits" \
  --ref_map_azi "${REFERENCE_PREFIX}.azimuth.fits" \
  --ref_map_disambig "${REFERENCE_PREFIX}.disambig.fits" \
  --ref_map_vlos_mag "${REFERENCE_PREFIX}.vlos_mag.fits" \
  --hpc_range "${HPC_RANGE[@]}" \
  --output "${RUN_PREFIX}_90s_${RUN_VERSION_90S}/${EVALUATION_SUBDIR}"

python3 -m pme.evaluation.spherical.load_ref_map \
  --input "${RUN_PREFIX}_720s_${RUN_VERSION_720S}/inversion.pme" \
  --ref_map_fld "${REFERENCE_PREFIX}.field.fits" \
  --ref_map_inc "${REFERENCE_PREFIX}.inclination.fits" \
  --ref_map_azi "${REFERENCE_PREFIX}.azimuth.fits" \
  --ref_map_disambig "${REFERENCE_PREFIX}.disambig.fits" \
  --ref_map_vlos_mag "${REFERENCE_PREFIX}.vlos_mag.fits" \
  --hpc_range "${HPC_RANGE[@]}" \
  --output "${RUN_PREFIX}_720s_${RUN_VERSION_720S}/${EVALUATION_SUBDIR}"
