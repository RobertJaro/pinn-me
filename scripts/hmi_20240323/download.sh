#!/bin/bash -l

#PBS -N hmi-download
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=08:00:00

set -euo pipefail
cd /glade/u/home/rjarolim/projects/PINN-ME

# March 23, 01:00–03:00 TAI, end excluded. JSOC interprets naive dates as TAI.
# One sequential export per product for the full two-hour interval.

python3 -m pme.data.download_hmi \
  --download_dir "/glade/work/rjarolim/data/hmi_stokes/20240323_0100_0300_90s" \
  --email "robert.jarolim@uni-graz.at" \
  --t_start "2024-03-23T01:00:00" --t_end "2024-03-23T03:00:00" \
  --series "S_90s" --cadence "90s"

python3 -m pme.data.download_hmi \
  --download_dir "/glade/work/rjarolim/data/hmi_stokes/20240323_0100_0300_720s" \
  --email "robert.jarolim@uni-graz.at" \
  --t_start "2024-03-23T01:00:00" --t_end "2024-03-23T03:00:00" \
  --series "S_720s" --cadence "720s"

python3 -m pme.data.download_hmi \
  --download_dir "/glade/work/rjarolim/data/hmi_stokes/20240323_0100_0300_reference_90s" \
  --email "robert.jarolim@uni-graz.at" \
  --t_start "2024-03-23T01:00:00" --t_end "2024-03-23T03:00:00" \
  --series "B_90s" --cadence "90s" \
  --segments "field,inclination,azimuth,disambig,vlos_mag"

python3 -m pme.data.download_hmi \
  --download_dir "/glade/work/rjarolim/data/hmi_stokes/20240323_0100_0300_reference_720s" \
  --email "robert.jarolim@uni-graz.at" \
  --t_start "2024-03-23T01:00:00" --t_end "2024-03-23T03:00:00" \
  --series "B_720s" --cadence "720s" \
  --segments "field,inclination,azimuth,disambig,vlos_mag"

# Download phase maps and prepare offline transmission profiles for both cadences.
python3 -m pme.data.hmi_transmission \
  --input "/glade/work/rjarolim/data/hmi_stokes/20240323_0100_0300_90s" \
          "/glade/work/rjarolim/data/hmi_stokes/20240323_0100_0300_720s" \
  --email "robert.jarolim@uni-graz.at" \
  --output "/glade/work/rjarolim/data/hmi_calibration/20240323_0100_0300"
