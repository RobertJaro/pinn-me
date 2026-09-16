#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

# Threads share calibration tables; lower this if full-disk images exhaust RAM.
export PROM3THEUS_PREP_WORKERS="${PROM3THEUS_PREP_WORKERS:-16}"

data_directory="/glade/work/rjarolim/data/prom3theus/2011_02"
email="robert.jarolim@uni-graz.at"
# Shared subframe center in heliographic Carrington coordinates.
longitude_deg=35.3
latitude_deg=-20
# Approximate common size in HMI pixels; AIA keeps its native sampling.
width_pixels=500
height_pixels=500

PYTHONPATH=src python -m prom3theus.cli.main prepare hmi "${data_directory}/hmi/full_disk" \
  --output "${data_directory}/hmi/prepared" \
  --longitude-deg "${longitude_deg}" \
  --latitude-deg "${latitude_deg}" \
  --width-pixels "${width_pixels}" \
  --height-pixels "${height_pixels}"

# Response resources are needed by the inversion, not by image cropping.
PYTHONPATH=src python -m prom3theus.cli.main prepare hmi-responses \
  "${data_directory}/hmi/prepared" \
  --output "${data_directory}/hmi/responses" --email "${email}"

PYTHONPATH=src python -m prom3theus.cli.main prepare aia "${data_directory}/aia/level1" \
  --output "${data_directory}/aia/prepared" \
  --calibration-directory "${data_directory}/aia/calibration" \
  --longitude-deg "${longitude_deg}" \
  --latitude-deg "${latitude_deg}" \
  --width-pixels "${width_pixels}" \
  --height-pixels "${height_pixels}"
