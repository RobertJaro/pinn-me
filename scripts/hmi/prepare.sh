#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

PYTHONPATH=src python -m prom3theus.cli.main prepare hmi-subframes data/hmi/2024-03-23/full_disk \
  --output data/hmi/2024-03-23/subframe \
  --longitude-deg 215 \
  --latitude-deg -12 \
  --width-pixels 1024 \
  --height-pixels 512

PYTHONPATH=src python -m prom3theus.cli.main prepare hmi-responses /glade/work/rjarolim/data/hmi_stokes/20240323_720s_subframe \
  --output /glade/work/rjarolim/data/hmi_calibration/20240323 \
  --email "robert.jarolim@uni-graz.at"
