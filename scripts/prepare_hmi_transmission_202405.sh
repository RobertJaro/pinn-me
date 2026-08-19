#!/bin/bash -l
set -euo pipefail

cd /glade/u/home/rjarolim/projects/PINN-ME

python3 -m pme.data.hmi_transmission \
  --input "/glade/work/rjarolim/data/hmi_stokes/202405_12h" \
  --email "robert.jarolim@uni-graz.at" \
  --output "/glade/work/rjarolim/data/hmi_calibration/202405"


python3 -m pme.data.hmi_transmission \
  --input "/glade/work/rjarolim/data/hmi_stokes/20240323_720s_subframe" \
  --email "robert.jarolim@uni-graz.at" \
  --output "/glade/work/rjarolim/data/hmi_calibration/20240323"
