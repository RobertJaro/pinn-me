#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

PYTHONPATH=src python -m prom3theus.muram.cli synthesize \
  --source-path /glade/derecho/scratch/rjarolim/muram_snapshot \
  --out /glade/derecho/scratch/rjarolim/muram_synthetic_observations/sdo \
  --iterations 0 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
  --output-format sdo-fits
