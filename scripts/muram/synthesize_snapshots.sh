#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

PYTHONPATH=src python -m prom3theus.muram.cli synthesize \
  --source-path /glade/derecho/scratch/rjarolim/muram_snapshot \
  --out /glade/derecho/scratch/rjarolim/muram_synthetic_observations/hinode_single_frame \
  --iterations 0 --output-format store \
  --instrument-config configs/muram/hinode/synthesis.yaml
